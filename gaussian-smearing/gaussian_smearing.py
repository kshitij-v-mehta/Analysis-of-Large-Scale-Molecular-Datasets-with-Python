"""
Apply Gaussian smearing on the AiSD-Ex dataset
Run this with 1 MPI process per node
"""
import glob
import os
import getpass
import shutil
import tarfile
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

from mpi4py import MPI
from mpi4py.futures import MPICommExecutor

import mpi_utils
import importlib
dftbuv2d = importlib.import_module("dftb-uv_2d")
from utils import draw_2Dmol


# Location of the original tar files
tarfiles_in = "./dataset"

# Where to store the new tar files containing the processed dataset
tarfiles_out = "./dataset_out"

# Set the location of the scratch space where tar files will be unpacked to /tmp/<username>
scratch_space_root = "/tmp/{}".format(getpass.getuser())


def test_scratch_space():
    # Ensure that you can create a directory in scratch space
    try:
        os.makedirs(scratch_space_root, exist_ok=True)
    except Exception as e:
        raise e


def get_tar_file_list():
    try:
        _tarfiles_list = glob.glob("{}/*.tar.gz".format(tarfiles_in))
        tarfiles_list = [tarf for tarf in _tarfiles_list if 'unprocessed' not in tarf]
        assert len(tarfiles_list) > 0, "No tar files found at {}".format(tarfiles_in)

        print("Found {} tar files in {}".format(len(tarfiles_list), tarfiles_in), flush=True)
        return tarfiles_list

    except Exception as e:
        raise e


def get_mol_dirs(tar_cwd):
    try:
        mol_dirs = [os.path.join(tar_cwd, mol_dir) for mol_dir in next(os.walk(tar_cwd))[1]]
        assert len(mol_dirs) > 0, "No molecule directories found in {}".format(tar_cwd)
        return mol_dirs

    except Exception as e:
        raise e


def create_new_tar(cwd, mol_dirs):
    # Tar the molecule directories again after the gaussian smearing is done

    t = None

    try:
        new_tar_f = os.path.basename(cwd) + "-gaussian-smearing.tar.gz"
        print("Creating new tar file {}".format(new_tar_f), flush=True)
        new_tar_f_scratch_loc = "{}/{}".format(scratch_space_root, new_tar_f)

        # Create the new tar file in /tmp first, and then copy it to the file system
        with tarfile.open(new_tar_f_scratch_loc, mode='x:gz') as t:
            for m in mol_dirs:
                t.add(m, arcname=os.path.basename(m))

        # Copy the tar files to the parallel file system
        new_tar_f_final_loc = "{}/{}".format(tarfiles_out, new_tar_f)
        shutil.move(new_tar_f_scratch_loc, new_tar_f_final_loc)

    except Exception as e:
        print(e)
        raise e

    finally:
        if t is not None:
            t.close()


def dftb_uv_2d(mol_dir):
    try:
        # print("Process {} with global rank {} on node {} received molecule {}"
        #       "".format(os.getpid(), MPI.COMM_WORLD.Get_rank(), MPI.Get_processor_name(), mol_dir), flush=True)

        # draw_2Dmol(MPI.COMM_SELF, mol_dir)
        dftbuv2d.smooth_spectrum(MPI.COMM_SELF, str(Path(mol_dir).parent), os.path.basename(mol_dir), None, 70.0, None, None)

        # Verify that the spectrum png file was created
        assert len(glob.glob("{}/abs_spectrum_*.png".format(mol_dir))) == 1, "spectrum png file not created"

        return None

    except Exception as e:
        print("Exception: {} for molecule {}".format(e, mol_dir), flush=True)
        print(traceback.format_exc(), flush=True)
        return mol_dir


def process_molecules_on_node(mol_dirs):
    # Distribute molecules from a tar file amongst all processes on the compute node

    try:
        nfailed = 0
        failed_molecules = []
        # Create a pool of processes and distribute molecules amongst them
        with ProcessPoolExecutor() as executor:
            for m in executor.map(dftb_uv_2d, mol_dirs):
                if m is not None:
                    nfailed += 1
                    failed_molecules.append(m)

        assert nfailed == 0, "dftb_uv_2d failed for {} molecules: {}".format(len(failed_molecules), failed_molecules)

    except Exception as e:
        print(e, flush=True)
        print(traceback.format_exc(), flush=True)
        raise e


def process_tarfile(tarfpath):
    """
    Unpack and process all molecules in a tar file
    :param fargs: A tuple of the type (tar file path, {'mpi_info': mpi_info})
    """
    cwd = None
    retval = 0
    try:
        print("Processing {} on {}".format(tarfpath, MPI.Get_processor_name()), flush=True)

        # Unpack the tar file in the working directory
        tarf_name = os.path.basename(tarfpath).split('.')[0]
        cwd = os.path.join(scratch_space_root, tarf_name)
        os.mkdir(cwd)

        with tarfile.open(tarfpath) as t:
            t.extractall(cwd)

        # Get all molecule_directories in the unpacked tar file
        mol_dirs = get_mol_dirs(cwd)
        print("Processing {} molecules found in {}".format(len(mol_dirs), os.path.basename(tarfpath)), flush=True)

        # Distribute molecule processing amongst processes on the node
        process_molecules_on_node(mol_dirs)

        # Tar everything up
        create_new_tar(cwd, mol_dirs)

        # Set return value to success
        retval = 0

    except Exception as e:
        print(e)
        print(traceback.format_exc(), flush=True)
        retval = 1
        # Don't raise the Exception, just return

    finally:
        try:
            shutil.rmtree(cwd)
        except Exception as e:
            print("Couldn't remove scratch space directory {} due to {}. Continuing ..".format(cwd, e), flush=True)
        finally:
            return retval


def main():
    tar_files = []
    try:
        mpi_info = mpi_utils.create_mpi_info()
        
        # Ensure 1 process per node has been spawned
        if mpi_info['local_size'] > 1:
            print("ERROR. Please spawn one process per node for optimal performance. Exiting.", flush=True)
            MPI.COMM_WORLD.Abort(1)

        # Test that you have write access to the scratch space
        test_scratch_space()

        # Read in the list of tar files
        if MPI.COMM_WORLD.Get_rank() == 0:
            tar_files = get_tar_file_list()

    except Exception as e:
        print(e, flush=True)
        print(traceback.format_exc())
        MPI.COMM_WORLD.Abort(1)

    try:
        # Process tar files in parallel using a manager-worker pattern
        nfailed = 0
        with MPICommExecutor() as executor:
            for m in executor.map(process_tarfile, tar_files, chunksize=1, unordered=True):
                nfailed += m

        # All done
        MPI.COMM_WORLD.Barrier()
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("All done. {}/{} tar files successfully processed. Exiting."
                  .format(len(tar_files)-nfailed, len(tar_files)))

    except Exception as e:
        print(e)
        print(traceback.format_exc(), flush=True)
        # Something went wrong processing a tar file. Don't exit, continue with the next one.


if __name__ == '__main__':
    main()

