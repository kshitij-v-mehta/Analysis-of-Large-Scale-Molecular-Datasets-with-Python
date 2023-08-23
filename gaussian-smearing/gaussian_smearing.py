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
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

from mpi4py import MPI
from mpi4py.futures import MPICommExecutor

import mpi_utils

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
        tarfiles_list = glob.glob("{}/*.tar.gz".format(tarfiles_in))
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


def dftb_uv_2d(mol_dir):
    try:
        # print("Process {} with global rank {} on node {} received molecule {}"
        #       "".format(os.getpid(), MPI.COMM_WORLD.Get_rank(), MPI.Get_processor_name(), mol_dir), flush=True)

        with open("{}/EXC-smooth.DAT".format(mol_dir), "w") as f:
            pass

        return

    except Exception as e:
        print(e)
        print(traceback.format_exc(), flush=True)
        raise e


def create_new_tar(cwd, mol_dirs):
    # Tar the molecule directories again after the gaussian smearing is done

    t = None

    try:
        new_tar_f = os.path.basename(cwd) + "-gaussian-smearing.tar.gz"
        print("Creating new tar file {}".format(new_tar_f), flush=True)
        new_tar_f_scratch_loc = "{}/{}".format(scratch_space_root, new_tar_f)

        # Create the new tar file in /tmp first, and then copy it to the file system
        t = tarfile.open(new_tar_f_scratch_loc, mode='x:gz')
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


def distribute_molecules_locally(mol_dirs):
    # Distribute molecules from a tar file amongst all processes on the compute node

    try:
        # Create a pool of processes and distribute molecules amongst them
        with ProcessPoolExecutor() as executor:
            executor.map(dftb_uv_2d, mol_dirs)

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
        distribute_molecules_locally(mol_dirs)

        # Tar everything up
        create_new_tar(cwd, mol_dirs)

    except Exception as e:
        print(e)
        print(traceback.format_exc(), flush=True)
        # Don't raise the Exception, just return

    finally:
        try:
            shutil.rmtree(cwd)
        except Exception as e:
            print("Couldn't remove scratch space directory {} due to {}. Continuing ..".format(cwd, e), flush=True)


def main():
    try:
        # mpi_info = mpi_utils.create_mpi_info()
        #
        # # Ensure 1 process per node has been spawned
        # if mpi_info['local_size'] > 1:
        #     print("ERROR. Please spawn one process per node for optimal performance. Exiting.", flush=True)
        #     MPI.COMM_WORLD.Abort(1)

        # Test that you have write access to the scratch space
        test_scratch_space()

        # Read in the list of tar files
        tar_files = []
        if MPI.COMM_WORLD.Get_rank() == 0:
            tar_files = get_tar_file_list()

        # Process tar files in parallel using a manager-worker pattern
        with MPICommExecutor() as executor:
            executor.map(process_tarfile, tar_files, chunksize=1, unordered=True)

        # All done
        MPI.COMM_WORLD.Barrier()
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("All done. Exiting.")

    except Exception as e:
        print(e, flush=True)
        print(traceback.format_exc())
        MPI.COMM_WORLD.Abort(1)


if __name__ == '__main__':
    main()

