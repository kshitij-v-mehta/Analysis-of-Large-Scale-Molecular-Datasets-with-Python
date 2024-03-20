from mpi4py.futures import MPICommExecutor
from mpi4py import MPI
import os, sys, traceback, glob
from pathlib import Path

from find_failed_molecules import check_orca_output
from convert_gen_to_xyz import generate_xyz_files
from my_orca_uv import smooth_spectrum


def orca_smooth_spectrum(mol_dir):
    try:
        # Return if the molecule has failed
        if check_orca_output(mol_dir) != 0:
            return
        
        # Generate the xyz file from the .gen file
        if os.path.exists(os.path.join(mol_dir, "geo_end.gen")):
            generate_xyz_files(mol_dir, None)
        
        # draw_2Dmol(MPI.COMM_SELF, mol_dir)
        smooth_spectrum(MPI.COMM_SELF, str(Path(mol_dir).parent), os.path.basename(mol_dir), None, 70.0, None, None)
        
        # Verify that the orca-smmoth.DAT file was created
        assert os.path.exists("{}/{}".format(mol_dir, "orca-smooth.DAT")), "EXC-smooth.DAT file not created"
        
        # Verify that the spectrum png file was created
        assert len(glob.glob("{}/abs_spectrum_*.png".format(mol_dir))) == 1, "spectrum png file not created"
        
        return None

    except Exception as e:
        print("{} for {}".format(e, os.path.basename(mol_dir)), flush=True)
        print(traceback.format_exc())


def main():
    try:
        assert len(sys.argv) == 2, "Provide full path to molecule directories"
        dataset = sys.argv[1]

        mol_dirs = None
        if MPI.COMM_WORLD.Get_rank() == 0:
            mol_dirs = [f.path for f in os.scandir(dataset) if f.is_dir() and f.name.startswith("mol_")]
            print("Found {} molecules".format(len(mol_dirs)), flush=True)

        with MPICommExecutor() as executor:
            executor.map(orca_smooth_spectrum, mol_dirs, chunksize=10, unordered=True)

        # mol_dirs = ["/lustre/orion/lrn026/world-shared/kmehta/datasets/gdb-9-ex/orca-eom-ccsd/untarred/GDB-9-Ex-ORCA-EOM-CCSD/mol_025426"]
        # for m in mol_dirs:
        #     orca_smooth_spectrum(m)

        MPI.COMM_WORLD.Barrier()
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("All done", flush=True)

    except Exception as e:
        print(e)
        print(traceback.format_exc(), flush=True)


if __name__ == '__main__':
    main()

