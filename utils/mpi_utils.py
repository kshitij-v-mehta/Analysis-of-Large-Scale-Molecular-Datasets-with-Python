from mpi4py import MPI
import sys
import re


def create_mpi_info():
    """
    WORKS ONLY ON ANDES
    Create a two-level MPI hierarchy of the global communicator MPI_COMM_WORLD and
    a node-local communicator to group processes by host.
    Returns a dictionary of global and node-local rank and size information.
    """
    try:
        comm_world = MPI.COMM_WORLD
        global_rank = comm_world.Get_rank()
        global_size = comm_world.Get_size()
        
        # Create communicator 'node_comm' of node-local processes
        hostname = MPI.Get_processor_name()  # e.g. andes154.olcf.ornl.gov
        # host_uid = int(hostname.split(".olcf.ornl.gov")[0].split("andes")[1])
        host_uid = int(re.sub("[^0-9]", "", hostname))
        
        node_comm = comm_world.Split(color=host_uid, key=global_rank)
        local_rank = node_comm.Get_rank()
        local_size = node_comm.Get_size()

        # Create communicator of all node-local root processes
        node_roots = []
        if local_rank == 0:
            node_roots.append(global_rank)
        
        node_roots_group = MPI.COMM_WORLD.group.Incl(node_roots)
        node_roots_comm = MPI.COMM_WORLD.Create_group(node_roots_group)

        # print("{} {}".format(local_rank, node_roots_comm == MPI.COMM_NULL))

        mpi_info = {'node_comm'       : node_comm,
                    'node_roots_comm' : node_roots_comm,
                    'local_size'      : local_size,
                    'global_size'     : global_size,
                    'local_rank'      : local_rank,
                    'global_rank'     : global_rank}
        
        # print("Global rank {} is local rank {} on host {}".format(global_rank, local_rank, hostname))
        print(mpi_info)
        return mpi_info
 
    
    except Exception as e:
        print(e, flush=True)
        raise e


if __name__ == '__main__':
    mpi_info = create_mpi_info()
    print(mpi_info)

