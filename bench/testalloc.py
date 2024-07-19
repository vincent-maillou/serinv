import cupy as cp
import mpi4py.MPI as MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N_GPU_PER_NODE = 8

def set_device(
    rank_id: int,
    n_gpus_per_node: int,
    debug: bool = False,
):
    """
    Set the device for the current process.

    Parameters
    ----------
    rank_id : int
        The rank of the current process.
    n_gpus_per_node int: int
        The number of GPUs per node.
    debug : bool, optional
        Whether to print debug information.
    """
    device_id = rank_id % n_gpus_per_node
    cp.cuda.Device(device_id).use()
    if debug:
        print(f"Rank {rank_id} of {size} is using the GPU {device_id}")

    return device_id


if __name__ == "__main__":
    device_id = set_device(rank, N_GPU_PER_NODE, debug=True)