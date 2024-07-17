import cupy as cp
import mpi4py.MPI as MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print(f"Hello from rank {rank} of {size}, I'm using the GPU {cp.cuda.Device().id}")