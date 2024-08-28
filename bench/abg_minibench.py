import numpy as np
from mpi4py import MPI
import time

comm_rank = MPI.COMM_WORLD.Get_rank()
comm_size = MPI.COMM_WORLD.Get_size()


if __name__ == "__main__":
    # abg of all reduce
    b_sizes = [256, 512, 1024]
    n_iterations = 100

    timings = np.zeros((len(b_sizes), n_iterations))

    print(f"Bench Allreduce for P={comm_size}")
    for b_i in b_sizes:
        b_init = np.random.rand(b_i)
        for i in range(n_iterations):
            b = b_init.copy()

            tic = time.perf_counter()
            MPI.COMM_WORLD.Allreduce(
                MPI.IN_PLACE,
                b,
                op=MPI.SUM,
            )
            toc = time.perf_counter()

            timings[b_i, i] = toc - tic

        if comm_rank == 0:
            print(f"    b_size: {b_i}")
            print(f"        mean: {np.mean(timings[b_i])}")
            print(f"        std: {np.std(timings[b_i])}")

    np.save(f"abg_allreduce_timings_{comm_size}.npy", timings)
