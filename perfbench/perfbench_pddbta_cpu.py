import time

tic = time.perf_counter()
import numpy as np
from mpi4py import MPI
import argparse

from serinv.utils import (
    allocate_ddbtax_permutation_buffers,
)
from serinv.wrappers import (
    pddbtasc,
    pddbtasci,
    allocate_ddbtars,
)


def get_partition_size_bta(n: int, p: int, balancing_ratio: float):
    # Also need to ensure that the partition size is greater than 3
    middle_process_partition_size = int(np.ceil(n / (p - 1 + balancing_ratio)))

    if middle_process_partition_size < 3:
        middle_process_partition_size = 3

    first_partition_size = n - (p - 1) * middle_process_partition_size

    extras = 0
    if first_partition_size < 3:
        Neach_section, extras = divmod(n, p)
        first_partition_size = middle_process_partition_size = Neach_section

    partition_sizes = []
    for i in range(p):
        if i == 0:
            partition_sizes.append(first_partition_size)
        else:
            partition_sizes.append(middle_process_partition_size)
        if extras > 0:
            partition_sizes[-1] += 1
            extras -= 1

    assert np.sum(partition_sizes) == n

    return partition_sizes

def generate_dataset(
    n_blocks: int,
    diagonal_blocksize: int,
    arrowhead_blocksize: int,
    n_processes: int,
    load_balancing_ratio: float,
    bsym: bool,
    quadratic: bool,
    dtype=np.float64,
):
    arrow_colsum = np.zeros((arrowhead_blocksize), dtype=dtype)
    partition_sizes = []
    partition_sizes = get_partition_size_bta(
        n=n_blocks, p=n_processes, balancing_ratio=load_balancing_ratio
    )

    rc = (1.0 + 1.0j) if dtype == np.complex128 else 1.0

    A_arrow_tip_block = rc * np.random.rand(
        arrowhead_blocksize, arrowhead_blocksize
    )
    A_arrow_tip_block[:, :] += np.diag(
        arrow_colsum + np.sum(A_arrow_tip_block[:, :], axis=1)
    )
    if quadratic:
        B_arrow_tip_block = rc * np.random.rand(
            arrowhead_blocksize, arrowhead_blocksize
        )
        B_arrow_tip_block[:, :] += np.diag(
            arrow_colsum + np.sum(B_arrow_tip_block[:, :], axis=1)
        )

    print(f"Generating sequential quadratic dataset...", flush=True)
    for p in range(n_processes):
        if MPI.COMM_WORLD.rank == p:
            n_blocks_pi = partition_sizes[p]

            print(f"    - Generating A", flush=True)
            A_diagonal_blocks_pi = rc * np.random.rand(
                n_blocks_pi, diagonal_blocksize, diagonal_blocksize
            )
            A_lower_arrow_blocks_pi = rc * np.random.rand(
                n_blocks_pi, arrowhead_blocksize, diagonal_blocksize
            )
            A_upper_arrow_blocks_pi = rc * np.random.rand(
                n_blocks_pi, diagonal_blocksize, arrowhead_blocksize
            )
            if p == n_processes - 1:
                A_lower_diagonal_blocks_pi = rc * np.random.rand(
                    n_blocks_pi - 1, diagonal_blocksize, diagonal_blocksize
                )
                A_upper_diagonal_blocks_pi = rc * np.random.rand(
                    n_blocks_pi - 1, diagonal_blocksize, diagonal_blocksize
                )
            else:
                A_lower_diagonal_blocks_pi = rc * np.random.rand(
                    n_blocks_pi, diagonal_blocksize, diagonal_blocksize
                )
                A_upper_diagonal_blocks_pi = rc * np.random.rand(
                    n_blocks_pi, diagonal_blocksize, diagonal_blocksize
                )
            for i in range(A_diagonal_blocks_pi.shape[0]):
                colsum = np.sum(A_diagonal_blocks_pi[i], axis=1) - np.diag(
                    A_diagonal_blocks_pi[i]
                )
                if i > 0:
                    colsum += np.sum(A_lower_diagonal_blocks_pi[i - 1], axis=1)
                if i + np.cumsum(partition_sizes)[p] < n_blocks - 1:
                    colsum += np.sum(A_upper_diagonal_blocks_pi[i], axis=0)
                colsum += np.sum(A_upper_arrow_blocks_pi[i], axis=1)
                A_diagonal_blocks_pi[i] += np.diag(colsum)
                arrow_colsum[:] += np.sum(A_lower_arrow_blocks_pi[i], axis=1)

            A = {
                "A_diagonal_blocks": A_diagonal_blocks_pi,
                "A_lower_diagonal_blocks": A_lower_diagonal_blocks_pi,
                "A_upper_diagonal_blocks": A_upper_diagonal_blocks_pi,
                "A_lower_arrow_blocks": A_lower_arrow_blocks_pi,
                "A_upper_arrow_blocks": A_upper_arrow_blocks_pi,
                "A_arrow_tip_block": A_arrow_tip_block,
            }

            if quadratic:
                print(f"    - Generating B (Quadratic Equation)", flush=True)
                B_diagonal_blocks_pi = rc * np.random.rand(
                    n_blocks_pi, diagonal_blocksize, diagonal_blocksize
                )
                B_lower_arrow_blocks_pi = rc * np.random.rand(
                    n_blocks_pi, arrowhead_blocksize, diagonal_blocksize
                )
                B_upper_arrow_blocks_pi = rc * np.random.rand(
                    n_blocks_pi, diagonal_blocksize, arrowhead_blocksize
                )
                if p == n_processes - 1:
                    B_lower_diagonal_blocks_pi = rc * np.random.rand(
                        n_blocks_pi - 1, diagonal_blocksize, diagonal_blocksize
                    )
                    B_upper_diagonal_blocks_pi = rc * np.random.rand(
                        n_blocks_pi - 1, diagonal_blocksize, diagonal_blocksize
                    )
                else:
                    B_lower_diagonal_blocks_pi = rc * np.random.rand(
                        n_blocks_pi, diagonal_blocksize, diagonal_blocksize
                    )
                    B_upper_diagonal_blocks_pi = rc * np.random.rand(
                        n_blocks_pi, diagonal_blocksize, diagonal_blocksize
                    )
                for i in range(B_diagonal_blocks_pi.shape[0]):
                    colsum = np.sum(B_diagonal_blocks_pi[i], axis=1) - np.diag(
                        B_diagonal_blocks_pi[i]
                    )
                    if i > 0:
                        colsum += np.sum(B_lower_diagonal_blocks_pi[i - 1], axis=1)
                    if i + np.cumsum(partition_sizes)[p] < n_blocks - 1:
                        colsum += np.sum(B_upper_diagonal_blocks_pi[i], axis=0)
                    colsum += np.sum(B_upper_arrow_blocks_pi[i], axis=1)
                    B_diagonal_blocks_pi[i] += np.diag(colsum)
                    arrow_colsum[:] += np.sum(B_lower_arrow_blocks_pi[i], axis=1)

                if bsym:
                    for i in range(n_blocks_pi):
                        B_diagonal_blocks_pi[i] = (
                            B_diagonal_blocks_pi[i] + B_diagonal_blocks_pi[i].conj().T
                        ) / 2
                        if i + np.cumsum(partition_sizes)[p] < n_blocks - 1:
                            B_upper_diagonal_blocks_pi[i] = (
                                B_lower_diagonal_blocks_pi[i].conj().T
                            )
                        B_upper_arrow_blocks_pi[i] = B_lower_arrow_blocks_pi[i].conj().T

                B = {
                    "B_diagonal_blocks": B_diagonal_blocks_pi,
                    "B_lower_diagonal_blocks": B_lower_diagonal_blocks_pi,
                    "B_upper_diagonal_blocks": B_upper_diagonal_blocks_pi,
                    "B_lower_arrow_blocks": B_lower_arrow_blocks_pi,
                    "B_upper_arrow_blocks": B_upper_arrow_blocks_pi,
                    "B_arrow_tip_block": B_arrow_tip_block,
                }
            else:
                B = None

    return A, B

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--b",
        type=int,
        default=128,
        help="an integer for the diagonal block size",
    )
    parser.add_argument(
        "--a",
        type=int,
        default=0,
        help="an integer for the diagonal block size",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=8,
        help="an integer for the number of diagonal blocks",
    )
    parser.add_argument(
        "--bsym",
        type=bool,
        default=True,
        help="whether the matrix is symmetric or not",
    )
    parser.add_argument(
        "--lb",
        type=float,
        default=1,
        help="load balancing ratio for the distributed case",
    )
    parser.add_argument(
        "--q",
        type=int,
        help="wether to run the quadratic or not",
    )
    args = parser.parse_args()
    toc = time.perf_counter()
    print(f"rank: {MPI.COMM_WORLD.rank}, import and parsing took: {toc - tic:.5f} sec", flush=True)

    quadratic = False if args.q == 0 else True
    n_iterations = 10
    n_warmups = 2

    tic = time.perf_counter()
    A, B = generate_dataset(
        n_blocks = args.n,
        diagonal_blocksize = args.b,
        arrowhead_blocksize = args.a,
        n_processes = MPI.COMM_WORLD.size,
        load_balancing_ratio = args.lb,
        bsym = args.bsym,
        quadratic = quadratic,
    )
    toc = time.perf_counter()
    print(f"rank: {MPI.COMM_WORLD.rank}, dataset generation took: {toc - tic:.5f} sec", flush=True)

    A_diagonal_blocks_init = A["A_diagonal_blocks"]
    A_lower_diagonal_blocks_init = A["A_lower_diagonal_blocks"]
    A_upper_diagonal_blocks_init = A["A_upper_diagonal_blocks"]
    A_lower_arrow_blocks_init = A["A_lower_arrow_blocks"]
    A_upper_arrow_blocks_init = A["A_upper_arrow_blocks"]
    A_arrow_tip_block_init = A["A_arrow_tip_block"]

    # Init device arrays
    A_diagonal_blocks_cpu = np.empty_like(A_diagonal_blocks_init)
    A_lower_diagonal_blocks_cpu = np.empty_like(A_lower_diagonal_blocks_init)
    A_upper_diagonal_blocks_cpu = np.empty_like(A_upper_diagonal_blocks_init)
    A_lower_arrow_blocks_cpu = np.empty_like(A_lower_arrow_blocks_init)
    A_upper_arrow_blocks_cpu = np.empty_like(A_upper_arrow_blocks_init)
    A_arrow_tip_block_cpu = np.empty_like(A_arrow_tip_block_init)

    if quadratic:
        B_diagonal_blocks_init = B["B_diagonal_blocks"]
        B_lower_diagonal_blocks_init = B["B_lower_diagonal_blocks"]
        B_upper_diagonal_blocks_init = B["B_upper_diagonal_blocks"]
        B_lower_arrow_blocks_init = B["B_lower_arrow_blocks"]
        B_upper_arrow_blocks_init = B["B_upper_arrow_blocks"]
        B_arrow_tip_block_init = B["B_arrow_tip_block"]

        # Init device arrays
        B_diagonal_blocks_cpu = np.empty_like(B_diagonal_blocks_init)
        B_lower_diagonal_blocks_cpu = np.empty_like(B_lower_diagonal_blocks_init)
        B_upper_diagonal_blocks_cpu = np.empty_like(B_upper_diagonal_blocks_init)
        B_lower_arrow_blocks_cpu = np.empty_like(B_lower_arrow_blocks_init)
        B_upper_arrow_blocks_cpu = np.empty_like(B_upper_arrow_blocks_init)
        B_arrow_tip_block_cpu = np.empty_like(B_arrow_tip_block_init)

    t_pddbtasc = []
    t_comm = []
    t_pddbtasci = []

    for i in range(n_warmups + n_iterations):
        MPI.COMM_WORLD.Barrier()
        if MPI.COMM_WORLD.rank == 0:
            print(f"Iteration: {i+1}/{n_warmups+n_iterations}", flush=True)

        tic = time.perf_counter()
        buffer = allocate_ddbtax_permutation_buffers(
            A_lower_diagonal_blocks=A_lower_diagonal_blocks_cpu,
            quadratic=quadratic,
        )
        ddbtars = allocate_ddbtars(
            A_diagonal_blocks=A_diagonal_blocks_cpu,
            A_lower_diagonal_blocks=A_lower_diagonal_blocks_cpu,
            A_upper_diagonal_blocks=A_upper_diagonal_blocks_cpu,
            A_lower_arrow_blocks=A_lower_arrow_blocks_cpu,
            A_upper_arrow_blocks=A_upper_arrow_blocks_cpu,
            A_arrow_tip_block=A_arrow_tip_block_cpu,
            comm=MPI.COMM_WORLD,
            array_module="numpy",
            strategy="allgather",
            quadratic=quadratic,
        )

        A_diagonal_blocks_cpu[:] = A_diagonal_blocks_init
        A_lower_diagonal_blocks_cpu[:] = A_lower_diagonal_blocks_init
        A_upper_diagonal_blocks_cpu[:] = A_upper_diagonal_blocks_init
        A_lower_arrow_blocks_cpu[:] = A_lower_arrow_blocks_init
        A_upper_arrow_blocks_cpu[:] = A_upper_arrow_blocks_init
        A_arrow_tip_block_cpu[:] = A_arrow_tip_block_init

        if quadratic:
            B_diagonal_blocks_cpu[:] = B_diagonal_blocks_init
            B_lower_diagonal_blocks_cpu[:] = B_lower_diagonal_blocks_init
            B_upper_diagonal_blocks_cpu[:] = B_upper_diagonal_blocks_init
            B_lower_arrow_blocks_cpu[:] = B_lower_arrow_blocks_init
            B_upper_arrow_blocks_cpu[:] = B_upper_arrow_blocks_init
            B_arrow_tip_block_cpu[:] = B_arrow_tip_block_init

            rhs: dict = {
                "B_diagonal_blocks": B_diagonal_blocks_cpu,
                "B_lower_diagonal_blocks": B_lower_diagonal_blocks_cpu,
                "B_upper_diagonal_blocks": B_upper_diagonal_blocks_cpu,
                "B_lower_arrow_blocks": B_lower_arrow_blocks_cpu,
                "B_upper_arrow_blocks": B_upper_arrow_blocks_cpu,
                "B_arrow_tip_block": B_arrow_tip_block_cpu,
            }
        toc = time.perf_counter()
        print(f"rank: {MPI.COMM_WORLD.rank}, copying data + buffer alloc took: {toc - tic:.5f} sec", flush=True)

        MPI.COMM_WORLD.Barrier()
        tic = time.perf_counter()
        comm_time = pddbtasc(
            A_diagonal_blocks=A_diagonal_blocks_cpu,
            A_lower_diagonal_blocks=A_lower_diagonal_blocks_cpu,
            A_upper_diagonal_blocks=A_upper_diagonal_blocks_cpu,
            A_lower_arrow_blocks=A_lower_arrow_blocks_cpu,
            A_upper_arrow_blocks=A_upper_arrow_blocks_cpu,
            A_arrow_tip_block=A_arrow_tip_block_cpu,
            rhs=rhs if quadratic else None,
            quadratic=quadratic,
            buffers=buffer,
            ddbtars=ddbtars,
            comm=MPI.COMM_WORLD,
            strategy="allgather",
        )
        MPI.COMM_WORLD.Barrier()
        toc = time.perf_counter()
        elapsed = toc - tic
        if MPI.COMM_WORLD.rank == 0:
            print(f"rank: {MPI.COMM_WORLD.rank}, pddbtasc comm time: {comm_time:.5f} sec", flush=True)
            print(f"rank: {MPI.COMM_WORLD.rank}, pddbtasc took: {elapsed:.5f} sec", flush=True)
        if i >= n_warmups:
            t_comm.append(comm_time)
            t_pddbtasc.append(elapsed)

        MPI.COMM_WORLD.Barrier()
        tic = time.perf_counter()
        pddbtasci(
            A_diagonal_blocks=A_diagonal_blocks_cpu,
            A_lower_diagonal_blocks=A_lower_diagonal_blocks_cpu,
            A_upper_diagonal_blocks=A_upper_diagonal_blocks_cpu,
            A_lower_arrow_blocks=A_lower_arrow_blocks_cpu,
            A_upper_arrow_blocks=A_upper_arrow_blocks_cpu,
            A_arrow_tip_block=A_arrow_tip_block_cpu,
            comm=MPI.COMM_WORLD,
            rhs=rhs if quadratic else None,
            quadratic=quadratic,
            buffers=buffer,
            ddbtars=ddbtars,
        )
        MPI.COMM_WORLD.Barrier()
        toc = time.perf_counter()
        elapsed = toc - tic
        if MPI.COMM_WORLD.rank == 0:
            print(f"rank: {MPI.COMM_WORLD.rank}, pddbtasci took: {elapsed:.5f} sec", flush=True)
        if i >= n_warmups:
            t_pddbtasci.append(elapsed)

        rhs = None
        buffer = None
        ddbtrs = None

    MPI.COMM_WORLD.Barrier()
    print(f"rank: {MPI.COMM_WORLD.rank}, t_pddbtasc: {t_pddbtasc}", flush=True)
    print(f"rank: {MPI.COMM_WORLD.rank}, t_comm: {t_comm}", flush=True)
    print(f"rank: {MPI.COMM_WORLD.rank}, t_pddbtasci: {t_pddbtasci}", flush=True)

    print(
        f"rank: {MPI.COMM_WORLD.rank},avg t_pddbtasc: {np.mean(np.array(t_pddbtasc)):.5f} sec",
        flush=True,
    )
    print(
        f"rank: {MPI.COMM_WORLD.rank},avg t_comm: {np.mean(np.array(t_comm)):.5f} sec",
        flush=True,
    )
    print(
        f"rank: {MPI.COMM_WORLD.rank},avg t_pddbtasci: {np.mean(np.array(t_pddbtasci)):.5f} sec",
        flush=True,
    )
    print(
        f"rank: {MPI.COMM_WORLD.rank},avg total time: {np.mean(np.array(t_pddbtasc))+np.mean(np.array(t_comm))+np.mean(np.array(t_pddbtasci)):.5f} sec",
        flush=True,
    )
