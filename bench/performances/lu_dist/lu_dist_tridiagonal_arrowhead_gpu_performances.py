"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Lisa Gaedke-Merzhaeuser  (lisa.gaedke.merzhaeuser@usi.ch)
@date: 2024-03

Example of the lu_dist algorithm for tridiagonal arrowhead matrices.

Copyright 2023-2024 ETH Zurich and USI. All rights reserved.
"""

import cupy as cp
import numpy as np
import time
from mpi4py import MPI
import mkl

from sdr.lu_dist.lu_dist_tridiagonal_arrowhead_gpu import (
    lu_dist_tridiagonal_arrowhead_gpu,
)
from sdr.utils.matrix_generation import generate_tridiag_arrowhead_arrays
from sdr.utils.dist_utils import (
    get_partitions_indices,
    extract_partition_tridiagonal_arrowhead_array,
    extract_bridges_tridiagonal_array,
)

# from sdr.utils.gpu_utils import set_device

PATH_TO_SAVE = "./"
N_WARMUPS = 3
N_RUNS = 10

N_GPU_PER_NODE = 1

if __name__ == "__main__":
    # ----- Populate the blocks list HERE -----
    l_nblocks = [512]
    # ----- Populate the diagonal blocksizes list HERE -----
    l_diag_blocksize = [600]
    # ----- Populate the arrow blocksizes list HERE -----
    l_arrow_blocksize = [100]
    diagonal_dominant = True
    symmetric = False
    seed = 63

    runs_timings = []
    runs_sections = []

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    # set_device(comm_rank, N_GPU_PER_NODE)

    # STRONG SCALLING
    # partition_sizes = [l_nblocks[i] // comm_size for i in range(len(l_nblocks))]

    # WEAK SCALLING
    partition_sizes = [256]

    for partition_size in partition_sizes:
        for diag_blocksize in l_diag_blocksize:
            for arrow_blocksize in l_arrow_blocksize:

                if partition_size < 3:
                    raise ValueError(
                        "Each processes should have at least 3 blocks to perfrome the middle factorization."
                    )

                if comm_rank == 0:
                    print(
                        "Starting data generation for:\n    partition_size: ",
                        partition_size,
                        " diag_blocksize: ",
                        diag_blocksize,
                        " arrow_blocksize: ",
                        arrow_blocksize,
                    )

                t_data_gen_start = time.perf_counter_ns()
                A_diagonal_blocks_ref = np.random.rand(
                    diag_blocksize, partition_size * diag_blocksize
                )
                A_lower_diagonal_blocks_ref = np.random.rand(
                    diag_blocksize, (partition_size - 1) * diag_blocksize
                )
                A_upper_diagonal_blocks_ref = np.random.rand(
                    diag_blocksize, (partition_size - 1) * diag_blocksize
                )
                A_arrow_bottom_blocks_ref = np.random.rand(
                    arrow_blocksize, partition_size * diag_blocksize
                )
                A_arrow_right_blocks_ref = np.random.rand(
                    partition_size * diag_blocksize,
                    arrow_blocksize,
                )

                # Make diagonally dominant
                for i in range(partition_size):
                    A_diagonal_blocks_ref[
                        :, i * diag_blocksize : (i + 1) * diag_blocksize
                    ] += np.eye(diag_blocksize) * (3 * diag_blocksize + arrow_blocksize)

                matrix_elements_size = (
                    partition_size * comm_size * diag_blocksize + arrow_blocksize
                )
                A_arrow_tip_block_ref = (
                    np.random.rand(arrow_blocksize, arrow_blocksize)
                    + np.eye(arrow_blocksize) * matrix_elements_size
                )

                A_bridges_lower = np.random.rand(
                    diag_blocksize, (comm_size - 1) * diag_blocksize
                )
                A_bridges_upper = np.random.rand(
                    diag_blocksize, (comm_size - 1) * diag_blocksize
                )
                t_data_gen_stop = time.perf_counter_ns()
                t_data_gen = t_data_gen_stop - t_data_gen_start

                input_mem_size = (
                    A_diagonal_blocks_ref.size * A_diagonal_blocks_ref.itemsize
                    + A_lower_diagonal_blocks_ref.size
                    * A_lower_diagonal_blocks_ref.itemsize
                    + A_upper_diagonal_blocks_ref.size
                    * A_upper_diagonal_blocks_ref.itemsize
                    + A_arrow_bottom_blocks_ref.size
                    * A_arrow_bottom_blocks_ref.itemsize
                    + A_arrow_right_blocks_ref.size * A_arrow_right_blocks_ref.itemsize
                    + A_arrow_tip_block_ref.size * A_arrow_tip_block_ref.itemsize
                    + A_bridges_lower.size * A_bridges_lower.itemsize
                    + A_bridges_upper.size * A_bridges_upper.itemsize
                )
                if comm_rank == 0:
                    print(
                        "    Input (partition_size) memory size: ",
                        input_mem_size * 1e-9,
                        "GB",
                    )
                    print("    Data generation took: ", t_data_gen * 1e-9, "s")

                headers = {}
                headers["N_WARMUPS"] = N_WARMUPS
                headers["N_RUNS"] = N_RUNS
                headers["MKL_NUM_THREADS"] = mkl.get_max_threads()
                headers["GPU_DEVICE_ID"] = cp.cuda.get_device_id()
                headers["COMM_SIZE"] = comm_size
                headers["COMM_RANK"] = comm_rank
                headers["nblocks"] = partition_size * comm_size
                headers["partition_size"] = partition_size
                headers["partition_memory"] = input_mem_size * 1e-9
                headers["blocksize"] = diag_blocksize
                headers["arrow_blocksize"] = arrow_blocksize
                headers["symmetric"] = symmetric
                headers["diagonal_dominant"] = diagonal_dominant
                headers["seed"] = seed
                headers["total_runtime"] = 0.0

                for i in range(N_WARMUPS + N_RUNS):
                    A_diagonal_blocks_local = A_diagonal_blocks_ref.copy()
                    A_lower_diagonal_blocks_local = A_lower_diagonal_blocks_ref.copy()
                    A_upper_diagonal_blocks_local = A_upper_diagonal_blocks_ref.copy()
                    A_arrow_bottom_blocks_local = A_arrow_bottom_blocks_ref.copy()
                    A_arrow_right_blocks_local = A_arrow_right_blocks_ref.copy()
                    A_arrow_tip_block_local = A_arrow_tip_block_ref.copy()
                    A_bridges_lower_local = A_bridges_lower.copy()
                    A_bridges_upper_local = A_bridges_upper.copy()

                    if comm_rank == 0:
                        if i < N_WARMUPS:
                            print("    Warmup run ", i, " ...", end="", flush=True)
                        else:
                            print(
                                "    Starting run ",
                                i - N_WARMUPS,
                                " ...",
                                end="",
                                flush=True,
                            )

                    comm.Barrier()
                    t_run_start = time.perf_counter_ns()
                    (
                        X_diagonal_blocks_local,
                        X_lower_diagonal_blocks_local,
                        X_upper_diagonal_blocks_local,
                        X_arrow_bottom_blocks_local,
                        X_arrow_right_blocks_local,
                        X_arrow_tip_block_local,
                        X_bridges_lower,
                        X_bridges_upper,
                        timings,
                        sections,
                    ) = lu_dist_tridiagonal_arrowhead_gpu(
                        A_diagonal_blocks_local,
                        A_lower_diagonal_blocks_local,
                        A_upper_diagonal_blocks_local,
                        A_arrow_bottom_blocks_local,
                        A_arrow_right_blocks_local,
                        A_arrow_tip_block_local,
                        A_bridges_lower,
                        A_bridges_upper,
                    )
                    comm.Barrier()
                    t_run_stop = time.perf_counter_ns()
                    t_run = t_run_stop - t_run_start

                    headers["total_runtime"] = t_run

                    if comm_rank == 0:
                        print(" run took: ", t_run * 1e-9, "s")

                    if i >= N_WARMUPS:
                        runs_timings.append({**headers, **timings})
                        runs_sections.append({**headers, **sections})

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save the timings and nblocks and blocksize
    runs_timings = np.array(runs_timings)
    print("runs_timings: ", runs_timings)
    np.save(
        PATH_TO_SAVE
        + f"lu_dist_tridiagonal_arrowhead_gpu_nprocesses_{comm_size}_rank_{comm_rank}_timings_{timestamp}.npy",
        runs_timings,
    )

    runs_sections = np.array(runs_sections)
    print("runs_sections:", runs_sections)
    np.save(
        PATH_TO_SAVE
        + f"lu_dist_tridiagonal_arrowhead_gpu_nprocesses_{comm_size}_rank_{comm_rank}_sections_{timestamp}.npy",
        runs_sections,
    )
