"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Lisa Gaedke-Merzhaeuser  (lisa.gaedke.merzhaeuser@usi.ch)
@date: 2024-02

Tests for lu tridiagonal matrices selected factorization routine.

Copyright 2023-2024 ETH Zurich and USI. All rights reserved.
"""

import numpy as np
import time

from sdr.lu.lu_factorize import lu_factorize_tridiag
from sdr.utils.matrix_generation import generate_tridiag_array

PATH_TO_SAVE = "./"
N_WARMUPS = 3
N_RUNS = 10

if __name__ == "__main__":
    # ----- Populate the blocks list HERE -----
    l_nblocks = [128]
    # ----- Populate the blocksizes list HERE -----
    l_blocksize = [100]
    symmetric = False
    diagonal_dominant = True
    seed = 63

    runs_timings = []

    for nblocks in l_nblocks:
        for blocksize in l_blocksize:

            print(
                "Starting data generation for:\n    nblocks: ",
                nblocks,
                " blocksize: ",
                blocksize,
            )

            t_data_gen_start = time.perf_counter_ns()
            (
                A_diagonal_blocks_ref,
                A_lower_diagonal_blocks_ref,
                A_upper_diagonal_blocks_ref,
            ) = generate_tridiag_array(
                nblocks, blocksize, symmetric, diagonal_dominant, seed
            )
            t_data_gen_stop = time.perf_counter_ns()
            t_data_gen = t_data_gen_stop - t_data_gen_start

            input_mem_size = (
                A_diagonal_blocks_ref.size * A_diagonal_blocks_ref.itemsize
                + A_lower_diagonal_blocks_ref.size
                * A_lower_diagonal_blocks_ref.itemsize
                + A_upper_diagonal_blocks_ref.size
                * A_upper_diagonal_blocks_ref.itemsize
            )
            print("    Input data memory size: ", input_mem_size * 1e-9, "GB")
            print("    Data generation took: ", t_data_gen * 1e-9, "s")

            headers = {}
            headers["N_WARMUPS"] = N_WARMUPS
            headers["N_RUNS"] = N_RUNS
            headers["nblocks"] = nblocks
            headers["blocksize"] = blocksize
            headers["symmetric"] = symmetric
            headers["diagonal_dominant"] = diagonal_dominant
            headers["seed"] = seed
            headers["total_runtime"] = 0.0

            for i in range(N_WARMUPS + N_RUNS):
                A_diagonal_blocks = A_diagonal_blocks_ref.copy()
                A_lower_diagonal_blocks = A_lower_diagonal_blocks_ref.copy()
                A_upper_diagonal_blocks = A_upper_diagonal_blocks_ref.copy()

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
                t_run_start = time.perf_counter_ns()
                (
                    L_diagonal_blocks,
                    L_lower_diagonal_blocks,
                    U_diagonal_blocks,
                    U_upper_diagonal_blocks,
                    timings,
                ) = lu_factorize_tridiag(
                    A_diagonal_blocks,
                    A_lower_diagonal_blocks,
                    A_upper_diagonal_blocks,
                )
                t_run_stop = time.perf_counter_ns()
                t_run = t_run_stop - t_run_start

                headers["total_runtime"] = t_run

                print(" run took: ", t_run * 1e-9, "s")

                if i >= N_WARMUPS:
                    runs_timings.append({**headers, **timings})

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save the timings and nblocks and blocksize
    runs_timings = np.array(runs_timings)
    print(runs_timings)
    np.save(
        PATH_TO_SAVE + f"lu_factorize_tridiag_timings_{timestamp}.npy", runs_timings
    )
