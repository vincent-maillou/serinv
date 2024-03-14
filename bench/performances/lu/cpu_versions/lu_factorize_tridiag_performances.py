"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Lisa Gaedke-Merzhaeuser  (lisa.gaedke.merzhaeuser@usi.ch)
@date: 2024-02

Tests for lu tridiagonal matrices selected factorization routine.

Copyright 2023-2024 ETH Zurich and USI. All rights reserved.
"""

import numpy as np

from sdr.lu.lu_factorize import lu_factorize_tridiag
from sdr.utils.matrix_generation import generate_tridiag_array

PATH_TO_SAVE = "../../"
N_WARMUPS = 3
N_RUNS = 10

# Testing of block tridiagonal lu
if __name__ == "__main__":
    # ----- Populate the blocks list HERE -----
    l_nblocks = [5]
    # ----- Populate the blocksizes list HERE -----
    l_blocksize = [2]
    symmetric = False
    diagonal_dominant = True
    seed = 63

    runs_timings = []

    for nblocks in l_nblocks:
        for blocksize in l_blocksize:

            (
                A_diagonal_blocks_ref,
                A_lower_diagonal_blocks_ref,
                A_upper_diagonal_blocks_ref,
            ) = generate_tridiag_array(
                nblocks, blocksize, symmetric, diagonal_dominant, seed
            )

            headers = {}
            headers["N_WARMUPS"] = N_WARMUPS
            headers["N_RUNS"] = N_RUNS
            headers["nblocks"] = nblocks
            headers["blocksize"] = blocksize
            headers["symmetric"] = symmetric
            headers["diagonal_dominant"] = diagonal_dominant
            headers["seed"] = seed

            for i in range(N_WARMUPS + N_RUNS):
                A_diagonal_blocks = A_diagonal_blocks_ref.copy()
                A_lower_diagonal_blocks = A_lower_diagonal_blocks_ref.copy()
                A_upper_diagonal_blocks = A_upper_diagonal_blocks_ref.copy()

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

                if i >= N_WARMUPS:
                    runs_timings.append({**headers, **timings})

    # Save the timings and nblocks and blocksize
    runs_timings = np.array(runs_timings)
    print(runs_timings)
    np.save(PATH_TO_SAVE + "lu_factorize_tridiag_timings.npy", runs_timings)
