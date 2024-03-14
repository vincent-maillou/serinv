"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Lisa Gaedke-Merzhaeuser  (lisa.gaedke.merzhaeuser@usi.ch)
@date: 2024-02

Tests for lu tridiagonal arrowhead matrices selected factorization routine.

Copyright 2023-2024 ETH Zurich and USI. All rights reserved.
"""

import numpy as np

from sdr.lu.lu_factorize import lu_factorize_tridiag_arrowhead
from sdr.utils.matrix_generation import generate_tridiag_arrowhead_arrays

PATH_TO_SAVE = "../../"
N_WARMUPS = 3
N_RUNS = 10

# Testing of block tridiagonal arrowhead lu
if __name__ == "__main__":
    nblocks = 5
    diag_blocksize = 3
    arrow_blocksize = 2
    symmetric = False
    diagonal_dominant = True
    seed = 63

    (
        A_diagonal_blocks_ref,
        A_lower_diagonal_blocks_ref,
        A_upper_diagonal_blocks_ref,
        A_arrow_bottom_blocks_ref,
        A_arrow_right_blocks_ref,
        A_arrow_tip_block_ref,
    ) = generate_tridiag_arrowhead_arrays(
        nblocks, diag_blocksize, arrow_blocksize, symmetric, diagonal_dominant, seed
    )

    headers = {}
    headers["nblocks"] = nblocks
    headers["blocksize"] = diag_blocksize
    headers["arrow_blocksize"] = arrow_blocksize
    headers["symmetric"] = symmetric
    headers["diagonal_dominant"] = diagonal_dominant
    headers["seed"] = seed
    runs_timings = [headers]

    for i in range(N_WARMUPS + N_RUNS):
        A_diagonal_blocks = A_diagonal_blocks_ref.copy()
        A_lower_diagonal_blocks = A_lower_diagonal_blocks_ref.copy()
        A_upper_diagonal_blocks = A_upper_diagonal_blocks_ref.copy()
        A_arrow_bottom_blocks = A_arrow_bottom_blocks_ref.copy()
        A_arrow_right_blocks = A_arrow_right_blocks_ref.copy()
        A_arrow_tip_block = A_arrow_tip_block_ref.copy()

        (
            L_diagonal_blocks,
            L_lower_diagonal_blocks,
            L_arrow_bottom_blocks,
            U_diagonal_blocks,
            U_upper_diagonal_blocks,
            U_arrow_right_blocks,
            timings,
        ) = lu_factorize_tridiag_arrowhead(
            A_diagonal_blocks,
            A_lower_diagonal_blocks,
            A_upper_diagonal_blocks,
            A_arrow_bottom_blocks,
            A_arrow_right_blocks,
            A_arrow_tip_block,
        )

        if i >= N_WARMUPS:
            runs_timings.append(timings)

    # Save the timings and nblocks and blocksize
    runs_timings = np.array(runs_timings)
    print(runs_timings)
    np.save(PATH_TO_SAVE + "lu_factorize_tridiag_arrowhead_timings.npy", runs_timings)
