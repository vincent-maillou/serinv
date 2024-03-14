"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Lisa Gaedke-Merzhaeuser  (lisa.gaedke.merzhaeuser@usi.ch)
@date: 2023-11

Tests for lu selected inversion routines.

Copyright 2023-2024 ETH Zurich and USI. All rights reserved.
"""

import numpy as np

from sdr.lu.lu_factorize import lu_factorize_tridiag_arrowhead
from sdr.lu.lu_selected_inversion import lu_sinv_tridiag_arrowhead
from sdr.utils.matrix_generation import generate_tridiag_arrowhead_arrays


PATH_TO_SAVE = "../../"
N_WARMUPS = 3
N_RUNS = 10

# Testing of block tridiagonal lu sinv
if __name__ == "__main__":
    nblocks = 6
    diag_blocksize = 3
    arrow_blocksize = 2
    symmetric = False
    diagonal_dominant = True
    seed = 63

    (
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_upper_diagonal_blocks,
        A_arrow_bottom_blocks,
        A_arrow_right_blocks,
        A_arrow_tip_block,
    ) = generate_tridiag_arrowhead_arrays(
        nblocks, diag_blocksize, arrow_blocksize, symmetric, diagonal_dominant, seed
    )

    (
        L_diagonal_blocks_ref,
        L_lower_diagonal_blocks_ref,
        L_arrow_bottom_blocks_ref,
        U_diagonal_blocks_ref,
        U_upper_diagonal_blocks_ref,
        U_arrow_right_blocks_ref,
        _,
    ) = lu_factorize_tridiag_arrowhead(
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_upper_diagonal_blocks,
        A_arrow_bottom_blocks,
        A_arrow_right_blocks,
        A_arrow_tip_block,
    )

    headers = {}
    headers["N_WARMUPS"] = N_WARMUPS
    headers["N_RUNS"] = N_RUNS
    headers["nblocks"] = nblocks
    headers["blocksize"] = diag_blocksize
    headers["arrow_blocksize"] = arrow_blocksize
    headers["symmetric"] = symmetric
    headers["diagonal_dominant"] = diagonal_dominant
    headers["seed"] = seed
    runs_timings = [headers]

    for i in range(N_WARMUPS + N_RUNS):
        L_diagonal_blocks = L_diagonal_blocks_ref.copy()
        L_lower_diagonal_blocks = L_lower_diagonal_blocks_ref.copy()
        L_arrow_bottom_blocks = L_arrow_bottom_blocks_ref.copy()
        U_diagonal_blocks = U_diagonal_blocks_ref.copy()
        U_upper_diagonal_blocks = U_upper_diagonal_blocks_ref.copy()
        U_arrow_right_blocks = U_arrow_right_blocks_ref.copy()

        (
            X_sdr_diagonal_blocks,
            X_sdr_lower_diagonal_blocks,
            X_sdr_upper_diagonal_blocks,
            X_sdr_arrow_bottom_blocks,
            X_sdr_arrow_right_blocks,
            X_sdr_arrow_tip_block,
            timings,
        ) = lu_sinv_tridiag_arrowhead(
            L_diagonal_blocks,
            L_lower_diagonal_blocks,
            L_arrow_bottom_blocks,
            U_diagonal_blocks,
            U_upper_diagonal_blocks,
            U_arrow_right_blocks,
        )

        if i >= N_WARMUPS:
            runs_timings.append({**headers, **timings})

    # Save the timings and nblocks and blocksize
    runs_timings = np.array(runs_timings)
    print(runs_timings)
    np.save(PATH_TO_SAVE + "lu_sinv_tridiag_arrowhead_timings.npy", runs_timings)
