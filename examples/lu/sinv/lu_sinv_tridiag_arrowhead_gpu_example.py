"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Lisa Gaedke-Merzhaeuser  (lisa.gaedke.merzhaeuser@usi.ch)
@date: 2023-11

Tests for lu selected inversion routines.

Copyright 2023-2024 ETH Zurich and USI. All rights reserved.
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la

from sdr.lu.lu_factorize_gpu import lu_factorize_tridiag_arrowhead_gpu
from sdr.lu.lu_selected_inversion_gpu import lu_sinv_tridiag_arrowhead_gpu
from sdr.utils import matrix_generation
from sdr.utils.matrix_transform import (
    cut_to_blocktridiag_arrowhead,
    from_arrowhead_arrays_to_dense,
    from_dense_to_arrowhead_arrays,
)

# Example of block tridiagonal lu sinv
if __name__ == "__main__":
    nblocks = 6
    diag_blocksize = 3
    arrow_blocksize = 2
    symmetric = False
    diagonal_dominant = True
    seed = 63

    A = matrix_generation.generate_tridiag_arrowhead_dense(
        nblocks, diag_blocksize, arrow_blocksize, symmetric, diagonal_dominant, seed
    )

    # --- Inversion ---

    X_ref = la.inv(A)
    X_ref = cut_to_blocktridiag_arrowhead(X_ref, diag_blocksize, arrow_blocksize)

    (
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_upper_diagonal_blocks,
        A_arrow_bottom_blocks,
        A_arrow_right_blocks,
        A_arrow_tip_block,
    ) = from_dense_to_arrowhead_arrays(A, diag_blocksize, arrow_blocksize)

    (
        L_diagonal_blocks,
        L_lower_diagonal_blocks,
        L_arrow_bottom_blocks,
        U_diagonal_blocks,
        U_upper_diagonal_blocks,
        U_arrow_right_blocks,
    ) = lu_factorize_tridiag_arrowhead_gpu(
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_upper_diagonal_blocks,
        A_arrow_bottom_blocks,
        A_arrow_right_blocks,
        A_arrow_tip_block,
    )

    (
        X_sdr_diagonal_blocks,
        X_sdr_lower_diagonal_blocks,
        X_sdr_upper_diagonal_blocks,
        X_sdr_arrow_bottom_blocks,
        X_sdr_arrow_right_blocks,
        X_sdr_arrow_tip_block,
    ) = lu_sinv_tridiag_arrowhead_gpu(
        L_diagonal_blocks,
        L_lower_diagonal_blocks,
        L_arrow_bottom_blocks,
        U_diagonal_blocks,
        U_upper_diagonal_blocks,
        U_arrow_right_blocks,
    )

    X_sdr = from_arrowhead_arrays_to_dense(
        X_sdr_diagonal_blocks,
        X_sdr_lower_diagonal_blocks,
        X_sdr_upper_diagonal_blocks,
        X_sdr_arrow_bottom_blocks,
        X_sdr_arrow_right_blocks,
        X_sdr_arrow_tip_block,
    )

    X_diff = X_ref - X_sdr

    fig, ax = plt.subplots(1, 3)
    ax[0].set_title("X_ref: Scipy reference inversion")
    ax[0].matshow(X_ref)
    ax[1].set_title("X_sdr: LU selected inversion")
    ax[1].matshow(X_sdr)
    ax[2].set_title("X_diff: Difference between X_ref and X_sdr")
    ax[2].matshow(X_diff)
    fig.colorbar(ax[2].matshow(X_diff), ax=ax[2], label="Relative error", shrink=0.4)

    plt.show()
