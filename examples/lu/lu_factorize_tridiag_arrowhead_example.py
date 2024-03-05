"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Lisa Gaedke-Merzhaeuser  (lisa.gaedke.merzhaeuser@usi.ch)
@date: 2024-02

Tests for lu tridiagonal arrowhead matrices selected factorization routine.

Copyright 2023-2024 ETH Zurich and USI. All rights reserved.
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la

from sdr.lu.lu_factorize import lu_factorize_tridiag_arrowhead
from sdr.utils import matrix_generation
from sdr.utils.matrix_transform import (
    from_arrowhead_arrays_to_dense,
    from_dense_to_arrowhead_arrays,
)

# Testing of block tridiagonal arrowhead lu
if __name__ == "__main__":
    nblocks = 5
    diag_blocksize = 3
    arrow_blocksize = 2
    symmetric = False
    diagonal_dominant = True
    seed = 63

    A = matrix_generation.generate_tridiag_arrowhead_dense(
        nblocks, diag_blocksize, arrow_blocksize, symmetric, diagonal_dominant, seed
    )

    # --- Decomposition ---

    P_ref, L_ref, U_ref = la.lu(A, permute_l=False)

    if not np.allclose(P_ref, np.eye(A.shape[0])):
        plt.matshow(P_ref)
        plt.title("WARNING: Permutation matrix should be identity")

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
    ) = lu_factorize_tridiag_arrowhead(
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_upper_diagonal_blocks,
        A_arrow_bottom_blocks,
        A_arrow_right_blocks,
        A_arrow_tip_block,
    )

    L_sdr = from_arrowhead_arrays_to_dense(
        L_diagonal_blocks,
        L_lower_diagonal_blocks,
        np.zeros((diag_blocksize, (nblocks - 1) * diag_blocksize)),
        L_arrow_bottom_blocks[:, :-arrow_blocksize],
        np.zeros(((nblocks - 1) * diag_blocksize, arrow_blocksize)),
        L_arrow_bottom_blocks[:, -arrow_blocksize:],
    )

    U_sdr = from_arrowhead_arrays_to_dense(
        U_diagonal_blocks,
        np.zeros((diag_blocksize, (nblocks - 1) * diag_blocksize)),
        U_upper_diagonal_blocks,
        np.zeros((arrow_blocksize, (nblocks - 1) * diag_blocksize)),
        U_arrow_right_blocks[:-arrow_blocksize, :],
        U_arrow_right_blocks[-arrow_blocksize:, :],
    )

    fig, ax = plt.subplots(2, 3)
    ax[0, 0].set_title("L_ref: Scipy lower factor")
    ax[0, 0].matshow(L_ref)
    ax[1, 0].set_title("U_ref: Scipy upper factor")
    ax[1, 0].matshow(U_ref)

    ax[0, 1].set_title("L_sdr: SDR lower factor")
    ax[0, 1].matshow(L_sdr)
    ax[1, 1].set_title("U_sdr: SDR upper factor")
    ax[1, 1].matshow(U_sdr)

    L_diff = L_ref - L_sdr
    U_diff = U_ref - U_sdr
    ax[0, 2].set_title("L: Difference between L_ref and L_sdr")
    ax[0, 2].matshow(L_diff)
    ax[1, 2].set_title("U: Difference between U_ref and U_sdr")
    ax[1, 2].matshow(U_diff)
    fig.colorbar(ax[0, 2].matshow(L_diff), ax=ax[0, 2], label="Relative error")
    fig.colorbar(ax[1, 2].matshow(U_diff), ax=ax[1, 2], label="Relative error")

    plt.show()

    assert np.allclose(L_ref, L_sdr)
    assert np.allclose(U_ref, U_sdr)
