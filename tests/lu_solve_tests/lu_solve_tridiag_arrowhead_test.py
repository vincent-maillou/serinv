"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Lisa Gaedke-Merzhaeuser  (lisa.gaedke.merzhaeuser@usi.ch)
@date: 2023-11

Tests for lu selected solving routines.

Copyright 2023-2024 ETH Zurich and USI. All rights reserved.
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import pytest
import copy as cp

from sdr.lu.lu_factorize import lu_factorize_tridiag_arrowhead
from sdr.lu.lu_solve import lu_solve_tridiag_arrowhead
from sdr.utils import matrix_generation
from sdr.utils.matrix_transform import (
    from_dense_to_arrowhead_arrays,
)

# Testing of block tridiagonal lu
if __name__ == "__main__":
    nblocks = 5
    diag_blocksize = 3
    arrow_blocksize = 2
    symmetric = True
    diagonal_dominant = True
    seed = 63
    n_rhs = 1

    A = matrix_generation.generate_tridiag_arrowhead_dense(
        nblocks, diag_blocksize, arrow_blocksize, symmetric, diagonal_dominant, seed
    )

    A_copy = cp.deepcopy(A)

    # --- Factorization LU ---
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

    fig, ax = plt.subplots(2, 3)
    ax[0, 0].set_title("L_diagonal_blocks")
    ax[0, 0].matshow(L_diagonal_blocks)
    ax[0, 1].set_title("L_lower_diagonal_blocks")
    ax[0, 1].matshow(L_lower_diagonal_blocks)
    ax[0, 2].set_title("L_arrow_bottom_blocks")
    ax[0, 2].matshow(L_arrow_bottom_blocks)

    ax[1, 0].set_title("U_diagonal_blocks")
    ax[1, 0].matshow(U_diagonal_blocks)
    ax[1, 1].set_title("U_upper_diagonal_blocks")
    ax[1, 1].matshow(U_upper_diagonal_blocks)
    ax[1, 2].set_title("U_arrow_right_blocks")
    ax[1, 2].matshow(U_arrow_right_blocks)
    plt.show()

    # --- Solving ---
    B = np.random.randn(A.shape[0], n_rhs)

    Y_sdr, X_sdr = lu_solve_tridiag_arrowhead(
        L_diagonal_blocks,
        L_lower_diagonal_blocks,
        L_arrow_bottom_blocks,
        U_diagonal_blocks,
        U_upper_diagonal_blocks,
        U_arrow_right_blocks,
        B,
    )

    # X_solve_ref = np.linalg.solve(A_copy, B)

    P, L, U = la.lu(A_copy)

    Y_L_solve_ref = la.solve_triangular(L, B, lower=True)
    X_U_solve_ref = la.solve_triangular(U, Y_L_solve_ref, lower=False)

    A_inv_ref = la.inv(A_copy)
    X_inv_ref = A_inv_ref @ B


    """ fig, ax = plt.subplots(1, 3)
    fig.suptitle("Reference X comparison")
    ax[0].set_title("X_solve_ref")
    ax[0].matshow(X_solve_ref)
    ax[1].set_title("X_U_solve_ref")
    ax[1].matshow(X_U_solve_ref)
    ax[2].set_title("X_inv_ref")
    ax[2].matshow(X_inv_ref)
    plt.show() """

    fig, ax = plt.subplots(1, 3)
    fig.suptitle("")
    ax[0].set_title("Y_L_solve_ref")
    ax[0].matshow(Y_L_solve_ref)
    ax[1].set_title("Y_sdr")
    ax[1].matshow(Y_sdr)
    Y_diff = Y_L_solve_ref - Y_sdr
    ax[2].set_title("Y_diff")
    ax[2].matshow(Y_diff)
    plt.show()

    # np.testing.assert_allclose(Y_L_solve_ref, Y_sdr)


    """ fig, ax = plt.subplots(1, 3)
    ax[0].set_title("X_ref: Reference lu solver")
    ax[0].matshow(X_ref)

    ax[1].set_title("X_sdr: Selected lu solver")
    ax[1].matshow(X_sdr)

    X_diff = X_ref - X_sdr
    ax[2].set_title("X_diff: Difference between X_ref and X_sdr")
    ax[2].matshow(X_diff)
    fig.colorbar(ax[2].matshow(X_diff), ax=ax[2], label="Relative error", shrink=0.4) """

    plt.show()


# @pytest.mark.cpu
# @pytest.mark.mpi_skip()
# @pytest.mark.parametrize(
#     "nblocks, diag_blocksize, arrow_blocksize, nrhs",
#     [
#         (2, 2, 2, 1),
#         (2, 3, 2, 2),
#         (2, 2, 3, 5),
#         (10, 2, 2, 1),
#         (10, 3, 2, 4),
#         (10, 2, 3, 8),
#         (10, 10, 2, 1),
#         (10, 2, 10, 1),
#     ]
# )
# def test_lu_slv_tridiag_arrowhead(
#     nblocks: int,
#     diag_blocksize: int,
#     arrow_blocksize: int,
#     nrhs: int,
# ):
#     symmetric = False
#     diagonal_dominant = True
#     seed = 63

#     A = matrix_generation.generate_tridiag_arrowhead_dense(
#         nblocks, diag_blocksize, arrow_blocksize, symmetric, diagonal_dominant,
#         seed
#     )

#     lu_ref, p_ref = la.lu_factor(A)
#     L_sdr, U_sdr = lu_dcmp_tridiag_arrowhead(A, diag_blocksize, arrow_blocksize)

#     B = np.random.randn(A.shape[0], nrhs)

#     X_ref = la.lu_solve((lu_ref, p_ref), B)
#     X_sdr = lu_slv_tridiag_arrowhead(L_sdr, U_sdr, B, diag_blocksize, arrow_blocksize)

#     assert np.allclose(X_ref, X_sdr)
