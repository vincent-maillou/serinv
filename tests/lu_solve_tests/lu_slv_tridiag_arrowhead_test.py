"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Lisa Gaedke-Merzhaeuser  (lisa.gaedke.merzhaeuser@usi.ch)
@date: 2023-11

Tests for lu selected solving routines.

Copyright 2023-2024 ETH Zurich and USI. All rights reserved.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy.linalg as la

from sdr.lu.lu_solve import lu_slv_tridiag_arrowhead
from sdr.utils import matrix_generation

# Testing of block tridiagonal lu
if __name__ == "__main__":
    nblocks = 5
    diag_blocksize = 3
    arrow_blocksize = 2
    symmetric = True
    diagonal_dominant = True
    seed = 63

    A = matrix_generation.generate_tridiag_arrowhead_dense(
        nblocks, diag_blocksize, arrow_blocksize, symmetric, diagonal_dominant, seed
    )

    # P_ref, L_ref, U_ref = la.lu(A)
    lu_ref, p_ref = la.lu_factor(A)
    L_sdr, U_sdr = lu_dcmp_tridiag_arrowhead(A, diag_blocksize, arrow_blocksize)

    n_rhs = 1
    B = np.random.randn(A.shape[0], n_rhs)

    # --- Solving ---

    X_ref = la.lu_solve((lu_ref, p_ref), B)
    # Is equivalent to..
    # Y_ref = la.solve_triangular(L_ref, B, lower=True)
    # X_ref = la.solve_triangular(U_ref, Y_ref, lower=False)

    fig, ax = plt.subplots(1, 3)
    ax[0].set_title("X_ref: Reference lu solver")
    ax[0].matshow(X_ref)

    X_sdr = lu_slv_tridiag_arrowhead(L_sdr, U_sdr, B, diag_blocksize, arrow_blocksize)
    ax[1].set_title("X_sdr: Selected lu solver")
    ax[1].matshow(X_sdr)

    X_diff = X_ref - X_sdr
    ax[2].set_title("X_diff: Difference between X_ref and X_sdr")
    ax[2].matshow(X_diff)
    fig.colorbar(ax[2].matshow(X_diff), ax=ax[2], label="Relative error", shrink=0.4)

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
