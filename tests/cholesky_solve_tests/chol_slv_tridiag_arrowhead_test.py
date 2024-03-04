"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Lisa Gaedke-Merzhaeuser  (lisa.gaedke.merzhaeuser@usi.ch)
@date: 2023-11

Tests for cholesky selected solving routines.

Copyright 2023-2024 ETH Zurich and USI. All rights reserved.
"""

from sdr.utils import matrix_generation
from sdr.cholesky.cholesky_decompose import chol_dcmp_tridiag_arrowhead
from sdr.cholesky.cholesky_solve import chol_slv_tridiag_arrowhead

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import pytest


# Testing of block tridiagonal arrowhead cholesky
if __name__ == "__main__":
    nblocks = 5
    diag_blocksize = 3
    arrow_blocksize = 2
    symmetric = True
    diagonal_dominant = True
    seed = 63

    A = matrix_generation.generate_tridiag_arrowhead_dense(
        nblocks, diag_blocksize, arrow_blocksize, symmetric, diagonal_dominant, 
        seed
    )

    L_ref = la.cholesky(A, lower=True)
    L_sdr = chol_dcmp_tridiag_arrowhead(A, diag_blocksize, arrow_blocksize)

    n_rhs = 3
    B = np.random.randn(A.shape[0], n_rhs)


    # --- Solving ---

    #X_ref = la.cho_solve((L_ref, True), B)
    # Is equivalent to..
    Y_ref = la.solve_triangular(L_ref, B, lower=True)
    X_ref = la.solve_triangular(L_ref.T, Y_ref, lower=False)

    fig, ax = plt.subplots(1, 3)
    ax[0].set_title("X_ref: Reference cholesky solver")
    ax[0].matshow(X_ref)

    X_sdr = chol_slv_tridiag_arrowhead(L_sdr, B, diag_blocksize, arrow_blocksize)
    ax[1].set_title("X_sdr: Selected cholesky solver")
    ax[1].matshow(X_sdr)

    X_diff = X_ref - X_sdr
    ax[2].set_title("X_diff: Difference between X_ref and X_sdr")
    ax[2].matshow(X_diff)
    fig.colorbar(ax[2].matshow(X_diff), ax=ax[2], label="Relative error", shrink=0.4)

    plt.show()
    
    
@pytest.mark.mpi_skip()
@pytest.mark.parametrize(
    "nblocks, diag_blocksize, arrow_blocksize, nrhs", 
    [
        (2, 2, 2, 1),
        (2, 3, 2, 2),
        (2, 2, 3, 5),
        (10, 2, 2, 1),
        (10, 3, 2, 4),
        (10, 2, 3, 8),
        (10, 10, 2, 1),
        (10, 2, 10, 1),
    ]
)
def test_cholesky_slv_tridiag_arrowhead(
    nblocks: int, 
    diag_blocksize: int, 
    arrow_blocksize: int, 
    nrhs: int,
):
    symmetric = True
    diagonal_dominant = True
    seed = 63

    A = matrix_generation.generate_tridiag_arrowhead_dense(
        nblocks, diag_blocksize, arrow_blocksize, symmetric, diagonal_dominant, 
        seed
    )

    L_ref = la.cholesky(A, lower=True)
    L_sdr = chol_dcmp_tridiag_arrowhead(A, diag_blocksize, arrow_blocksize)
    
    n_rhs = 1
    B = np.random.randn(A.shape[0], n_rhs)
    
    X_ref = la.cho_solve((L_ref, True), B)
    X_sdr = chol_slv_tridiag_arrowhead(L_sdr, B, diag_blocksize, arrow_blocksize)

    assert np.allclose(X_ref, X_sdr)
