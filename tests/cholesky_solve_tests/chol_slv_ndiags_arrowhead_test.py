# Copyright 2023-2024 ETH Zurich and USI. All rights reserved.

import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy.linalg as la

from serinv.cholesky.cholesky_factorize import chol_dcmp_ndiags_arrowhead
from serinv.cholesky.cholesky_solve import chol_slv_ndiags_arrowhead
from serinv.utils import matrix_generation_dense

# Testing of block tridiagonal arrowhead cholesky
if __name__ == "__main__":
    nblocks = 5
    ndiags = 7
    diag_blocksize = 3
    arrow_blocksize = 2
    symmetric = True
    diagonal_dominant = True
    seed = 63

    A = matrix_generation_dense.generate_blocks_banded_arrowhead_dense(
        nblocks,
        ndiags,
        diag_blocksize,
        arrow_blocksize,
        symmetric,
        diagonal_dominant,
        seed,
    )

    L_ref = la.cholesky(A, lower=True)
    L_sdr = chol_dcmp_ndiags_arrowhead(A, ndiags, diag_blocksize, arrow_blocksize)

    nrhs = 4
    B = np.random.randn(A.shape[0], nrhs)

    # --- Solving ---

    X_ref = la.cho_solve((L_ref, True), B)
    # Is equivalent to..
    # Y_ref = la.solve_triangular(L_ref, B, lower=True)
    # X_ref = la.solve_triangular(L_ref.T, Y_ref, lower=False)

    fig, ax = plt.subplots(1, 3)
    ax[0].set_title("X_ref: Reference cholesky solver")
    ax[0].matshow(X_ref)

    X_sdr = chol_slv_ndiags_arrowhead(L_sdr, B, ndiags, diag_blocksize, arrow_blocksize)
    ax[1].set_title("X_sdr: Selected cholesky solver")
    ax[1].matshow(X_sdr)

    X_diff = X_ref - X_sdr
    ax[2].set_title("X_diff: Difference between X_ref and X_sdr")
    ax[2].matshow(X_diff)
    fig.colorbar(ax[2].matshow(X_diff), ax=ax[2], label="Relative error", shrink=0.4)

    plt.show()

    fig, ax = plt.subplots(1, 3)
    ax[0].set_title("X_ref: Reference cholesky solver")
    ax[0].matshow(X_ref)

    X_sdr = chol_slv_ndiags_arrowhead(
        L_sdr, B, ndiags, diag_blocksize, arrow_blocksize, overwrite=True
    )
    ax[1].set_title("X_sdr: Selected cholesky solver. Overwrite = True")
    ax[1].matshow(X_sdr)

    X_diff = X_ref - X_sdr
    ax[2].set_title("X_diff: Difference between X_ref and X_sdr")
    ax[2].matshow(X_diff)
    fig.colorbar(ax[2].matshow(X_diff), ax=ax[2], label="Relative error", shrink=0.4)

    plt.show()


@pytest.mark.cpu
@pytest.mark.mpi_skip()
@pytest.mark.parametrize(
    "nblocks, ndiags, diag_blocksize, arrow_blocksize, nrhs",
    [
        (2, 1, 1, 2, 1),
        (3, 3, 2, 1, 3),
        (4, 5, 1, 2, 5),
        (5, 7, 2, 1, 1),
        (15, 1, 3, 1, 2),
        (15, 3, 1, 2, 1),
        (15, 5, 3, 1, 6),
        (15, 7, 1, 2, 2),
    ],
)
@pytest.mark.parametrize("overwrite", [True, False])
def test_cholesky_slv_ndiags_arrowhead(
    nblocks: int,
    ndiags: int,
    diag_blocksize: int,
    arrow_blocksize: int,
    nrhs: int,
    overwrite: bool,
):
    symmetric = True
    diagonal_dominant = True
    seed = 63

    A = matrix_generation_dense.generate_blocks_banded_arrowhead_dense(
        nblocks,
        ndiags,
        diag_blocksize,
        arrow_blocksize,
        symmetric,
        diagonal_dominant,
        seed,
    )

    L_ref = la.cholesky(A, lower=True)
    L_sdr = chol_dcmp_ndiags_arrowhead(A, ndiags, diag_blocksize, arrow_blocksize)

    B = np.random.randn(A.shape[0], nrhs)

    X_ref = la.cho_solve((L_ref, True), B)
    X_sdr = chol_slv_ndiags_arrowhead(
        L_sdr, B, ndiags, diag_blocksize, arrow_blocksize, overwrite
    )

    if overwrite:
        assert np.allclose(X_ref, X_sdr) and B.ctypes.data == X_sdr.ctypes.data
    else:
        assert np.allclose(X_ref, X_sdr) and B.ctypes.data != X_sdr.ctypes.data
