"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Lisa Gaedke-Merzhaeuser  (lisa.gaedke.merzhaeuser@usi.ch)
@date: 2023-11

Tests for cholesky selected solving routines.

Copyright 2023-2024 ETH Zurich and USI. All rights reserved.
"""

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as npla
import pytest
import scipy.linalg as la

from sdr.cholesky.cholesky_factorize import (
    cholesky_factorize_block_tridiagonal_arrowhead,
)
from sdr.cholesky.cholesky_solve import cholesky_solve_block_tridiagonal_arrowhead
from sdr.utils import matrix_generation_dense
from sdr.utils.matrix_transformation_arrays import (
    convert_block_tridiagonal_arrowhead_dense_to_arrays,
)

# Testing of block tridiagonal arrowhead cholesky
if __name__ == "__main__":
    nblocks = 5
    diag_blocksize = 4
    arrow_blocksize = 2
    symmetric = True
    diagonal_dominant = True
    seed = 63

    A = matrix_generation_dense.generate_block_tridiagonal_arrowhead_dense(
        nblocks, diag_blocksize, arrow_blocksize, symmetric, diagonal_dominant, seed
    )

    (
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        _,
        A_arrow_bottom_blocks,
        _,
        A_arrow_tip_block,
    ) = convert_block_tridiagonal_arrowhead_dense_to_arrays(
        A, diag_blocksize, arrow_blocksize
    )

    (
        L_diagonal_blocks,
        L_lower_diagonal_blocks,
        L_arrow_bottom_blocks,
        L_arrow_tip_block,
    ) = cholesky_factorize_block_tridiagonal_arrowhead(
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_arrow_bottom_blocks,
        A_arrow_tip_block,
    )

    n_rhs = 3
    B = np.random.randn(A.shape[0], n_rhs)

    X_ref = npla.solve(A, B)

    X_sdr = cholesky_solve_block_tridiagonal_arrowhead(
        L_diagonal_blocks,
        L_lower_diagonal_blocks,
        L_arrow_bottom_blocks,
        L_arrow_tip_block,
        B,
    )

    fig, ax = plt.subplots(1, 3)
    ax[0].set_title("X_ref")
    ax[0].matshow(X_ref)

    ax[1].set_title("X_sdr")
    ax[1].matshow(X_sdr)

    X_diff = X_ref - X_sdr
    ax[2].set_title("X_diff: Difference between X_ref and X_sdr")
    ax[2].matshow(X_diff)
    fig.colorbar(ax[2].matshow(X_diff), ax=ax[2], label="Relative error", shrink=0.4)

    plt.show()


@pytest.mark.cpu
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
    ],
)
@pytest.mark.parametrize("overwrite", [True, False])
def test_cholesky_slv_tridiag_arrowhead(
    nblocks: int,
    diag_blocksize: int,
    arrow_blocksize: int,
    nrhs: int,
    overwrite: bool,
):
    symmetric = True
    diagonal_dominant = True
    seed = 63

    A = matrix_generation_dense.generate_block_tridiagonal_arrowhead_dense(
        nblocks, diag_blocksize, arrow_blocksize, symmetric, diagonal_dominant, seed
    )

    L_ref = la.cholesky(A, lower=True)
    L_sdr = chol_dcmp_tridiag_arrowhead(A, diag_blocksize, arrow_blocksize)

    n_rhs = 1
    B = np.random.randn(A.shape[0], n_rhs)

    X_ref = la.cho_solve((L_ref, True), B)
    X_sdr = chol_slv_tridiag_arrowhead(
        L_sdr, B, diag_blocksize, arrow_blocksize, overwrite
    )

    if overwrite:
        assert np.allclose(X_ref, X_sdr) and B.ctypes.data == X_sdr.ctypes.data
    else:
        assert np.allclose(X_ref, X_sdr) and B.ctypes.data != X_sdr.ctypes.data
