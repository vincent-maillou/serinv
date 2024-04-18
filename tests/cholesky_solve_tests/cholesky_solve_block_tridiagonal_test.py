"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Lisa Gaedke-Merzhaeuser  (lisa.gaedke.merzhaeuser@usi.ch)
@date: 2023-11

Tests for cholesky selected solving routines.

Copyright 2023-2024 ETH Zurich and USI. All rights reserved.
"""

import numpy as np
import numpy.linalg as npla
import pytest

from sdr.cholesky.cholesky_factorize import cholesky_factorize_block_tridiagonal
from sdr.cholesky.cholesky_solve import cholesky_solve_block_tridiagonal
from sdr.utils import matrix_generation_dense
from sdr.utils.matrix_transformation_arrays import (
    convert_block_tridiagonal_dense_to_arrays,
)


@pytest.mark.cpu
@pytest.mark.mpi_skip()
@pytest.mark.parametrize(
    "nblocks, blocksize, n_rhs",
    [
        (2, 2, 1),
        (10, 2, 3),
        (100, 2, 4),
        (2, 3, 1),
        (10, 3, 2),
        (100, 3, 1),
        (2, 100, 5),
        (5, 100, 2),
        (10, 100, 1),
    ],
)
def test_cholesky_slv_tridiag(
    nblocks: int,
    blocksize: int,
    n_rhs: int,
):
    symmetric = True
    diagonal_dominant = True
    seed = 63

    A = matrix_generation_dense.generate_block_tridiagonal_dense(
        nblocks, blocksize, symmetric, diagonal_dominant, seed
    )

    (A_diagonal_blocks, A_lower_diagonal_blocks, _) = (
        convert_block_tridiagonal_dense_to_arrays(A, blocksize)
    )

    (
        L_diagonal_blocks,
        L_lower_diagonal_blocks,
    ) = cholesky_factorize_block_tridiagonal(
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
    )

    B = np.random.randn(A.shape[0], n_rhs)

    # --- Solving ---

    X_ref = npla.solve(A, B)

    X_sdr = cholesky_solve_block_tridiagonal(
        L_diagonal_blocks,
        L_lower_diagonal_blocks,
        B,
    )

    assert np.allclose(X_ref, X_sdr)
