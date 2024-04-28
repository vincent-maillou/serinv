# Copyright 2023-2024 ETH Zurich and USI. All rights reserved.

import numpy as np
import pytest
import scipy.linalg as la

from sdr.cholesky.cholesky_factorize import cholesky_factorize_block_tridiagonal
from sdr.cholesky.cholesky_selected_inversion import cholesky_sinv_block_tridiagonal
from sdr.utils import matrix_generation_dense
from sdr.utils.matrix_transformation_arrays import (
    convert_block_tridiagonal_dense_to_arrays,
)


@pytest.mark.cpu
@pytest.mark.mpi_skip()
@pytest.mark.parametrize(
    "nblocks, blocksize",
    [
        (2, 2),
        (10, 2),
        (100, 2),
        (2, 3),
        (10, 3),
        (100, 3),
        (2, 100),
        (5, 100),
        (10, 100),
    ],
)
def test_cholesky_sinv_tridiag(
    nblocks: int,
    blocksize: int,
):
    symmetric = True
    diagonal_dominant = True
    seed = 63

    A = matrix_generation_dense.generate_block_tridiagonal_dense(
        nblocks, blocksize, symmetric, diagonal_dominant, seed
    )

    # --- Inversion ---

    X_ref = la.inv(A)

    (X_diagonal_blocks_ref, X_lower_diagonal_blocks_ref, _) = (
        convert_block_tridiagonal_dense_to_arrays(X_ref, blocksize)
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

    (
        X_diagonal_blocks,
        X_lower_diagonal_blocks,
    ) = cholesky_sinv_block_tridiagonal(
        L_diagonal_blocks,
        L_lower_diagonal_blocks,
    )

    assert np.allclose(X_lower_diagonal_blocks_ref, X_lower_diagonal_blocks)
    assert np.allclose(X_diagonal_blocks_ref, X_diagonal_blocks)
