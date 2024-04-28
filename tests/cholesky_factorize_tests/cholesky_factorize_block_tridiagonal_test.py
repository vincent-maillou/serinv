# Copyright 2023-2024 ETH Zurich and USI. All rights reserved.

import numpy as np
import pytest
import scipy.linalg as la

from serinv.cholesky.cholesky_factorize import cholesky_factorize_block_tridiagonal
from serinv.utils.matrix_generation_dense import generate_block_tridiagonal_dense
from serinv.utils.matrix_transformation_arrays import (
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
def test_cholesky_decompose_tridiag(
    nblocks: int,
    blocksize: int,
):
    symmetric = True
    diagonal_dominant = True
    seed = 63

    A = generate_block_tridiagonal_dense(
        nblocks, blocksize, symmetric, diagonal_dominant, seed
    )

    L_ref = la.cholesky(A, lower=True)

    (L_diagonal_blocks_ref, L_lower_diagonal_blocks_ref, _) = (
        convert_block_tridiagonal_dense_to_arrays(L_ref, blocksize)
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

    assert np.allclose(L_diagonal_blocks, L_diagonal_blocks_ref)
    assert np.allclose(L_lower_diagonal_blocks, L_lower_diagonal_blocks_ref)
