# Copyright 2023-2024 ETH Zurich and USI. All rights reserved.

import numpy as np
import pytest

from serinv.lu.lu_factorize import lu_factorize_tridiag
from serinv.lu.lu_solve import lu_solve_tridiag
from serinv.utils import matrix_generation_dense
from serinv.utils.matrix_transformation_arrays import (
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
def test_lu_solve_tridiag(nblocks: int, blocksize: int, n_rhs: int):
    symmetric = False
    diagonal_dominant = True
    seed = 63

    A = matrix_generation_dense.generate_block_tridiagonal_dense(
        nblocks, blocksize, symmetric, diagonal_dominant, seed
    )

    # --- Factorization LU ---
    (
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_upper_diagonal_blocks,
    ) = convert_block_tridiagonal_dense_to_arrays(A, blocksize)

    (
        L_diagonal_blocks,
        L_lower_diagonal_blocks,
        U_diagonal_blocks,
        U_upper_diagonal_blocks,
    ) = lu_factorize_tridiag(
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_upper_diagonal_blocks,
    )

    # --- Solving ---
    B = np.random.randn(A.shape[0], n_rhs)
    X_ref = np.linalg.solve(A, B)
    X_sdr = lu_solve_tridiag(
        L_diagonal_blocks,
        L_lower_diagonal_blocks,
        U_diagonal_blocks,
        U_upper_diagonal_blocks,
        B,
    )

    assert np.allclose(X_ref, X_sdr)
