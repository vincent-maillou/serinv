# Copyright 2023-2024 ETH Zurich and USI. All rights reserved.

import numpy as np
import pytest
import scipy.linalg as la

from serinv.lu.lu_factorize import lu_factorize_tridiag
from serinv.lu.lu_selected_inversion import lu_sinv_tridiag
from serinv.utils import matrix_generation_dense
from serinv.utils.matrix_transformation_dense import (
    zeros_to_block_tridiagonal_shape,
    convert_block_tridiagonal_arrays_to_dense,
)
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
def test_lu_sinv_tridiag(
    nblocks: int,
    blocksize: int,
):
    symmetric = False
    diagonal_dominant = True
    seed = 63

    A = matrix_generation_dense.generate_block_tridiagonal_dense(
        nblocks, blocksize, symmetric, diagonal_dominant, seed
    )

    # --- Inversion ---

    X_ref = la.inv(A)
    X_ref = zeros_to_block_tridiagonal_shape(X_ref, blocksize)

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
        _,
    ) = lu_factorize_tridiag(
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_upper_diagonal_blocks,
    )

    (
        X_sdr_diagonal_blocks,
        X_sdr_lower_diagonal_blocks,
        X_sdr_upper_diagonal_blocks,
        _,
    ) = lu_sinv_tridiag(
        L_diagonal_blocks,
        L_lower_diagonal_blocks,
        U_diagonal_blocks,
        U_upper_diagonal_blocks,
    )

    X_sdr_dense = convert_block_tridiagonal_arrays_to_dense(
        X_sdr_diagonal_blocks,
        X_sdr_lower_diagonal_blocks,
        X_sdr_upper_diagonal_blocks,
    )

    assert np.allclose(X_ref, X_sdr_dense)
