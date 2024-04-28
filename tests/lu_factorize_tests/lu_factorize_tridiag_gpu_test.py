# Copyright 2023-2024 ETH Zurich and USI. All rights reserved.

import sys

import numpy as np
import pytest
import scipy.linalg as la

try:
    from sdr.lu.lu_factorize_gpu import lu_factorize_tridiag_gpu

except ImportError:
    pass

from sdr.utils import matrix_generation_dense
from sdr.utils.matrix_transformation_dense import (
    convert_block_tridiagonal_arrays_to_dense,
)
from sdr.utils.matrix_transformation_arrays import (
    convert_block_tridiagonal_dense_to_arrays,
)


@pytest.mark.skipif(
    "cupy" not in sys.modules, reason="requires a working cupy installation"
)
@pytest.mark.gpu
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
def test_lu_decompose_tridiag_gpu(
    nblocks: int,
    blocksize: int,
):
    symmetric = False
    diagonal_dominant = True
    seed = 63

    A = matrix_generation_dense.generate_block_tridiagonal_dense(
        nblocks, blocksize, symmetric, diagonal_dominant, seed
    )

    # --- Decomposition ---

    P_ref, L_ref, U_ref = la.lu(A)

    if not np.allclose(P_ref, np.eye(A.shape[0])):
        L_ref = P_ref @ L_ref

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
    ) = lu_factorize_tridiag_gpu(
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_upper_diagonal_blocks,
    )

    L_sdr_dense = convert_block_tridiagonal_arrays_to_dense(
        L_diagonal_blocks,
        L_lower_diagonal_blocks,
        np.zeros((blocksize, (nblocks - 1) * blocksize)),
    )

    U_sdr_dense = convert_block_tridiagonal_arrays_to_dense(
        U_diagonal_blocks,
        np.zeros((blocksize, (nblocks - 1) * blocksize)),
        U_upper_diagonal_blocks,
    )

    assert np.allclose(L_ref, L_sdr_dense)
    assert np.allclose(U_ref, U_sdr_dense)
