# Copyright 2023-2024 ETH Zurich and USI. All rights reserved.

import numpy as np
import pytest

from serinv.utils.matrix_transformation_dense import make_dense_matrix_symmetric


N_DIAG_BLOCKS = 5
DIAG_BLOCKSIZE = 4


# Use pytest mark parametrize to test differents type complex and float
@pytest.mark.parametrize(
    "dtype",
    [
        np.float64,
        np.complex128,
    ],
)
def test_make_dense_matrix_diagonally_dominante(dtype):
    matrix_size = N_DIAG_BLOCKS * DIAG_BLOCKSIZE
    A = np.random.rand(matrix_size, matrix_size).astype(dtype)

    if np.iscomplexobj(A):
        A = A + 1j * np.random.rand(matrix_size, matrix_size).astype(dtype)

    A_symm = make_dense_matrix_symmetric(A)

    for i in range(matrix_size):
        row_sum = 0.0
        for j in range(i, matrix_size):
            assert A_symm[i, j] == A_symm[j, i]
