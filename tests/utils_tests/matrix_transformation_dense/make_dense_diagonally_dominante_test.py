# Copyright 2023-2024 ETH Zurich and USI. All rights reserved.

import numpy as np

from serinv.utils.matrix_transformation_dense import (
    make_dense_matrix_diagonally_dominante,
)


N_DIAG_BLOCKS = 5
DIAG_BLOCKSIZE = 4


def test_make_dense_matrix_diagonally_dominante():
    matrix_size = N_DIAG_BLOCKS * DIAG_BLOCKSIZE
    A = np.random.rand(matrix_size, matrix_size)

    A_dominante = make_dense_matrix_diagonally_dominante(A)

    for i in range(matrix_size):
        row_sum = 0.0
        for j in range(matrix_size):
            if i != j:
                row_sum += abs(A_dominante[i, j])
        assert abs(A_dominante[i, i]) > row_sum
