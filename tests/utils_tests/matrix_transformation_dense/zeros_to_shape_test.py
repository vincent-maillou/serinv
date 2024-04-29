# Copyright 2023-2024 ETH Zurich and USI. All rights reserved.

import numpy as np

from serinv.utils.matrix_transformation_dense import (
    zeros_to_block_tridiagonal_shape,
    zeros_to_block_tridiagonal_arrowhead_shape,
    zeros_to_blocks_banded_shape,
    zeros_to_blocks_banded_arrowhead_shape,
)

N_DIAG_BLOCKS = 5
DIAG_BLOCKSIZE = 4
ARROWHEAD_BLOCKSIZE = 2
N_BLOCKS_BANDED = 5


def test_zeros_to_block_tridiagonal_shape():
    matrix_size = N_DIAG_BLOCKS * DIAG_BLOCKSIZE
    A = np.random.rand(matrix_size, matrix_size)

    A_tridiag = zeros_to_block_tridiagonal_shape(A, DIAG_BLOCKSIZE)

    assert A_tridiag.shape == (matrix_size, matrix_size)

    nonzero_indices = np.nonzero(A_tridiag)
    for i, j in zip(*nonzero_indices):
        cond_1 = j <= i + 2 * DIAG_BLOCKSIZE
        cond_2 = j >= i - 2 * DIAG_BLOCKSIZE
        assert cond_1 and cond_2


def test_zeros_to_block_tridiagonal_arrowhead_shape():
    matrix_size = N_DIAG_BLOCKS * DIAG_BLOCKSIZE + ARROWHEAD_BLOCKSIZE
    A = np.random.rand(matrix_size, matrix_size)

    A_tridiag_arrowhead = zeros_to_block_tridiagonal_arrowhead_shape(
        A, DIAG_BLOCKSIZE, ARROWHEAD_BLOCKSIZE
    )

    assert A_tridiag_arrowhead.shape == (matrix_size, matrix_size)

    nonzero_indices = np.nonzero(A_tridiag_arrowhead)
    for i, j in zip(*nonzero_indices):
        cond_1 = j <= i + 2 * DIAG_BLOCKSIZE
        cond_2 = j >= i - 2 * DIAG_BLOCKSIZE
        cond_3 = j >= matrix_size - ARROWHEAD_BLOCKSIZE
        cond_4 = i >= matrix_size - ARROWHEAD_BLOCKSIZE
        assert cond_1 and cond_2 or cond_3 or cond_4


def test_zeros_to_blocks_banded_shape():
    matrix_size = N_DIAG_BLOCKS * DIAG_BLOCKSIZE
    A = np.random.rand(matrix_size, matrix_size)

    A_tridiag = zeros_to_blocks_banded_shape(A, N_BLOCKS_BANDED, DIAG_BLOCKSIZE)

    assert A_tridiag.shape == (matrix_size, matrix_size)

    nonzero_indices = np.nonzero(A_tridiag)
    for i, j in zip(*nonzero_indices):
        cond_1 = j <= i + (N_BLOCKS_BANDED // 2 + 1) * DIAG_BLOCKSIZE
        cond_2 = j >= i - (N_BLOCKS_BANDED // 2 + 1) * DIAG_BLOCKSIZE
        assert cond_1 and cond_2


def test_zeros_to_blocks_banded_arrowhead_shape():
    matrix_size = N_DIAG_BLOCKS * DIAG_BLOCKSIZE + ARROWHEAD_BLOCKSIZE
    A = np.random.rand(matrix_size, matrix_size)

    A_tridiag_arrowhead = zeros_to_blocks_banded_arrowhead_shape(
        A, N_BLOCKS_BANDED, DIAG_BLOCKSIZE, ARROWHEAD_BLOCKSIZE
    )

    assert A_tridiag_arrowhead.shape == (matrix_size, matrix_size)

    nonzero_indices = np.nonzero(A_tridiag_arrowhead)
    for i, j in zip(*nonzero_indices):
        cond_1 = j <= i + (N_BLOCKS_BANDED // 2 + 1) * DIAG_BLOCKSIZE
        cond_2 = j >= i - (N_BLOCKS_BANDED // 2 + 1) * DIAG_BLOCKSIZE
        cond_3 = j >= matrix_size - ARROWHEAD_BLOCKSIZE
        cond_4 = i >= matrix_size - ARROWHEAD_BLOCKSIZE
        assert cond_1 and cond_2 or cond_3 or cond_4
