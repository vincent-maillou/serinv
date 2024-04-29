# Copyright 2023-2024 ETH Zurich and USI. All rights reserved.

import numpy as np

from serinv.utils.matrix_transformation_arrays import (
    make_arrays_block_tridiagonal_diagonally_dominante,
    make_arrays_block_tridiagonal_arrowhead_diagonally_dominante,
)

from serinv.utils.matrix_transformation_dense import (
    convert_block_tridiagonal_arrays_to_dense,
    convert_block_tridiagonal_arrowhead_arrays_to_dense,
)


N_DIAG_BLOCKS = 5
DIAG_BLOCKSIZE = 4
ARROWHEAD_BLOCKSIZE = 2


def test_make_arrays_block_tridiagonal_diagonally_dominante():
    matrix_size = N_DIAG_BLOCKS * DIAG_BLOCKSIZE
    A_diagonal_blocks = np.random.rand(DIAG_BLOCKSIZE, DIAG_BLOCKSIZE * N_DIAG_BLOCKS)
    A_lower_diagonal_blocks = np.random.rand(
        DIAG_BLOCKSIZE, DIAG_BLOCKSIZE * (N_DIAG_BLOCKS - 1)
    )
    A_upper_diagonal_blocks = np.random.rand(
        DIAG_BLOCKSIZE, DIAG_BLOCKSIZE * (N_DIAG_BLOCKS - 1)
    )

    (A_diagonal_blocks) = make_arrays_block_tridiagonal_diagonally_dominante(
        A_diagonal_blocks, A_lower_diagonal_blocks, A_upper_diagonal_blocks
    )

    A_dense = convert_block_tridiagonal_arrays_to_dense(
        A_diagonal_blocks, A_lower_diagonal_blocks, A_upper_diagonal_blocks
    )

    for i in range(matrix_size):
        row_sum = 0.0
        for j in range(matrix_size):
            if i != j:
                row_sum += abs(A_dense[i, j])
        assert abs(A_dense[i, i]) > row_sum


def test_make_arrays_block_tridiagonal_arrowhead_diagonally_dominante():
    matrix_size = N_DIAG_BLOCKS * DIAG_BLOCKSIZE + ARROWHEAD_BLOCKSIZE

    A_diag_blocks = np.random.rand(DIAG_BLOCKSIZE, DIAG_BLOCKSIZE * N_DIAG_BLOCKS)
    A_upper_blocks = np.random.rand(
        DIAG_BLOCKSIZE, DIAG_BLOCKSIZE * (N_DIAG_BLOCKS - 1)
    )
    A_lower_blocks = np.random.rand(
        DIAG_BLOCKSIZE, DIAG_BLOCKSIZE * (N_DIAG_BLOCKS - 1)
    )
    A_arrow_bottom_blocks = np.random.rand(
        ARROWHEAD_BLOCKSIZE, DIAG_BLOCKSIZE * N_DIAG_BLOCKS
    )
    A_arrow_right_blocks = np.random.rand(
        DIAG_BLOCKSIZE * N_DIAG_BLOCKS, ARROWHEAD_BLOCKSIZE
    )
    A_arrow_tip_block = np.random.rand(ARROWHEAD_BLOCKSIZE, ARROWHEAD_BLOCKSIZE)

    (A_diag_blocks, A_arrow_tip_block) = (
        make_arrays_block_tridiagonal_arrowhead_diagonally_dominante(
            A_diag_blocks,
            A_lower_blocks,
            A_upper_blocks,
            A_arrow_bottom_blocks,
            A_arrow_right_blocks,
            A_arrow_tip_block,
        )
    )

    A_dense = convert_block_tridiagonal_arrowhead_arrays_to_dense(
        A_diag_blocks,
        A_lower_blocks,
        A_upper_blocks,
        A_arrow_bottom_blocks,
        A_arrow_right_blocks,
        A_arrow_tip_block,
    )

    for i in range(matrix_size):
        row_sum = 0.0
        for j in range(matrix_size):
            if i != j:
                row_sum += abs(A_dense[i, j])
        assert abs(A_dense[i, i]) > row_sum
