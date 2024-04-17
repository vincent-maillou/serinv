import numpy as np

from sdr.utils.matrix_transformation_dense import (
    convert_block_tridiagonal_arrays_to_dense,
    convert_block_tridiagonal_arrowhead_arrays_to_dense,
)

N_DIAG_BLOCKS = 5
DIAG_BLOCKSIZE = 4
ARROWHEAD_BLOCKSIZE = 2


def test_zeros_to_block_tridiagonal_shape():
    matrix_size = N_DIAG_BLOCKS * DIAG_BLOCKSIZE

    A_diag_blocks = np.random.rand(DIAG_BLOCKSIZE, DIAG_BLOCKSIZE * N_DIAG_BLOCKS)
    A_upper_blocks = np.random.rand(
        DIAG_BLOCKSIZE, DIAG_BLOCKSIZE * (N_DIAG_BLOCKS - 1)
    )
    A_lower_blocks = np.random.rand(
        DIAG_BLOCKSIZE, DIAG_BLOCKSIZE * (N_DIAG_BLOCKS - 1)
    )

    A_dense_tridiag = convert_block_tridiagonal_arrays_to_dense(
        A_diag_blocks, A_lower_blocks, A_upper_blocks
    )

    assert A_dense_tridiag.shape == (matrix_size, matrix_size)

    nonzero_indices = np.nonzero(A_dense_tridiag)
    for i, j in zip(*nonzero_indices):
        cond_1 = j <= i + 2 * DIAG_BLOCKSIZE
        cond_2 = j >= i - 2 * DIAG_BLOCKSIZE
        assert cond_1 and cond_2


def test_zeros_to_block_tridiagonal_arrowhead_shape():
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

    A_dense_tridiag_arrowhead = convert_block_tridiagonal_arrowhead_arrays_to_dense(
        A_diag_blocks,
        A_lower_blocks,
        A_upper_blocks,
        A_arrow_bottom_blocks,
        A_arrow_right_blocks,
        A_arrow_tip_block,
    )

    assert A_dense_tridiag_arrowhead.shape == (matrix_size, matrix_size)

    nonzero_indices = np.nonzero(A_dense_tridiag_arrowhead)
    for i, j in zip(*nonzero_indices):
        cond_1 = j <= i + 2 * DIAG_BLOCKSIZE
        cond_2 = j >= i - 2 * DIAG_BLOCKSIZE
        cond_3 = j >= matrix_size - ARROWHEAD_BLOCKSIZE
        cond_4 = i >= matrix_size - ARROWHEAD_BLOCKSIZE
        assert cond_1 and cond_2 or cond_3 or cond_4
