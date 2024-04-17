import numpy as np

from sdr.utils.matrix_transformation_arrays import (
    make_arrays_block_tridiagonal_symmetric,
    make_arrays_block_tridiagonal_arrowhead_symmetric,
)

N_DIAG_BLOCKS = 5
DIAG_BLOCKSIZE = 4
ARROWHEAD_BLOCKSIZE = 2


def test_make_arrays_block_tridiagonal_symmetric():
    A_diagonal_blocks = np.random.rand(DIAG_BLOCKSIZE, DIAG_BLOCKSIZE * N_DIAG_BLOCKS)
    A_lower_diagonal_blocks = np.random.rand(
        DIAG_BLOCKSIZE, DIAG_BLOCKSIZE * (N_DIAG_BLOCKS - 1)
    )
    A_upper_diagonal_blocks = np.random.rand(
        DIAG_BLOCKSIZE, DIAG_BLOCKSIZE * (N_DIAG_BLOCKS - 1)
    )

    (A_diagonal_blocks, A_lower_diagonal_blocks, A_upper_diagonal_blocks) = (
        make_arrays_block_tridiagonal_symmetric(
            A_diagonal_blocks, A_lower_diagonal_blocks, A_upper_diagonal_blocks
        )
    )

    for i in range(N_DIAG_BLOCKS):
        assert np.allclose(
            A_diagonal_blocks[:, i * DIAG_BLOCKSIZE : (i + 1) * DIAG_BLOCKSIZE],
            A_diagonal_blocks[:, i * DIAG_BLOCKSIZE : (i + 1) * DIAG_BLOCKSIZE].T,
        )
        if i < N_DIAG_BLOCKS - 1:
            assert np.allclose(
                A_lower_diagonal_blocks[
                    :, i * DIAG_BLOCKSIZE : (i + 1) * DIAG_BLOCKSIZE
                ],
                A_upper_diagonal_blocks[
                    :, i * DIAG_BLOCKSIZE : (i + 1) * DIAG_BLOCKSIZE
                ].T,
            )


def test_make_arrays_block_tridiagonal_arrowhead_symmetric():
    A_diagonal_blocks = np.random.rand(DIAG_BLOCKSIZE, DIAG_BLOCKSIZE * N_DIAG_BLOCKS)
    A_lower_diagonal_blocks = np.random.rand(
        DIAG_BLOCKSIZE, DIAG_BLOCKSIZE * (N_DIAG_BLOCKS - 1)
    )
    A_upper_diagonal_blocks = np.random.rand(
        DIAG_BLOCKSIZE, DIAG_BLOCKSIZE * (N_DIAG_BLOCKS - 1)
    )
    A_arrow_bottom_blocks = np.random.rand(
        ARROWHEAD_BLOCKSIZE, DIAG_BLOCKSIZE * N_DIAG_BLOCKS
    )
    A_arrow_right_blocks = np.random.rand(
        DIAG_BLOCKSIZE * N_DIAG_BLOCKS, ARROWHEAD_BLOCKSIZE
    )
    A_arrow_tip_block = np.random.rand(ARROWHEAD_BLOCKSIZE, ARROWHEAD_BLOCKSIZE)

    (
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_upper_diagonal_blocks,
        A_arrow_bottom_blocks,
        A_arrow_right_blocks,
        A_arrow_tip_block,
    ) = make_arrays_block_tridiagonal_arrowhead_symmetric(
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_upper_diagonal_blocks,
        A_arrow_bottom_blocks,
        A_arrow_right_blocks,
        A_arrow_tip_block,
    )

    for i in range(N_DIAG_BLOCKS):
        assert np.allclose(
            A_diagonal_blocks[:, i * DIAG_BLOCKSIZE : (i + 1) * DIAG_BLOCKSIZE],
            A_diagonal_blocks[:, i * DIAG_BLOCKSIZE : (i + 1) * DIAG_BLOCKSIZE].T,
        )
        if i < N_DIAG_BLOCKS - 1:
            assert np.allclose(
                A_lower_diagonal_blocks[
                    :, i * DIAG_BLOCKSIZE : (i + 1) * DIAG_BLOCKSIZE
                ],
                A_upper_diagonal_blocks[
                    :, i * DIAG_BLOCKSIZE : (i + 1) * DIAG_BLOCKSIZE
                ].T,
            )
            assert np.allclose(
                A_arrow_bottom_blocks[:, i * DIAG_BLOCKSIZE : (i + 1) * DIAG_BLOCKSIZE],
                A_arrow_right_blocks[
                    i * DIAG_BLOCKSIZE : (i + 1) * DIAG_BLOCKSIZE, :
                ].T,
            )

    assert np.allclose(A_arrow_tip_block, A_arrow_tip_block.T)
