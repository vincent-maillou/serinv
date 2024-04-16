import numpy as np

from sdr.utils.matrix_transformation_arrays import (
    make_arrays_block_tridiagonal_symmetric,
)

N_DIAG_BLOCKS = 5
DIAG_BLOCKSIZE = 4


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
