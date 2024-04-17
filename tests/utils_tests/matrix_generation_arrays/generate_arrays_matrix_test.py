import numpy as np
import pytest

from sdr.utils.matrix_generation_arrays import (
    generate_block_tridiagonal_arrays,
    generate_block_tridiagonal_arrowhead_arrays,
)

from sdr.utils.matrix_transformation_dense import (
    convert_block_tridiagonal_arrays_to_dense,
    convert_block_tridiagonal_arrowhead_arrays_to_dense,
)

N_DIAG_BLOCKS = 5
DIAG_BLOCKSIZE = 4
ARROWHEAD_BLOCKSIZE = 2


@pytest.mark.parametrize(
    "seed",
    [
        None,
        63,
    ],
)
@pytest.mark.parametrize(
    "symmetric",
    [
        False,
        True,
    ],
)
def test_generate_block_tridiagonal_arrays(
    seed,
    symmetric,
):
    A_diagonal_blocks, A_lower_diagonal_blocks, A_upper_diagonal_blocks = (
        generate_block_tridiagonal_arrays(
            N_DIAG_BLOCKS,
            DIAG_BLOCKSIZE,
            symmetric=symmetric,
            seed=seed,
        )
    )

    A_dense_tridiag = convert_block_tridiagonal_arrays_to_dense(
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_upper_diagonal_blocks,
    )

    nonzero_indices = np.nonzero(A_dense_tridiag)
    for i, j in zip(*nonzero_indices):
        cond_1 = j <= i + 2 * DIAG_BLOCKSIZE
        cond_2 = j >= i - 2 * DIAG_BLOCKSIZE
        assert cond_1 and cond_2
        if symmetric:
            assert A_dense_tridiag[i, j] == A_dense_tridiag[j, i]


@pytest.mark.parametrize(
    "seed",
    [
        None,
        63,
    ],
)
@pytest.mark.parametrize(
    "symmetric",
    [
        False,
        True,
    ],
)
def test_generate_block_tridiagonal_arrowhead_arrays(
    seed,
    symmetric,
):
    matrix_size = N_DIAG_BLOCKS * DIAG_BLOCKSIZE + ARROWHEAD_BLOCKSIZE

    (
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_upper_diagonal_blocks,
        A_arrow_bottom_blocks,
        A_arrow_right_blocks,
        A_arrow_tip_block,
    ) = generate_block_tridiagonal_arrowhead_arrays(
        N_DIAG_BLOCKS + 1,
        DIAG_BLOCKSIZE,
        ARROWHEAD_BLOCKSIZE,
        symmetric=symmetric,
        seed=seed,
    )

    A_dense_tridiag_arrowhead = convert_block_tridiagonal_arrowhead_arrays_to_dense(
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_upper_diagonal_blocks,
        A_arrow_bottom_blocks,
        A_arrow_right_blocks,
        A_arrow_tip_block,
    )

    nonzero_indices = np.nonzero(A_dense_tridiag_arrowhead)
    for i, j in zip(*nonzero_indices):
        cond_1 = j <= i + 2 * DIAG_BLOCKSIZE
        cond_2 = j >= i - 2 * DIAG_BLOCKSIZE
        cond_3 = j >= matrix_size - ARROWHEAD_BLOCKSIZE
        cond_4 = i >= matrix_size - ARROWHEAD_BLOCKSIZE
        assert cond_1 and cond_2 or cond_3 or cond_4
        if symmetric:
            assert A_dense_tridiag_arrowhead[i, j] == A_dense_tridiag_arrowhead[j, i]
