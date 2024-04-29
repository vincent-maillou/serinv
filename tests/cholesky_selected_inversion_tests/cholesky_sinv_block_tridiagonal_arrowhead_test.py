# Copyright 2023-2024 ETH Zurich and USI. All rights reserved.

import numpy as np
import pytest
import scipy.linalg as la

from serinv.cholesky.cholesky_factorize import (
    cholesky_factorize_block_tridiagonal_arrowhead,
)
from serinv.cholesky.cholesky_selected_inversion import (
    cholesky_sinv_block_tridiagonal_arrowhead,
)
from serinv.utils import matrix_generation_dense
from serinv.utils.matrix_transformation_arrays import (
    convert_block_tridiagonal_arrowhead_dense_to_arrays,
)


@pytest.mark.cpu
@pytest.mark.mpi_skip()
@pytest.mark.parametrize(
    "nblocks, diag_blocksize, arrow_blocksize",
    [
        (2, 2, 2),
        (2, 3, 2),
        (2, 2, 3),
        (10, 2, 2),
        (10, 3, 2),
        (10, 2, 3),
        (10, 10, 2),
        (10, 2, 10),
    ],
)
def test_cholesky_sinv_tridiag_arrowhead(
    nblocks: int,
    diag_blocksize: int,
    arrow_blocksize: int,
):
    symmetric = True
    diagonal_dominant = True
    seed = 63

    A = matrix_generation_dense.generate_block_tridiagonal_arrowhead_dense(
        nblocks, diag_blocksize, arrow_blocksize, symmetric, diagonal_dominant, seed
    )

    X_ref = la.inv(A)

    (
        X_diagonal_blocks_ref,
        X_lower_diagonal_blocks_ref,
        _,
        X_arrow_bottom_blocks_ref,
        _,
        X_arrow_tip_block_ref,
    ) = convert_block_tridiagonal_arrowhead_dense_to_arrays(
        X_ref, diag_blocksize, arrow_blocksize
    )

    (
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        _,
        A_arrow_bottom_blocks,
        _,
        A_arrow_tip_block,
    ) = convert_block_tridiagonal_arrowhead_dense_to_arrays(
        A, diag_blocksize, arrow_blocksize
    )

    (
        L_diagonal_blocks,
        L_lower_diagonal_blocks,
        L_arrow_bottom_blocks,
        L_arrow_tip_block,
    ) = cholesky_factorize_block_tridiagonal_arrowhead(
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_arrow_bottom_blocks,
        A_arrow_tip_block,
    )

    (
        X_diagonal_blocks,
        X_lower_diagonal_blocks,
        X_arrow_bottom_blocks,
        X_arrow_tip_block,
    ) = cholesky_sinv_block_tridiagonal_arrowhead(
        L_diagonal_blocks,
        L_lower_diagonal_blocks,
        L_arrow_bottom_blocks,
        L_arrow_tip_block,
    )

    assert np.allclose(X_diagonal_blocks_ref, X_diagonal_blocks)
    assert np.allclose(X_lower_diagonal_blocks_ref, X_lower_diagonal_blocks)
    assert np.allclose(X_arrow_bottom_blocks_ref, X_arrow_bottom_blocks)
    assert np.allclose(X_arrow_tip_block_ref, X_arrow_tip_block)
