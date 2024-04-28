# Copyright 2023-2024 ETH Zurich and USI. All rights reserved.

import numpy as np
import numpy.linalg as npla
import copy as cp
import pytest

from sdr.cholesky.cholesky_factorize import (
    cholesky_factorize_block_tridiagonal_arrowhead,
)
from sdr.cholesky.cholesky_solve import cholesky_solve_block_tridiagonal_arrowhead
from sdr.utils import matrix_generation_dense
from sdr.utils.matrix_transformation_arrays import (
    convert_block_tridiagonal_arrowhead_dense_to_arrays,
)


@pytest.mark.cpu
@pytest.mark.mpi_skip()
@pytest.mark.parametrize(
    "nblocks, diag_blocksize, arrow_blocksize, n_rhs",
    [
        (2, 2, 2, 1),
        (2, 3, 2, 2),
        (2, 2, 3, 5),
        (10, 2, 2, 1),
        (10, 3, 2, 4),
        (10, 2, 3, 8),
        (10, 10, 2, 1),
        (10, 2, 10, 1),
    ],
)
def test_cholesky_slv_tridiag_arrowhead(
    nblocks: int,
    diag_blocksize: int,
    arrow_blocksize: int,
    n_rhs: int,
):
    symmetric = True
    diagonal_dominant = True
    seed = 63

    A = matrix_generation_dense.generate_block_tridiagonal_arrowhead_dense(
        nblocks, diag_blocksize, arrow_blocksize, symmetric, diagonal_dominant, seed
    )

    A_copy = cp.deepcopy(A)

    (
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        _,
        A_arrow_bottom_blocks,
        _,
        A_arrow_tip_block,
    ) = convert_block_tridiagonal_arrowhead_dense_to_arrays(
        A_copy, diag_blocksize, arrow_blocksize
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

    B = np.random.randn(A.shape[0], n_rhs)

    X_ref = npla.solve(A, B)

    X_sdr = cholesky_solve_block_tridiagonal_arrowhead(
        L_diagonal_blocks,
        L_lower_diagonal_blocks,
        L_arrow_bottom_blocks,
        L_arrow_tip_block,
        B,
    )

    assert np.allclose(X_ref, X_sdr)
