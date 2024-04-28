# Copyright 2023-2024 ETH Zurich and USI. All rights reserved.

import numpy as np
import pytest
import copy as cp

from serinv.lu.lu_factorize import lu_factorize_tridiag_arrowhead
from serinv.lu.lu_solve import lu_solve_tridiag_arrowhead
from serinv.utils import matrix_generation_dense
from serinv.utils.matrix_transformation_arrays import (
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
def test_lu_slv_tridiag_arrowhead(
    nblocks: int,
    diag_blocksize: int,
    arrow_blocksize: int,
    n_rhs: int,
):
    symmetric = False
    diagonal_dominant = True
    seed = 63

    A = matrix_generation_dense.generate_block_tridiagonal_arrowhead_dense(
        nblocks, diag_blocksize, arrow_blocksize, symmetric, diagonal_dominant, seed
    )

    A_copy = cp.deepcopy(A)

    # --- Factorization LU ---
    (
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_upper_diagonal_blocks,
        A_arrow_bottom_blocks,
        A_arrow_right_blocks,
        A_arrow_tip_block,
    ) = convert_block_tridiagonal_arrowhead_dense_to_arrays(
        A, diag_blocksize, arrow_blocksize
    )

    (
        L_diagonal_blocks,
        L_lower_diagonal_blocks,
        L_arrow_bottom_blocks,
        U_diagonal_blocks,
        U_upper_diagonal_blocks,
        U_arrow_right_blocks,
    ) = lu_factorize_tridiag_arrowhead(
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_upper_diagonal_blocks,
        A_arrow_bottom_blocks,
        A_arrow_right_blocks,
        A_arrow_tip_block,
    )

    # --- Solving ---
    B = np.random.randn(A.shape[0], n_rhs)

    X_sdr = lu_solve_tridiag_arrowhead(
        L_diagonal_blocks,
        L_lower_diagonal_blocks,
        L_arrow_bottom_blocks,
        U_diagonal_blocks,
        U_upper_diagonal_blocks,
        U_arrow_right_blocks,
        B,
    )

    X_solve_ref = np.linalg.solve(A_copy, B)

    np.testing.assert_allclose(X_solve_ref, X_sdr)
