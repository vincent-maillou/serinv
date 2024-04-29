# Copyright 2023-2024 ETH Zurich and USI. All rights reserved.

import numpy as np
import pytest
import scipy.linalg as la

from serinv.lu.lu_factorize import lu_factorize_tridiag_arrowhead
from serinv.lu.lu_selected_inversion import lu_sinv_tridiag_arrowhead
from serinv.utils import matrix_generation_dense
from serinv.utils.matrix_transformation_dense import (
    zeros_to_block_tridiagonal_arrowhead_shape,
    convert_block_tridiagonal_arrowhead_arrays_to_dense,
)
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
def test_lu_sinv_tridiag_arrowhead(
    nblocks: int,
    diag_blocksize: int,
    arrow_blocksize: int,
):
    symmetric = False
    diagonal_dominant = True
    seed = 63

    A = matrix_generation_dense.generate_block_tridiagonal_arrowhead_dense(
        nblocks, diag_blocksize, arrow_blocksize, symmetric, diagonal_dominant, seed
    )

    # --- Inversion ---

    X_ref = la.inv(A)
    X_ref = zeros_to_block_tridiagonal_arrowhead_shape(
        X_ref, diag_blocksize, arrow_blocksize
    )

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
        _,
    ) = lu_factorize_tridiag_arrowhead(
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_upper_diagonal_blocks,
        A_arrow_bottom_blocks,
        A_arrow_right_blocks,
        A_arrow_tip_block,
    )

    (
        X_sdr_diagonal_blocks,
        X_sdr_lower_diagonal_blocks,
        X_sdr_upper_diagonal_blocks,
        X_sdr_arrow_bottom_blocks,
        X_sdr_arrow_right_blocks,
        X_sdr_arrow_tip_block,
        _,
    ) = lu_sinv_tridiag_arrowhead(
        L_diagonal_blocks,
        L_lower_diagonal_blocks,
        L_arrow_bottom_blocks,
        U_diagonal_blocks,
        U_upper_diagonal_blocks,
        U_arrow_right_blocks,
    )

    X_sdr = convert_block_tridiagonal_arrowhead_arrays_to_dense(
        X_sdr_diagonal_blocks,
        X_sdr_lower_diagonal_blocks,
        X_sdr_upper_diagonal_blocks,
        X_sdr_arrow_bottom_blocks,
        X_sdr_arrow_right_blocks,
        X_sdr_arrow_tip_block,
    )

    assert np.allclose(X_ref, X_sdr)
