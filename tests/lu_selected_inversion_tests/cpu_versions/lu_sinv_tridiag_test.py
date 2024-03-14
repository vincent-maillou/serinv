"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Lisa Gaedke-Merzhaeuser  (lisa.gaedke.merzhaeuser@usi.ch)
@date: 2023-11

Tests for lu selected inversion routines.

Copyright 2023-2024 ETH Zurich and USI. All rights reserved.
"""

import numpy as np
import pytest
import scipy.linalg as la

from sdr.lu.lu_factorize import lu_factorize_tridiag
from sdr.lu.lu_selected_inversion import lu_sinv_tridiag
from sdr.utils import matrix_generation
from sdr.utils.matrix_transform import (
    cut_to_blocktridiag,
    from_dense_to_tridiagonal_arrays,
    from_tridiagonal_arrays_to_dense,
)


@pytest.mark.cpu
@pytest.mark.mpi_skip()
@pytest.mark.parametrize(
    "nblocks, blocksize",
    [
        (2, 2),
        (10, 2),
        (100, 2),
        (2, 3),
        (10, 3),
        (100, 3),
        (2, 100),
        (5, 100),
        (10, 100),
    ],
)
def test_lu_sinv_tridiag(
    nblocks: int,
    blocksize: int,
):
    symmetric = False
    diagonal_dominant = True
    seed = 63

    A = matrix_generation.generate_tridiag_dense(
        nblocks, blocksize, symmetric, diagonal_dominant, seed
    )

    # --- Inversion ---

    X_ref = la.inv(A)
    X_ref = cut_to_blocktridiag(X_ref, blocksize)

    (
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_upper_diagonal_blocks,
    ) = from_dense_to_tridiagonal_arrays(A, blocksize)

    (
        L_diagonal_blocks,
        L_lower_diagonal_blocks,
        U_diagonal_blocks,
        U_upper_diagonal_blocks,
        _,
    ) = lu_factorize_tridiag(
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_upper_diagonal_blocks,
    )

    (
        X_sdr_diagonal_blocks,
        X_sdr_lower_diagonal_blocks,
        X_sdr_upper_diagonal_blocks,
        _,
    ) = lu_sinv_tridiag(
        L_diagonal_blocks,
        L_lower_diagonal_blocks,
        U_diagonal_blocks,
        U_upper_diagonal_blocks,
    )

    X_sdr_dense = from_tridiagonal_arrays_to_dense(
        X_sdr_diagonal_blocks,
        X_sdr_lower_diagonal_blocks,
        X_sdr_upper_diagonal_blocks,
    )

    assert np.allclose(X_ref, X_sdr_dense)
