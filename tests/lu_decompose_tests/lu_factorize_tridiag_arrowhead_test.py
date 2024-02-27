"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Lisa Gaedke-Merzhaeuser  (lisa.gaedke.merzhaeuser@usi.ch)
@date: 2024-02

Tests for lu tridiagonal arrowhead matrices selected factorization routine.

Copyright 2023-2024 ETH Zurich and USI. All rights reserved.
"""

from sdr.utils import matrix_generation
from sdr.utils.matrix_transform import from_dense_to_arrowhead_arrays, from_arrowhead_arrays_to_dense
from sdr.lu.lu_factorize import lu_factorize_tridiag_arrowhead

import numpy as np
import scipy.linalg as la
import pytest

    
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
    ]
)
def test_lu_decompose_tridiag_arrowhead(
    nblocks: int, 
    diag_blocksize: int, 
    arrow_blocksize: int, 
):
    symmetric = False
    diagonal_dominant = True
    seed = 63

    A = matrix_generation.generate_tridiag_arrowhead_dense(
        nblocks, diag_blocksize, arrow_blocksize, symmetric, diagonal_dominant, 
        seed
    )

    # --- Decomposition ---

    P_ref, L_ref, U_ref = la.lu(A)

    if np.allclose(P_ref, np.eye(A.shape[0])):
        L_ref = P_ref @ L_ref

    (
        A_diagonal_blocks, 
        A_lower_diagonal_blocks, 
        A_upper_diagonal_blocks, 
        A_arrow_bottom_blocks, 
        A_arrow_right_blocks, 
        A_arrow_tip_block
    ) = from_dense_to_arrowhead_arrays(A, diag_blocksize, arrow_blocksize)
    
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
        A_arrow_tip_block
    )
    
    L_sdr = from_arrowhead_arrays_to_dense(
        L_diagonal_blocks, 
        L_lower_diagonal_blocks, 
        np.zeros((diag_blocksize, (nblocks-1)*diag_blocksize)),
        L_arrow_bottom_blocks[:, :-arrow_blocksize], 
        np.zeros(((nblocks-1)*diag_blocksize, arrow_blocksize)),
        L_arrow_bottom_blocks[:, -arrow_blocksize:],
    )
    
    U_sdr = from_arrowhead_arrays_to_dense(
        U_diagonal_blocks, 
        np.zeros((diag_blocksize, (nblocks-1)*diag_blocksize)), 
        U_upper_diagonal_blocks, 
        np.zeros((arrow_blocksize, (nblocks-1)*diag_blocksize)),
        U_arrow_right_blocks[:-arrow_blocksize, :], 
        U_arrow_right_blocks[-arrow_blocksize:, :],
    )

    assert np.allclose(L_ref, L_sdr)
    assert np.allclose(U_ref, U_sdr)
