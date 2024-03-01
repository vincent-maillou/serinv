"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Lisa Gaedke-Merzhaeuser  (lisa.gaedke.merzhaeuser@usi.ch)
@date: 2024-03

Test of the middle process part of lu_dist algorithm for tridiagonal arrowhead 
matrices.

Copyright 2023-2024 ETH Zurich and USI. All rights reserved.
"""

from sdr.utils.matrix_generation import generate_tridiag_arrowhead_dense
from sdr.utils.matrix_transform import from_dense_to_arrowhead_arrays
from sdr.lu_dist.lu_dist_tridiagonal_arrowhead import middle_factorize, middle_sinv

import numpy as np
import copy as cp
import pytest


@pytest.mark.mpi_skip()
@pytest.mark.parametrize(
    "nblocks, diag_blocksize, arrow_blocksize", 
    [
        (3, 2, 2),
        (3, 3, 2),
        (3, 2, 3),
        (10, 2, 2),
        (10, 3, 2),
        (10, 2, 3),
        (10, 10, 2),
        (10, 2, 10),
    ]
)
def test_lu_dist_middle_process(
    nblocks: int, 
    diag_blocksize: int, 
    arrow_blocksize: int, 
):
    diagonal_dominant = True
    symmetric = False
    seed = 63
    
    A = generate_tridiag_arrowhead_dense(
        nblocks, 
        diag_blocksize, 
        arrow_blocksize, 
        symmetric, 
        diagonal_dominant, 
        seed
    )

    # ----- Reference -----
    A_ref = cp.deepcopy(A)

    X_ref = np.linalg.inv(A_ref)
    
    (
        X_ref_diagonal_blocks, 
        X_ref_lower_diagonal_blocks, 
        X_ref_upper_diagonal_blocks, 
        X_ref_arrow_bottom_blocks, 
        X_ref_arrow_right_blocks, 
        X_ref_arrow_tip_block,
    ) = from_dense_to_arrowhead_arrays(
        X_ref, 
        diag_blocksize, 
        arrow_blocksize
    )
    # ---------------------

    (
        A_diagonal_blocks, 
        A_lower_diagonal_blocks, 
        A_upper_diagonal_blocks, 
        A_arrow_bottom_blocks, 
        A_arrow_right_blocks, 
        A_arrow_tip_block,
    ) = from_dense_to_arrowhead_arrays(
        A, 
        diag_blocksize, 
        arrow_blocksize
    )

    # Arrays that store the update of the 2sided pattern for the middle processes
    A_top_2sided_arrow_blocks_local = np.zeros((diag_blocksize, nblocks * diag_blocksize), dtype=A_diagonal_blocks.dtype)
    A_left_2sided_arrow_blocks_local = np.zeros((nblocks * diag_blocksize, diag_blocksize, ), dtype=A_diagonal_blocks.dtype)

    A_top_2sided_arrow_blocks_local[:, :diag_blocksize] = A_diagonal_blocks[:, :diag_blocksize]
    A_top_2sided_arrow_blocks_local[:, diag_blocksize:2*diag_blocksize] = A_upper_diagonal_blocks[:, :diag_blocksize]
    
    A_left_2sided_arrow_blocks_local[:diag_blocksize, :] = A_diagonal_blocks[:, :diag_blocksize]
    A_left_2sided_arrow_blocks_local[diag_blocksize:2*diag_blocksize, :] = A_lower_diagonal_blocks[:, :diag_blocksize]

    (
        L_diagonal_blocks, 
        L_lower_diagonal_blocks, 
        L_arrow_bottom_blocks, 
        L_upper_2sided_arrow_blocks,
        U_diagonal_blocks, 
        U_upper_diagonal_blocks, 
        U_arrow_right_blocks, 
        U_left_2sided_arrow_blocks,
        Update_arrow_tip
    ) = middle_factorize(
        A_diagonal_blocks, 
        A_lower_diagonal_blocks, 
        A_upper_diagonal_blocks, 
        A_arrow_bottom_blocks, 
        A_arrow_right_blocks, 
        A_top_2sided_arrow_blocks_local,
        A_left_2sided_arrow_blocks_local,
        A_arrow_tip_block
    )


    # # Create and inverse the reduced system created by the last reduced block
    # # and the tip of the arrowhead.

    # reduced_system = np.zeros((2 * diag_blocksize + arrow_blocksize, 2 * diag_blocksize + arrow_blocksize))
    # global_arrow_tip = np.zeros((arrow_blocksize, arrow_blocksize))
    # global_arrow_tip = A_arrow_tip + Update_arrow_tip
    
    # # (top, top)
    # reduced_system[0:diag_blocksize, 0:diag_blocksize] = A_local[0:diag_blocksize, 0:diag_blocksize]
    # # (top, nblocks)
    # reduced_system[0:diag_blocksize, -diag_blocksize-arrow_blocksize:-arrow_blocksize] = A_local[0:diag_blocksize, -diag_blocksize:]
    # # (top, ndb+1)
    # reduced_system[0:diag_blocksize, -arrow_blocksize:] = A_arrow_right[0:diag_blocksize, :]
    # # (nblocks, top)
    # reduced_system[-diag_blocksize-arrow_blocksize:-arrow_blocksize, 0:diag_blocksize] = A_local[-diag_blocksize:, 0:diag_blocksize]
    # # (ndb+1, top)
    # reduced_system[-arrow_blocksize:, 0:diag_blocksize] = A_arrow_bottom[:, 0:diag_blocksize]
    # # (nblocks, nblocks)
    # reduced_system[-diag_blocksize-arrow_blocksize:-arrow_blocksize, -diag_blocksize-arrow_blocksize:-arrow_blocksize] = A_local[-diag_blocksize:, -diag_blocksize:]    
    # # (nblocks, ndb+1)
    # reduced_system[-diag_blocksize-arrow_blocksize:-arrow_blocksize, -arrow_blocksize:] = A_arrow_right[-diag_blocksize:, :]
    # # (ndb+1, nblocks)
    # reduced_system[-arrow_blocksize:, -diag_blocksize-arrow_blocksize:-arrow_blocksize] = A_arrow_bottom[:, -diag_blocksize:]
    # # (ndb+1, ndb+1)
    # reduced_system[-arrow_blocksize:, -arrow_blocksize:] = global_arrow_tip
    
    # reduced_system_inv = np.linalg.inv(reduced_system)

    # X_local = np.zeros_like(A_local)
    # X_arrow_bottom = np.zeros_like(A_arrow_bottom)
    # X_arrow_right = np.zeros_like(A_arrow_right)
    # X_global_arrow_tip = np.zeros_like(global_arrow_tip)
    
    # # (top, top)
    # X_local[0:diag_blocksize, 0:diag_blocksize] = reduced_system_inv[0:diag_blocksize, 0:diag_blocksize]
    # # (top, nblocks)
    # X_local[0:diag_blocksize, -diag_blocksize:] = reduced_system_inv[0:diag_blocksize, -diag_blocksize-arrow_blocksize:-arrow_blocksize]
    # # (top, ndb+1)
    # X_arrow_right[0:diag_blocksize, :] = reduced_system_inv[0:diag_blocksize, -arrow_blocksize:]
    # # (nblocks, top)
    # X_local[-diag_blocksize:, 0:diag_blocksize] = reduced_system_inv[-diag_blocksize-arrow_blocksize:-arrow_blocksize, 0:diag_blocksize]
    # # (ndb+1, top)
    # X_arrow_bottom[:, 0:diag_blocksize] = reduced_system_inv[-arrow_blocksize:, 0:diag_blocksize]
    # # (nblocks, nblocks)
    # X_local[-diag_blocksize:, -diag_blocksize:] = reduced_system_inv[-diag_blocksize-arrow_blocksize:-arrow_blocksize, -diag_blocksize-arrow_blocksize:-arrow_blocksize] 
    # # (nblocks, ndb+1)
    # X_arrow_right[-diag_blocksize:, :] = reduced_system_inv[-diag_blocksize-arrow_blocksize:-arrow_blocksize, -arrow_blocksize:]
    # # (ndb+1, nblocks)
    # X_arrow_bottom[:, -diag_blocksize:] = reduced_system_inv[-arrow_blocksize:, -diag_blocksize-arrow_blocksize:-arrow_blocksize]
    # # (ndb+1, ndb+1)
    # X_global_arrow_tip = reduced_system_inv[-arrow_blocksize:, -arrow_blocksize:]

    
    # # ----- Selected inversion part -----
    # (
    #     X_local, 
    #     X_arrow_bottom, 
    #     X_arrow_right
    # ) = middle_sinv(
    #     X_local,
    #     X_arrow_bottom,
    #     X_arrow_right,
    #     X_global_arrow_tip,
    #     L_local,
    #     U_local,
    #     L_arrow_bottom,
    #     U_arrow_right,
    #     diag_blocksize,
    # )
    
    # X_ref_local = matrix_transform.cut_to_blocktridiag(X_ref_local, diag_blocksize)
    # X_local = matrix_transform.cut_to_blocktridiag(X_local, diag_blocksize)
    
    # assert np.allclose(X_ref_local, X_local)
    # assert np.allclose(X_ref_arrow_bottom, X_arrow_bottom)
    # assert np.allclose(X_ref_arrow_right, X_arrow_right)
    # assert np.allclose(X_ref_arrow_tip, X_global_arrow_tip)
    
    
if __name__ == "__main__":
    test_lu_dist_middle_process(10, 10, 2)