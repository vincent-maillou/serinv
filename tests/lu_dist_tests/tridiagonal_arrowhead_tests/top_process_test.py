"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Lisa Gaedke-Merzhaeuser  (lisa.gaedke.merzhaeuser@usi.ch)
@date: 2024-03

Test of the top process part of lu_dist algorithm for tridiagonal arrowhead 
matrices.

Copyright 2023-2024 ETH Zurich and USI. All rights reserved.
"""

from sdr.utils import matrix_generation
from sdr.utils import matrix_transform
from sdr.utils import dist_utils
from sdr.lu_dist.lu_dist_tridiagonal_arrowhead import top_factorize, top_sinv   

import numpy as np
import copy as cp
import pytest


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
    ]
)
def test_lu_dist_top_process(
    nblocks: int, 
    diag_blocksize: int, 
    arrow_blocksize: int, 
):
    diagonal_dominant = True
    symmetric = False
    seed = 63
    
    A = matrix_generation.generate_tridiag_arrowhead_dense(
        nblocks, diag_blocksize, arrow_blocksize, symmetric, diagonal_dominant, seed
    )

    # ----- Reference -----
    A_ref = cp.deepcopy(A)

    X_ref = np.linalg.inv(A_ref)
    
    X_ref_local, X_ref_arrow_bottom, X_ref_arrow_right = dist_utils.extract_partition(
        X_ref,
        0,
        nblocks-1,
        diag_blocksize,
        arrow_blocksize,
    )
    
    X_ref_arrow_tip = X_ref[-arrow_blocksize:, -arrow_blocksize:]
    # ---------------------

    A_local, A_arrow_bottom, A_arrow_right = dist_utils.extract_partition(
        A,
        0,
        nblocks-1,
        diag_blocksize,
        arrow_blocksize,
    )

    A_arrow_tip = A[-arrow_blocksize:, -arrow_blocksize:]

    (
        A_local, 
        A_arrow_bottom, 
        A_arrow_right,
        Update_arrow_tip, 
        L_local, 
        U_local, 
        L_arrow_bottom, 
        U_arrow_right,
    ) = top_factorize(
        A_local,
        A_arrow_bottom,
        A_arrow_right,
        diag_blocksize,
        arrow_blocksize,
    )
    
    
    # Create and inverse the reduced system created by the last reduced block
    # and the tip of the arrowhead.

    X_sdr_local = np.zeros_like(A_local)
    X_sdr_arrow_bottom = np.zeros_like(A_arrow_bottom)
    X_sdr_arrow_right = np.zeros_like(A_arrow_right)
    X_sdr_global_arrow_tip = np.zeros_like(Update_arrow_tip)

    reduced_system = np.zeros((diag_blocksize+arrow_blocksize, diag_blocksize+arrow_blocksize))
    reduced_system[0:diag_blocksize, 0:diag_blocksize] = A_local[-diag_blocksize:, -diag_blocksize:]
    reduced_system[-arrow_blocksize:, -arrow_blocksize:] = A_arrow_tip + Update_arrow_tip    
    reduced_system[0:diag_blocksize, -arrow_blocksize:] = A_arrow_right[-diag_blocksize:, :]
    reduced_system[-arrow_blocksize:, 0:diag_blocksize] = A_arrow_bottom[:, -diag_blocksize:]
    
    reduced_system_inv = np.linalg.inv(reduced_system)
    
    X_sdr_local[-diag_blocksize:, -diag_blocksize:] = reduced_system_inv[0:diag_blocksize, 0:diag_blocksize]
    X_sdr_arrow_bottom[:, -diag_blocksize:] = reduced_system_inv[-arrow_blocksize:, 0:diag_blocksize]
    X_sdr_arrow_right[-diag_blocksize:, :] = reduced_system_inv[0:diag_blocksize, -arrow_blocksize:]
    X_sdr_global_arrow_tip = reduced_system_inv[-arrow_blocksize:, -arrow_blocksize:]

    
    # ----- Selected inversion part -----
    X_sdr_local, X_sdr_arrow_bottom, X_sdr_arrow_right = top_sinv(
        X_sdr_local,
        X_sdr_arrow_bottom,
        X_sdr_arrow_right,
        X_sdr_global_arrow_tip,
        L_local,
        U_local,
        L_arrow_bottom,
        U_arrow_right,
        diag_blocksize,
    )
  
    X_ref_local = matrix_transform.cut_to_blocktridiag(X_ref_local, diag_blocksize)
    
    assert np.allclose(X_ref_local, X_sdr_local)
    assert np.allclose(X_ref_arrow_bottom, X_sdr_arrow_bottom)
    assert np.allclose(X_ref_arrow_right, X_sdr_arrow_right)
    assert np.allclose(X_ref_arrow_tip, X_sdr_global_arrow_tip)
    