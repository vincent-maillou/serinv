"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Lisa Gaedke-Merzhaeuser  (lisa.gaedke.merzhaeuser@usi.ch)
@author: Alexandros Nikolaos Ziogas (alziogas@iis.ee.ethz.ch)
@date: 2023-11

Tests for lu selected decompositions routines.

Copyright 2023 ETH Zurich and USI. All rights reserved.
"""

import copy
from sdr.utils import matrix_generation
from sdr.lu.lu_decompose import lu_dcmp_ndiags_bsize_one

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import pytest



# Testing of block n-diagonals lu
if __name__ == "__main__":
    np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))
    nblocks = 12
    ndiags = 7
    blocksize = 1
    symmetric = False
    diagonal_dominant = True
    seed = 63

    A_origin = matrix_generation.generate_block_ndiags(
        nblocks, ndiags, blocksize, symmetric, diagonal_dominant, seed
    )
    A = copy.deepcopy(A_origin)

    # --- Decomposition ---

    P_ref, L_ref, U_ref = la.lu(A)

    plt.matshow(P_ref)

    fig, ax = plt.subplots(2, 3)
    ax[0, 0].set_title("L_ref: Scipy lower factor")
    ax[0, 0].matshow(L_ref)
    ax[1, 0].set_title("U_ref: Scipy upper factor")
    ax[1, 0].matshow(U_ref)

    L_sdr, U_sdr = lu_dcmp_ndiags_bsize_one(A, ndiags)
    ax[0, 1].set_title("L_sdr: SDR lower factor")
    ax[0, 1].matshow(L_sdr)
    ax[1, 1].set_title("U_sdr: SDR upper factor")
    ax[1, 1].matshow(U_sdr)

    L_diff = L_ref - L_sdr
    U_diff = U_ref - U_sdr
    ax[0, 2].set_title("L: Difference between L_ref and L_sdr")
    ax[0, 2].matshow(L_diff)
    ax[1, 2].set_title("U: Difference between U_ref and U_sdr")
    ax[1, 2].matshow(U_diff)
    fig.colorbar(ax[0, 2].matshow(L_diff), ax=ax[0, 2], label="Relative error")
    fig.colorbar(ax[1, 2].matshow(U_diff), ax=ax[1, 2], label="Relative error")
    
    plt.show() 


    
@pytest.mark.parametrize(
    "nblocks, ndiags", 
    [
        (4, 3),
        (6, 5),
        (8, 7),
        (60, 3),
        (90, 5),
        (120, 7),
    ]
)
def test_lu_decompose_ndiags_bsize_one(
    nblocks, 
    ndiags
):
    symmetric = False
    diagonal_dominant = True
    seed = 63
    blocksize = 1

    A = matrix_generation.generate_block_ndiags(
        nblocks, ndiags, blocksize, symmetric, diagonal_dominant, seed
    )

    # --- Decomposition ---

    P_ref, L_ref, U_ref = la.lu(A)

    if np.allclose(P_ref, np.eye(A.shape[0])):
        L_ref = P_ref @ L_ref

    L_sdr, U_sdr = lu_dcmp_ndiags_bsize_one(A, ndiags)
    
    assert np.allclose(L_ref, L_sdr)
    assert np.allclose(U_ref, U_sdr)