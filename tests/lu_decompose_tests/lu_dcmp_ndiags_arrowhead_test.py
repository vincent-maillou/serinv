"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Lisa Gaedke-Merzhaeuser  (lisa.gaedke.merzhaeuser@usi.ch)
@date: 2023-11

Tests for lu selected decompositions routines.

Copyright 2023 ETH Zurich and USI. All rights reserved.
"""

from sdr.utils import matrix_generation
from sdr.lu.lu_decompose import lu_dcmp_ndiags_arrowhead

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import pytest


# Testing of block n-diagonals arrowhead lu
if __name__ == "__main__":
    nblocks = 4
    ndiags = 3
    diag_blocksize = 2
    arrow_blocksize = 2
    symmetric = False
    diagonal_dominant = True
    seed = 63

    A = matrix_generation.generate_ndiags_arrowhead(
        nblocks, ndiags, diag_blocksize, arrow_blocksize, symmetric, 
        diagonal_dominant, seed
    )

    plt.matshow(A)

    # --- Decomposition ---

    P_ref, L_ref, U_ref = la.lu(A)
    # check that LU is not permuted
    if not np.array_equal(P_ref, np.eye(P_ref.shape[0])):
        raise ValueError("Reference LU solution is permuted!")

    plt.matshow(P_ref)

    fig, ax = plt.subplots(2, 3)
    ax[0, 0].set_title("L_ref: Scipy lower factor")
    ax[0, 0].matshow(L_ref)
    ax[1, 0].set_title("U_ref: Scipy upper factor")
    ax[1, 0].matshow(U_ref)

    L_sdr, U_sdr = lu_dcmp_ndiags_arrowhead(A, ndiags, diag_blocksize, arrow_blocksize)
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
    "nblocks, ndiags, diag_blocksize, arrow_blocksize", 
    [
        (2, 1, 1, 2),
        (3, 3, 2, 1),
        (4, 5, 1, 2),
        # (5, 7, 2, 1), TODO: The routine is not working when the matrix is full because of it's numbers of off-diagonals
        (15, 1, 3, 1),
        (15, 3, 1, 2),
        (15, 5, 3, 1),
        (15, 7, 1, 2),
    ]
)
def test_lu_decompose_ndiags_arrowhead(
    nblocks, 
    ndiags, 
    diag_blocksize, 
    arrow_blocksize
):
    symmetric = False
    diagonal_dominant = True
    seed = 63

    A = matrix_generation.generate_ndiags_arrowhead(
        nblocks, ndiags, diag_blocksize, arrow_blocksize, symmetric, 
        diagonal_dominant, seed
    )

    # --- Decomposition ---

    P_ref, L_ref, U_ref = la.lu(A)

    if np.allclose(P_ref, np.eye(A.shape[0])):
        L_ref = P_ref @ L_ref

    L_sdr, U_sdr = lu_dcmp_ndiags_arrowhead(A, nblocks, diag_blocksize, arrow_blocksize)
    
    assert np.allclose(L_ref, L_sdr)
    assert np.allclose(U_ref, U_sdr)
