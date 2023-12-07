"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Lisa Gaedke-Merzhaeuser  (lisa.gaedke.merzhaeuser@usi.ch)
@date: 2023-11

Tests for lu selected decompositions routines.

Copyright 2023 ETH Zurich and USI. All rights reserved.
"""

from sdr.utils import matrix_generation
from sdr.lu.lu_decompose import lu_dcmp_tridiag

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import pytest


# Testing of block tridiagonal lu
if __name__ == "__main__":
    nblocks = 5
    blocksize = 2
    symmetric = False
    diagonal_dominant = True
    seed = 63

    A = matrix_generation.generate_blocktridiag(
        nblocks, blocksize, symmetric, diagonal_dominant, seed
    )


    # --- Decomposition ---
    # permute_l is default but to raise awareness that this would fail otherwise ...
    P_ref, L_ref, U_ref = la.lu(A, permute_l=False)

    plt.matshow(P_ref)
    ## TODO: we need P to be identity ...

    fig, ax = plt.subplots(2, 3)
    ax[0, 0].set_title("L_ref: Scipy lower factor")
    ax[0, 0].matshow(L_ref)
    ax[1, 0].set_title("U_ref: Scipy upper factor")
    ax[1, 0].matshow(U_ref)

    L_sdr, U_sdr = lu_dcmp_tridiag(A, blocksize)
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
    ]
)
def test_lu_decompose_tridiag(
    nblocks: int,
    blocksize: int,  
):
    symmetric = False
    diagonal_dominant = True
    seed = 63

    A = matrix_generation.generate_blocktridiag(
        nblocks, blocksize, symmetric, diagonal_dominant, seed
    )

    P_ref, L_ref, U_ref = la.lu(A)
    L_sdr, U_sdr = lu_dcmp_tridiag(A, blocksize)
            
    assert np.allclose(L_ref, L_sdr) and np.allclose(U_ref, U_sdr)
