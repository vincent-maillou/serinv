"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Lisa Gaedke-Merzhaeuser  (lisa.gaedke.merzhaeuser@usi.ch)
@date: 2023-11

Tests for cholesky selected decompositions routines.

Copyright 2023 ETH Zurich and USI. All rights reserved.
"""

from sdr.utils import matrix_generation
from sdr.cholesky.cholesky_decompose import chol_dcmp_tridiag

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import pytest
import ctypes



# Testing of block tridiagonal cholesky
if __name__ == "__main__":
    nblocks = 5
    blocksize = 2
    symmetric = True
    diagonal_dominant = True
    seed = 63

    A = matrix_generation.generate_blocktridiag(
        nblocks, blocksize, symmetric, diagonal_dominant, seed
    )


    # --- Decomposition ---

    fig, ax = plt.subplots(1, 3)
    L_ref = la.cholesky(A, lower=True)
    ax[0].set_title("L_ref: Reference cholesky decomposition")
    ax[0].matshow(L_ref)

    L_sdr = chol_dcmp_tridiag(A, blocksize)
    ax[1].set_title("L_sdr: Selected cholesky decomposition")
    ax[1].matshow(L_sdr)

    L_diff = L_ref - L_sdr
    ax[2].set_title("L_diff: Difference between ref_chol and sel_chol")
    ax[2].matshow(L_diff)
    fig.colorbar(ax[2].matshow(L_diff), ax=ax[2], label="Relative error", shrink=0.4)

    plt.show()
    
    # Run with overwrite = True functionality
    L_sdr = chol_dcmp_tridiag(A, blocksize, overwrite=True)
    print("Run with overwrite :  True")
    print("memory address A   : ", A.ctypes.data)
    print("memory address L   : ", L_sdr.ctypes.data)
    print("L_ref == L_sdr     : ", np.allclose(L_ref, L_sdr))




@pytest.mark.parametrize(
    "nblocks, blocksize, overwrite", 
    [
        (2, 2, False),
        (10, 2, False),
        (100, 2, False),
        (2, 3, False),
        (10, 3, False),
        (100, 3, False),
        (2, 100, False),
        (5, 100, False),
        (10, 100, False),
        (2, 2, True),
        (10, 2, True),
        (100, 2, True),
        (2, 3, True),
        (10, 3, True),
        (100, 3, True),
        (2, 100, True),
        (5, 100, True),
        (10, 100, True),
    ]
)
def test_cholesky_decompose_tridiag(
    nblocks: int,
    blocksize: int,  
    overwrite: bool,
):
    symmetric = True
    diagonal_dominant = True
    seed = 63

    A = matrix_generation.generate_blocktridiag(
        nblocks, blocksize, symmetric, diagonal_dominant, seed
    )

    L_ref = la.cholesky(A, lower=True)
    L_sdr = chol_dcmp_tridiag(A, blocksize, overwrite=overwrite)

    if overwrite:
        assert np.allclose(L_ref, L_sdr) and A.ctypes.data == L_sdr.ctypes.data
    else: 
        assert np.allclose(L_ref, L_sdr) and A.ctypes.data != L_sdr.ctypes.data 
