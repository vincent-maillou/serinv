"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Lisa Gaedke-Merzhaeuser  (lisa.gaedke.merzhaeuser@usi.ch)
@date: 2023-11

Tests for cholesky decompositions routines.

Copyright 2023 ETH Zurich and USI. All rights reserved.
"""

from sdr.utils import matrix_transform
from sdr.utils import matrix_generation

from sdr.cholesky.cholesky_decompose import chol_dcmp_tridiag, chol_dcmp_tridia_arrowhead

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt



# Testing of block tridiagonal cholesky
""" if __name__ == "__main__":
    nblocks = 5
    blocksize = 2
    symmetric = True
    seed = 63

    A = matrix_generation.generate_blocktridiag(
        nblocks, blocksize, symmetric, seed
    )


    # --- Decomposition ---

    fig, ax = plt.subplots(1, 3)
    ax[0].set_title("A: Initial matrix")
    ax[0].matshow(A)

    L_ref = la.cholesky(A, lower=True)
    ax[1].set_title("L: Reference cholesky decomposition")
    ax[1].matshow(L_ref)

    L = chol_dcmp_tridiag(A, blocksize)
    ax[2].set_title("L: Selected cholesky decomposition")
    ax[2].matshow(L)

    plt.show() """



# Testing of block tridiagonal arrowhead cholesky
""" if __name__ == "__main__":
    nblocks = 5
    diag_blocksize = 3
    arrow_blocksize = 2
    symmetric = True
    diagonal_dominant = True
    seed = 63

    A = matrix_generation.generate_blocktridiag_arrowhead(
        nblocks, diag_blocksize, arrow_blocksize, symmetric, diagonal_dominant, 
        seed
    )


    # --- Decomposition ---

    fig, ax = plt.subplots(1, 3)
    ax[0].set_title("A: Initial matrix")
    ax[0].matshow(A)

    L_ref = la.cholesky(A, lower=True)
    ax[1].set_title("L: Reference cholesky decomposition")
    ax[1].matshow(L_ref)

    L = chol_dcmp_tridia_arrowhead(A, diag_blocksize, arrow_blocksize)
    ax[2].set_title("L: Selected cholesky decomposition")
    ax[2].matshow(L)

    plt.show() """
