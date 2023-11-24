"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Lisa Gaedke-Merzhaeuser  (lisa.gaedke.merzhaeuser@usi.ch)
@date: 2023-11

Tests for cholesky decompositions routines.

Copyright 2023 ETH Zurich and USI. All rights reserved.
"""

from sdr.utils import matrix_transform
from sdr.utils import matrix_generation

from sdr.cholesky.cholesky_decompose import chol_dcmp_tridiag

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt


if __name__ == "__main__":
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

    plt.show()


    # --- Solver ---

    fig, ax = plt.subplots(1, 3)
    ax[0].set_title("A: Initial matrix")
    ax[0].matshow(A)

    A_inv_ref = np.linalg.inv(A)
    A_inv_ref_cut = matrix_transform.cut_to_blocktridiag(A_inv_ref, blocksize)
    ax[1].set_title("A_inv_ref: Reference inversion and cut")
    ax[1].matshow(A_inv_ref_cut)

    #plt.show()