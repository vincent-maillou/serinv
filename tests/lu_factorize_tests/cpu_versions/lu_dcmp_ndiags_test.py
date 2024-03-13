"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Lisa Gaedke-Merzhaeuser  (lisa.gaedke.merzhaeuser@usi.ch)
@date: 2023-11

Tests for lu selected decompositions routines.

Copyright 2023-2024 ETH Zurich and USI. All rights reserved.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy.linalg as la

from sdr.lu.lu_decompose import lu_dcmp_ndiags
from sdr.utils import matrix_generation

# Testing of block n-diagonals lu
if __name__ == "__main__":
    nblocks = 6
    ndiags = 7
    blocksize = 2
    symmetric = False
    diagonal_dominant = True
    seed = 63

    A = matrix_generation.generate_block_ndiags_dense(
        nblocks, ndiags, blocksize, symmetric, diagonal_dominant, seed
    )

    # --- Decomposition ---

    P_ref, L_ref, U_ref = la.lu(A)

    plt.matshow(P_ref)

    fig, ax = plt.subplots(2, 3)
    ax[0, 0].set_title("L_ref: Scipy lower factor")
    ax[0, 0].matshow(L_ref)
    ax[1, 0].set_title("U_ref: Scipy upper factor")
    ax[1, 0].matshow(U_ref)

    L_sdr, U_sdr = lu_dcmp_ndiags(A, ndiags, blocksize)
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


@pytest.mark.cpu
@pytest.mark.mpi_skip()
@pytest.mark.parametrize(
    "nblocks, ndiags, blocksize",
    [
        (2, 3, 2),
        (3, 5, 2),
        (4, 7, 2),
        (20, 3, 3),
        (30, 5, 3),
        (40, 7, 3),
    ],
)
def test_lu_decompose_ndiags(nblocks, ndiags, blocksize):
    symmetric = False
    diagonal_dominant = True
    seed = 63

    A = matrix_generation.generate_block_ndiags_dense(
        nblocks, ndiags, blocksize, symmetric, diagonal_dominant, seed
    )

    # --- Decomposition ---

    P_ref, L_ref, U_ref = la.lu(A)

    if np.allclose(P_ref, np.eye(A.shape[0])):
        L_ref = P_ref @ L_ref

    L_sdr, U_sdr = lu_dcmp_ndiags(A, ndiags, blocksize)

    assert np.allclose(L_ref, L_sdr)
    assert np.allclose(U_ref, U_sdr)
