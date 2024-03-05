"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Lisa Gaedke-Merzhaeuser  (lisa.gaedke.merzhaeuser@usi.ch)
@date: 2023-11

Tests for cholesky selected decompositions routines.

Copyright 2023-2024 ETH Zurich and USI. All rights reserved.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy.linalg as la

from sdr.cholesky.cholesky_decompose import chol_dcmp_tridiag_arrowhead
from sdr.utils import matrix_generation

# Testing of block tridiagonal arrowhead cholesky
if __name__ == "__main__":
    nblocks = 5
    diag_blocksize = 3
    arrow_blocksize = 2
    symmetric = True
    diagonal_dominant = True
    seed = 63

    A = matrix_generation.generate_tridiag_arrowhead_dense(
        nblocks, diag_blocksize, arrow_blocksize, symmetric, diagonal_dominant, seed
    )

    # --- Decomposition ---

    fig, ax = plt.subplots(1, 3)
    L_ref = la.cholesky(A, lower=True)
    ax[0].set_title("L_ref: Reference cholesky decomposition")
    ax[0].matshow(L_ref)

    L_sdr = chol_dcmp_tridiag_arrowhead(A, diag_blocksize, arrow_blocksize)
    ax[1].set_title("L_sdr: Selected cholesky decomposition")
    ax[1].matshow(L_sdr)

    L_diff = L_ref - L_sdr
    ax[2].set_title("L_diff: Difference between L_ref and L_sdr")
    ax[2].matshow(L_diff)
    fig.colorbar(ax[2].matshow(L_diff), ax=ax[2], label="Relative error", shrink=0.4)

    plt.show()

    # Run with overwrite = True functionality
    L_sdr = chol_dcmp_tridiag_arrowhead(
        A, diag_blocksize, arrow_blocksize, overwrite=True
    )
    print("Run with overwrite :  True")
    print("memory address A   : ", A.ctypes.data)
    print("memory address L   : ", L_sdr.ctypes.data)
    print("L_ref == L_sdr     : ", np.allclose(L_ref, L_sdr))


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
    ],
)
@pytest.mark.parametrize("overwrite", [True, False])
def test_cholesky_decompose_tridiag_arrowhead(
    nblocks: int,
    diag_blocksize: int,
    arrow_blocksize: int,
    overwrite: bool,
):
    symmetric = True
    diagonal_dominant = True
    seed = 63

    A = matrix_generation.generate_tridiag_arrowhead_dense(
        nblocks, diag_blocksize, arrow_blocksize, symmetric, diagonal_dominant, seed
    )

    L_ref = la.cholesky(A, lower=True)
    L_sdr = chol_dcmp_tridiag_arrowhead(A, diag_blocksize, arrow_blocksize, overwrite)

    if overwrite:
        assert np.allclose(L_ref, L_sdr) and A.ctypes.data == L_sdr.ctypes.data
    else:
        assert np.allclose(L_ref, L_sdr) and A.ctypes.data != L_sdr.ctypes.data
