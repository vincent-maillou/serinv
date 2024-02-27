"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Lisa Gaedke-Merzhaeuser  (lisa.gaedke.merzhaeuser@usi.ch)
@date: 2023-11

Tests for cholesky selected inversion routines.

Copyright 2023-2024 ETH Zurich and USI. All rights reserved.
"""

from sdr.utils import matrix_generation
from sdr.utils.matrix_transform import cut_to_blocktridiag_arrowhead
from sdr.cholesky.cholesky_decompose import chol_dcmp_tridiag_arrowhead
from sdr.cholesky.cholesky_selected_inversion import chol_sinv_tridiag_arrowhead

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import pytest



# Testing of block tridiagonal cholesky sinv
if __name__ == "__main__":
    nblocks = 5
    diag_blocksize = 3
    arrow_blocksize = 2
    symmetric = True
    diagonal_dominant = True
    seed = 63

    A = matrix_generation.generate_tridiag_arrowhead_dense(
        nblocks, diag_blocksize, arrow_blocksize, symmetric, diagonal_dominant, 
        seed
    )

    # --- Inversion ---

    X_ref = la.inv(A)
    X_ref = cut_to_blocktridiag_arrowhead(X_ref, diag_blocksize, arrow_blocksize)

    L_sdr = chol_dcmp_tridiag_arrowhead(A, diag_blocksize, arrow_blocksize)

    fig, ax = plt.subplots(1, 3)
    ax[0].set_title("X_ref: Cholesky reference inversion")
    ax[0].matshow(X_ref)

    X_sdr = chol_sinv_tridiag_arrowhead(L_sdr, diag_blocksize, arrow_blocksize)
    ax[1].set_title("X_sdr: Cholesky selected inversion")
    ax[1].matshow(X_sdr)

    X_diff = X_ref - X_sdr
    ax[2].set_title("X_diff: Difference between X_ref and X_sdr")
    ax[2].matshow(X_diff)
    fig.colorbar(ax[2].matshow(X_diff), ax=ax[2], label="Relative error", shrink=0.4)

    plt.show()



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
def test_cholesky_sinv_tridiag_arrowhead(
    nblocks: int, 
    diag_blocksize: int, 
    arrow_blocksize: int, 
):
    symmetric = True
    diagonal_dominant = True
    seed = 63

    A = matrix_generation.generate_tridiag_arrowhead_dense(
        nblocks, diag_blocksize, arrow_blocksize, symmetric, diagonal_dominant, 
        seed
    )

    # --- Inversion ---

    X_ref = la.inv(A)
    X_ref = cut_to_blocktridiag_arrowhead(X_ref, diag_blocksize, arrow_blocksize)

    L_sdr = chol_dcmp_tridiag_arrowhead(A, diag_blocksize, arrow_blocksize)
    X_sdr = chol_sinv_tridiag_arrowhead(L_sdr, diag_blocksize, arrow_blocksize)

    assert np.allclose(X_ref, X_sdr)
    