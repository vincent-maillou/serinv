"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Lisa Gaedke-Merzhaeuser  (lisa.gaedke.merzhaeuser@usi.ch)
@date: 2023-11

Tests for lu selected inversion routines.

Copyright 2023-2024 ETH Zurich and USI. All rights reserved.
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la

from sdr.lu.lu_factorize import lu_factorize_tridiag
from sdr.lu.lu_selected_inversion import lu_sinv_tridiag
from sdr.utils import matrix_generation_dense
from sdr.utils.matrix_transformation import (
    cut_to_blocktridiag,
    from_dense_to_tridiagonal_arrays,
    from_tridiagonal_arrays_to_dense,
)

# Example of block tridiagonal lu sinv
if __name__ == "__main__":
    nblocks = 5
    blocksize = 2
    symmetric = False
    diagonal_dominant = True
    seed = 63

    A = matrix_generation_dense.generate_tridiag_dense(
        nblocks, blocksize, symmetric, diagonal_dominant, seed
    )

    # --- Inversion ---

    X_ref = la.inv(A)
    X_ref = cut_to_blocktridiag(X_ref, blocksize)

    (
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_upper_diagonal_blocks,
    ) = from_dense_to_tridiagonal_arrays(A, blocksize)

    (
        L_diagonal_blocks,
        L_lower_diagonal_blocks,
        U_diagonal_blocks,
        U_upper_diagonal_blocks,
    ) = lu_factorize_tridiag(
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_upper_diagonal_blocks,
    )

    (
        X_sdr_diagonal_blocks,
        X_sdr_lower_diagonal_blocks,
        X_sdr_upper_diagonal_blocks,
    ) = lu_sinv_tridiag(
        L_diagonal_blocks,
        L_lower_diagonal_blocks,
        U_diagonal_blocks,
        U_upper_diagonal_blocks,
    )

    X_sdr_dense = from_tridiagonal_arrays_to_dense(
        X_sdr_diagonal_blocks,
        X_sdr_lower_diagonal_blocks,
        X_sdr_upper_diagonal_blocks,
    )

    X_diff = X_ref - X_sdr_dense

    fig, ax = plt.subplots(1, 3)
    ax[0].set_title("X_ref: Scipy reference inversion")
    ax[0].matshow(X_ref)
    ax[1].set_title("X_sdr: LU selected inversion")
    ax[1].matshow(X_sdr_dense)
    ax[2].set_title("X_diff: Difference between X_ref and X_sdr")
    ax[2].matshow(X_diff)
    fig.colorbar(ax[2].matshow(X_diff), ax=ax[2], label="Relative error", shrink=0.4)

    plt.show()

    assert np.allclose(X_ref, X_sdr_dense)
