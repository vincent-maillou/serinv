# Copyright 2023-2024 ETH Zurich and USI. All rights reserved.

import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy.linalg as la

from sdr.lu.lu_decompose import lu_dcmp_ndiags
from sdr.lu.lu_selected_inversion import lu_sinv_ndiags
from sdr.utils import matrix_generation_dense
from sdr.utils.matrix_transformation_dense import zeros_to_blocks_banded_shape

# Testing of block tridiagonal lu sinv
if __name__ == "__main__":
    nblocks = 8
    ndiags = 7
    blocksize = 2
    symmetric = False
    diagonal_dominant = True
    seed = 63

    A = matrix_generation_dense.generate_blocks_banded_dense(
        nblocks, ndiags, blocksize, symmetric, diagonal_dominant, seed
    )

    # --- Inversion ---

    X_ref = la.inv(A)
    X_ref = zeros_to_blocks_banded_shape(X_ref, ndiags, blocksize)

    L_sdr, U_sdr = lu_dcmp_ndiags(A, ndiags, blocksize)

    fig, ax = plt.subplots(1, 3)
    ax[0].set_title("X_ref: Scipy reference inversion")
    ax[0].matshow(X_ref)

    X_sdr = lu_sinv_ndiags(L_sdr, U_sdr, ndiags, blocksize)
    ax[1].set_title("X_sdr: LU selected inversion")
    ax[1].matshow(X_sdr)

    X_diff = X_ref - X_sdr
    ax[2].set_title("X_diff: Difference between X_ref and X_sdr")
    ax[2].matshow(X_diff)
    fig.colorbar(ax[2].matshow(X_diff), ax=ax[2], label="Relative error", shrink=0.4)

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
def test_lu_sinv_ndiags(nblocks, ndiags, blocksize):
    symmetric = False
    diagonal_dominant = True
    seed = 63

    A = matrix_generation_dense.generate_blocks_banded_dense(
        nblocks, ndiags, blocksize, symmetric, diagonal_dominant, seed
    )

    # --- Inversion ---

    X_ref = la.inv(A)
    X_ref = zeros_to_blocks_banded_shape(X_ref, ndiags, blocksize)

    L_sdr, U_sdr = lu_dcmp_ndiags(A, ndiags, blocksize)
    X_sdr = lu_sinv_ndiags(L_sdr, U_sdr, ndiags, blocksize)

    assert np.allclose(X_ref, X_sdr)
