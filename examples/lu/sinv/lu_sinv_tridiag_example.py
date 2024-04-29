# Copyright 2023-2024 ETH Zurich and USI. All rights reserved.

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la

from serinv.lu.lu_factorize import lu_factorize_tridiag
from serinv.lu.lu_selected_inversion import lu_sinv_tridiag
from serinv.utils import matrix_generation_dense
from serinv.utils.matrix_transformation_dense import (
    zeros_to_block_tridiagonal_shape,
    convert_block_tridiagonal_dense_to_arrays,
    convert_block_tridiagonal_arrays_to_dense,
)

# Example of block tridiagonal lu sinv
if __name__ == "__main__":
    nblocks = 5
    blocksize = 2
    symmetric = False
    diagonal_dominant = True
    seed = 63

    A = matrix_generation_dense.generate_block_tridiagonal_dense(
        nblocks, blocksize, symmetric, diagonal_dominant, seed
    )

    # --- Inversion ---

    X_ref = la.inv(A)
    X_ref = zeros_to_block_tridiagonal_shape(X_ref, blocksize)

    (
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_upper_diagonal_blocks,
    ) = convert_block_tridiagonal_dense_to_arrays(A, blocksize)

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

    X_sdr_dense = convert_block_tridiagonal_arrays_to_dense(
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
