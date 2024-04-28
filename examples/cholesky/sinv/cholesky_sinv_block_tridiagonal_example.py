# Copyright 2023-2024 ETH Zurich and USI. All rights reserved.

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la

from sdr.cholesky.cholesky_factorize import cholesky_factorize_block_tridiagonal
from sdr.cholesky.cholesky_selected_inversion import cholesky_sinv_block_tridiagonal
from sdr.utils import matrix_generation_dense
from sdr.utils.matrix_transformation_arrays import (
    convert_block_tridiagonal_dense_to_arrays,
)


# Testing of block tridiagonal cholesky sinv
if __name__ == "__main__":
    nblocks = 5
    blocksize = 3
    symmetric = True
    diagonal_dominant = True
    seed = 63

    A = matrix_generation_dense.generate_block_tridiagonal_dense(
        nblocks, blocksize, symmetric, diagonal_dominant, seed
    )

    # --- Inversion ---

    X_ref = la.inv(A)

    (X_diagonal_blocks_ref, X_lower_diagonal_blocks_ref, _) = (
        convert_block_tridiagonal_dense_to_arrays(X_ref, blocksize)
    )

    (A_diagonal_blocks, A_lower_diagonal_blocks, _) = (
        convert_block_tridiagonal_dense_to_arrays(A, blocksize)
    )

    (
        L_diagonal_blocks,
        L_lower_diagonal_blocks,
    ) = cholesky_factorize_block_tridiagonal(
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
    )

    (
        X_diagonal_blocks,
        X_lower_diagonal_blocks,
    ) = cholesky_sinv_block_tridiagonal(
        L_diagonal_blocks,
        L_lower_diagonal_blocks,
    )

    assert np.allclose(X_lower_diagonal_blocks_ref, X_lower_diagonal_blocks)
    assert np.allclose(X_diagonal_blocks_ref, X_diagonal_blocks)

    fig, ax = plt.subplots(2, 2)
    ax[0, 0].matshow(X_diagonal_blocks_ref)
    ax[0, 0].set_title("X_diagonal_blocks_ref")
    ax[0, 1].matshow(X_lower_diagonal_blocks_ref)
    ax[0, 1].set_title("X_lower_diagonal_blocks_ref")
    ax[1, 0].matshow(X_diagonal_blocks)
    ax[1, 0].set_title("X_diagonal_blocks")
    ax[1, 1].matshow(X_lower_diagonal_blocks)
    ax[1, 1].set_title("X_lower_diagonal_blocks")
    plt.show()
