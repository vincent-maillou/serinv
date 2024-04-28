# Copyright 2023-2024 ETH Zurich and USI. All rights reserved.

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as npla

from sdr.cholesky.cholesky_factorize import cholesky_factorize_block_tridiagonal
from sdr.cholesky.cholesky_solve import cholesky_solve_block_tridiagonal
from sdr.utils import matrix_generation_dense
from sdr.utils.matrix_transformation_arrays import (
    convert_block_tridiagonal_dense_to_arrays,
)

# Testing of block tridiagonal cholesky
if __name__ == "__main__":
    nblocks = 5
    blocksize = 2
    symmetric = True
    diagonal_dominant = True
    seed = 63

    A = matrix_generation_dense.generate_block_tridiagonal_dense(
        nblocks, blocksize, symmetric, diagonal_dominant, seed
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

    n_rhs = 2
    B = np.random.randn(A.shape[0], n_rhs)

    # --- Solving ---

    X_ref = npla.solve(A, B)

    X_sdr = cholesky_solve_block_tridiagonal(
        L_diagonal_blocks,
        L_lower_diagonal_blocks,
        B,
    )

    assert np.allclose(X_ref, X_sdr)

    fig, ax = plt.subplots(1, 3)
    ax[0].set_title("X_ref")
    ax[0].matshow(X_ref)

    ax[1].set_title("X_sdr: Selected cholesky solver")
    ax[1].matshow(X_sdr)

    X_diff = X_ref - X_sdr
    ax[2].set_title("X_diff: Difference between X_ref and X_sdr")
    ax[2].matshow(X_diff)
    fig.colorbar(ax[2].matshow(X_diff), ax=ax[2], label="Relative error", shrink=0.4)

    plt.show()
