# Copyright 2023-2024 ETH Zurich and USI. All rights reserved.

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la

from serinv.cholesky.cholesky_factorize import cholesky_factorize_block_tridiagonal
from serinv.utils.matrix_generation_dense import generate_block_tridiagonal_dense
from serinv.utils.matrix_transformation_arrays import (
    convert_block_tridiagonal_dense_to_arrays,
)

if __name__ == "__main__":
    nblocks = 5
    blocksize = 4
    symmetric = True
    diagonal_dominant = True
    seed = 63

    A = generate_block_tridiagonal_dense(
        nblocks, blocksize, symmetric, diagonal_dominant, seed
    )

    L_ref = la.cholesky(A, lower=True)

    (L_diagonal_blocks_ref, L_lower_diagonal_blocks_ref, _) = (
        convert_block_tridiagonal_dense_to_arrays(L_ref, blocksize)
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

    assert np.allclose(L_diagonal_blocks, L_diagonal_blocks_ref)
    assert np.allclose(L_lower_diagonal_blocks, L_lower_diagonal_blocks_ref)

    fig, ax = plt.subplots(2, 2)
    ax[0, 0].matshow(L_diagonal_blocks_ref)
    ax[0, 0].set_title("L_ref diagonal blocks")
    ax[0, 1].matshow(L_lower_diagonal_blocks_ref)
    ax[0, 1].set_title("L_ref lower diagonal blocks")
    ax[1, 0].matshow(L_diagonal_blocks)
    ax[1, 0].set_title("L diagonal blocks")
    ax[1, 1].matshow(L_lower_diagonal_blocks)
    ax[1, 1].set_title("L lower diagonal blocks")
    plt.show()
