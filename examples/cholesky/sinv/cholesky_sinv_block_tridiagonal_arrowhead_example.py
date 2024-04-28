# Copyright 2023-2024 ETH Zurich and USI. All rights reserved.

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la

from serinv.cholesky.cholesky_factorize import (
    cholesky_factorize_block_tridiagonal_arrowhead,
)
from serinv.cholesky.cholesky_selected_inversion import (
    cholesky_sinv_block_tridiagonal_arrowhead,
)
from serinv.utils import matrix_generation_dense
from serinv.utils.matrix_transformation_arrays import (
    convert_block_tridiagonal_arrowhead_dense_to_arrays,
)

# Testing of block tridiagonal cholesky sinv
if __name__ == "__main__":
    nblocks = 5
    diag_blocksize = 3
    arrow_blocksize = 2
    symmetric = True
    diagonal_dominant = True
    seed = 63

    A = matrix_generation_dense.generate_block_tridiagonal_arrowhead_dense(
        nblocks, diag_blocksize, arrow_blocksize, symmetric, diagonal_dominant, seed
    )

    X_ref = la.inv(A)

    (
        X_diagonal_blocks_ref,
        X_lower_diagonal_blocks_ref,
        _,
        X_arrow_bottom_blocks_ref,
        _,
        X_arrow_tip_block_ref,
    ) = convert_block_tridiagonal_arrowhead_dense_to_arrays(
        X_ref, diag_blocksize, arrow_blocksize
    )

    (
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        _,
        A_arrow_bottom_blocks,
        _,
        A_arrow_tip_block,
    ) = convert_block_tridiagonal_arrowhead_dense_to_arrays(
        A, diag_blocksize, arrow_blocksize
    )

    (
        L_diagonal_blocks,
        L_lower_diagonal_blocks,
        L_arrow_bottom_blocks,
        L_arrow_tip_block,
    ) = cholesky_factorize_block_tridiagonal_arrowhead(
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_arrow_bottom_blocks,
        A_arrow_tip_block,
    )

    (
        X_diagonal_blocks,
        X_lower_diagonal_blocks,
        X_arrow_bottom_blocks,
        X_arrow_tip_block,
    ) = cholesky_sinv_block_tridiagonal_arrowhead(
        L_diagonal_blocks,
        L_lower_diagonal_blocks,
        L_arrow_bottom_blocks,
        L_arrow_tip_block,
    )

    assert np.allclose(X_diagonal_blocks_ref, X_diagonal_blocks)
    assert np.allclose(X_lower_diagonal_blocks_ref, X_lower_diagonal_blocks)
    assert np.allclose(X_arrow_bottom_blocks_ref, X_arrow_bottom_blocks)
    assert np.allclose(X_arrow_tip_block_ref, X_arrow_tip_block)

    fig, ax = plt.subplots(2, 4)
    ax[0, 0].matshow(X_diagonal_blocks_ref)
    ax[0, 0].set_title("X_diagonal_blocks_ref")
    ax[0, 1].matshow(X_lower_diagonal_blocks_ref)
    ax[0, 1].set_title("X_lower_diagonal_blocks_ref")
    ax[0, 2].matshow(X_arrow_bottom_blocks_ref)
    ax[0, 2].set_title("X_arrow_bottom_blocks_ref")
    ax[0, 3].matshow(X_arrow_tip_block_ref)
    ax[0, 3].set_title("X_arrow_tip_block_ref")

    ax[1, 0].matshow(X_diagonal_blocks)
    ax[1, 0].set_title("X_diagonal_blocks")
    ax[1, 1].matshow(X_lower_diagonal_blocks)
    ax[1, 1].set_title("X_lower_diagonal_blocks")
    ax[1, 2].matshow(X_arrow_bottom_blocks)
    ax[1, 2].set_title("X_arrow_bottom_blocks")
    ax[1, 3].matshow(X_arrow_tip_block)
    ax[1, 3].set_title("X_arrow_tip_block")
    plt.show()
