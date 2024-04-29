# Copyright 2023-2024 ETH Zurich and USI. All rights reserved.

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la

from serinv.cholesky.cholesky_factorize import (
    cholesky_factorize_block_tridiagonal_arrowhead,
)
from serinv.utils import matrix_generation_dense
from serinv.utils.matrix_transformation_arrays import (
    convert_block_tridiagonal_arrowhead_dense_to_arrays,
)


# Testing of block tridiagonal arrowhead cholesky
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

    L_ref = la.cholesky(A, lower=True)

    (
        L_diagonal_blocks_ref,
        L_lower_diagonal_blocks_ref,
        _,
        L_arrow_bottom_blocks_ref,
        _,
        L_arrow_tip_block_ref,
    ) = convert_block_tridiagonal_arrowhead_dense_to_arrays(
        L_ref, diag_blocksize, arrow_blocksize
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

    assert np.allclose(L_diagonal_blocks_ref, L_diagonal_blocks)
    assert np.allclose(L_lower_diagonal_blocks_ref, L_lower_diagonal_blocks)
    assert np.allclose(L_arrow_bottom_blocks_ref, L_arrow_bottom_blocks)
    assert np.allclose(L_arrow_tip_block_ref, L_arrow_tip_block)

    fig, ax = plt.subplots(2, 4)
    ax[0, 0].matshow(L_diagonal_blocks_ref)
    ax[0, 0].set_title("L_diagonal_blocks_ref")
    ax[0, 1].matshow(L_lower_diagonal_blocks_ref)
    ax[0, 1].set_title("L_lower_diagonal_blocks_ref")
    ax[0, 2].matshow(L_arrow_bottom_blocks_ref)
    ax[0, 2].set_title("L_arrow_bottom_blocks_ref")
    ax[0, 3].matshow(L_arrow_tip_block_ref)
    ax[0, 3].set_title("L_arrow_tip_block_ref")

    ax[1, 0].matshow(L_diagonal_blocks)
    ax[1, 0].set_title("L_diagonal_blocks")
    ax[1, 1].matshow(L_lower_diagonal_blocks)
    ax[1, 1].set_title("L_lower_diagonal_blocks")
    ax[1, 2].matshow(L_arrow_bottom_blocks)
    ax[1, 2].set_title("L_arrow_bottom_blocks")
    ax[1, 3].matshow(L_arrow_tip_block)
    ax[1, 3].set_title("L_arrow_tip_block")
    plt.show()
