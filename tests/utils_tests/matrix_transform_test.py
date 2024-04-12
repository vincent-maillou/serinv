"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Lisa Gaedke-Merzhaeuser  (lisa.gaedke.merzhaeuser@usi.ch)
@date: 2023-11

Tests for matrix transformations routines.

Copyright 2023-2024 ETH Zurich and USI. All rights reserved.
"""

import matplotlib.pyplot as plt
import numpy as np

from sdr.utils import matrix_generation_dense, matrix_transformation_dense

if __name__ == "__main__":
    nblocks = 5
    blocksize = 2
    symmetric = True
    diagonal_dominant = True
    seed = 63

    A = matrix_generation_dense.generate_block_tridiagonal_dense(
        nblocks, blocksize, symmetric, diagonal_dominant, seed
    )

    plt.matshow(A)

    A_inv = np.linalg.inv(A)

    plt.matshow(A_inv)

    A_cut = matrix_transformation_dense.zeros_to_block_tridiagonal_shape(
        A_inv, blocksize
    )

    plt.matshow(A_cut)

    plt.show()


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

    plt.matshow(A)

    A_inv = np.linalg.inv(A)

    plt.matshow(A_inv)

    A_cut = matrix_transformation_dense.zeros_to_block_tridiagonal_arrowhead_shape(
        A_inv, diag_blocksize, arrow_blocksize
    )

    plt.matshow(A_cut)

    plt.show()
