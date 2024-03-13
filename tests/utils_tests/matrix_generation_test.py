"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Lisa Gaedke-Merzhaeuser  (lisa.gaedke.merzhaeuser@usi.ch)
@date: 2023-11

Tests for matrix generations routines.

Copyright 2023-2024 ETH Zurich and USI. All rights reserved.
"""

import matplotlib.pyplot as plt

from sdr.utils import matrix_generation

if __name__ == "__main__":
    nblocks = 5
    blocksize = 2
    symmetric = True
    diagonal_dominant = True
    seed = 63

    A = matrix_generation.generate_tridiag_dense(
        nblocks, blocksize, symmetric, diagonal_dominant, seed
    )

    plt.matshow(A)
    plt.title("Block tridiagonal matrix \n blocksize = " + str(blocksize))
    plt.show()


if __name__ == "__main__":
    nblocks = 5
    diag_blocksize = 3
    arrow_blocksize = 2
    symmetric = True
    diagonal_dominant = True
    seed = 63

    A = matrix_generation.generate_tridiag_arrowhead_dense(
        nblocks, diag_blocksize, arrow_blocksize, symmetric, diagonal_dominant, seed
    )

    plt.matshow(A)
    plt.title("Block tridiagonal arrowhead matrix \n blocksize = " + str(blocksize))
    plt.show()


if __name__ == "__main__":
    nblocks = 6
    ndiags = 5
    blocksize = 2
    symmetric = True
    diagonal_dominant = True
    seed = 63

    A = matrix_generation.generate_block_ndiags_dense(
        nblocks, ndiags, blocksize, symmetric, diagonal_dominant, seed
    )

    plt.matshow(A)
    plt.title(
        "Block " + str(ndiags) + "-diagonals matrix \n blocksize = " + str(blocksize)
    )
    plt.show()


if __name__ == "__main__":
    nblocks = 7
    ndiags = 5
    diag_blocksize = 3
    arrow_blocksize = 2
    symmetric = True
    diagonal_dominant = True
    seed = 63

    A = matrix_generation.generate_ndiags_arrowhead_dense(
        nblocks,
        ndiags,
        diag_blocksize,
        arrow_blocksize,
        symmetric,
        diagonal_dominant,
        seed,
    )

    plt.matshow(A)
    plt.title(
        "Block "
        + str(ndiags)
        + "-diagonals arrowhead matrix \n diag_blocksize = "
        + str(diag_blocksize)
        + "\n arrow_blocksize = "
        + str(arrow_blocksize)
    )
    plt.show()
