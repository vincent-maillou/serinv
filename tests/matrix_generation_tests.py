"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Lisa Gaedke-Merzhaeuser  (lisa.gaedke.merzhaeuser@usi.ch)
@date: 2023-11

Tests for matrix generations routines.

Copyright 2023 ETH Zurich and USI. All rights reserved.
"""

from sdr.utils import matrix_generation

import matplotlib.pyplot as plt


if __name__ == "__main__":
    nblocks = 5
    blocksize = 2
    symmetric = True
    diagonal_dominant = True
    seed = 63

    A = matrix_generation.generate_blocktridiag(
        nblocks, blocksize, symmetric, diagonal_dominant, seed
    )

    plt.matshow(A)
    plt.show()



if __name__ == "__main__":
    nblocks = 5
    diag_blocksize = 3
    arrow_blocksize = 2
    symmetric = True
    diagonal_dominant = True
    seed = 63

    A = matrix_generation.generate_blocktridiag_arrowhead(
        nblocks, diag_blocksize, arrow_blocksize, symmetric, diagonal_dominant, 
        seed
    )

    plt.matshow(A)
    plt.show()
    