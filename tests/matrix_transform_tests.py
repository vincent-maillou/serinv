"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Lisa Gaedke-Merzhaeuser  (lisa.gaedke.merzhaeuser@usi.ch)
@date: 2023-11

Tests for matrix transformations routines.

Copyright 2023 ETH Zurich and USI. All rights reserved.
"""

from sdr.utils import matrix_transform
from sdr.utils import matrix_generation

import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    nblocks = 5
    blocksize = 2
    symmetric = True
    seed = 63

    A = matrix_generation.generate_blocktridiag(
        nblocks, blocksize, symmetric, seed
    )

    plt.matshow(A)

    A_inv = np.linalg.inv(A)

    plt.matshow(A_inv)

    A_cut = matrix_transform.cut_to_blocktridiag(A_inv, blocksize)

    plt.matshow(A_cut)

    plt.show()



if __name__ == "__main__":
    nblocks = 5
    diag_blocksize = 3
    arrow_blocksize = 2
    symmetric = True
    seed = 63

    A = matrix_generation.generate_blocktridiag_arrowhead(
        nblocks, diag_blocksize, arrow_blocksize, symmetric, seed
    )

    plt.matshow(A)

    A_inv = np.linalg.inv(A)

    plt.matshow(A_inv)

    A_cut = matrix_transform.cut_to_blocktridiag_arrowhead(A_inv, diag_blocksize, arrow_blocksize)

    plt.matshow(A_cut)

    plt.show()
    