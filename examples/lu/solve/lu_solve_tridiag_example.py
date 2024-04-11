"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Lisa Gaedke-Merzhaeuser  (lisa.gaedke.merzhaeuser@usi.ch)
@date: 2024-02

Example for lu tridiagonal matrices selected factorization routine.

Copyright 2023-2024 ETH Zurich and USI. All rights reserved.
"""

import matplotlib.pyplot as plt
import numpy as np

from sdr.lu.lu_factorize import lu_factorize_tridiag
from sdr.lu.lu_solve import lu_solve_tridiag
from sdr.utils import matrix_generation_dense
from sdr.utils.matrix_transformation_dense import (
    from_dense_to_tridiagonal_arrays,
)

# Example of block tridiagonal lu
if __name__ == "__main__":
    nblocks = 5
    blocksize = 2
    symmetric = False
    diagonal_dominant = True
    seed = 63
    n_rhs = 1

    A = matrix_generation_dense.generate_tridiag_dense(
        nblocks, blocksize, symmetric, diagonal_dominant, seed
    )

    # --- Factorization LU ---
    (
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_upper_diagonal_blocks,
    ) = from_dense_to_tridiagonal_arrays(A, blocksize)

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

    # --- Solving ---
    B = np.random.randn(A.shape[0], n_rhs)
    X_ref = np.linalg.solve(A, B)

    fig, ax = plt.subplots(1, 3)
    ax[0].set_title("X_ref: Reference numpy solver")
    ax[0].matshow(X_ref)

    X_sdr = lu_solve_tridiag(
        L_diagonal_blocks,
        L_lower_diagonal_blocks,
        U_diagonal_blocks,
        U_upper_diagonal_blocks,
        B, 
    )
    ax[1].set_title("X_sdr: Selected lu solver")
    ax[1].matshow(X_sdr)

    X_diff = X_ref - X_sdr
    ax[2].set_title("X_diff: Difference between X_ref and X_sdr")
    ax[2].matshow(X_diff)
    fig.colorbar(ax[2].matshow(X_diff), ax=ax[2], label="Relative error", shrink=0.4)

    plt.show()
