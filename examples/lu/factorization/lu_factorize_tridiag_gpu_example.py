"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Lisa Gaedke-Merzhaeuser  (lisa.gaedke.merzhaeuser@usi.ch)
@date: 2024-03

Tests for lu tridiagonal matrices selected factorization routine, on GPU.

Copyright 2023-2024 ETH Zurich and USI. All rights reserved.
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la

from sdr.lu.lu_factorize_gpu import lu_factorize_tridiag_gpu
from sdr.utils import matrix_generation_dense
from sdr.utils.matrix_transformation_dense import (
    convert_block_tridiagonal_dense_to_arrays,
    convert_block_tridiagonal_arrays_to_dense,
)

# Example of block tridiagonal lu
if __name__ == "__main__":
    nblocks = 5
    blocksize = 3
    symmetric = False
    diagonal_dominant = True
    seed = 63

    A = matrix_generation_dense.generate_block_tridiagonal_dense(
        nblocks, blocksize, symmetric, diagonal_dominant, seed
    )

    # --- Decomposition ---
    # permute_l is default but to raise awareness that this would fail otherwise ...
    P_ref, L_ref, U_ref = la.lu(A, permute_l=False)

    if not np.allclose(P_ref, np.eye(A.shape[0])):
        plt.matshow(P_ref)
        plt.title("WARNING: Permutation matrix should be identity")

    (
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_upper_diagonal_blocks,
    ) = convert_block_tridiagonal_dense_to_arrays(A, blocksize)

    (
        L_diagonal_blocks,
        L_lower_diagonal_blocks,
        U_diagonal_blocks,
        U_upper_diagonal_blocks,
    ) = lu_factorize_tridiag_gpu(
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_upper_diagonal_blocks,
    )

    L_sdr_dense = convert_block_tridiagonal_arrays_to_dense(
        L_diagonal_blocks,
        L_lower_diagonal_blocks,
        np.zeros((blocksize, (nblocks - 1) * blocksize)),
    )

    U_sdr_dense = convert_block_tridiagonal_arrays_to_dense(
        U_diagonal_blocks,
        np.zeros((blocksize, (nblocks - 1) * blocksize)),
        U_upper_diagonal_blocks,
    )

    fig, ax = plt.subplots(2, 3)
    ax[0, 0].set_title("L_ref: Scipy lower factor")
    ax[0, 0].matshow(L_ref)
    ax[1, 0].set_title("U_ref: Scipy upper factor")
    ax[1, 0].matshow(U_ref)
    ax[0, 1].set_title("L_sdr_dense: SDR lower factor")
    ax[0, 1].matshow(L_sdr_dense)
    ax[1, 1].set_title("U_sdr_dense: SDR upper factor")
    ax[1, 1].matshow(U_sdr_dense)

    L_diff = L_ref - L_sdr_dense
    U_diff = U_ref - U_sdr_dense
    ax[0, 2].set_title("L: Difference between L_ref and L_sdr_dense")
    ax[0, 2].matshow(L_diff)
    ax[1, 2].set_title("U: Difference between U_ref and U_sdr_dense")
    ax[1, 2].matshow(U_diff)
    fig.colorbar(ax[0, 2].matshow(L_diff), ax=ax[0, 2], label="Relative error")
    fig.colorbar(ax[1, 2].matshow(U_diff), ax=ax[1, 2], label="Relative error")

    plt.show()

    assert np.allclose(L_ref, L_sdr_dense)
    assert np.allclose(U_ref, U_sdr_dense)
