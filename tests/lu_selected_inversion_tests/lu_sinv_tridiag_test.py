"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Lisa Gaedke-Merzhaeuser  (lisa.gaedke.merzhaeuser@usi.ch)
@date: 2023-11

Tests for lu selected inversion routines.

Copyright 2023-2024 ETH Zurich and USI. All rights reserved.
"""

from sdr.utils import matrix_generation
from sdr.utils.matrix_transform import cut_to_blocktridiag
from sdr.lu.lu_selected_inversion import lu_sinv_tridiag

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import pytest



# Testing of block tridiagonal lu sinv
if __name__ == "__main__":
    nblocks = 5
    blocksize = 2
    symmetric = False
    diagonal_dominant = True
    seed = 63

    A = matrix_generation.generate_tridiag_dense(
        nblocks, blocksize, symmetric, diagonal_dominant, seed
    )


    # --- Inversion ---

    X_ref = la.inv(A)
    X_ref = cut_to_blocktridiag(X_ref, blocksize)

    L_sdr, U_sdr = lu_dcmp_tridiag(A, blocksize)

    fig, ax = plt.subplots(1, 3)
    ax[0].set_title("X_ref: Scipy reference inversion")
    ax[0].matshow(X_ref)

    X_sdr = lu_sinv_tridiag(L_sdr, U_sdr, blocksize)
    ax[1].set_title("X_sdr: LU selected inversion")
    ax[1].matshow(X_sdr)

    X_diff = X_ref - X_sdr
    ax[2].set_title("X_diff: Difference between X_ref and X_sdr")
    ax[2].matshow(X_diff)
    fig.colorbar(ax[2].matshow(X_diff), ax=ax[2], label="Relative error", shrink=0.4)

    plt.show()

    

# @pytest.mark.parametrize(
#     "nblocks, blocksize", 
#     [
#         (2, 2),
#         (10, 2),
#         (100, 2),
#         (2, 3),
#         (10, 3),
#         (100, 3),
#         (2, 100),
#         (5, 100),
#         (10, 100),
#     ]
# )
# def test_lu_sinv_tridiag(
#     nblocks: int,
#     blocksize: int,  
# ):
#     symmetric = False
#     diagonal_dominant = True
#     seed = 63

#     A = matrix_generation.generate_tridiag_dense(
#         nblocks, blocksize, symmetric, diagonal_dominant, seed
#     )

#     # --- Inversion ---

#     X_ref = la.inv(A)
#     X_ref = cut_to_blocktridiag(X_ref, blocksize)

#     L_sdr, U_sdr = lu_dcmp_tridiag(A, blocksize)
#     X_sdr = lu_sinv_tridiag(L_sdr, U_sdr, blocksize)

#     assert np.allclose(X_ref, X_sdr)
