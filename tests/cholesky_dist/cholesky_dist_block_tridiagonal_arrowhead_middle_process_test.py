"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Lisa Gaedke-Merzhaeuser  (lisa.gaedke.merzhaeuser@usi.ch)
@date: 2024-03

Test of the top process part of lu_dist algorithm for tridiagonal arrowhead 
matrices.

Copyright 2023-2024 ETH Zurich and USI. All rights reserved.
"""

import copy as cp

import numpy as np
import pytest

from sdr.cholesky_dist.cholesky_dist_block_tridiagonal_arrowhead import (
    middle_factorize,
    middle_sinv,
)
from sdr.utils.matrix_generation_dense import generate_block_tridiagonal_arrowhead_dense
from sdr.utils.matrix_transformation_arrays import (
    convert_block_tridiagonal_arrowhead_dense_to_arrays,
)


@pytest.mark.cpu
@pytest.mark.mpi_skip()
@pytest.mark.parametrize(
    "nblocks, diag_blocksize, arrow_blocksize",
    [
        (2, 2, 2),
        (2, 3, 2),
        (2, 2, 3),
        (10, 2, 2),
        (10, 3, 2),
        (10, 2, 3),
        (10, 10, 2),
        (10, 2, 10),
    ],
)
def test_cholesky_dist_middle_process(
    nblocks: int,
    diag_blocksize: int,
    arrow_blocksize: int,
):
    diagonal_dominant = True
    symmetric = True
    seed = 63

    A = generate_block_tridiagonal_arrowhead_dense(
        nblocks, diag_blocksize, arrow_blocksize, symmetric, diagonal_dominant, seed
    )

    # ----- Reference -----
    A_ref = cp.deepcopy(A)

    X_ref = np.linalg.inv(A_ref)

    (
        X_ref_diagonal_blocks,
        X_ref_lower_diagonal_blocks,
        _,
        X_ref_arrow_bottom_blocks,
        _,
        X_ref_arrow_tip_block,
    ) = convert_block_tridiagonal_arrowhead_dense_to_arrays(
        X_ref, diag_blocksize, arrow_blocksize
    )
    # ---------------------

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

    n_diag_blocks = nblocks - 1

    # Arrays that store the update of the 2sided pattern for the middle processes
    A_top_2sided_arrow_blocks_local = np.zeros(
        (diag_blocksize, n_diag_blocks * diag_blocksize), dtype=A_diagonal_blocks.dtype
    )
    A_top_2sided_arrow_blocks_local[:, :diag_blocksize] = A_diagonal_blocks[
        :, :diag_blocksize
    ]
    A_top_2sided_arrow_blocks_local[:, diag_blocksize : 2 * diag_blocksize] = (
        A_lower_diagonal_blocks[:, :diag_blocksize].T
    )

    (
        L_diagonal_blocks,
        L_lower_diagonal_blocks,
        L_arrow_bottom_blocks,
        L_upper_2sided_arrow_blocks,
        Update_arrow_tip,
    ) = middle_factorize(
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_arrow_bottom_blocks,
        A_top_2sided_arrow_blocks_local,
        A_arrow_tip_block,
    )

    reduced_system = np.zeros(
        (2 * diag_blocksize + arrow_blocksize, 2 * diag_blocksize + arrow_blocksize)
    )

    # (top, top)
    reduced_system[0:diag_blocksize, 0:diag_blocksize] = A_diagonal_blocks[
        :, 0:diag_blocksize
    ]
    # (top, nblocks)
    reduced_system[
        0:diag_blocksize, -diag_blocksize - arrow_blocksize : -arrow_blocksize
    ] = A_top_2sided_arrow_blocks_local[:, -diag_blocksize:]
    # (top, ndb+1)
    reduced_system[0:diag_blocksize, -arrow_blocksize:] = A_arrow_bottom_blocks[
        :, :diag_blocksize
    ].T
    # (nblocks, top)
    reduced_system[
        -diag_blocksize - arrow_blocksize : -arrow_blocksize, 0:diag_blocksize
    ] = A_top_2sided_arrow_blocks_local[:, -diag_blocksize:].T
    # (ndb+1, top)
    reduced_system[-arrow_blocksize:, 0:diag_blocksize] = A_arrow_bottom_blocks[
        :, :diag_blocksize
    ]
    # (nblocks, nblocks)
    reduced_system[
        -diag_blocksize - arrow_blocksize : -arrow_blocksize,
        -diag_blocksize - arrow_blocksize : -arrow_blocksize,
    ] = A_diagonal_blocks[:, -diag_blocksize:]
    # (nblocks, ndb+1)
    reduced_system[
        -diag_blocksize - arrow_blocksize : -arrow_blocksize, -arrow_blocksize:
    ] = A_arrow_bottom_blocks[:, -diag_blocksize:].T
    # (ndb+1, nblocks)
    reduced_system[
        -arrow_blocksize:, -diag_blocksize - arrow_blocksize : -arrow_blocksize
    ] = A_arrow_bottom_blocks[:, -diag_blocksize:]
    # (ndb+1, ndb+1)
    reduced_system[-arrow_blocksize:, -arrow_blocksize:] = (
        A_arrow_tip_block + Update_arrow_tip
    )

    reduced_system_inv = np.linalg.inv(reduced_system)

    import matplotlib.pyplot as plt

    plt.matshow(X_ref)
    plt.matshow(reduced_system_inv)
    plt.show()


if __name__ == "__main__":
    test_cholesky_dist_middle_process(10, 4, 2)
