"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Lisa Gaedke-Merzhaeuser  (lisa.gaedke.merzhaeuser@usi.ch)
@date: 2024-03

Test of the middle process part of lu_dist algorithm for tridiagonal arrowhead 
matrices.

Copyright 2023-2024 ETH Zurich and USI. All rights reserved.
"""

from copy import deepcopy

import numpy as np
import pytest

try:
    import cupy
except ImportError:
    import sys

    sys.exit()

from sdr.lu_dist.lu_dist_tridiagonal_arrowhead_gpu import (
    middle_factorize_gpu,
    middle_sinv_gpu,
)
from sdr.utils.matrix_generation import generate_tridiag_arrowhead_dense
from sdr.utils.matrix_transform import from_dense_to_arrowhead_arrays


@pytest.mark.gpu
@pytest.mark.mpi_skip()
@pytest.mark.parametrize(
    "nblocks, diag_blocksize, arrow_blocksize",
    [
        (3, 2, 2),
        (3, 3, 2),
        (3, 2, 3),
        (10, 2, 2),
        (10, 3, 2),
        (10, 2, 3),
        (10, 10, 2),
        (10, 2, 10),
    ],
)
def test_lu_dist_middle_process(
    nblocks: int,
    diag_blocksize: int,
    arrow_blocksize: int,
):
    diagonal_dominant = True
    symmetric = False
    seed = 63

    A = generate_tridiag_arrowhead_dense(
        nblocks, diag_blocksize, arrow_blocksize, symmetric, diagonal_dominant, seed
    )

    # ----- Reference -----
    A_ref = deepcopy(A)

    X_ref = np.linalg.inv(A_ref)

    (
        X_ref_diagonal_blocks,
        X_ref_lower_diagonal_blocks,
        X_ref_upper_diagonal_blocks,
        X_ref_arrow_bottom_blocks,
        X_ref_arrow_right_blocks,
        X_ref_arrow_tip_block,
    ) = from_dense_to_arrowhead_arrays(X_ref, diag_blocksize, arrow_blocksize)
    # ---------------------

    (
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_upper_diagonal_blocks,
        A_arrow_bottom_blocks,
        A_arrow_right_blocks,
        A_arrow_tip_block,
    ) = from_dense_to_arrowhead_arrays(A, diag_blocksize, arrow_blocksize)

    n_diag_blocks = nblocks - 1

    # Arrays that store the update of the 2sided pattern for the middle processes
    A_top_2sided_arrow_blocks_local = np.zeros(
        (diag_blocksize, n_diag_blocks * diag_blocksize), dtype=A_diagonal_blocks.dtype
    )
    A_left_2sided_arrow_blocks_local = np.zeros(
        (n_diag_blocks * diag_blocksize, diag_blocksize), dtype=A_diagonal_blocks.dtype
    )

    A_top_2sided_arrow_blocks_local[:, :diag_blocksize] = A_diagonal_blocks[
        :, :diag_blocksize
    ]
    A_top_2sided_arrow_blocks_local[
        :, diag_blocksize : 2 * diag_blocksize
    ] = A_upper_diagonal_blocks[:, :diag_blocksize]

    A_left_2sided_arrow_blocks_local[:diag_blocksize, :] = A_diagonal_blocks[
        :, :diag_blocksize
    ]
    A_left_2sided_arrow_blocks_local[
        diag_blocksize : 2 * diag_blocksize, :
    ] = A_lower_diagonal_blocks[:, :diag_blocksize]

    (
        L_diagonal_blocks,
        L_lower_diagonal_blocks,
        L_arrow_bottom_blocks,
        L_upper_2sided_arrow_blocks,
        U_diagonal_blocks,
        U_upper_diagonal_blocks,
        U_arrow_right_blocks,
        U_left_2sided_arrow_blocks,
        Update_arrow_tip,
        A_diagonal_blocks_local_updated,
        A_arrow_bottom_blocks_local_updated,
        A_arrow_right_blocks_local_updated,
        A_top_2sided_arrow_blocks_local_updated,
        A_left_2sided_arrow_blocks_local_updated,
    ) = middle_factorize_gpu(
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_upper_diagonal_blocks,
        A_arrow_bottom_blocks,
        A_arrow_right_blocks,
        A_top_2sided_arrow_blocks_local,
        A_left_2sided_arrow_blocks_local,
        A_arrow_tip_block,
    )

    # Create and inverse the reduced system created by the last reduced block
    # and the tip of the arrowhead.

    reduced_system = np.zeros(
        (2 * diag_blocksize + arrow_blocksize, 2 * diag_blocksize + arrow_blocksize)
    )

    # (top, top)
    reduced_system[
        0:diag_blocksize, 0:diag_blocksize
    ] = A_diagonal_blocks_local_updated[:, 0:diag_blocksize]
    # (top, nblocks)
    reduced_system[
        0:diag_blocksize, -diag_blocksize - arrow_blocksize : -arrow_blocksize
    ] = A_top_2sided_arrow_blocks_local_updated[:, -diag_blocksize:]
    # (top, ndb+1)
    reduced_system[
        0:diag_blocksize, -arrow_blocksize:
    ] = A_arrow_right_blocks_local_updated[:diag_blocksize, :]
    # (nblocks, top)
    reduced_system[
        -diag_blocksize - arrow_blocksize : -arrow_blocksize, 0:diag_blocksize
    ] = A_left_2sided_arrow_blocks_local_updated[-diag_blocksize:, :]
    # (ndb+1, top)
    reduced_system[
        -arrow_blocksize:, 0:diag_blocksize
    ] = A_arrow_bottom_blocks_local_updated[:, :diag_blocksize]
    # (nblocks, nblocks)
    reduced_system[
        -diag_blocksize - arrow_blocksize : -arrow_blocksize,
        -diag_blocksize - arrow_blocksize : -arrow_blocksize,
    ] = A_diagonal_blocks_local_updated[:, -diag_blocksize:]
    # (nblocks, ndb+1)
    reduced_system[
        -diag_blocksize - arrow_blocksize : -arrow_blocksize, -arrow_blocksize:
    ] = A_arrow_right_blocks_local_updated[-diag_blocksize:, :]
    # (ndb+1, nblocks)
    reduced_system[
        -arrow_blocksize:, -diag_blocksize - arrow_blocksize : -arrow_blocksize
    ] = A_arrow_bottom_blocks_local_updated[:, -diag_blocksize:]
    # (ndb+1, ndb+1)
    reduced_system[-arrow_blocksize:, -arrow_blocksize:] = (
        A_arrow_tip_block + Update_arrow_tip
    )

    reduced_system_inv = np.linalg.inv(reduced_system)

    X_sdr_diagonal_blocks = np.zeros_like(A_diagonal_blocks)
    X_sdr_lower_diagonal_blocks = np.zeros_like(A_lower_diagonal_blocks)
    X_sdr_upper_diagonal_blocks = np.zeros_like(A_upper_diagonal_blocks)
    X_sdr_arrow_bottom_blocks = np.zeros_like(A_arrow_bottom_blocks)
    X_sdr_arrow_right_blocks = np.zeros_like(A_arrow_right_blocks)
    X_sdr_top_2sided_arrow_blocks_local = np.zeros_like(A_top_2sided_arrow_blocks_local)
    X_sdr_left_2sided_arrow_blocks_local = np.zeros_like(
        A_left_2sided_arrow_blocks_local
    )
    X_sdr_global_arrow_tip_block = np.zeros_like(A_arrow_tip_block)

    # (top, top)
    X_sdr_diagonal_blocks[:, 0:diag_blocksize] = reduced_system_inv[
        0:diag_blocksize, 0:diag_blocksize
    ]
    # (top, nblocks)
    X_sdr_top_2sided_arrow_blocks_local[:, -diag_blocksize:] = reduced_system_inv[
        0:diag_blocksize, -diag_blocksize - arrow_blocksize : -arrow_blocksize
    ]
    # (top, ndb+1)
    X_sdr_arrow_right_blocks[:diag_blocksize, :] = reduced_system_inv[
        0:diag_blocksize, -arrow_blocksize:
    ]
    # (nblocks, top)
    X_sdr_left_2sided_arrow_blocks_local[-diag_blocksize:, :] = reduced_system_inv[
        -diag_blocksize - arrow_blocksize : -arrow_blocksize, 0:diag_blocksize
    ]
    # (ndb+1, top)
    X_sdr_arrow_bottom_blocks[:, :diag_blocksize] = reduced_system_inv[
        -arrow_blocksize:, 0:diag_blocksize
    ]
    # (nblocks, nblocks)
    X_sdr_diagonal_blocks[:, -diag_blocksize:] = reduced_system_inv[
        -diag_blocksize - arrow_blocksize : -arrow_blocksize,
        -diag_blocksize - arrow_blocksize : -arrow_blocksize,
    ]
    # (nblocks, ndb+1)
    X_sdr_arrow_right_blocks[-diag_blocksize:, :] = reduced_system_inv[
        -diag_blocksize - arrow_blocksize : -arrow_blocksize, -arrow_blocksize:
    ]
    # (ndb+1, nblocks)
    X_sdr_arrow_bottom_blocks[:, -diag_blocksize:] = reduced_system_inv[
        -arrow_blocksize:, -diag_blocksize - arrow_blocksize : -arrow_blocksize
    ]
    # (ndb+1, ndb+1)
    X_sdr_global_arrow_tip_block = reduced_system_inv[
        -arrow_blocksize:, -arrow_blocksize:
    ]

    # ----- Selected inversion part -----
    (
        X_sdr_diagonal_blocks,
        X_sdr_lower_diagonal_blocks,
        X_sdr_upper_diagonal_blocks,
        X_sdr_arrow_bottom_blocks,
        X_sdr_arrow_right_blocks,
        X_sdr_global_arrow_tip_block,
    ) = middle_sinv_gpu(
        X_sdr_diagonal_blocks,
        X_sdr_lower_diagonal_blocks,
        X_sdr_upper_diagonal_blocks,
        X_sdr_arrow_bottom_blocks,
        X_sdr_arrow_right_blocks,
        X_sdr_top_2sided_arrow_blocks_local,
        X_sdr_left_2sided_arrow_blocks_local,
        X_sdr_global_arrow_tip_block,
        L_diagonal_blocks,
        L_lower_diagonal_blocks,
        L_arrow_bottom_blocks,
        L_upper_2sided_arrow_blocks,
        U_diagonal_blocks,
        U_upper_diagonal_blocks,
        U_arrow_right_blocks,
        U_left_2sided_arrow_blocks,
    )

    assert np.allclose(X_ref_diagonal_blocks, X_sdr_diagonal_blocks)
    assert np.allclose(X_ref_arrow_bottom_blocks, X_sdr_arrow_bottom_blocks)
    assert np.allclose(X_ref_arrow_right_blocks, X_sdr_arrow_right_blocks)
    assert np.allclose(X_ref_arrow_tip_block, X_sdr_global_arrow_tip_block)
    assert np.allclose(X_ref_lower_diagonal_blocks, X_sdr_lower_diagonal_blocks)
    assert np.allclose(X_ref_upper_diagonal_blocks, X_sdr_upper_diagonal_blocks)


if __name__ == "__main__":
    test_lu_dist_middle_process(10, 10, 2)
