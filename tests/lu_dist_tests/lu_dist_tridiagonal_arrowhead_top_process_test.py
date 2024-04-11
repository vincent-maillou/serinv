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

from sdr.lu_dist.lu_dist_tridiagonal_arrowhead import top_factorize, top_sinv
from sdr.utils.matrix_generation_dense import generate_tridiag_arrowhead_dense
from sdr.utils.matrix_transformation_arrays import from_dense_to_arrowhead_arrays


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
def test_lu_dist_top_process(
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
    A_ref = cp.deepcopy(A)

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

    (
        L_diagonal_blocks,
        L_lower_diagonal_blocks,
        L_arrow_bottom_blocks,
        U_diagonal_blocks,
        U_upper_diagonal_blocks,
        U_arrow_right_blocks,
        Update_arrow_tip,
    ) = top_factorize(
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_upper_diagonal_blocks,
        A_arrow_bottom_blocks,
        A_arrow_right_blocks,
        A_arrow_tip_block,
    )

    # Create and inverse the reduced system created by the last reduced block
    # and the tip of the arrowhead.

    reduced_system = np.zeros(
        (diag_blocksize + arrow_blocksize, diag_blocksize + arrow_blocksize)
    )
    reduced_system[0:diag_blocksize, 0:diag_blocksize] = A_diagonal_blocks[
        :, -diag_blocksize:
    ]
    reduced_system[-arrow_blocksize:, -arrow_blocksize:] = (
        A_arrow_tip_block + Update_arrow_tip
    )
    reduced_system[0:diag_blocksize, -arrow_blocksize:] = A_arrow_right_blocks[
        -diag_blocksize:, :
    ]
    reduced_system[-arrow_blocksize:, 0:diag_blocksize] = A_arrow_bottom_blocks[
        :, -diag_blocksize:
    ]

    reduced_system_inv = np.linalg.inv(reduced_system)

    X_sdr_diagonal_blocks = np.zeros_like(A_diagonal_blocks)
    X_sdr_lower_diagonal_blocks = np.zeros_like(A_lower_diagonal_blocks)
    X_sdr_upper_diagonal_blocks = np.zeros_like(A_upper_diagonal_blocks)
    X_sdr_arrow_bottom_blocks = np.zeros_like(A_arrow_bottom_blocks)
    X_sdr_arrow_right_blocks = np.zeros_like(A_arrow_right_blocks)
    X_sdr_global_arrow_tip = np.zeros_like(A_arrow_tip_block)

    X_sdr_diagonal_blocks[:, -diag_blocksize:] = reduced_system_inv[
        0:diag_blocksize, 0:diag_blocksize
    ]
    X_sdr_arrow_bottom_blocks[:, -diag_blocksize:] = reduced_system_inv[
        -arrow_blocksize:, 0:diag_blocksize
    ]
    X_sdr_arrow_right_blocks[-diag_blocksize:, :] = reduced_system_inv[
        0:diag_blocksize, -arrow_blocksize:
    ]
    X_sdr_global_arrow_tip = reduced_system_inv[-arrow_blocksize:, -arrow_blocksize:]

    # ----- Selected inversion part -----
    (
        X_sdr_diagonal_blocks,
        X_sdr_lower_diagonal_blocks,
        X_sdr_upper_diagonal_blocks,
        X_sdr_arrow_bottom_blocks,
        X_sdr_arrow_right_blocks,
        X_sdr_global_arrow_tip,
    ) = top_sinv(
        X_sdr_diagonal_blocks,
        X_sdr_lower_diagonal_blocks,
        X_sdr_upper_diagonal_blocks,
        X_sdr_arrow_bottom_blocks,
        X_sdr_arrow_right_blocks,
        X_sdr_global_arrow_tip,
        L_diagonal_blocks,
        L_lower_diagonal_blocks,
        L_arrow_bottom_blocks,
        U_diagonal_blocks,
        U_upper_diagonal_blocks,
        U_arrow_right_blocks,
    )

    assert np.allclose(X_ref_diagonal_blocks, X_sdr_diagonal_blocks)
    assert np.allclose(X_ref_lower_diagonal_blocks, X_sdr_lower_diagonal_blocks)
    assert np.allclose(X_ref_upper_diagonal_blocks, X_sdr_upper_diagonal_blocks)
    assert np.allclose(X_ref_arrow_bottom_blocks, X_sdr_arrow_bottom_blocks)
    assert np.allclose(X_ref_arrow_right_blocks, X_sdr_arrow_right_blocks)
    assert np.allclose(X_ref_arrow_tip_block, X_sdr_global_arrow_tip)


if __name__ == "__main__":
    test_lu_dist_top_process(10, 10, 2)
