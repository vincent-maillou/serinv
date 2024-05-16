# Copyright 2023-2024 ETH Zurich & USI. All rights reserved.

import numpy as np
import pytest

from serinv.sequential.cpu import scddbtaf
from serinv.sequential.cpu import scddbtasi


@pytest.mark.parametrize(
    "diagonal_blocksize, arrowhead_blocksize, n_diag_blocks",
    [(3, 2, 5), (2, 3, 3), (5, 5, 2)],
)
def test_scddbtaf(
    dd_bta,
    bta_dense_to_arrays_factory,
    diagonal_blocksize,
    arrowhead_blocksize,
    n_diag_blocks,
):

    X_ref = np.linalg.inv(dd_bta)

    (
        X_diagonal_blocks_ref,
        X_lower_diagonal_blocks_ref,
        X_upper_diagonal_blocks_ref,
        X_arrow_bottom_blocks_ref,
        X_arrow_right_blocks_ref,
        X_arrow_tip_block_ref,
    ) = bta_dense_to_arrays_factory(
        X_ref, diagonal_blocksize, arrowhead_blocksize, n_diag_blocks
    )

    (
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_upper_diagonal_blocks,
        A_arrow_bottom_blocks,
        A_arrow_right_blocks,
        A_arrow_tip_block,
    ) = bta_dense_to_arrays_factory(
        dd_bta, diagonal_blocksize, arrowhead_blocksize, n_diag_blocks
    )

    (
        L_diagonal_blocks,
        L_lower_diagonal_blocks,
        L_arrow_bottom_blocks,
        L_arrow_tip_block,
        U_diagonal_blocks,
        U_upper_diagonal_blocks,
        U_arrow_right_blocks,
        U_arrow_tip_block,
    ) = scddbtaf(
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_upper_diagonal_blocks,
        A_arrow_bottom_blocks,
        A_arrow_right_blocks,
        A_arrow_tip_block,
    )

    (
        X_diagonal_blocks_serinv,
        X_lower_diagonal_blocks_serinv,
        X_upper_diagonal_blocks_serinv,
        X_arrow_bottom_blocks_serinv,
        X_arrow_right_blocks_serinv,
        X_arrow_tip_block_serinv,
    ) = scddbtasi(
        L_diagonal_blocks,
        L_lower_diagonal_blocks,
        L_arrow_bottom_blocks,
        L_arrow_tip_block,
        U_diagonal_blocks,
        U_upper_diagonal_blocks,
        U_arrow_right_blocks,
        U_arrow_tip_block,
    )

    assert np.allclose(X_diagonal_blocks_ref, X_diagonal_blocks_serinv)
    assert np.allclose(X_lower_diagonal_blocks_ref, X_lower_diagonal_blocks_serinv)
    assert np.allclose(X_upper_diagonal_blocks_ref, X_upper_diagonal_blocks_serinv)
    assert np.allclose(X_arrow_bottom_blocks_ref, X_arrow_bottom_blocks_serinv)
    assert np.allclose(X_arrow_right_blocks_ref, X_arrow_right_blocks_serinv)
    assert np.allclose(X_arrow_tip_block_ref, X_arrow_tip_block_serinv)
