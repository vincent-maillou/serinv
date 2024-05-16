# Copyright 2023-2024 ETH Zurich & USI. All rights reserved.

import numpy as np
import scipy.linalg as la
import pytest

from serinv.sequential.cpu import scddbtaf


@pytest.mark.parametrize(
    "diagonal_blocksize, arrowhead_blocksize, n_diag_blocks",
    [(3, 2, 5), (2, 3, 3), (5, 5, 2)],
)
def test_scddbtaf(
    dd_bta,
    bta_dense_to_arrays_factory,
    bta_arrays_to_dense_factory,
    diagonal_blocksize,
    arrowhead_blocksize,
    n_diag_blocks,
):

    P_ref, L_ref, U_ref = la.lu(dd_bta)

    assert np.allclose(P_ref, np.eye(P_ref.shape[0]))

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

    L_serinv = bta_arrays_to_dense_factory(
        L_diagonal_blocks,
        L_lower_diagonal_blocks,
        np.zeros_like(A_upper_diagonal_blocks),
        L_arrow_bottom_blocks,
        np.zeros_like(A_arrow_right_blocks),
        L_arrow_tip_block,
    )

    U_serinv = bta_arrays_to_dense_factory(
        U_diagonal_blocks,
        np.zeros_like(A_lower_diagonal_blocks),
        U_upper_diagonal_blocks,
        np.zeros_like(A_arrow_bottom_blocks),
        U_arrow_right_blocks,
        U_arrow_tip_block,
    )

    assert np.allclose(L_ref, L_serinv)
    assert np.allclose(U_ref, U_serinv)
