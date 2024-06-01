# Copyright 2023-2024 ETH Zurich. All rights reserved.

try:
    import cupy as cp
    import cupyx.scipy.linalg as cu_la

    CUPY_AVAIL = True

except:
    CUPY_AVAIL = False

import numpy as np
import scipy.linalg as np_la

import pytest

from serinv.sequential import ddbtaf


@pytest.mark.parametrize("diagonal_blocksize", [2, 3])
@pytest.mark.parametrize("arrowhead_blocksize", [2, 3])
@pytest.mark.parametrize("n_diag_blocks", [1, 2, 3])
@pytest.mark.parametrize("device_array", [False, True])
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_ddbtaf(
    dd_bta,
    bta_dense_to_arrays,
    bta_arrays_to_dense,
    diagonal_blocksize,
    arrowhead_blocksize,
    n_diag_blocks,
):
    la = np_la
    if CUPY_AVAIL:
        xp = cp.get_array_module(dd_bta)
        if xp == cp:
            la = cu_la
    else:
        xp = np

    P_ref, L_ref, U_ref = la.lu(dd_bta)

    assert xp.allclose(P_ref, xp.eye(P_ref.shape[0]))

    (
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_upper_diagonal_blocks,
        A_arrow_bottom_blocks,
        A_arrow_right_blocks,
        A_arrow_tip_block,
    ) = bta_dense_to_arrays(
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
    ) = ddbtaf(
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_upper_diagonal_blocks,
        A_arrow_bottom_blocks,
        A_arrow_right_blocks,
        A_arrow_tip_block,
    )

    L_serinv = bta_arrays_to_dense(
        L_diagonal_blocks,
        L_lower_diagonal_blocks,
        xp.zeros_like(A_upper_diagonal_blocks),
        L_arrow_bottom_blocks,
        xp.zeros_like(A_arrow_right_blocks),
        L_arrow_tip_block,
    )

    U_serinv = bta_arrays_to_dense(
        U_diagonal_blocks,
        xp.zeros_like(A_lower_diagonal_blocks),
        U_upper_diagonal_blocks,
        xp.zeros_like(A_arrow_bottom_blocks),
        U_arrow_right_blocks,
        U_arrow_tip_block,
    )

    assert xp.allclose(L_ref, L_serinv)
    assert xp.allclose(U_ref, U_serinv)
