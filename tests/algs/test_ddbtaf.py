# Copyright 2023-2024 ETH Zurich. All rights reserved.

try:
    import cupy as cp
    import cupyx.scipy.linalg as cu_la

    CUPY_AVAIL = True

except ImportError:
    CUPY_AVAIL = False

import numpy as np
import pytest
import scipy.linalg as np_la

from serinv.algs import ddbtaf


@pytest.mark.mpi_skip()
@pytest.mark.parametrize("diagonal_blocksize", [2, 3])
@pytest.mark.parametrize("arrowhead_blocksize", [2, 3])
@pytest.mark.parametrize("n_diag_blocks", [1, 2, 3, 4])
@pytest.mark.parametrize("device_array", [False, True], ids=["host", "device"])
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
        LU_diagonal_blocks,
        LU_lower_diagonal_blocks,
        LU_upper_diagonal_blocks,
        LU_arrow_bottom_blocks,
        LU_arrow_right_blocks,
        LU_arrow_tip_block,
        P_diag,
        P_tip,
    ) = ddbtaf(
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_upper_diagonal_blocks,
        A_arrow_bottom_blocks,
        A_arrow_right_blocks,
        A_arrow_tip_block,
    )

    LU_serinv = bta_arrays_to_dense(
        LU_diagonal_blocks,
        LU_lower_diagonal_blocks,
        LU_upper_diagonal_blocks,
        LU_arrow_bottom_blocks,
        LU_arrow_right_blocks,
        LU_arrow_tip_block,
    )

    assert xp.allclose(L_ref, xp.tril(LU_serinv, -1) + xp.eye(LU_serinv.shape[0]))
    assert xp.allclose(U_ref, xp.triu(LU_serinv))
