# Copyright 2023-2024 ETH Zurich. All rights reserved.

try:
    import cupy as cp

    CUPY_AVAIL = True

except ImportError:
    CUPY_AVAIL = False

import numpy as np
import pytest

from serinv.algs import ddbtaf, ddbtasi


@pytest.mark.mpi_skip()
@pytest.mark.parametrize("diagonal_blocksize", [2, 3])
@pytest.mark.parametrize("arrowhead_blocksize", [2, 3])
@pytest.mark.parametrize("n_diag_blocks", [1, 2, 3, 4])
@pytest.mark.parametrize("device_array", [False, True], ids=["host", "device"])
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_ddbtasi(
    dd_bta,
    bta_dense_to_arrays,
    diagonal_blocksize,
    arrowhead_blocksize,
    n_diag_blocks,
):
    if CUPY_AVAIL:
        xp = cp.get_array_module(dd_bta)
    else:
        xp = np

    X_ref = xp.linalg.inv(dd_bta)

    (
        X_diagonal_blocks_ref,
        X_lower_diagonal_blocks_ref,
        X_upper_diagonal_blocks_ref,
        X_arrow_bottom_blocks_ref,
        X_arrow_right_blocks_ref,
        X_arrow_tip_block_ref,
    ) = bta_dense_to_arrays(
        X_ref, diagonal_blocksize, arrowhead_blocksize, n_diag_blocks
    )

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

    (
        X_diagonal_blocks_serinv,
        X_lower_diagonal_blocks_serinv,
        X_upper_diagonal_blocks_serinv,
        X_arrow_bottom_blocks_serinv,
        X_arrow_right_blocks_serinv,
        X_arrow_tip_block_serinv,
    ) = ddbtasi(
        L_diagonal_blocks,
        L_lower_diagonal_blocks,
        L_arrow_bottom_blocks,
        L_arrow_tip_block,
        U_diagonal_blocks,
        U_upper_diagonal_blocks,
        U_arrow_right_blocks,
        U_arrow_tip_block,
    )

    assert xp.allclose(X_diagonal_blocks_ref, X_diagonal_blocks_serinv)
    assert xp.allclose(X_lower_diagonal_blocks_ref, X_lower_diagonal_blocks_serinv)
    assert xp.allclose(X_upper_diagonal_blocks_ref, X_upper_diagonal_blocks_serinv)
    assert xp.allclose(X_arrow_bottom_blocks_ref, X_arrow_bottom_blocks_serinv)
    assert xp.allclose(X_arrow_right_blocks_ref, X_arrow_right_blocks_serinv)
    assert xp.allclose(X_arrow_tip_block_ref, X_arrow_tip_block_serinv)
