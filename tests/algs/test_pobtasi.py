# Copyright 2023-2024 ETH Zurich. All rights reserved.

try:
    import cupy as cp

    CUPY_AVAIL = True

except:
    CUPY_AVAIL = False

import numpy as np

import pytest

from serinv.algs import pobtaf, pobtasi


@pytest.mark.parametrize("diagonal_blocksize", [2, 3])
@pytest.mark.parametrize("arrowhead_blocksize", [2, 3])
@pytest.mark.parametrize("n_diag_blocks", [1, 2, 3])
@pytest.mark.parametrize("device_array", [False, True])
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_pobtasi(
    dd_bta,
    bta_dense_to_arrays,
    bta_symmetrize,
    diagonal_blocksize,
    arrowhead_blocksize,
    n_diag_blocks,
):
    if CUPY_AVAIL:
        xp = cp.get_array_module(dd_bta)
    else:
        xp = np

    A = bta_symmetrize(dd_bta)

    X_ref = xp.linalg.inv(A)

    (
        X_diagonal_blocks_ref,
        X_lower_diagonal_blocks_ref,
        _,
        X_arrow_bottom_blocks_ref,
        _,
        X_arrow_tip_block_ref,
    ) = bta_dense_to_arrays(
        X_ref, diagonal_blocksize, arrowhead_blocksize, n_diag_blocks
    )

    (
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        _,
        A_arrow_bottom_blocks,
        _,
        A_arrow_tip_block,
    ) = bta_dense_to_arrays(A, diagonal_blocksize, arrowhead_blocksize, n_diag_blocks)

    (
        L_diagonal_blocks,
        L_lower_diagonal_blocks,
        L_arrow_bottom_blocks,
        L_arrow_tip_block,
    ) = pobtaf(
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_arrow_bottom_blocks,
        A_arrow_tip_block,
    )

    (
        X_diagonal_blocks_serinv,
        X_lower_diagonal_blocks_serinv,
        X_arrow_bottom_blocks_serinv,
        X_arrow_tip_block_serinv,
    ) = pobtasi(
        L_diagonal_blocks,
        L_lower_diagonal_blocks,
        L_arrow_bottom_blocks,
        L_arrow_tip_block,
    )

    assert xp.allclose(X_diagonal_blocks_ref, X_diagonal_blocks_serinv)
    assert xp.allclose(X_lower_diagonal_blocks_ref, X_lower_diagonal_blocks_serinv)
    assert xp.allclose(X_arrow_bottom_blocks_ref, X_arrow_bottom_blocks_serinv)
    assert xp.allclose(X_arrow_tip_block_ref, X_arrow_tip_block_serinv)