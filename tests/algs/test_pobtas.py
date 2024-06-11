# Copyright 2023-2024 ETH Zurich. All rights reserved.

try:
    import cupy as cp

    CUPY_AVAIL = True

except:
    CUPY_AVAIL = False

import numpy as np

import pytest

from serinv.algs import pobtaf, pobtas


@pytest.mark.parametrize("diagonal_blocksize", [2, 3])
@pytest.mark.parametrize("arrowhead_blocksize", [2, 3])
@pytest.mark.parametrize("n_diag_blocks", [1, 2, 3])
@pytest.mark.parametrize("device_array", [False, True])
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("n_rhs", [1, 2, 3])
def test_pobtas(
    dd_bta,
    b_rhs,
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
    B = b_rhs

    X_ref = xp.linalg.solve(A, B)

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

    X_serinv = pobtas(
        L_diagonal_blocks,
        L_lower_diagonal_blocks,
        L_arrow_bottom_blocks,
        L_arrow_tip_block,
        B,
    )

    assert xp.allclose(X_serinv, X_ref)
