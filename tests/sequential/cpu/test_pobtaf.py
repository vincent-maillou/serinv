# Copyright 2023-2024 ETH Zurich. All rights reserved.

try:
    import cupy as cp

    CUPY_AVAIL = True

except:
    CUPY_AVAIL = False

import numpy as np

import pytest

from serinv.sequential import pobtaf


@pytest.mark.parametrize("diagonal_blocksize", [2, 3])
@pytest.mark.parametrize("arrowhead_blocksize", [2, 3])
@pytest.mark.parametrize("n_diag_blocks", [1, 2, 3])
@pytest.mark.parametrize("device_array", [False, True])
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_pobtaf(
    dd_bta,
    bta_dense_to_arrays,
    bta_arrays_to_dense,
    bta_symmetrize,
    diagonal_blocksize,
    arrowhead_blocksize,
    n_diag_blocks,
):
    if CUPY_AVAIL:
        xp = cp.get_array_module(dd_bta)
    else:
        xp = np

    L_ref = np.linalg.cholesky(bta_symmetrize(dd_bta))

    (
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_upper_diagonal_blocks,
        A_arrow_bottom_blocks,
        A_arrow_right_blocks,
        A_arrow_tip_block,
    ) = bta_dense_to_arrays(
        bta_symmetrize(dd_bta), diagonal_blocksize, arrowhead_blocksize, n_diag_blocks
    )

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

    L_serinv = bta_arrays_to_dense(
        L_diagonal_blocks,
        L_lower_diagonal_blocks,
        np.zeros_like(A_upper_diagonal_blocks),
        L_arrow_bottom_blocks,
        np.zeros_like(A_arrow_right_blocks),
        L_arrow_tip_block,
    )

    assert np.allclose(L_ref, L_serinv)
