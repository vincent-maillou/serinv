# Copyright 2023-2024 ETH Zurich. All rights reserved.

try:
    import cupy as cp
    import cupyx as cpx

    CUPY_AVAIL = True

except ImportError:
    CUPY_AVAIL = False

import numpy as np

import pytest

from serinv.algs import pobtasinv


@pytest.mark.mpi_skip()
@pytest.mark.parametrize("diagonal_blocksize", [2, 3])
@pytest.mark.parametrize("arrowhead_blocksize", [2, 3])
@pytest.mark.parametrize("n_diag_blocks", [1, 2, 3, 4])
@pytest.mark.parametrize("device_array", [False, True], ids=["host", "device"])
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("device_streaming", [False, True])
def test_pobtasinv(
    dd_bta,
    bta_symmetrize,
    bta_dense_to_arrays,
    diagonal_blocksize,
    arrowhead_blocksize,
    n_diag_blocks,
    device_array,
    device_streaming,
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

    if CUPY_AVAIL and device_streaming and not device_array:
        A_diagonal_blocks_pinned = cpx.zeros_like_pinned(A_diagonal_blocks)
        A_diagonal_blocks_pinned[:, :, :] = A_diagonal_blocks[:, :, :]
        A_lower_diagonal_blocks_pinned = cpx.zeros_like_pinned(A_lower_diagonal_blocks)
        A_lower_diagonal_blocks_pinned[:, :, :] = A_lower_diagonal_blocks[:, :, :]
        A_arrow_bottom_blocks_pinned = cpx.zeros_like_pinned(A_arrow_bottom_blocks)
        A_arrow_bottom_blocks_pinned[:, :, :] = A_arrow_bottom_blocks[:, :, :]
        A_arrow_tip_block_pinned = cpx.zeros_like_pinned(A_arrow_tip_block)
        A_arrow_tip_block_pinned[:, :] = A_arrow_tip_block[:, :]

        A_diagonal_blocks = A_diagonal_blocks_pinned
        A_lower_diagonal_blocks = A_lower_diagonal_blocks_pinned
        A_arrow_bottom_blocks = A_arrow_bottom_blocks_pinned
        A_arrow_tip_block = A_arrow_tip_block_pinned

    (
        X_diagonal_blocks_serinv,
        X_lower_diagonal_blocks_serinv,
        X_arrow_bottom_blocks_serinv,
        X_arrow_tip_block_serinv,
    ) = pobtasinv(
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_arrow_bottom_blocks,
        A_arrow_tip_block,
        device_streaming,
    )

    # Check algorithm validity
    assert np.allclose(X_diagonal_blocks_ref, X_diagonal_blocks_serinv)
    assert np.allclose(X_lower_diagonal_blocks_ref, X_lower_diagonal_blocks_serinv)
    assert np.allclose(X_arrow_bottom_blocks_ref, X_arrow_bottom_blocks_serinv)
    assert np.allclose(X_arrow_tip_block_ref, X_arrow_tip_block_serinv)

    # Check for in-place operations
    if device_array:
        assert X_diagonal_blocks_serinv.data == A_diagonal_blocks.data
        assert X_lower_diagonal_blocks_serinv.data == A_lower_diagonal_blocks.data
        assert X_arrow_bottom_blocks_serinv.data == A_arrow_bottom_blocks.data
        assert X_arrow_tip_block_serinv.data == A_arrow_tip_block.data
    else:
        assert X_diagonal_blocks_serinv.ctypes.data == A_diagonal_blocks.ctypes.data
        assert (
            X_lower_diagonal_blocks_serinv.ctypes.data
            == A_lower_diagonal_blocks.ctypes.data
        )
        assert (
            X_arrow_bottom_blocks_serinv.ctypes.data
            == A_arrow_bottom_blocks.ctypes.data
        )
        assert X_arrow_tip_block_serinv.ctypes.data == A_arrow_tip_block.ctypes.data
