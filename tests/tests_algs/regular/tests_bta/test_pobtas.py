# Copyright 2023-2025 ETH Zurich. All rights reserved.

import numpy as np
import pytest

from serinv import backend_flags, _get_module_from_array
from ....testing_utils import bta_dense_to_arrays, dd_bta, symmetrize, rhs

from serinv.algs import pobtaf, pobtas

if backend_flags["cupy_avail"]:
    import cupyx as cpx


@pytest.mark.mpi_skip()
@pytest.mark.parametrize("n_rhs", [1, 2, 3])
def test_pobtas(
    n_rhs: int,
    diagonal_blocksize: int,
    arrowhead_blocksize: int,
    n_diag_blocks: int,
    array_type: str,
    dtype: np.dtype,
):
    
    
    A = dd_bta(
        diagonal_blocksize,
        arrowhead_blocksize,
        n_diag_blocks,
        device_array=True if array_type == "device" else False,
        dtype=dtype,
    )

    symmetrize(A)

    xp, _ = _get_module_from_array(A)

    B = rhs(
        n_rhs,
        diagonal_blocksize,
        arrowhead_blocksize,
        n_diag_blocks,
        device_array=True if array_type == "device" else False,
        dtype=dtype,
    )

    X_ref = xp.linalg.solve(A.copy(), B.copy())

    (
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        _,
        A_lower_arrow_blocks,
        _,
        A_arrow_tip_block,
    ) = bta_dense_to_arrays(A, diagonal_blocksize, arrowhead_blocksize, n_diag_blocks)

    if backend_flags["cupy_avail"] and array_type == "streaming":
        A_diagonal_blocks_pinned = cpx.zeros_like_pinned(A_diagonal_blocks)
        A_diagonal_blocks_pinned[:, :, :] = A_diagonal_blocks[:, :, :]
        A_lower_diagonal_blocks_pinned = cpx.zeros_like_pinned(A_lower_diagonal_blocks)
        A_lower_diagonal_blocks_pinned[:, :, :] = A_lower_diagonal_blocks[:, :, :]
        A_lower_arrow_blocks_pinned = cpx.zeros_like_pinned(A_lower_arrow_blocks)
        A_lower_arrow_blocks_pinned[:, :, :] = A_lower_arrow_blocks[:, :, :]
        A_arrow_tip_block_pinned = cpx.zeros_like_pinned(A_arrow_tip_block)
        A_arrow_tip_block_pinned[:, :] = A_arrow_tip_block[:, :]
        B_pinned = cpx.zeros_like_pinned(B)
        B_pinned[:, :] = B[:, :]

        A_diagonal_blocks = A_diagonal_blocks_pinned
        A_lower_diagonal_blocks = A_lower_diagonal_blocks_pinned
        A_lower_arrow_blocks = A_lower_arrow_blocks_pinned
        A_arrow_tip_block = A_arrow_tip_block_pinned
        B = B_pinned

    pobtaf(
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_lower_arrow_blocks,
        A_arrow_tip_block,
    )

    # Forward solve: Y=L^{-1}B
    pobtas(
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_lower_arrow_blocks,
        A_arrow_tip_block,
        B,
        trans="N",
        device_streaming=True if array_type == "streaming" else False,
    )

    # Backward solve: X=L^{-T}Y
    pobtas(
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_lower_arrow_blocks,
        A_arrow_tip_block,
        B,
        trans="C",
    )

    assert xp.allclose(B, X_ref)
