# Copyright 2023-2025 ETH Zurich. All rights reserved.

import numpy as np
import pytest

from ....conftest import ARRAY_TYPE as ARRAY_TYPE

from serinv import backend_flags, _get_module_from_array
from ....testing_utils import bta_dense_to_arrays, dd_bta, symmetrize

from serinv.algs import pobtaf

if backend_flags["cupy_avail"]:
    ARRAY_TYPE.extend(
        [
            pytest.param("streaming", id="streaming"),
        ]
    )

if backend_flags["cupy_avail"]:
    import cupyx as cpx


@pytest.mark.mpi_skip()
def test_pobtaf(
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

    L_ref = xp.linalg.cholesky(A.copy())

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

        A_diagonal_blocks = A_diagonal_blocks_pinned
        A_lower_diagonal_blocks = A_lower_diagonal_blocks_pinned
        A_lower_arrow_blocks = A_lower_arrow_blocks_pinned
        A_arrow_tip_block = A_arrow_tip_block_pinned
        raise ValueError("Streaming")

    pobtaf(
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_lower_arrow_blocks,
        A_arrow_tip_block,
        device_streaming=True if array_type == "streaming" else False,
    )

    (
        L_diagonal_blocks_ref,
        L_lower_diagonal_blocks_ref,
        _,
        L_lower_arrow_blocks_ref,
        _,
        L_arrow_tip_block_ref,
    ) = bta_dense_to_arrays(
        L_ref, diagonal_blocksize, arrowhead_blocksize, n_diag_blocks
    )

    # Check algorithm validity
    assert xp.allclose(L_diagonal_blocks_ref, A_diagonal_blocks)
    assert xp.allclose(L_lower_diagonal_blocks_ref, A_lower_diagonal_blocks)
    assert xp.allclose(L_lower_arrow_blocks_ref, A_lower_arrow_blocks)
    assert xp.allclose(L_arrow_tip_block_ref, A_arrow_tip_block)
