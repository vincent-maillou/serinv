# Copyright 2023-2025 ETH Zurich. All rights reserved.

import numpy as np
import pytest

from serinv import backend_flags, _get_module_from_array
from ....testing_utils import bta_dense_to_arrays, dd_bta, symmetrize

from serinv.utils import allocate_pobtax_permutation_buffers
from serinv.algs import pobtaf, pobtasi

if backend_flags["cupy_avail"]:
    import cupyx as cpx

@pytest.mark.mpi_skip()
@pytest.mark.parametrize("type_of_equation", ["AX=I", "AXA.T=B"])
def test_ddbtasc_permuted(
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

    X_ref = xp.linalg.inv(A.copy())

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

    if backend_flags["cupy_avail"] and array_type == "streaming":
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

    # Allocate permutation buffer
    buffer = allocate_pobtax_permutation_buffers(
        A_diagonal_blocks=A_diagonal_blocks,
        device_streaming=True if array_type == "streaming" else False,
    )

    pobtaf(
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_arrow_bottom_blocks,
        A_arrow_tip_block,
        buffer=buffer,
        device_streaming=True if array_type == "streaming" else False,
    )


    (
        X_diagonal_blocks_ref,
        X_lower_diagonal_blocks_ref,
        X_upper_diagonal_blocks_ref,
        X_lower_arrow_blocks_ref,
        X_upper_arrow_blocks_ref,
        X_arrow_tip_block_ref,
    ) = bta_dense_to_arrays(
        X_ref, diagonal_blocksize, arrowhead_blocksize, n_diag_blocks
    )

    # Check algorithm validity
    # For permuted algorithm we need to make a reduced system,
    # invert it and add the buffer blocks back
    A_reduced = xp.empty(
        (
            2 * diagonal_blocksize + arrowhead_blocksize,
            2 * diagonal_blocksize + arrowhead_blocksize,
        ),
        dtype=dtype,
    )

    # Map blocks to the reduced system
    # 1st diagonal block
    A_reduced[:diagonal_blocksize, :diagonal_blocksize] = A_diagonal_blocks[0]
    # Last diagonal block
    A_reduced[
        diagonal_blocksize : 2 * diagonal_blocksize,
        diagonal_blocksize : 2 * diagonal_blocksize,
    ] = A_diagonal_blocks[-1]
    # Arrow tip block
    A_reduced[-arrowhead_blocksize:, -arrowhead_blocksize:] = A_arrow_tip_block

    # Lower diagonal block
    A_reduced[diagonal_blocksize : 2 * diagonal_blocksize, :diagonal_blocksize] = (
        A_upper_buffer_blocks[-1]
    )

    # Upper arrow blocks
    A_reduced[:diagonal_blocksize, -arrowhead_blocksize:] = A_upper_arrow_blocks[0]
    A_reduced[diagonal_blocksize : 2 * diagonal_blocksize, -arrowhead_blocksize:] = (
        A_upper_arrow_blocks[-1]
    )

    # Lower arrow blocks
    A_reduced[-arrowhead_blocksize:, :diagonal_blocksize] = A_lower_arrow_blocks[0]
    A_reduced[-arrowhead_blocksize:, diagonal_blocksize : 2 * diagonal_blocksize] = (
        A_lower_arrow_blocks[-1]
    )

    X_reduced = xp.linalg.inv(A_reduced)

    # Map back to the original system
    # 1st diagonal block
    A_diagonal_blocks[0] = X_reduced[:diagonal_blocksize, :diagonal_blocksize]
    # Last diagonal block
    A_diagonal_blocks[-1] = X_reduced[
        diagonal_blocksize : 2 * diagonal_blocksize,
        diagonal_blocksize : 2 * diagonal_blocksize,
    ]
    # Arrow tip block
    A_arrow_tip_block[:, :] = X_reduced[-arrowhead_blocksize:, -arrowhead_blocksize:]

    # Upper diagonal block
    A_lower_buffer_blocks[-1] = X_reduced[
        :diagonal_blocksize, diagonal_blocksize : 2 * diagonal_blocksize
    ]
    # Lower diagonal block
    A_upper_buffer_blocks[-1] = X_reduced[
        diagonal_blocksize : 2 * diagonal_blocksize, :diagonal_blocksize
    ]

    # Upper arrow blocks
    A_upper_arrow_blocks[0] = X_reduced[:diagonal_blocksize, -arrowhead_blocksize:]
    A_upper_arrow_blocks[-1] = X_reduced[
        diagonal_blocksize : 2 * diagonal_blocksize, -arrowhead_blocksize:
    ]

    # Lower arrow blocks
    A_lower_arrow_blocks[0] = X_reduced[-arrowhead_blocksize:, :diagonal_blocksize]
    A_lower_arrow_blocks[-1] = X_reduced[
        -arrowhead_blocksize:, diagonal_blocksize : 2 * diagonal_blocksize
    ]

    pobtasi(
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_arrow_tip_block,
        buffers=buffer,
    )

    assert xp.allclose(X_diagonal_blocks_ref, A_diagonal_blocks)
    assert xp.allclose(X_lower_diagonal_blocks_ref, A_lower_diagonal_blocks)
    assert xp.allclose(X_arrow_tip_block_ref, A_arrow_tip_block)
