# Copyright 2023-2025 ETH Zurich. All rights reserved.

import numpy as np
import pytest

from ....conftest import ARRAY_TYPE as ARRAY_TYPE

from serinv import backend_flags, _get_module_from_array
from ....testing_utils import bta_dense_to_arrays, dd_bta, symmetrize

from serinv.utils import allocate_pobtax_permutation_buffers
from serinv.algs import pobtaf, pobtasi

if backend_flags["cupy_avail"]:
    ARRAY_TYPE.extend(
        [
            pytest.param("streaming", id="streaming"),
        ]
    )

if backend_flags["cupy_avail"]:
    import cupyx as cpx

@pytest.fixture(params=ARRAY_TYPE, autouse=True)
def array_type(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.mark.mpi_skip()
def test_pobtasi_permuted(
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
        X_lower_arrow_blocks_ref,
        _,
        X_arrow_tip_block_ref,
    ) = bta_dense_to_arrays(
        X_ref, diagonal_blocksize, arrowhead_blocksize, n_diag_blocks
    )

    (
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        _,
        A_lower_arrow_blocks,
        _,
        A_arrow_tip_block,
    ) = bta_dense_to_arrays(
        A.copy(), diagonal_blocksize, arrowhead_blocksize, n_diag_blocks
    )

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

    # Allocate permutation buffer
    buffer = allocate_pobtax_permutation_buffers(
        A_diagonal_blocks=A_diagonal_blocks,
        device_streaming=True if array_type == "streaming" else False,
    )

    pobtaf(
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_lower_arrow_blocks,
        A_arrow_tip_block,
        buffer=buffer,
        device_streaming=True if array_type == "streaming" else False,
    )

    # Check algorithm validity
    # For permuted algorithm we need to make a reduced system,
    # invert it and add the buffer blocks back
    _A_diagonal_blocks = xp.zeros(
        (2, diagonal_blocksize, diagonal_blocksize), dtype=dtype
    )
    _A_lower_diagonal_blocks = xp.zeros(
        (1, diagonal_blocksize, diagonal_blocksize), dtype=dtype
    )
    _A_lower_arrow_blocks = xp.zeros(
        (2, arrowhead_blocksize, diagonal_blocksize), dtype=dtype
    )
    _A_arrow_tip_block = xp.zeros(
        (arrowhead_blocksize, arrowhead_blocksize), dtype=dtype
    )

    _A_diagonal_blocks[0] = A_diagonal_blocks[0]
    _A_diagonal_blocks[1] = A_diagonal_blocks[-1]
    _A_lower_diagonal_blocks[0] = buffer[-1].conj().T
    _A_lower_arrow_blocks[0] = A_lower_arrow_blocks[0]
    _A_lower_arrow_blocks[1] = A_lower_arrow_blocks[-1]
    _A_arrow_tip_block[:, :] = A_arrow_tip_block

    pobtaf(
        _A_diagonal_blocks,
        _A_lower_diagonal_blocks,
        _A_lower_arrow_blocks,
        _A_arrow_tip_block,
    )

    pobtasi(
        _A_diagonal_blocks,
        _A_lower_diagonal_blocks,
        _A_lower_arrow_blocks,
        _A_arrow_tip_block,
    )

    print(X_diagonal_blocks_ref)
    print(_A_diagonal_blocks)
    # Verify that the reduced system is already correct
    assert xp.allclose(X_arrow_tip_block_ref, _A_arrow_tip_block)
    assert xp.allclose(X_diagonal_blocks_ref[0], _A_diagonal_blocks[0])
    assert xp.allclose(X_diagonal_blocks_ref[-1], _A_diagonal_blocks[-1])
    assert xp.allclose(X_lower_arrow_blocks_ref[0], _A_lower_arrow_blocks[0])
    assert xp.allclose(X_lower_arrow_blocks_ref[-1], _A_lower_arrow_blocks[-1])

    # Map back the correct reduced system to the original system
    A_diagonal_blocks[0] = _A_diagonal_blocks[0]
    A_diagonal_blocks[-1] = _A_diagonal_blocks[1]
    buffer[-1] = _A_lower_diagonal_blocks[0].conj().T
    A_lower_arrow_blocks[0] = _A_lower_arrow_blocks[0]
    A_lower_arrow_blocks[-1] = _A_lower_arrow_blocks[1]
    A_arrow_tip_block[:, :] = _A_arrow_tip_block

    pobtasi(
        L_diagonal_blocks=A_diagonal_blocks,
        L_lower_diagonal_blocks=A_lower_diagonal_blocks,
        L_lower_arrow_blocks=A_lower_arrow_blocks,
        L_arrow_tip_block=A_arrow_tip_block,
        buffer=buffer,
    )

    assert xp.allclose(X_arrow_tip_block_ref, A_arrow_tip_block)
    assert xp.allclose(X_diagonal_blocks_ref, A_diagonal_blocks)
    assert xp.allclose(X_lower_diagonal_blocks_ref, A_lower_diagonal_blocks)
    assert xp.allclose(X_lower_arrow_blocks_ref, A_lower_arrow_blocks)
