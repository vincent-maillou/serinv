# Copyright 2023-2025 ETH Zurich. All rights reserved.

import numpy as np
import pytest

from serinv import backend_flags, _get_module_from_array
from ....testing_utils import bt_dense_to_arrays, dd_bt, symmetrize

from serinv.utils import allocate_pobtx_permutation_buffers
from serinv.algs import pobtf, pobtsi

if backend_flags["cupy_avail"]:
    import cupyx as cpx


@pytest.mark.mpi_skip()
def test_pobtsi_permuted(
    diagonal_blocksize: int,
    n_diag_blocks: int,
    array_type: str,
    dtype: np.dtype,
):
    A = dd_bt(
        diagonal_blocksize,
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
    ) = bt_dense_to_arrays(X_ref, diagonal_blocksize, n_diag_blocks)

    (
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        _,
    ) = bt_dense_to_arrays(A, diagonal_blocksize, n_diag_blocks)

    if backend_flags["cupy_avail"] and array_type == "streaming":
        A_diagonal_blocks_pinned = cpx.zeros_like_pinned(A_diagonal_blocks)
        A_diagonal_blocks_pinned[:, :, :] = A_diagonal_blocks[:, :, :]
        A_lower_diagonal_blocks_pinned = cpx.zeros_like_pinned(A_lower_diagonal_blocks)
        A_lower_diagonal_blocks_pinned[:, :, :] = A_lower_diagonal_blocks[:, :, :]

        A_diagonal_blocks = A_diagonal_blocks_pinned
        A_lower_diagonal_blocks = A_lower_diagonal_blocks_pinned

    buffer: dict = allocate_pobtx_permutation_buffers(
        A_diagonal_blocks=A_diagonal_blocks,
    )

    pobtf(
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        buffer=buffer,
    )

    # Check algorithm validity
    # For permuted algorithm we need to make a reduced system
    # of size 2x2, invert it and add the buffer blocks back
    _A_diagonal_blocks = xp.zeros(
        (2, diagonal_blocksize, diagonal_blocksize), dtype=dtype
    )
    _A_lower_diagonal_blocks = xp.zeros(
        (1, diagonal_blocksize, diagonal_blocksize), dtype=dtype
    )

    _A_diagonal_blocks[0] = A_diagonal_blocks[0]
    _A_diagonal_blocks[1] = A_diagonal_blocks[-1]
    _A_lower_diagonal_blocks[0] = buffer[-1].conj().T

    pobtf(
        _A_diagonal_blocks,
        _A_lower_diagonal_blocks,
    )

    pobtsi(
        _A_diagonal_blocks,
        _A_lower_diagonal_blocks,
    )

    # Verify that the reduced system is already correct
    assert xp.allclose(X_diagonal_blocks_ref[0], _A_diagonal_blocks[0])
    assert xp.allclose(X_diagonal_blocks_ref[-1], _A_diagonal_blocks[-1])

    A_diagonal_blocks[0] = _A_diagonal_blocks[0]
    A_diagonal_blocks[-1] = _A_diagonal_blocks[1]
    buffer[-1] = _A_lower_diagonal_blocks[0].conj().T

    pobtsi(
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        buffers=buffer,
    )

    # assert xp.allclose(X_diagonal_blocks_ref, A_diagonal_blocks)
    # assert xp.allclose(X_lower_diagonal_blocks_ref, A_lower_diagonal_blocks)
