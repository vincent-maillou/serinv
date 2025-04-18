# Copyright 2023-2025 ETH Zurich. All rights reserved.

import numpy as np
import pytest

from serinv import backend_flags, _get_module_from_array
from ....testing_utils import bt_dense_to_arrays, dd_bt, symmetrize

from serinv.algs import pobtf, pobtsi

if backend_flags["cupy_avail"]:
    import cupyx as cpx


@pytest.mark.mpi_skip()
def test_pobtsi(
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

    pobtf(
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        device_streaming=True if array_type == "streaming" else False,
    )

    pobtsi(
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        device_streaming=True if array_type == "streaming" else False,
    )

    assert xp.allclose(X_diagonal_blocks_ref, A_diagonal_blocks)
    assert xp.allclose(X_lower_diagonal_blocks_ref, A_lower_diagonal_blocks)
