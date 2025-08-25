# Copyright 2023-2025 ETH Zurich. All rights reserved.

import numpy as np
import pytest

from serinv import _get_module_from_array
from ....testing_utils import bt_dense_to_arrays, dd_bt

from serinv.algs import ddbtsc
from serinv.utils import allocate_ddbtx_permutation_buffers


@pytest.mark.mpi_skip()
@pytest.mark.parametrize("type_of_equation", ["AX=I", "AXA.T=B"])
def test_ddbtsc_permuted(
    diagonal_blocksize: int,
    n_diag_blocks: int,
    array_type: str,
    dtype: np.dtype,
    type_of_equation: str,
):
    A = dd_bt(
        diagonal_blocksize,
        n_diag_blocks,
        device_array=True if array_type == "device" else False,
        dtype=dtype,
    )

    xp, _ = _get_module_from_array(A)

    (
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_upper_diagonal_blocks,
    ) = bt_dense_to_arrays(A, diagonal_blocksize, n_diag_blocks)

    X_ref = xp.linalg.inv(A.copy())

    if type_of_equation == "AX=I":
        rhs = None
        quadratic = None
    elif type_of_equation == "AX=B":
        ...
    elif type_of_equation == "AXA.T=B":
        B = dd_bt(
            diagonal_blocksize,
            n_diag_blocks,
            device_array=True if array_type == "device" else False,
            dtype=dtype,
        )

        (
            B_diagonal_blocks,
            B_lower_diagonal_blocks,
            B_upper_diagonal_blocks,
        ) = bt_dense_to_arrays(B, diagonal_blocksize, n_diag_blocks)

        rhs = {
            "B_diagonal_blocks": B_diagonal_blocks,
            "B_lower_diagonal_blocks": B_lower_diagonal_blocks,
            "B_upper_diagonal_blocks": B_upper_diagonal_blocks,
        }

        quadratic = True

    buffers: dict = allocate_ddbtx_permutation_buffers(
        A_lower_diagonal_blocks=A_lower_diagonal_blocks,
        quadratic=True if type_of_equation == "AXA.T=B" else False,
    )

    ddbtsc(
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_upper_diagonal_blocks,
        rhs=rhs,
        quadratic=quadratic,
        buffers=buffers,
    )

    (
        X_diagonal_blocks_ref,
        X_lower_diagonal_blocks_ref,
        X_upper_diagonal_blocks_ref,
    ) = bt_dense_to_arrays(X_ref, diagonal_blocksize, n_diag_blocks)

    # Check algorithm validity
    # For permuted algorithm we need to make a reduced system
    # of size 2x2, invert it and add the buffer blocks back
    A_reduced = xp.empty((2 * diagonal_blocksize, 2 * diagonal_blocksize), dtype=dtype)

    # Map blocks to the reduced system
    A_reduced[:diagonal_blocksize, :diagonal_blocksize] = A_diagonal_blocks[0]
    A_reduced[-diagonal_blocksize:, -diagonal_blocksize:] = A_diagonal_blocks[-1]
    A_reduced[:diagonal_blocksize, -diagonal_blocksize:] = buffers[
        "A_lower_buffer_blocks"
    ][-1]
    A_reduced[-diagonal_blocksize:, :diagonal_blocksize] = buffers[
        "A_upper_buffer_blocks"
    ][-1]

    X_reduced = xp.linalg.inv(A_reduced)

    # Map back to the original system
    A_diagonal_blocks[0] = X_reduced[:diagonal_blocksize, :diagonal_blocksize]
    A_diagonal_blocks[-1] = X_reduced[-diagonal_blocksize:, -diagonal_blocksize:]
    buffers["A_upper_buffer_blocks"][-1] = X_reduced[
        -diagonal_blocksize:, :diagonal_blocksize
    ]
    buffers["A_lower_buffer_blocks"][-1] = X_reduced[
        :diagonal_blocksize, -diagonal_blocksize:
    ]

    assert xp.allclose(X_diagonal_blocks_ref[-1], A_diagonal_blocks[-1])
    assert xp.allclose(X_diagonal_blocks_ref[0], A_diagonal_blocks[0])

    if type_of_equation == "AX=B":
        ...
    elif type_of_equation == "AXA.T=B":
        Xl_ref = X_ref @ B @ X_ref.conj().T

        (
            Xl_diagonal_blocks_ref,
            Xl_lower_diagonal_blocks_ref,
            Xl_upper_diagonal_blocks_ref,
        ) = bt_dense_to_arrays(Xl_ref, diagonal_blocksize, n_diag_blocks)

        B_reduced = xp.empty(
            (2 * diagonal_blocksize, 2 * diagonal_blocksize), dtype=dtype
        )

        # Map blocks to the reduced system
        B_reduced[:diagonal_blocksize, :diagonal_blocksize] = B_diagonal_blocks[0]
        B_reduced[-diagonal_blocksize:, -diagonal_blocksize:] = B_diagonal_blocks[-1]
        B_reduced[:diagonal_blocksize, -diagonal_blocksize:] = buffers[
            "B_lower_buffer_blocks"
        ][-1]
        B_reduced[-diagonal_blocksize:, :diagonal_blocksize] = buffers[
            "B_upper_buffer_blocks"
        ][-1]

        Xl_reduced = X_reduced @ B_reduced @ X_reduced.conj().T

        # Map back to the original system
        B_diagonal_blocks[0] = Xl_reduced[:diagonal_blocksize, :diagonal_blocksize]
        B_diagonal_blocks[-1] = Xl_reduced[-diagonal_blocksize:, -diagonal_blocksize:]
        buffers["B_upper_buffer_blocks"][-1] = Xl_reduced[
            -diagonal_blocksize:, :diagonal_blocksize
        ]
        buffers["B_lower_buffer_blocks"][-1] = Xl_reduced[
            :diagonal_blocksize, -diagonal_blocksize:
        ]

        assert xp.allclose(Xl_diagonal_blocks_ref[-1], B_diagonal_blocks[-1])
        assert xp.allclose(Xl_diagonal_blocks_ref[0], B_diagonal_blocks[0])
