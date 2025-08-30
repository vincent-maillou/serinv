# Copyright 2023-2025 ETH Zurich. All rights reserved.

import numpy as np
import pytest

from serinv import _get_module_from_array
from ....testing_utils import bta_dense_to_arrays, dd_bta, symmetrize

from serinv.algs import ddbtasc, ddbtasci


@pytest.mark.mpi_skip()
@pytest.mark.parametrize("type_of_equation", ["AX=I", "AXA.T=B"])
def test_ddbtasci_permuted(
    diagonal_blocksize: int,
    arrowhead_blocksize: int,
    n_diag_blocks: int,
    array_type: str,
    dtype: np.dtype,
    type_of_equation: str,
):
    A = dd_bta(
        diagonal_blocksize,
        arrowhead_blocksize,
        n_diag_blocks,
        device_array=True if array_type == "device" else False,
        dtype=dtype,
    )

    xp, _ = _get_module_from_array(A)

    (
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_upper_diagonal_blocks,
        A_lower_arrow_blocks,
        A_upper_arrow_blocks,
        A_arrow_tip_block,
    ) = bta_dense_to_arrays(A, diagonal_blocksize, arrowhead_blocksize, n_diag_blocks)

    A_lower_buffer_blocks = xp.zeros_like(A_lower_diagonal_blocks)
    A_upper_buffer_blocks = xp.zeros_like(A_upper_diagonal_blocks)
    buffers: dict = {
        "A_lower_buffer_blocks": A_lower_buffer_blocks,
        "A_upper_buffer_blocks": A_upper_buffer_blocks,
    }

    X_ref = xp.linalg.inv(A.copy())

    if type_of_equation == "AX=I":
        rhs = None
        quadratic = None
    elif type_of_equation == "AX=B":
        ...
    elif type_of_equation == "AXA.T=B":
        B = dd_bta(
            diagonal_blocksize,
            arrowhead_blocksize,
            n_diag_blocks,
            device_array=True if array_type == "device" else False,
            dtype=dtype,
        )

        symmetrize(B)

        (
            B_diagonal_blocks,
            B_lower_diagonal_blocks,
            B_upper_diagonal_blocks,
            B_lower_arrow_blocks,
            B_upper_arrow_blocks,
            B_arrow_tip_block,
        ) = bta_dense_to_arrays(
            B, diagonal_blocksize, arrowhead_blocksize, n_diag_blocks
        )

        B_lower_buffer_blocks = xp.zeros_like(B_lower_diagonal_blocks)
        B_upper_buffer_blocks = xp.zeros_like(B_upper_diagonal_blocks)
        buffers["B_lower_buffer_blocks"] = B_lower_buffer_blocks
        buffers["B_upper_buffer_blocks"] = B_upper_buffer_blocks

        rhs = {
            "B_diagonal_blocks": B_diagonal_blocks,
            "B_lower_diagonal_blocks": B_lower_diagonal_blocks,
            "B_upper_diagonal_blocks": B_upper_diagonal_blocks,
            "B_lower_arrow_blocks": B_lower_arrow_blocks,
            "B_upper_arrow_blocks": B_upper_arrow_blocks,
            "B_arrow_tip_block": B_arrow_tip_block,
        }

        quadratic = True

    ddbtasc(
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_upper_diagonal_blocks,
        A_lower_arrow_blocks,
        A_upper_arrow_blocks,
        A_arrow_tip_block,
        rhs=rhs,
        quadratic=quadratic,
        buffers=buffers,
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

    # Upper diagonal block
    A_reduced[:diagonal_blocksize, diagonal_blocksize : 2 * diagonal_blocksize] = (
        A_lower_buffer_blocks[-1]
    )
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

    if type_of_equation == "AX=B":
        ...
    elif type_of_equation == "AXA.T=B":
        Xl_ref = X_ref @ B @ X_ref.conj().T

        (
            Xl_diagonal_blocks_ref,
            Xl_lower_diagonal_blocks_ref,
            Xl_upper_diagonal_blocks_ref,
            Xl_lower_arrow_blocks_ref,
            Xl_upper_arrow_blocks_ref,
            Xl_arrow_tip_block_ref,
        ) = bta_dense_to_arrays(
            Xl_ref, diagonal_blocksize, arrowhead_blocksize, n_diag_blocks
        )

        B_reduced = xp.empty(
            (
                2 * diagonal_blocksize + arrowhead_blocksize,
                2 * diagonal_blocksize + arrowhead_blocksize,
            ),
            dtype=dtype,
        )

        # Map blocks to the reduced system
        # 1st diagonal block
        B_reduced[:diagonal_blocksize, :diagonal_blocksize] = B_diagonal_blocks[0]
        # Last diagonal block
        B_reduced[
            diagonal_blocksize : 2 * diagonal_blocksize,
            diagonal_blocksize : 2 * diagonal_blocksize,
        ] = B_diagonal_blocks[-1]
        # Arrow tip block
        B_reduced[-arrowhead_blocksize:, -arrowhead_blocksize:] = B_arrow_tip_block

        # Upper diagonal block
        B_reduced[:diagonal_blocksize, diagonal_blocksize : 2 * diagonal_blocksize] = (
            B_lower_buffer_blocks[-1]
        )
        # Lower diagonal block
        B_reduced[diagonal_blocksize : 2 * diagonal_blocksize, :diagonal_blocksize] = (
            B_upper_buffer_blocks[-1]
        )

        # Upper arrow blocks
        B_reduced[:diagonal_blocksize, -arrowhead_blocksize:] = B_upper_arrow_blocks[0]
        B_reduced[
            diagonal_blocksize : 2 * diagonal_blocksize, -arrowhead_blocksize:
        ] = B_upper_arrow_blocks[-1]

        # Lower arrow blocks
        B_reduced[-arrowhead_blocksize:, :diagonal_blocksize] = B_lower_arrow_blocks[0]
        B_reduced[
            -arrowhead_blocksize:, diagonal_blocksize : 2 * diagonal_blocksize
        ] = B_lower_arrow_blocks[-1]

        Xl_reduced = X_reduced @ B_reduced @ X_reduced.conj().T

        # Map back to the original system
        # 1st diagonal block
        B_diagonal_blocks[0] = Xl_reduced[:diagonal_blocksize, :diagonal_blocksize]
        # Last diagonal block
        B_diagonal_blocks[-1] = Xl_reduced[
            diagonal_blocksize : 2 * diagonal_blocksize,
            diagonal_blocksize : 2 * diagonal_blocksize,
        ]
        # Arrow tip block
        B_arrow_tip_block[:, :] = Xl_reduced[
            -arrowhead_blocksize:, -arrowhead_blocksize:
        ]

        # Upper diagonal block
        B_lower_buffer_blocks[-1] = Xl_reduced[
            :diagonal_blocksize, diagonal_blocksize : 2 * diagonal_blocksize
        ]
        # Lower diagonal block
        B_upper_buffer_blocks[-1] = Xl_reduced[
            diagonal_blocksize : 2 * diagonal_blocksize, :diagonal_blocksize
        ]

        # Upper arrow blocks
        B_upper_arrow_blocks[0] = Xl_reduced[:diagonal_blocksize, -arrowhead_blocksize:]
        B_upper_arrow_blocks[-1] = Xl_reduced[
            diagonal_blocksize : 2 * diagonal_blocksize, -arrowhead_blocksize:
        ]

        # Lower arrow blocks
        B_lower_arrow_blocks[0] = Xl_reduced[-arrowhead_blocksize:, :diagonal_blocksize]
        B_lower_arrow_blocks[-1] = Xl_reduced[
            -arrowhead_blocksize:, diagonal_blocksize : 2 * diagonal_blocksize
        ]

    ddbtasci(
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_upper_diagonal_blocks,
        A_lower_arrow_blocks,
        A_upper_arrow_blocks,
        A_arrow_tip_block,
        rhs=rhs,
        quadratic=quadratic,
        buffers=buffers,
    )

    assert xp.allclose(X_diagonal_blocks_ref, A_diagonal_blocks)
    assert xp.allclose(X_lower_diagonal_blocks_ref, A_lower_diagonal_blocks)
    assert xp.allclose(X_upper_diagonal_blocks_ref, A_upper_diagonal_blocks)
    assert xp.allclose(X_lower_arrow_blocks_ref, A_lower_arrow_blocks)
    assert xp.allclose(X_upper_arrow_blocks_ref, A_upper_arrow_blocks)
    assert xp.allclose(X_arrow_tip_block_ref, A_arrow_tip_block)

    if type_of_equation == "AX=B":
        ...
    elif type_of_equation == "AXA.T=B":
        assert xp.allclose(Xl_diagonal_blocks_ref, B_diagonal_blocks)
        assert xp.allclose(Xl_lower_diagonal_blocks_ref, B_lower_diagonal_blocks)
        assert xp.allclose(Xl_upper_diagonal_blocks_ref, B_upper_diagonal_blocks)
        assert xp.allclose(Xl_lower_arrow_blocks_ref, B_lower_arrow_blocks)
        assert xp.allclose(Xl_upper_arrow_blocks_ref, B_upper_arrow_blocks)
        assert xp.allclose(Xl_arrow_tip_block_ref, B_arrow_tip_block)
