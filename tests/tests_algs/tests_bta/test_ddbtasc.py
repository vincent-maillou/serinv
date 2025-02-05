# Copyright 2023-2025 ETH Zurich. All rights reserved.

import numpy as np
import pytest

from serinv import backend_flags, _get_module_from_array
from ...testing_utils import bta_dense_to_arrays, dd_bta, symmetrize

from serinv.algs import ddbtasc

# @pytest.mark.parametrize("type_of_equation", ["AX=I", "AX=B", "AXA.T=B"])

@pytest.mark.mpi_skip()
@pytest.mark.parametrize("type_of_equation", ["AX=I", "AXA.T=B"])
def test_ddbtasc(
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
        ) = bta_dense_to_arrays(B, diagonal_blocksize, arrowhead_blocksize, n_diag_blocks)
        
        rhs = {
            "B_diagonal_blocks": B_diagonal_blocks,
            "B_lower_diagonal_blocks": B_lower_diagonal_blocks,
            "B_upper_diagonal_blocks": B_upper_diagonal_blocks,
            "B_lower_arrow_blocks": B_lower_arrow_blocks,
            "B_upper_arrow_blocks": B_upper_arrow_blocks,
            "B_arrow_tip_block": B_arrow_tip_block,
        }
        quadratic = True

    (
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_upper_diagonal_blocks,
        A_lower_arrow_blocks,
        A_upper_arrow_blocks,
        A_arrow_tip_block,
    ) = bta_dense_to_arrays(A, diagonal_blocksize, arrowhead_blocksize, n_diag_blocks)

    ddbtasc(
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_upper_diagonal_blocks,
        A_lower_arrow_blocks,
        A_upper_arrow_blocks,
        A_arrow_tip_block,
        rhs=rhs,
        quadratic=quadratic,
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
    assert xp.allclose(X_arrow_tip_block_ref, A_arrow_tip_block)

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
        ) = bta_dense_to_arrays(Xl_ref, diagonal_blocksize, arrowhead_blocksize, n_diag_blocks)

        assert xp.allclose(Xl_arrow_tip_block_ref, B_arrow_tip_block)