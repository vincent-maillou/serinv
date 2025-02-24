# Copyright 2023-2025 ETH Zurich. All rights reserved.

import numpy as np
import pytest

from serinv import _get_module_from_array
from ....testing_utils import bta_dense_to_arrays, dd_bta, symmetrize, rhs

from serinv.algs import pobtaf, pobtas


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

    pobtaf(
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_lower_arrow_blocks,
        A_arrow_tip_block,
    )

    pobtas(
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_lower_arrow_blocks,
        A_arrow_tip_block,
        B,
    )

    assert xp.allclose(B, X_ref)
