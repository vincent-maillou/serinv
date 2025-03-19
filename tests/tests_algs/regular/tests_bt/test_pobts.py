# Copyright 2023-2025 ETH Zurich. All rights reserved.

import numpy as np
import pytest

from serinv import _get_module_from_array
from ....testing_utils import bt_dense_to_arrays, dd_bt, symmetrize, rhs

from serinv.algs import pobtf, pobts


@pytest.mark.mpi_skip()
@pytest.mark.parametrize("n_rhs", [1, 2, 3])
def test_pobts(
    n_rhs: int,
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

    arrowhead_blocksize = 0
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
    ) = bt_dense_to_arrays(A, diagonal_blocksize, n_diag_blocks)

    pobtf(
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
    )

    # Forward solve: Y=L^{-1}B
    pobts(
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        B,
        trans="N",
    )

    # Backward solve: X=L^{-T}Y
    pobts(
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        B,
        trans="C",
    )

    assert xp.allclose(B, X_ref)
