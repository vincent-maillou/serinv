# Copyright 2023-2025 ETH Zurich. All rights reserved.

import numpy as np
import pytest

from serinv import _get_module_from_array
from ....testing_utils import bt_dense_to_arrays, dd_bt, symmetrize

from serinv.algs import ddbtsc, ddbtsci


@pytest.mark.mpi_skip()
@pytest.mark.parametrize("type_of_equation", ["AX=I", "AXA.T=B"])
@pytest.mark.parametrize("direction", ["downward", "upward"])
def test_ddbtsci(
    diagonal_blocksize: int,
    n_diag_blocks: int,
    array_type: str,
    dtype: np.dtype,
    type_of_equation: str,
    direction: str,
):
    A = dd_bt(
        diagonal_blocksize,
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
        B = dd_bt(
            diagonal_blocksize,
            n_diag_blocks,
            device_array=True if array_type == "device" else False,
            dtype=dtype,
        )

        symmetrize(B)

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

    (
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_upper_diagonal_blocks,
    ) = bt_dense_to_arrays(A, diagonal_blocksize, n_diag_blocks)

    ddbtsc(
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_upper_diagonal_blocks,
        rhs=rhs,
        quadratic=quadratic,
        direction=direction,
    )

    ddbtsci(
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_upper_diagonal_blocks,
        rhs=rhs,
        quadratic=quadratic,
        direction=direction,
    )

    (
        X_diagonal_blocks_ref,
        X_lower_diagonal_blocks_ref,
        X_upper_diagonal_blocks_ref,
    ) = bt_dense_to_arrays(X_ref, diagonal_blocksize, n_diag_blocks)

    # Check algorithm validity
    assert xp.allclose(X_diagonal_blocks_ref, A_diagonal_blocks)
    assert xp.allclose(X_lower_diagonal_blocks_ref, A_lower_diagonal_blocks)
    assert xp.allclose(X_upper_diagonal_blocks_ref, A_upper_diagonal_blocks)

    if type_of_equation == "AX=B":
        ...
    elif type_of_equation == "AXA.T=B":
        Xl_ref = X_ref @ B @ X_ref.conj().T

        (
            Xl_diagonal_blocks_ref,
            Xl_lower_diagonal_blocks_ref,
            Xl_upper_diagonal_blocks_ref,
        ) = bt_dense_to_arrays(Xl_ref, diagonal_blocksize, n_diag_blocks)

        assert xp.allclose(Xl_diagonal_blocks_ref, B_diagonal_blocks)
        assert xp.allclose(Xl_lower_diagonal_blocks_ref, B_lower_diagonal_blocks)
        assert xp.allclose(Xl_upper_diagonal_blocks_ref, B_upper_diagonal_blocks)
