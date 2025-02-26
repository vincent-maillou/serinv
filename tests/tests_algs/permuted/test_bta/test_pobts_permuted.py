# Copyright 2023-2025 ETH Zurich. All rights reserved.

import numpy as np
import pytest

from serinv import _get_module_from_array
from ....testing_utils import bt_dense_to_arrays, dd_bt, symmetrize, rhs

from serinv.utils import allocate_pobtx_permutation_buffers
from serinv.algs import pobtf, pobts


@pytest.mark.mpi_skip()
@pytest.mark.parametrize("n_rhs", [1, 2, 3])
def test_pobts_permuted(
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

    B = rhs(
        n_rhs=n_rhs,
        diagonal_blocksize=diagonal_blocksize,
        arrowhead_blocksize=0,
        n_diag_blocks=n_diag_blocks,
        device_array=True if array_type == "device" else False,
        dtype=dtype,
    )
    X_ref = xp.linalg.solve(A.copy(), B.copy())

    (
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        _,
    ) = bt_dense_to_arrays(
        A.copy(), diagonal_blocksize, n_diag_blocks
    )

    # Allocate permutation buffer
    buffer = allocate_pobtx_permutation_buffers(
        A_diagonal_blocks=A_diagonal_blocks,
    )

    pobtf(
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        buffer=buffer,
    )

    # For permuted algorithm we need to make a reduced system,
    # invert it and add the buffer blocks back
    _A_diagonal_blocks = xp.zeros(
        (2, diagonal_blocksize, diagonal_blocksize), dtype=dtype
    )
    _A_lower_diagonal_blocks = xp.zeros(
        (1, diagonal_blocksize, diagonal_blocksize), dtype=dtype
    )

    _A_diagonal_blocks[0] = A_diagonal_blocks[0]
    _A_diagonal_blocks[1] = A_diagonal_blocks[-1]
    _A_lower_diagonal_blocks[0] = buffer[-1].conj().T

    # Finish the factorization (on the reduced system)
    pobtf(
        _A_diagonal_blocks,
        _A_lower_diagonal_blocks,
    )

    A_diagonal_blocks[0] = _A_diagonal_blocks[0]
    A_diagonal_blocks[-1] = _A_diagonal_blocks[1]
    buffer[-1] = _A_lower_diagonal_blocks[0].conj().T

    # Forward solve: Y=L^{-1}B
    pobts(
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        B,
        buffer=buffer,
        trans="N",
    )

    # Backward solve: X=L^{-T}Y
    pobts(
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        B,
        buffer=buffer,
        trans="C",
    )

    assert xp.allclose(X_ref, B)
