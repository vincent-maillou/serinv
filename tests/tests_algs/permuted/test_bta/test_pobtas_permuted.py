# Copyright 2023-2025 ETH Zurich. All rights reserved.

import numpy as np
import pytest

from serinv import _get_module_from_array
from ....testing_utils import bta_dense_to_arrays, dd_bta, symmetrize, rhs

from serinv.utils import allocate_pobtax_permutation_buffers
from serinv.algs import pobtaf, pobtas


@pytest.mark.mpi_skip()
@pytest.mark.parametrize("n_rhs", [1, 2, 3])
def test_pobtas_permuted(
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
    ) = bta_dense_to_arrays(
        A.copy(), diagonal_blocksize, arrowhead_blocksize, n_diag_blocks
    )

    # Allocate permutation buffer
    buffer = allocate_pobtax_permutation_buffers(
        A_diagonal_blocks=A_diagonal_blocks,
    )

    pobtaf(
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_lower_arrow_blocks,
        A_arrow_tip_block,
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

    # Finish the factorization (on the reduced system)
    pobtaf(
        _A_diagonal_blocks,
        _A_lower_diagonal_blocks,
        _A_lower_arrow_blocks,
        _A_arrow_tip_block,
    )

    A_diagonal_blocks[0] = _A_diagonal_blocks[0]
    A_diagonal_blocks[-1] = _A_diagonal_blocks[1]
    buffer[-1] = _A_lower_diagonal_blocks[0].conj().T
    A_lower_arrow_blocks[0] = _A_lower_arrow_blocks[0]
    A_lower_arrow_blocks[-1] = _A_lower_arrow_blocks[1]
    A_arrow_tip_block[:, :] = _A_arrow_tip_block


    pobtas(
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_lower_arrow_blocks,
        A_arrow_tip_block,
        B,
        buffer=buffer,
    )

    assert xp.allclose(X_ref, B)