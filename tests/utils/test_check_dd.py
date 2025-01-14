# Copyright 2023-2025 ETH Zurich. All rights reserved.

import numpy as np
import pytest

from serinv import CUPY_AVAIL, _get_module_from_array
from ..testing_utils import bta_dense_to_arrays, dd_bta

from serinv.utils.check_dd import check_block_dd, check_ddbta

device_array = [False]
if CUPY_AVAIL:
    device_array += [True]

@pytest.mark.mpi_skip()
@pytest.mark.parametrize("diagonal_blocksize", [2, 3])
@pytest.mark.parametrize("arrowhead_blocksize", [2, 3])
@pytest.mark.parametrize("n_diag_blocks", [1, 2, 3, 4])
@pytest.mark.parametrize("device_array", device_array)
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_check_block_dd(
    diagonal_blocksize,
    arrowhead_blocksize,
    n_diag_blocks,
    device_array,
    dtype,
):

    A = dd_bta(
        diagonal_blocksize,
        arrowhead_blocksize,
        n_diag_blocks,
        device_array=device_array,
        dtype=dtype,
    )

    xp, _ = _get_module_from_array(A)

    (
        A_diagonal_blocks,
        _,
        _,
        _,
        _,
        _,
    ) = bta_dense_to_arrays(A, diagonal_blocksize, arrowhead_blocksize, n_diag_blocks)

    block_dd = check_block_dd(A_diagonal_blocks)

    assert xp.all(block_dd)


@pytest.mark.mpi_skip()
@pytest.mark.parametrize("diagonal_blocksize", [2, 3])
@pytest.mark.parametrize("arrowhead_blocksize", [2, 3])
@pytest.mark.parametrize("n_diag_blocks", [1, 2, 3, 4])
@pytest.mark.parametrize("device_array", device_array)
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_check_ddbta(
    diagonal_blocksize,
    arrowhead_blocksize,
    n_diag_blocks,
    device_array,
    dtype,
):
    A = dd_bta(
        diagonal_blocksize,
        arrowhead_blocksize,
        n_diag_blocks,
        device_array=device_array,
        dtype=dtype,
    )

    xp, _ = _get_module_from_array(A)

    (
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_upper_diagonal_blocks,
        A_arrow_bottom_blocks,
        A_arrow_right_blocks,
        A_arrow_tip_block,
    ) = bta_dense_to_arrays(A, diagonal_blocksize, arrowhead_blocksize, n_diag_blocks)

    ddbta = check_ddbta(
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_upper_diagonal_blocks,
        A_arrow_bottom_blocks,
        A_arrow_right_blocks,
        A_arrow_tip_block,
    )

    assert xp.all(ddbta)
