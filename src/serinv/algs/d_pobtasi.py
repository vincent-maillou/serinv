# Copyright 2023-2024 ETH Zurich. All rights reserved.

try:
    import cupy as cp
    import cupyx.scipy.linalg as cu_la

    CUPY_AVAIL = True

except:
    CUPY_AVAIL = False

import numpy as np
import scipy.linalg as np_la
from numpy.typing import ArrayLike
from mpi4py import MPI

comm_rank = MPI.COMM_WORLD.Get_rank()
comm_size = MPI.COMM_WORLD.Get_size()

from serinv.algs import pobtasinv


def d_pobtasi(
    L_diagonal_blocks_local: ArrayLike,
    L_lower_diagonal_blocks_local: ArrayLike,
    L_arrow_bottom_blocks_local: ArrayLike,
    L_arrow_tip_block_global: ArrayLike,
    device_streaming: bool = False,
) -> tuple[
    ArrayLike,
    ArrayLike,
    ArrayLike,
    ArrayLike,
]:
    """Perform a distributed selected-inversion of a block tridiagonal with
    arrowhead matrix.

    Note:
    -----
    - Will overwrite the inputs and store the results in them (in-place).
    - If a device array is given, the algorithm will run on the GPU.

    Parameters
    ----------
    L_diagonal_blocks_local : ArrayLike
        Local slice of the diagonal blocks of L.
    L_lower_diagonal_blocks_local : ArrayLike
        Local slice of the lower diagonal blocks of L.
    L_arrow_bottom_blocks_local : ArrayLike
        Local slice of the arrow bottom blocks of L.
    L_arrow_tip_block_global : ArrayLike
        Arrow tip block of L.
    device_streaming : bool
        Whether to use streamed GPU computation.

    Returns
    -------
    X_diagonal_blocks_local : ArrayLike
        Local slice of the diagonal blocks of X.
    X_lower_diagonal_blocks_local : ArrayLike
        Local slice of the lower diagonal blocks of X.
    X_arrow_bottom_blocks_local : ArrayLike
        Local slice of the arrow bottom blocks of X.
    X_arrow_tip_block_global : ArrayLike
        Arrow tip block of X.
    """

    if (
        CUPY_AVAIL
        and cp.get_array_module(L_diagonal_blocks_local) == np
        and device_streaming
    ):
        return _streaming_d_pobtasi(
            L_diagonal_blocks_local,
            L_lower_diagonal_blocks_local,
            L_arrow_bottom_blocks_local,
            L_arrow_tip_block_global,
        )

    return _d_pobtasi(
        L_diagonal_blocks_local,
        L_lower_diagonal_blocks_local,
        L_arrow_bottom_blocks_local,
        L_arrow_tip_block_global,
    )


def _d_pobtasi(
    L_diagonal_blocks_local: ArrayLike,
    L_lower_diagonal_blocks_local: ArrayLike,
    L_arrow_bottom_blocks_local: ArrayLike,
    L_arrow_tip_block_global: ArrayLike,
) -> tuple[
    ArrayLike,
    ArrayLike,
    ArrayLike,
    ArrayLike,
]:
    la = np_la
    if CUPY_AVAIL:
        xp = cp.get_array_module(L_diagonal_blocks_local)
        if xp == cp:
            la = cu_la
    else:
        xp = np


def _streaming_d_pobtasi(
    L_diagonal_blocks_local: ArrayLike,
    L_lower_diagonal_blocks_local: ArrayLike,
    L_arrow_bottom_blocks_local: ArrayLike,
    L_arrow_tip_block_global: ArrayLike,
) -> tuple[
    ArrayLike,
    ArrayLike,
    ArrayLike,
    ArrayLike,
]:
    raise NotImplementedError
