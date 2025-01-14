# Copyright 2023-2025 ETH Zurich. All rights reserved.

from serinv import (
    ArrayLike,
)

from serinv.algs import pobtasi
from serinv.wrappers.ppobtars import map_ppobtars_to_ppobtax

from mpi4py import MPI

comm_rank = MPI.COMM_WORLD.Get_rank()
comm_size = MPI.COMM_WORLD.Get_size()


def ppobtasi(
    L_diagonal_blocks: ArrayLike,
    L_lower_diagonal_blocks: ArrayLike,
    L_arrow_bottom_blocks: ArrayLike,
    L_arrow_tip_block: ArrayLike,
    _L_diagonal_blocks: ArrayLike,
    _L_lower_diagonal_blocks: ArrayLike,
    _L_lower_arrow_blocks: ArrayLike,
    L_permutation_buffer: ArrayLike,
    **kwargs,
):
    """Perform a selected inversion of a block tridiagonal with arrowhead matrix (pointing downward by convention).

    Parameters
    ----------
    L_diagonal_blocks : ArrayLike
        Diagonal blocks of the Cholesky factor of the matrix.
    L_lower_diagonal_blocks : ArrayLike
        Lower diagonal blocks of the Cholesky factor of the matrix.
    L_arrow_bottom_blocks : ArrayLike
        Arrow bottom blocks of the Cholesky factor of the matrix.
    L_arrow_tip_block : ArrayLike
        Arrow tip block of the Cholesky factor of the matrix.
    _L_diagonal_blocks : ArrayLike
        Diagonal blocks of the reduced system.
    _L_lower_diagonal_blocks : ArrayLike
        Lower diagonal blocks of the reduced system.
    _L_lower_arrow_blocks : ArrayLike
        Arrow bottom blocks of the reduced system.
    L_permutation_buffer : ArrayLike
        Buffer array for the permuted arrowhead.

    Keyword Arguments
    -----------------
    device_streaming : bool, optional
        If True, the algorithm will run on the GPU. (default: False)

    Note:
    -----
    - Will overwrite the inputs and store the results in them (in-place).
    - If a device array is given, the algorithm will run on the GPU.

    Currently implemented:
    ----------------------
    |              | Natural | Permuted |
    | ------------ | ------- | -------- |
    | Direct-array | x       | x        |
    | Streaming    | x       | x        |
    """
    device_streaming: bool = kwargs.get("device_streaming", False)
    strategy: str = kwargs.get("strategy", "allgather")

    # Selected-inversion of the reduced system
    pobtasi(
        _L_diagonal_blocks,
        _L_lower_diagonal_blocks,
        _L_lower_arrow_blocks,
        L_arrow_tip_block,
        device_streaming=device_streaming,
    )

    # Map result of the reduced system back to the original system
    map_ppobtars_to_ppobtax(
        L_diagonal_blocks=L_diagonal_blocks,
        L_lower_diagonal_blocks=L_lower_diagonal_blocks,
        L_arrow_bottom_blocks=L_arrow_bottom_blocks,
        L_arrow_tip_block=L_arrow_tip_block,
        L_permutation_buffer=L_permutation_buffer,
        _L_diagonal_blocks=_L_diagonal_blocks,
        _L_lower_diagonal_blocks=_L_lower_diagonal_blocks,
        _L_lower_arrow_blocks=_L_lower_arrow_blocks,
        _L_tip_update=L_arrow_tip_block,
        strategy=strategy,
    )

    # Parallel selected inversion of the original system
    if comm_rank == 0:
        pobtasi(
            L_diagonal_blocks,
            L_lower_diagonal_blocks,
            L_arrow_bottom_blocks,
            L_arrow_tip_block,
            device_streaming=device_streaming,
            inverse_last_block=False,
        )
    else:
        pobtasi(
            L_diagonal_blocks,
            L_lower_diagonal_blocks,
            L_arrow_bottom_blocks,
            L_arrow_tip_block,
            device_streaming=device_streaming,
            buffer=L_permutation_buffer,
        )
