# Copyright 2023-2025 ETH Zurich. All rights reserved.

import time


def print_time(rank, start, index):
    end = time.perf_counter()
    print(f"Rank {rank} ({index}) {end - start:.5f} sec", flush=True)
    index += 1
    return end, index


from serinv import (
    ArrayLike,
    CUPY_AVAIL,
    _get_module_from_array,
)

if CUPY_AVAIL:
    import cupyx as cpx

from mpi4py import MPI


from serinv.algs import pobtaf
from .ppobtars import (
    allocate_permutation_buffer,
    allocate_ppobtars,
    map_ppobtax_to_ppobtars,
    aggregate_ppobtars,
    allocate_pinned_pobtars,
)

comm_rank = MPI.COMM_WORLD.Get_rank()
comm_size = MPI.COMM_WORLD.Get_size()


def ppobtaf(
    A_diagonal_blocks: ArrayLike,
    A_lower_diagonal_blocks: ArrayLike,
    A_arrow_bottom_blocks: ArrayLike,
    A_arrow_tip_block: ArrayLike,
    **kwargs,
) -> ArrayLike:
    """Perform the parallel factorization of a block tridiagonal with arrowhead matrix
    (pointing downward by convention).

    Parameters
    ----------
    A_diagonal_blocks : ArrayLike
        The diagonal blocks of the block tridiagonal with arrowhead matrix.
    A_lower_diagonal_blocks : ArrayLike
        The lower diagonal blocks of the block tridiagonal with arrowhead matrix.
    A_arrow_bottom_blocks : ArrayLike
        The arrow bottom blocks of the block tridiagonal with arrowhead matrix.
    A_arrow_tip_block : ArrayLike
        The arrow tip block of the block tridiagonal with arrowhead matrix.

    Returns
    -------
    _L_diagonal_blocks : ArrayLike
        Diagonal blocks of the reduced system.
    _L_lower_diagonal_blocks : ArrayLike
        Lower diagonal blocks of the reduced system.
    _L_lower_arrow_blocks : ArrayLike
        Arrow bottom blocks of the reduced system.
    buffer : ArrayLike
        Buffer array for the permuted arrowhead.

    Keyword Arguments
    -----------------
    device_streaming : bool, optional
        If True, the algorithm will perform host-device streaming. (default: False)
    strategy : str, optional
        The communication strategy to use. (default: "allgather")
    root : int, optional
        The root rank for the communication strategy. (default: 0)

    Note:
    -----
    - The matrix, A, is assumed to be symmetric positive definite.
    - Will overwrite the inputs and store the results in them (in-place).
    - If a device array is given, the algorithm will run on the GPU.

    Currently implemented:
    ----------------------
    |              | Natural | Permuted |
    | ------------ | ------- | -------- |
    | Direct-array | x       | x        |
    | Streaming    | x       | x        |
    """

    start, index = time.perf_counter(), 0

    if comm_size == 1:
        raise ValueError("The number of MPI processes must be greater than 1.")

    xp, _ = _get_module_from_array(arr=A_diagonal_blocks)

    # Check for optional parameters
    device_streaming: bool = kwargs.get("device_streaming", False)
    strategy: str = kwargs.get("strategy", "allgather")
    root: int = kwargs.get("root", 0)

    # Check for given permutation buffer
    A_permutation_buffer: ArrayLike = kwargs.get("A_permutation_buffer", None)

    if comm_rank != 0:
        if A_permutation_buffer is None:
            A_permutation_buffer = allocate_permutation_buffer(
                A_diagonal_blocks=A_diagonal_blocks,
                device_streaming=device_streaming,
            )
        else:
            assert A_permutation_buffer.shape == A_diagonal_blocks.shape

    # start, index = print_time(comm_rank, start, index)

    # Check for given reduced system buffers
    _L_diagonal_blocks: ArrayLike = kwargs.get("_L_diagonal_blocks", None)
    _L_lower_diagonal_blocks: ArrayLike = kwargs.get("_L_lower_diagonal_blocks", None)
    _L_lower_arrow_blocks: ArrayLike = kwargs.get("_L_lower_arrow_blocks", None)
    _L_tip_update: ArrayLike = kwargs.get("_L_tip_update", None)

    # If one of the reduced system buffers is not given, allocate them all
    if any(
        buffers is None
        for buffers in [
            _L_diagonal_blocks,
            _L_lower_diagonal_blocks,
            _L_lower_arrow_blocks,
            _L_tip_update,
        ]
    ):
        (
            _L_diagonal_blocks,
            _L_lower_diagonal_blocks,
            _L_lower_arrow_blocks,
            _L_tip_update,
        ) = allocate_ppobtars(
            A_diagonal_blocks=A_diagonal_blocks,
            A_lower_diagonal_blocks=A_lower_diagonal_blocks,
            A_arrow_bottom_blocks=A_arrow_bottom_blocks,
            A_arrow_tip_block=A_arrow_tip_block,
            comm_size=comm_size,
            array_module=xp.__name__,
            device_streaming=device_streaming,
            strategy=strategy,
        )
    
    # start, index = print_time(comm_rank, start, index)

    # Perform the parallel factorization
    if comm_rank == 0:
        pobtaf(
            A_diagonal_blocks,
            A_lower_diagonal_blocks,
            A_arrow_bottom_blocks,
            _L_tip_update,
            device_streaming=device_streaming,
            factorize_last_block=False,
        )
    else:
        pobtaf(
            A_diagonal_blocks,
            A_lower_diagonal_blocks,
            A_arrow_bottom_blocks,
            _L_tip_update,
            device_streaming=device_streaming,
            buffer=A_permutation_buffer,
        )

    start, index = print_time(comm_rank, start, index)

    map_ppobtax_to_ppobtars(
        _L_diagonal_blocks=_L_diagonal_blocks,
        _L_lower_diagonal_blocks=_L_lower_diagonal_blocks,
        _L_lower_arrow_blocks=_L_lower_arrow_blocks,
        _L_tip_update=_L_tip_update,
        A_diagonal_blocks=A_diagonal_blocks,
        A_lower_diagonal_blocks=A_lower_diagonal_blocks,
        A_arrow_bottom_blocks=A_arrow_bottom_blocks,
        A_arrow_tip_block=A_arrow_tip_block,
        A_permutation_buffer=A_permutation_buffer,
        strategy=strategy,
    )

    # start, index = print_time(comm_rank, start, index)

    if xp.__name__ == "cupy":
        # Check for given pinned memory buffers
        _L_diagonal_blocks_h: ArrayLike = kwargs.get("_L_diagonal_blocks_h", None)
        _L_lower_diagonal_blocks_h: ArrayLike = kwargs.get(
            "_L_lower_diagonal_blocks_h", None
        )
        _L_lower_arrow_blocks_h: ArrayLike = kwargs.get("_L_lower_arrow_blocks_h", None)
        _L_tip_update_h: ArrayLike = kwargs.get("_L_tip_update_h", None)

        if any(
            buffers is None
            for buffers in [
                _L_diagonal_blocks_h,
                _L_lower_diagonal_blocks_h,
                _L_lower_arrow_blocks_h,
                _L_tip_update_h,
            ]
        ):
            (
                _L_diagonal_blocks_h,
                _L_lower_diagonal_blocks_h,
                _L_lower_arrow_blocks_h,
                _L_tip_update_h,
            ) = allocate_pinned_pobtars(
                _L_diagonal_blocks,
                _L_lower_diagonal_blocks,
                _L_lower_arrow_blocks,
                _L_tip_update,
            )

        _L_diagonal_blocks.get(out=_L_diagonal_blocks_h)
        _L_lower_diagonal_blocks.get(out=_L_lower_diagonal_blocks_h)
        _L_lower_arrow_blocks.get(out=_L_lower_arrow_blocks_h)
        _L_tip_update.get(out=_L_tip_update_h)

        aggregate_ppobtars(
            _L_diagonal_blocks=_L_diagonal_blocks_h,
            _L_lower_diagonal_blocks=_L_lower_diagonal_blocks_h,
            _L_lower_arrow_blocks=_L_lower_arrow_blocks_h,
            _L_tip_update=_L_tip_update_h,
            strategy=strategy,
            root=root if strategy == "gather-scatter" else None,
        )

        _L_diagonal_blocks.set(arr=_L_diagonal_blocks_h)
        _L_lower_diagonal_blocks.set(arr=_L_lower_diagonal_blocks_h)
        _L_lower_arrow_blocks.set(arr=_L_lower_arrow_blocks_h)
        _L_tip_update.set(arr=_L_tip_update_h)

    else:
        aggregate_ppobtars(
            _L_diagonal_blocks=_L_diagonal_blocks,
            _L_lower_diagonal_blocks=_L_lower_diagonal_blocks,
            _L_lower_arrow_blocks=_L_lower_arrow_blocks,
            _L_tip_update=_L_tip_update,
            strategy=strategy,
            root=root if strategy == "gather-scatter" else None,
        )
    
    start, index = print_time(comm_rank, start, index)

    A_arrow_tip_block[:, :] = A_arrow_tip_block[:, :] + _L_tip_update[:, :]

    # --- Factorize the reduced system ---
    if strategy == "gather-scatter":
        if comm_rank == root:
            pobtaf(
                _L_diagonal_blocks[1:],
                _L_lower_diagonal_blocks[1:-1],
                _L_lower_arrow_blocks[1:],
                A_arrow_tip_block,
                device_streaming=device_streaming,
            )
    else:
        if strategy == "allgather":
            _L_diagonal_blocks = _L_diagonal_blocks[1:]
            _L_lower_diagonal_blocks = _L_lower_diagonal_blocks[1:-1]
            _L_lower_arrow_blocks = _L_lower_arrow_blocks[1:]

        pobtaf(
            _L_diagonal_blocks,
            _L_lower_diagonal_blocks,
            _L_lower_arrow_blocks,
            A_arrow_tip_block,
            device_streaming=device_streaming,
        )
    
    start, index = print_time(comm_rank, start, index)

    return (
        _L_diagonal_blocks,
        _L_lower_diagonal_blocks,
        _L_lower_arrow_blocks,
        A_permutation_buffer,
    )
