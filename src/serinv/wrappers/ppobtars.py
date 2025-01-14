# Copyright 2023-2025 ETH Zurich. All rights reserved.

from mpi4py import MPI

from serinv import (
    ArrayLike,
    CUPY_AVAIL,
    _get_module_from_str,
    _get_module_from_array,
)

if CUPY_AVAIL:
    import cupyx as cpx

comm_rank = MPI.COMM_WORLD.Get_rank()
comm_size = MPI.COMM_WORLD.Get_size()


def allocate_permutation_buffer(
    A_diagonal_blocks: ArrayLike,
    device_streaming: bool,
):
    """Allocate the (permutation) buffers necessary for the parallel BTA algorithms.

    Parameters
    ----------
    A_diagonal_blocks : ArrayLike
        The diagonal blocks of the original system.
    device_streaming : bool
        If True, pinned host-arrays will be allocated

    Returns
    -------
    A_permutation_buffer : ArrayLike
        The permutation buffer needed for the parallel BTA algorithms.
    """
    xp, _ = _get_module_from_array(arr=A_diagonal_blocks)

    if device_streaming:
        empty_like = cpx.empty_like_pinned
    else:
        empty_like = xp.empty_like

    A_permutation_buffer = empty_like(A_diagonal_blocks)

    return A_permutation_buffer


def allocate_ppobtars(
    A_diagonal_blocks: ArrayLike,
    A_lower_diagonal_blocks: ArrayLike,
    A_arrow_bottom_blocks: ArrayLike,
    A_arrow_tip_block: ArrayLike,
    comm_size: int,
    array_module: str,
    device_streaming: bool = False,
    strategy: str = "allreduce",
):
    """Allocate the buffers necessary for the reduced system of the PPOBTARX algorithms.

    Parameters
    ----------
    A_diagonal_blocks : ArrayLike
        The diagonal blocks of the original system.
    A_lower_diagonal_blocks : ArrayLike
        The lower diagonal blocks of the original system.
    A_arrow_bottom_blocks : ArrayLike
        The arrow bottom blocks of the original system.
    A_arrow_tip_block : ArrayLike
        The arrow tip block of the original system.
    comm_size : int
        Number of MPI ranks.
    array_module : str
        Array module to use, expect "numpy" or "cupy".
    device_streaming : bool, optional
        If True, pinned host-arrays will be allocated

    Returns
    -------
    _L_diagonal_blocks : ArrayLike
        The diagonal blocks of the reduced system.
    _L_lower_diagonal_blocks : ArrayLike
        The lower diagonal blocks of the reduced system.
    _L_lower_arrow_blocks : ArrayLike
        The arrow bottom blocks of the reduced system.
    _L_tip_update : ArrayLike
        The arrow tip block of the reduced system.
    """
    xp, _ = _get_module_from_str(array_module)

    if device_streaming:
        zeros = cpx.zeros_pinned
        empty = cpx.empty_pinned
    else:
        zeros = xp.zeros
        empty = xp.empty

    if strategy == "allreduce":
        _n: int = 2 * comm_size - 1

        # In the case of an allreduce communication strategy, the buffers needs
        # to be allocated as zeros to avoid false-reduction.
        alloc = zeros

        _L_diagonal_blocks = alloc(
            (_n, A_diagonal_blocks[0].shape[0], A_diagonal_blocks[0].shape[1]),
            dtype=A_diagonal_blocks.dtype,
        )
        _L_lower_diagonal_blocks = alloc(
            (
                _n - 1,
                A_lower_diagonal_blocks[0].shape[0],
                A_lower_diagonal_blocks[0].shape[1],
            ),
            dtype=A_lower_diagonal_blocks.dtype,
        )
        _L_lower_arrow_blocks = alloc(
            (_n, A_arrow_bottom_blocks[0].shape[0], A_arrow_bottom_blocks[0].shape[1]),
            dtype=A_arrow_bottom_blocks.dtype,
        )
    elif strategy == "allgather":
        _n: int = 2 * comm_size
        alloc = zeros

        _L_diagonal_blocks = alloc(
            (_n, A_diagonal_blocks[0].shape[0], A_diagonal_blocks[0].shape[1]),
            dtype=A_diagonal_blocks.dtype,
        )
        _L_lower_diagonal_blocks = alloc(
            (
                _n,
                A_lower_diagonal_blocks[0].shape[0],
                A_lower_diagonal_blocks[0].shape[1],
            ),
            dtype=A_lower_diagonal_blocks.dtype,
        )
        _L_lower_arrow_blocks = alloc(
            (_n, A_arrow_bottom_blocks[0].shape[0], A_arrow_bottom_blocks[0].shape[1]),
            dtype=A_arrow_bottom_blocks.dtype,
        )
    else:
        raise ValueError("Unknown communication strategy.")

    _L_tip_update = zeros(
        (A_arrow_tip_block.shape[0], A_arrow_tip_block.shape[1]),
        dtype=A_arrow_tip_block.dtype,
    )

    return (
        _L_diagonal_blocks,
        _L_lower_diagonal_blocks,
        _L_lower_arrow_blocks,
        _L_tip_update,
    )


def map_ppobtax_to_ppobtars(
    _L_diagonal_blocks: ArrayLike,
    _L_lower_diagonal_blocks: ArrayLike,
    _L_lower_arrow_blocks: ArrayLike,
    _L_tip_update: ArrayLike,
    A_diagonal_blocks: ArrayLike,
    A_lower_diagonal_blocks: ArrayLike,
    A_arrow_bottom_blocks: ArrayLike,
    A_arrow_tip_block: ArrayLike,
    A_permutation_buffer: ArrayLike,
    strategy: str = "allreduce",
):
    """Map the the boundary blocks of the PPOBTAX algorithm to the reduced system.

    Parameters
    ----------
    _L_diagonal_blocks : ArrayLike
        The diagonal blocks of the reduced system.
    _L_lower_diagonal_blocks : ArrayLike
        The lower diagonal blocks of the reduced system.
    _L_lower_arrow_blocks : ArrayLike
        The arrow bottom blocks of the reduced system.
    _L_tip_update : ArrayLike
        The arrow tip block of the reduced system.
    A_diagonal_blocks : ArrayLike
        The diagonal blocks of the original system.
    A_lower_diagonal_blocks : ArrayLike
        The lower diagonal blocks of the original system.
    A_arrow_bottom_blocks : ArrayLike
        The arrow bottom blocks of the original system.
    A_arrow_tip_block : ArrayLike
        The arrow tip block of the original system.
    """

    if strategy == "allreduce":
        if comm_rank == 0:
            _L_diagonal_blocks[0] = A_diagonal_blocks[-1]
            _L_lower_diagonal_blocks[0] = A_lower_diagonal_blocks[-1]
            _L_lower_arrow_blocks[0] = A_arrow_bottom_blocks[-1]
        else:
            _L_diagonal_blocks[2 * comm_rank - 1] = A_diagonal_blocks[0]
            _L_diagonal_blocks[2 * comm_rank] = A_diagonal_blocks[-1]

            _L_lower_diagonal_blocks[2 * comm_rank - 1] = (
                A_permutation_buffer[-1].conj().T
            )
            if comm_rank != comm_size - 1:
                _L_lower_diagonal_blocks[2 * comm_rank] = A_lower_diagonal_blocks[-1]

            _L_lower_arrow_blocks[2 * comm_rank - 1] = A_arrow_bottom_blocks[0]
            _L_lower_arrow_blocks[2 * comm_rank] = A_arrow_bottom_blocks[-1]
    elif strategy == "allgather":
        if comm_rank == 0:
            _L_diagonal_blocks[1] = A_diagonal_blocks[-1]
            _L_lower_diagonal_blocks[1] = A_lower_diagonal_blocks[-1]
            _L_lower_arrow_blocks[1] = A_arrow_bottom_blocks[-1]
        else:
            _L_diagonal_blocks[2 * comm_rank] = A_diagonal_blocks[0]
            _L_diagonal_blocks[2 * comm_rank + 1] = A_diagonal_blocks[-1]

            _L_lower_diagonal_blocks[2 * comm_rank] = A_permutation_buffer[-1].conj().T
            if comm_rank < comm_size - 1:
                _L_lower_diagonal_blocks[2 * comm_rank + 1] = A_lower_diagonal_blocks[
                    -1
                ]

            _L_lower_arrow_blocks[2 * comm_rank] = A_arrow_bottom_blocks[0]
            _L_lower_arrow_blocks[2 * comm_rank + 1] = A_arrow_bottom_blocks[-1]
    else:
        raise ValueError("Unknown communication strategy.")

def map_ppobtars_to_ppobtax(
    L_diagonal_blocks: ArrayLike,
    L_lower_diagonal_blocks: ArrayLike,
    L_arrow_bottom_blocks: ArrayLike,
    L_arrow_tip_block: ArrayLike,
    L_permutation_buffer: ArrayLike,
    _L_diagonal_blocks: ArrayLike,
    _L_lower_diagonal_blocks: ArrayLike,
    _L_lower_arrow_blocks: ArrayLike,
    _L_tip_update: ArrayLike,
    strategy: str = "allreduce",
):
    """Map the reduced system back to the original system.

    Parameters
    ----------
    A_diagonal_blocks : ArrayLike
        The diagonal blocks of the original system.
    A_lower_diagonal_blocks : ArrayLike
        The lower diagonal blocks of the original system.
    A_arrow_bottom_blocks : ArrayLike
        The arrow bottom blocks of the original system.
    A_arrow_tip_block : ArrayLike
        The arrow tip block of the original system.
    A_permutation_buffer : ArrayLike
        The permutation buffer of the original system.
    _L_diagonal_blocks : ArrayLike
        The diagonal blocks of the reduced system.
    _L_lower_diagonal_blocks : ArrayLike
        The lower diagonal blocks of the reduced system.
    _L_lower_arrow_blocks : ArrayLike
        The arrow bottom blocks of the reduced system.
    _L_tip_update : ArrayLike
        The arrow tip block of the reduced system.
    strategy : str, optional
        Communication strategy to use. (default: "allreduce")
    """

    if strategy == "allreduce":
        if comm_rank == 0:
            L_diagonal_blocks[-1] = _L_diagonal_blocks[0]
            L_lower_diagonal_blocks[-1] = _L_lower_diagonal_blocks[0]
            L_arrow_bottom_blocks[-1] = _L_lower_arrow_blocks[0]
        else:
            L_diagonal_blocks[0] = _L_diagonal_blocks[2 * comm_rank - 1]
            L_diagonal_blocks[-1] = _L_diagonal_blocks[2 * comm_rank]

            L_permutation_buffer[-1] = (
                _L_lower_diagonal_blocks[2 * comm_rank - 1].conj().T
            )
            if comm_rank != comm_size - 1:
                L_lower_diagonal_blocks[-1] = _L_lower_diagonal_blocks[2 * comm_rank]

            L_arrow_bottom_blocks[0] = _L_lower_arrow_blocks[2 * comm_rank - 1]
            L_arrow_bottom_blocks[-1] = _L_lower_arrow_blocks[2 * comm_rank]
    elif strategy == "allgather":
        if comm_rank == 0:
            L_diagonal_blocks[-1] = _L_diagonal_blocks[1]
            L_lower_diagonal_blocks[-1] = _L_lower_diagonal_blocks[1]
            L_arrow_bottom_blocks[-1] = _L_lower_arrow_blocks[1]
        else:
            L_diagonal_blocks[0] = _L_diagonal_blocks[2 * comm_rank]
            L_diagonal_blocks[-1] = _L_diagonal_blocks[2 * comm_rank + 1]

            L_permutation_buffer[-1] = _L_lower_diagonal_blocks[2 * comm_rank].conj().T
            if comm_rank < comm_size - 1:
                L_lower_diagonal_blocks[-1] = _L_lower_diagonal_blocks[
                    2 * comm_rank + 1
                ]

            L_arrow_bottom_blocks[0] = _L_lower_arrow_blocks[2 * comm_rank]
            L_arrow_bottom_blocks[-1] = _L_lower_arrow_blocks[2 * comm_rank + 1]
    else:
        raise ValueError("Unknown communication strategy.")

def aggregate_ppobtars(
    _L_diagonal_blocks: ArrayLike,
    _L_lower_diagonal_blocks: ArrayLike,
    _L_lower_arrow_blocks: ArrayLike,
    _L_tip_update: ArrayLike,
    strategy: str = "allreduce",
):
    """Aggregate the reduced system.

    Parameters
    ----------
    _L_diagonal_blocks : ArrayLike
        The diagonal blocks of the reduced system.
    _L_lower_diagonal_blocks : ArrayLike
        The lower diagonal blocks of the reduced system.
    _L_lower_arrow_blocks : ArrayLike
        The arrow bottom blocks of the reduced system.
    _L_tip_update : ArrayLike
        The arrow tip block of the reduced system.
    strategy : str, optional
        Communication strategy to use. (default: "allreduce")
    """

    if strategy == "allreduce":
        MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, _L_diagonal_blocks, op=MPI.SUM)
        MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, _L_lower_diagonal_blocks, op=MPI.SUM)
        MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, _L_lower_arrow_blocks, op=MPI.SUM)
    elif strategy == "allgather":
        MPI.COMM_WORLD.Allgather(
            MPI.IN_PLACE,
            _L_diagonal_blocks,
        )
        MPI.COMM_WORLD.Allgather(
            MPI.IN_PLACE,
            _L_lower_diagonal_blocks,
        )
        MPI.COMM_WORLD.Allgather(
            MPI.IN_PLACE,
            _L_lower_arrow_blocks,
        )
    else:
        raise ValueError("Unknown communication strategy.")

    # The tip update allways need an allreduce operation.
    MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, _L_tip_update, op=MPI.SUM)

    MPI.COMM_WORLD.Barrier()
