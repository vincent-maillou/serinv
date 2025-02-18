# Copyright 2023-2025 ETH Zurich. All rights reserved.

from mpi4py import MPI

from serinv import (
    ArrayLike,
    backend_flags,
    _get_module_from_str,
    _get_module_from_array,
)

if backend_flags["cupy_avail"]:
    import cupyx as cpx

comm_rank = MPI.COMM_WORLD.Get_rank()
comm_size = MPI.COMM_WORLD.Get_size()


def allocate_pobtars(
    A_diagonal_blocks: ArrayLike,
    A_lower_diagonal_blocks: ArrayLike,
    A_arrow_bottom_blocks: ArrayLike,
    A_arrow_tip_block: ArrayLike,
    comm_size: int,
    array_module: str,
    device_streaming: bool = False,
    strategy: str = "allgather",
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
    strategy : str, optional
        Communication strategy to use. (default: "allgather")

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
        alloc = empty

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
    elif strategy == "gather-scatter":
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


def allocate_pinned_pobtars(
    _L_diagonal_blocks,
    _L_lower_diagonal_blocks,
    _L_lower_arrow_blocks,
    _L_arrow_tip_block,
):
    """Allocate pinned host-arrays for the reduced system of the PPOBTARX algorithms.

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

    Returns
    -------
    _L_diagonal_blocks_h : ArrayLike
        The diagonal blocks of the reduced system (host-pinned).
    _L_lower_diagonal_blocks_h : ArrayLike
        The lower diagonal blocks of the reduced system (host-pinned).
    _L_lower_arrow_blocks_h : ArrayLike
        The arrow bottom blocks of the reduced system (host-pinned).
    _L_tip_update_h : ArrayLike
        The arrow tip block of the reduced system (host-pinned).
    """
    _L_diagonal_blocks_h = cpx.zeros_like_pinned(_L_diagonal_blocks)
    _L_lower_diagonal_blocks_h = cpx.zeros_like_pinned(_L_lower_diagonal_blocks)
    _L_lower_arrow_blocks_h = cpx.zeros_like_pinned(_L_lower_arrow_blocks)
    _L_tip_update_h = cpx.zeros_like_pinned(_L_arrow_tip_block)

    return (
        _L_diagonal_blocks_h,
        _L_lower_diagonal_blocks_h,
        _L_lower_arrow_blocks_h,
        _L_tip_update_h,
    )


def map_ppobtax_to_pobtars(
    _L_diagonal_blocks: ArrayLike,
    _L_lower_diagonal_blocks: ArrayLike,
    _L_lower_arrow_blocks: ArrayLike,
    _L_tip_update: ArrayLike,
    A_diagonal_blocks: ArrayLike,
    A_lower_diagonal_blocks: ArrayLike,
    A_arrow_bottom_blocks: ArrayLike,
    A_arrow_tip_block: ArrayLike,
    buffer: ArrayLike,
    strategy: str = "allgather",
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

            _L_lower_diagonal_blocks[2 * comm_rank - 1] = buffer[-1].conj().T
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

            _L_lower_diagonal_blocks[2 * comm_rank] = buffer[-1].conj().T
            if comm_rank < comm_size - 1:
                _L_lower_diagonal_blocks[2 * comm_rank + 1] = A_lower_diagonal_blocks[
                    -1
                ]

            _L_lower_arrow_blocks[2 * comm_rank] = A_arrow_bottom_blocks[0]
            _L_lower_arrow_blocks[2 * comm_rank + 1] = A_arrow_bottom_blocks[-1]
    elif strategy == "gather-scatter":
        if comm_rank == 0:
            _L_diagonal_blocks[1] = A_diagonal_blocks[-1]
            _L_lower_diagonal_blocks[1] = A_lower_diagonal_blocks[-1]
            _L_lower_arrow_blocks[1] = A_arrow_bottom_blocks[-1]
        else:
            _L_diagonal_blocks[2 * comm_rank] = A_diagonal_blocks[0]
            _L_diagonal_blocks[2 * comm_rank + 1] = A_diagonal_blocks[-1]

            _L_lower_diagonal_blocks[2 * comm_rank] = buffer[-1].conj().T
            if comm_rank < comm_size - 1:
                _L_lower_diagonal_blocks[2 * comm_rank + 1] = A_lower_diagonal_blocks[
                    -1
                ]

            _L_lower_arrow_blocks[2 * comm_rank] = A_arrow_bottom_blocks[0]
            _L_lower_arrow_blocks[2 * comm_rank + 1] = A_arrow_bottom_blocks[-1]
    else:
        raise ValueError("Unknown communication strategy.")


def map_pobtars_to_ppobtax(
    L_diagonal_blocks: ArrayLike,
    L_lower_diagonal_blocks: ArrayLike,
    L_arrow_bottom_blocks: ArrayLike,
    L_arrow_tip_block: ArrayLike,
    buffer: ArrayLike,
    _L_diagonal_blocks: ArrayLike,
    _L_lower_diagonal_blocks: ArrayLike,
    _L_lower_arrow_blocks: ArrayLike,
    _L_tip_update: ArrayLike,
    strategy: str = "allgather",
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
    buffer : ArrayLike
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
        Communication strategy to use. (default: "allgather")
    """

    if strategy == "allreduce":
        if comm_rank == 0:
            L_diagonal_blocks[-1] = _L_diagonal_blocks[0]
            L_lower_diagonal_blocks[-1] = _L_lower_diagonal_blocks[0]
            L_arrow_bottom_blocks[-1] = _L_lower_arrow_blocks[0]
        else:
            L_diagonal_blocks[0] = _L_diagonal_blocks[2 * comm_rank - 1]
            L_diagonal_blocks[-1] = _L_diagonal_blocks[2 * comm_rank]

            buffer[-1] = _L_lower_diagonal_blocks[2 * comm_rank - 1].conj().T
            if comm_rank != comm_size - 1:
                L_lower_diagonal_blocks[-1] = _L_lower_diagonal_blocks[2 * comm_rank]

            L_arrow_bottom_blocks[0] = _L_lower_arrow_blocks[2 * comm_rank - 1]
            L_arrow_bottom_blocks[-1] = _L_lower_arrow_blocks[2 * comm_rank]
    elif strategy == "allgather":
        if comm_rank == 0:
            L_diagonal_blocks[-1] = _L_diagonal_blocks[0]
            L_lower_diagonal_blocks[-1] = _L_lower_diagonal_blocks[0]
            L_arrow_bottom_blocks[-1] = _L_lower_arrow_blocks[0]
        else:
            L_diagonal_blocks[0] = _L_diagonal_blocks[2 * comm_rank - 1]
            L_diagonal_blocks[-1] = _L_diagonal_blocks[2 * comm_rank]

            buffer[-1] = _L_lower_diagonal_blocks[2 * comm_rank - 1].conj().T
            if comm_rank != comm_size - 1:
                L_lower_diagonal_blocks[-1] = _L_lower_diagonal_blocks[2 * comm_rank]

            L_arrow_bottom_blocks[0] = _L_lower_arrow_blocks[2 * comm_rank - 1]
            L_arrow_bottom_blocks[-1] = _L_lower_arrow_blocks[2 * comm_rank]
    elif strategy == "gather-scatter":
        if comm_rank == 0:
            L_diagonal_blocks[-1] = _L_diagonal_blocks[1]
            L_lower_diagonal_blocks[-1] = _L_lower_diagonal_blocks[1]
            L_arrow_bottom_blocks[-1] = _L_lower_arrow_blocks[1]
        else:
            L_diagonal_blocks[0] = _L_diagonal_blocks[2 * comm_rank]
            L_diagonal_blocks[-1] = _L_diagonal_blocks[2 * comm_rank + 1]

            buffer[-1] = _L_lower_diagonal_blocks[2 * comm_rank].conj().T
            if comm_rank < comm_size - 1:
                L_lower_diagonal_blocks[-1] = _L_lower_diagonal_blocks[
                    2 * comm_rank + 1
                ]

            L_arrow_bottom_blocks[0] = _L_lower_arrow_blocks[2 * comm_rank]
            L_arrow_bottom_blocks[-1] = _L_lower_arrow_blocks[2 * comm_rank + 1]
    else:
        raise ValueError("Unknown communication strategy.")


def aggregate_pobtars(
    _L_diagonal_blocks: ArrayLike,
    _L_lower_diagonal_blocks: ArrayLike,
    _L_lower_arrow_blocks: ArrayLike,
    _L_tip_update: ArrayLike,
    strategy: str = "allgather",
    **kwargs,
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
        Communication strategy to use. (default: "allgather")
    """

    """ # For debug fill the reduced system with ones
    _L_diagonal_blocks.fill(1 * comm_rank + 1)
    _L_lower_diagonal_blocks.fill(1 * comm_rank + 1)
    _L_lower_arrow_blocks.fill(1 * comm_rank + 1)
    _L_tip_update.fill(1) """

    if strategy == "allreduce":
        MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, _L_diagonal_blocks, op=MPI.SUM)
        MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, _L_lower_diagonal_blocks, op=MPI.SUM)
        MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, _L_lower_arrow_blocks, op=MPI.SUM)
        MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, _L_tip_update, op=MPI.SUM)
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
        MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, _L_tip_update, op=MPI.SUM)
    elif strategy == "gather-scatter":
        root = kwargs.get("root", None)
        if root is None:
            raise ValueError(
                "The root rank must be given for gather-scatter communication strategy."
            )

        MPI.COMM_WORLD.Gather(
            sendbuf=(
                _L_diagonal_blocks[2 * comm_rank : 2 * (comm_rank + 1)]
                if comm_rank != root
                else MPI.IN_PLACE
            ),
            recvbuf=_L_diagonal_blocks if comm_rank == root else None,
            root=root,
        )
        MPI.COMM_WORLD.Gather(
            sendbuf=(
                _L_lower_diagonal_blocks[2 * comm_rank : 2 * (comm_rank + 1)]
                if comm_rank != root
                else MPI.IN_PLACE
            ),
            recvbuf=_L_lower_diagonal_blocks if comm_rank == root else None,
            root=root,
        )
        MPI.COMM_WORLD.Gather(
            sendbuf=(
                _L_lower_arrow_blocks[2 * comm_rank : 2 * (comm_rank + 1)]
                if comm_rank != root
                else MPI.IN_PLACE
            ),
            recvbuf=_L_lower_arrow_blocks if comm_rank == root else None,
            root=root,
        )
        MPI.COMM_WORLD.Reduce(
            sendbuf=_L_tip_update if comm_rank != root else MPI.IN_PLACE,
            recvbuf=_L_tip_update if comm_rank == root else None,
            op=MPI.SUM,
            root=root,
        )
    else:
        raise ValueError("Unknown communication strategy.")

    MPI.COMM_WORLD.Barrier()


def scatter_pobtars(
    _L_diagonal_blocks: ArrayLike,
    _L_lower_diagonal_blocks: ArrayLike,
    _L_lower_arrow_blocks: ArrayLike,
    L_arrow_tip_block: ArrayLike,
    strategy: str = "allgather",
    **kwargs,
):
    """Scatter the reduced system.

    Parameters
    ----------
    _L_diagonal_blocks : ArrayLike
        The diagonal blocks of the reduced system.
    _L_lower_diagonal_blocks : ArrayLike
        The lower diagonal blocks of the reduced system.
    _L_lower_arrow_blocks : ArrayLike
        The arrow bottom blocks of the reduced system.
    L_arrow_tip_block : ArrayLike
        The arrow tip block of the reduced system.
    strategy : str, optional
        Communication strategy to use. (default: "allgather")

    Keyword Arguments
    -----------------
    root : int, optional
        The root rank for the communication strategy. (default: 0)
    """
    if strategy == "allreduce":
        ...
    elif strategy == "allgather":
        ...
    elif strategy == "gather-scatter":
        root = kwargs.get("root", None)
        if root is None:
            raise ValueError(
                "The root rank must be given for gather-scatter communication strategy."
            )
        MPI.COMM_WORLD.Scatter(
            sendbuf=_L_diagonal_blocks if comm_rank == root else None,
            recvbuf=(
                _L_diagonal_blocks[2 * comm_rank : 2 * (comm_rank + 1)]
                if comm_rank != root
                else MPI.IN_PLACE
            ),
            root=root,
        )
        MPI.COMM_WORLD.Scatter(
            sendbuf=_L_lower_diagonal_blocks if comm_rank == root else None,
            recvbuf=(
                _L_lower_diagonal_blocks[2 * comm_rank : 2 * (comm_rank + 1)]
                if comm_rank != root
                else MPI.IN_PLACE
            ),
            root=root,
        )
        MPI.COMM_WORLD.Scatter(
            sendbuf=_L_lower_arrow_blocks if comm_rank == root else None,
            recvbuf=(
                _L_lower_arrow_blocks[2 * comm_rank : 2 * (comm_rank + 1)]
                if comm_rank != root
                else MPI.IN_PLACE
            ),
            root=root,
        )
        MPI.COMM_WORLD.Bcast(
            buf=L_arrow_tip_block,
            root=root,
        )
    else:
        raise ValueError("Unknown communication strategy.")

    MPI.COMM_WORLD.Barrier()
