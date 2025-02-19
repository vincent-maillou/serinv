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

    if strategy == "allgather":
        _n: int = 2 * comm_size
        alloc = empty

        _A_diagonal_blocks = alloc(
            (_n, A_diagonal_blocks[0].shape[0], A_diagonal_blocks[0].shape[1]),
            dtype=A_diagonal_blocks.dtype,
        )
        _A_lower_diagonal_blocks = alloc(
            (
                _n,
                A_lower_diagonal_blocks[0].shape[0],
                A_lower_diagonal_blocks[0].shape[1],
            ),
            dtype=A_lower_diagonal_blocks.dtype,
        )
        _A_lower_arrow_blocks = alloc(
            (_n, A_arrow_bottom_blocks[0].shape[0], A_arrow_bottom_blocks[0].shape[1]),
            dtype=A_arrow_bottom_blocks.dtype,
        )
    elif strategy == "gather-scatter":
        _n: int = 2 * comm_size
        alloc = zeros

        _A_diagonal_blocks = alloc(
            (_n, A_diagonal_blocks[0].shape[0], A_diagonal_blocks[0].shape[1]),
            dtype=A_diagonal_blocks.dtype,
        )
        _A_lower_diagonal_blocks = alloc(
            (
                _n,
                A_lower_diagonal_blocks[0].shape[0],
                A_lower_diagonal_blocks[0].shape[1],
            ),
            dtype=A_lower_diagonal_blocks.dtype,
        )
        _A_lower_arrow_blocks = alloc(
            (_n, A_arrow_bottom_blocks[0].shape[0], A_arrow_bottom_blocks[0].shape[1]),
            dtype=A_arrow_bottom_blocks.dtype,
        )
    else:
        raise ValueError("Unknown communication strategy.")

    _A_arrow_tip_block = zeros(
        (A_arrow_tip_block.shape[0], A_arrow_tip_block.shape[1]),
        dtype=A_arrow_tip_block.dtype,
    )

    # If needed, allocate the reduced system for communication
    if xp.__name__ == "cupy":
        _A_diagonal_blocks_comm = cpx.empty_like_pinned(_A_diagonal_blocks)
        _A_lower_diagonal_blocks_comm = cpx.empty_like_pinned(_A_lower_diagonal_blocks)
        _A_lower_arrow_blocks_comm = cpx.empty_like_pinned(_A_lower_arrow_blocks)
        _A_arrow_tip_block_comm = cpx.empty_like_pinned(_A_arrow_tip_block)
    else:
        _A_diagonal_blocks_comm = _A_diagonal_blocks
        _A_lower_diagonal_blocks_comm = _A_lower_diagonal_blocks
        _A_lower_arrow_blocks_comm = _A_lower_arrow_blocks
        _A_arrow_tip_block_comm = _A_arrow_tip_block

    pobtars: dict = {
        "A_diagonal_blocks": _A_diagonal_blocks,
        "A_lower_diagonal_blocks": _A_lower_diagonal_blocks,
        "A_lower_arrow_blocks": _A_lower_arrow_blocks,
        "A_arrow_tip_block": _A_arrow_tip_block,
        "A_diagonal_blocks_comm": _A_diagonal_blocks_comm,
        "A_lower_diagonal_blocks_comm": _A_lower_diagonal_blocks_comm,
        "A_lower_arrow_blocks_comm": _A_lower_arrow_blocks_comm,
        "A_arrow_tip_block_comm": _A_arrow_tip_block_comm,
    }

    return pobtars


def map_ppobtax_to_pobtars(
    A_diagonal_blocks: ArrayLike,
    A_lower_diagonal_blocks: ArrayLike,
    A_lower_arrow_blocks: ArrayLike,
    A_arrow_tip_block: ArrayLike,
    _A_diagonal_blocks: ArrayLike,
    _A_lower_diagonal_blocks: ArrayLike,
    _A_lower_arrow_blocks: ArrayLike,
    _A_arrow_tip_block: ArrayLike,
    buffer: ArrayLike,
    strategy: str = "allgather",
):
    """Map the the boundary blocks of the PPOBTAX algorithm to the reduced system.

    Parameters
    ----------
    A_diagonal_blocks : ArrayLike,
        The diagonal blocks of the original system.
    A_lower_diagonal_blocks : ArrayLike,
        The lower diagonal blocks of the original system.
    A_lower_arrow_blocks : ArrayLike,
        The arrow bottom blocks of the original system.
    A_arrow_tip_block : ArrayLike,
        The arrow tip block of the original system.
    _A_diagonal_blocks : ArrayLike,
        The diagonal blocks of the reduced system.
    _A_lower_diagonal_blocks : ArrayLike,
        The lower diagonal blocks of the reduced system.
    _A_lower_arrow_blocks : ArrayLike,
        The arrow bottom blocks of the reduced system.
    _A_arrow_tip_block : ArrayLike,
        The arrow tip block of the reduced system.
    buffer : ArrayLike,
        The permutation buffer used in the PPOBTAX algorithm.
    strategy : str, optional
        Communication strategy to use. (default: "allgather")
    """

    if strategy == "allgather":
        if comm_rank == 0:
            _A_diagonal_blocks[1] = A_diagonal_blocks[-1]
            _A_lower_diagonal_blocks[1] = A_lower_diagonal_blocks[-1]
            _A_lower_arrow_blocks[1] = A_lower_arrow_blocks[-1]
        else:
            _A_diagonal_blocks[2 * comm_rank] = A_diagonal_blocks[0]
            _A_diagonal_blocks[2 * comm_rank + 1] = A_diagonal_blocks[-1]

            _A_lower_diagonal_blocks[2 * comm_rank] = buffer[-1].conj().T
            if comm_rank < comm_size - 1:
                _A_lower_diagonal_blocks[2 * comm_rank + 1] = A_lower_diagonal_blocks[
                    -1
                ]

            _A_lower_arrow_blocks[2 * comm_rank] = A_lower_arrow_blocks[0]
            _A_lower_arrow_blocks[2 * comm_rank + 1] = A_lower_arrow_blocks[-1]
        _A_arrow_tip_block[:] = A_arrow_tip_block[:]
    elif strategy == "gather-scatter":
        if comm_rank == 0:
            _A_diagonal_blocks[1] = A_diagonal_blocks[-1]
            _A_lower_diagonal_blocks[1] = A_lower_diagonal_blocks[-1]
            _A_lower_arrow_blocks[1] = A_lower_arrow_blocks[-1]
        else:
            _A_diagonal_blocks[2 * comm_rank] = A_diagonal_blocks[0]
            _A_diagonal_blocks[2 * comm_rank + 1] = A_diagonal_blocks[-1]

            _A_lower_diagonal_blocks[2 * comm_rank] = buffer[-1].conj().T
            if comm_rank < comm_size - 1:
                _A_lower_diagonal_blocks[2 * comm_rank + 1] = A_lower_diagonal_blocks[
                    -1
                ]

            _A_lower_arrow_blocks[2 * comm_rank] = A_lower_arrow_blocks[0]
            _A_lower_arrow_blocks[2 * comm_rank + 1] = A_lower_arrow_blocks[-1]
        _A_arrow_tip_block[:] = A_arrow_tip_block[:]
    else:
        raise ValueError("Unknown communication strategy.")


def aggregate_pobtars(
    pobtars: dict,
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

    _A_diagonal_blocks: ArrayLike = pobtars.get("A_diagonal_blocks", None)
    _A_lower_diagonal_blocks: ArrayLike = pobtars.get("A_lower_diagonal_blocks", None)
    _A_lower_arrow_blocks: ArrayLike = pobtars.get("A_lower_arrow_blocks", None)
    _A_arrow_tip_block: ArrayLike = pobtars.get("A_arrow_tip_block", None)

    _A_diagonal_blocks_comm: ArrayLike = pobtars.get("A_diagonal_blocks_comm", None)
    _A_lower_diagonal_blocks_comm: ArrayLike = pobtars.get(
        "A_lower_diagonal_blocks_comm", None
    )
    _A_lower_arrow_blocks_comm: ArrayLike = pobtars.get(
        "A_lower_arrow_blocks_comm", None
    )
    _A_arrow_tip_block_comm: ArrayLike = pobtars.get("A_arrow_tip_block_comm", None)
    if any(
        x is None
        for x in [
            _A_diagonal_blocks,
            _A_lower_diagonal_blocks,
            _A_lower_arrow_blocks,
            _A_arrow_tip_block,
            _A_diagonal_blocks_comm,
            _A_lower_diagonal_blocks_comm,
            _A_lower_arrow_blocks_comm,
            _A_arrow_tip_block_comm,
        ]
    ):
        raise ValueError(
            "The reduced system `pobtars` doesn't contain the required arrays."
        )

    xp, _ = _get_module_from_array(arr=_A_diagonal_blocks)
    if xp.__name__ == "cupy":
        # We need to move the data of the reduced system from the GPU to the HOST pinned arrays.
        _A_diagonal_blocks.get(out=_A_diagonal_blocks_comm)
        _A_lower_diagonal_blocks.get(out=_A_lower_diagonal_blocks_comm)
        _A_lower_arrow_blocks.get(out=_A_lower_arrow_blocks_comm)
        _A_arrow_tip_block.get(out=_A_arrow_tip_block_comm)

        cpx.cuda.Stream.null.synchronize()

    if strategy == "allgather":
        MPI.COMM_WORLD.Allgather(
            MPI.IN_PLACE,
            _A_diagonal_blocks_comm,
        )
        MPI.COMM_WORLD.Allgather(
            MPI.IN_PLACE,
            _A_lower_diagonal_blocks_comm,
        )
        MPI.COMM_WORLD.Allgather(
            MPI.IN_PLACE,
            _A_lower_arrow_blocks_comm,
        )
        MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, _A_arrow_tip_block_comm, op=MPI.SUM)

        pobtars["A_diagonal_blocks_comm"] = _A_diagonal_blocks_comm[1:]
        pobtars["A_lower_diagonal_blocks_comm"] = _A_lower_diagonal_blocks_comm[1:-1]
        pobtars["A_lower_arrow_blocks_comm"] = _A_lower_arrow_blocks_comm[1:]

        pobtars["A_diagonal_blocks"] = _A_diagonal_blocks[1:]
        pobtars["A_lower_diagonal_blocks"] = _A_lower_diagonal_blocks[1:-1]
        pobtars["A_lower_arrow_blocks"] = _A_lower_arrow_blocks[1:]

    elif strategy == "gather-scatter":
        root = kwargs.get("root", None)
        if root is None:
            raise ValueError(
                "The root rank must be given for gather-scatter communication strategy."
            )

        MPI.COMM_WORLD.Gather(
            sendbuf=(
                _A_diagonal_blocks_comm[2 * comm_rank : 2 * (comm_rank + 1)]
                if comm_rank != root
                else MPI.IN_PLACE
            ),
            recvbuf=_A_diagonal_blocks_comm if comm_rank == root else None,
            root=root,
        )
        MPI.COMM_WORLD.Gather(
            sendbuf=(
                _A_lower_diagonal_blocks_comm[2 * comm_rank : 2 * (comm_rank + 1)]
                if comm_rank != root
                else MPI.IN_PLACE
            ),
            recvbuf=_A_lower_diagonal_blocks_comm if comm_rank == root else None,
            root=root,
        )
        MPI.COMM_WORLD.Gather(
            sendbuf=(
                _A_lower_arrow_blocks_comm[2 * comm_rank : 2 * (comm_rank + 1)]
                if comm_rank != root
                else MPI.IN_PLACE
            ),
            recvbuf=_A_lower_arrow_blocks_comm if comm_rank == root else None,
            root=root,
        )
        MPI.COMM_WORLD.Reduce(
            sendbuf=_A_arrow_tip_block_comm if comm_rank != root else MPI.IN_PLACE,
            recvbuf=_A_arrow_tip_block_comm if comm_rank == root else None,
            op=MPI.SUM,
            root=root,
        )

        pobtars["A_diagonal_blocks_comm"] = _A_diagonal_blocks_comm[1:]
        pobtars["A_lower_diagonal_blocks_comm"] = _A_lower_diagonal_blocks_comm[1:-1]
        pobtars["A_lower_arrow_blocks_comm"] = _A_lower_arrow_blocks_comm[1:]

        pobtars["A_diagonal_blocks"] = _A_diagonal_blocks[1:]
        pobtars["A_lower_diagonal_blocks"] = _A_lower_diagonal_blocks[1:-1]
        pobtars["A_lower_arrow_blocks"] = _A_lower_arrow_blocks[1:]
    else:
        raise ValueError("Unknown communication strategy.")

    MPI.COMM_WORLD.Barrier()

    if xp.__name__ == "cupy":
        # Need to put back the reduced system on the GPU
        _A_diagonal_blocks.set(arr=_A_diagonal_blocks_comm)
        _A_lower_diagonal_blocks.set(arr=_A_lower_diagonal_blocks_comm)
        _A_lower_arrow_blocks.set(arr=_A_lower_arrow_blocks_comm)
        _A_arrow_tip_block.set(arr=_A_arrow_tip_block_comm)

        cpx.cuda.Stream.null.synchronize()


def scatter_pobtars(
    pobtars: dict,
    strategy: str = "allgather",
    **kwargs,
):
    """Scatter the reduced system."""
    _A_diagonal_blocks: ArrayLike = pobtars.get("A_diagonal_blocks", None)
    _A_lower_diagonal_blocks: ArrayLike = pobtars.get("A_lower_diagonal_blocks", None)
    _A_lower_arrow_blocks: ArrayLike = pobtars.get("A_lower_arrow_blocks", None)
    _A_arrow_tip_block: ArrayLike = pobtars.get("A_arrow_tip_block", None)

    _A_diagonal_blocks_comm: ArrayLike = pobtars.get("A_diagonal_blocks_comm", None)
    _A_lower_diagonal_blocks_comm: ArrayLike = pobtars.get(
        "A_lower_diagonal_blocks_comm", None
    )
    _A_lower_arrow_blocks_comm: ArrayLike = pobtars.get(
        "A_lower_arrow_blocks_comm", None
    )
    _A_arrow_tip_block_comm: ArrayLike = pobtars.get("A_arrow_tip_block_comm", None)
    if any(
        x is None
        for x in [
            _A_diagonal_blocks,
            _A_lower_diagonal_blocks,
            _A_lower_arrow_blocks,
            _A_arrow_tip_block,
            _A_diagonal_blocks_comm,
            _A_lower_diagonal_blocks_comm,
            _A_lower_arrow_blocks_comm,
            _A_arrow_tip_block_comm,
        ]
    ):
        raise ValueError(
            "The reduced system `pobtars` doesn't contain the required arrays."
        )

    xp, _ = _get_module_from_array(arr=_A_diagonal_blocks)
    if strategy == "allgather":
        ...
    elif strategy == "gather-scatter":
        root = kwargs.get("root", None)
        if root is None:
            raise ValueError(
                "The root rank must be given for gather-scatter communication strategy."
            )

        if xp.__name__ == "cupy":
            if comm_rank == root:
                # If cupy array, need to move the data to host before initiating the communications
                _A_diagonal_blocks.get(out=_A_diagonal_blocks_comm)
                _A_lower_diagonal_blocks.get(out=_A_lower_diagonal_blocks_comm)
                _A_lower_arrow_blocks.get(out=_A_lower_arrow_blocks_comm)
                _A_arrow_tip_block.get(out=_A_arrow_tip_block_comm)

                cpx.cuda.Stream.null.synchronize()

        MPI.COMM_WORLD.Scatter(
            sendbuf=_A_diagonal_blocks_comm if comm_rank == root else None,
            recvbuf=(
                _A_diagonal_blocks_comm[2 * comm_rank : 2 * (comm_rank + 1)]
                if comm_rank != root
                else MPI.IN_PLACE
            ),
            root=root,
        )

        MPI.COMM_WORLD.Gather(
            sendbuf=(
                _A_diagonal_blocks_comm[2 * comm_rank : 2 * (comm_rank + 1)]
                if comm_rank != root
                else MPI.IN_PLACE
            ),
            recvbuf=_A_diagonal_blocks_comm if comm_rank == root else None,
            root=root,
        )

        MPI.COMM_WORLD.Scatter(
            sendbuf=_A_lower_diagonal_blocks_comm if comm_rank == root else None,
            recvbuf=(
                _A_lower_diagonal_blocks_comm[2 * comm_rank : 2 * (comm_rank + 1)]
                if comm_rank != root
                else MPI.IN_PLACE
            ),
            root=root,
        )
        MPI.COMM_WORLD.Scatter(
            sendbuf=_A_lower_arrow_blocks_comm if comm_rank == root else None,
            recvbuf=(
                _A_lower_arrow_blocks_comm[2 * comm_rank : 2 * (comm_rank + 1)]
                if comm_rank != root
                else MPI.IN_PLACE
            ),
            root=root,
        )
        MPI.COMM_WORLD.Bcast(
            buf=_A_arrow_tip_block_comm,
            root=root,
        )

        if xp.__name__ == "cupy":
            # Need to put back the reduced system on the GPU
            _A_diagonal_blocks.set(arr=_A_diagonal_blocks_comm)
            _A_lower_diagonal_blocks.set(arr=_A_lower_diagonal_blocks_comm)
            _A_lower_arrow_blocks.set(arr=_A_lower_arrow_blocks_comm)
            _A_arrow_tip_block.set(arr=_A_arrow_tip_block_comm)

            cpx.cuda.Stream.null.synchronize()

    else:
        raise ValueError("Unknown communication strategy.")

    # MPI.COMM_WORLD.Barrier()


def map_pobtars_to_ppobtax(
    A_diagonal_blocks: ArrayLike,
    A_lower_diagonal_blocks: ArrayLike,
    A_lower_arrow_blocks: ArrayLike,
    A_arrow_tip_block: ArrayLike,
    _A_diagonal_blocks: ArrayLike,
    _A_lower_diagonal_blocks: ArrayLike,
    _A_lower_arrow_blocks: ArrayLike,
    _A_arrow_tip_block: ArrayLike,
    strategy: str = "allgather",
    **kwargs,
):
    """Map the reduced system back to the original system.

    Parameters
    ----------
    A_diagonal_blocks : ArrayLike
        The diagonal blocks of the original system.
    A_lower_diagonal_blocks : ArrayLike
        The lower diagonal blocks of the original system.
    A_lower_arrow_blocks : ArrayLike
        The arrow bottom blocks of the original system.
    A_arrow_tip_block : ArrayLike
        The arrow tip block of the original system.
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
    buffer: dict = kwargs.get("buffer", None)

    if strategy == "allgather":
        if comm_rank == 0:
            A_diagonal_blocks[-1] = _A_diagonal_blocks[0]
            A_lower_diagonal_blocks[-1] = _A_lower_diagonal_blocks[0]
            A_lower_arrow_blocks[-1] = _A_lower_arrow_blocks[0]
        else:
            A_diagonal_blocks[0] = _A_diagonal_blocks[2 * comm_rank - 1]
            A_diagonal_blocks[-1] = _A_diagonal_blocks[2 * comm_rank]

            buffer[-1] = _A_lower_diagonal_blocks[2 * comm_rank - 1].conj().T
            if comm_rank != comm_size - 1:
                A_lower_diagonal_blocks[-1] = _A_lower_diagonal_blocks[2 * comm_rank]

            A_lower_arrow_blocks[0] = _A_lower_arrow_blocks[2 * comm_rank - 1]
            A_lower_arrow_blocks[-1] = _A_lower_arrow_blocks[2 * comm_rank]
        A_arrow_tip_block[:] = _A_arrow_tip_block[:]
    elif strategy == "gather-scatter":
        if comm_rank == 0:
            A_diagonal_blocks[-1] = _A_diagonal_blocks[1]
            A_lower_diagonal_blocks[-1] = _A_lower_diagonal_blocks[1]
            A_lower_arrow_blocks[-1] = _A_lower_arrow_blocks[1]
        else:
            A_diagonal_blocks[0] = _A_diagonal_blocks[2 * comm_rank]
            A_diagonal_blocks[-1] = _A_diagonal_blocks[2 * comm_rank + 1]

            buffer[-1] = _A_lower_diagonal_blocks[2 * comm_rank].conj().T
            if comm_rank < comm_size - 1:
                A_lower_diagonal_blocks[-1] = _A_lower_diagonal_blocks[
                    2 * comm_rank + 1
                ]

            A_lower_arrow_blocks[0] = _A_lower_arrow_blocks[2 * comm_rank]
            A_lower_arrow_blocks[-1] = _A_lower_arrow_blocks[2 * comm_rank + 1]
        A_arrow_tip_block[:] = _A_arrow_tip_block[:]
    else:
        raise ValueError("Unknown communication strategy.")
