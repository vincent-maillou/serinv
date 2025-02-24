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


def allocate_pobtrs(
    A_diagonal_blocks: ArrayLike,
    A_lower_diagonal_blocks: ArrayLike,
    comm: MPI.Comm,
    array_module: str,
    device_streaming: bool = False,
    strategy: str = "allgather",
):
    """Allocate the buffers necessary for the reduced system of the PpobtRX algorithms.

    Parameters
    ----------
    A_diagonal_blocks : ArrayLike
        The diagonal blocks of the original system.
    A_lower_diagonal_blocks : ArrayLike
        The lower diagonal blocks of the original system.
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
    pobtrs : dict
        Dictionary containing the reduced system arrays.
    """
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

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
    else:
        raise ValueError("Unknown communication strategy.")

    # If needed, allocate the reduced system for communication
    if xp.__name__ == "cupy":
        _A_diagonal_blocks_comm = cpx.empty_like_pinned(_A_diagonal_blocks)
        _A_lower_diagonal_blocks_comm = cpx.empty_like_pinned(_A_lower_diagonal_blocks)
    else:
        _A_diagonal_blocks_comm = _A_diagonal_blocks
        _A_lower_diagonal_blocks_comm = _A_lower_diagonal_blocks

    pobtrs: dict = {
        "A_diagonal_blocks": _A_diagonal_blocks,
        "A_lower_diagonal_blocks": _A_lower_diagonal_blocks,
        "A_diagonal_blocks_comm": _A_diagonal_blocks_comm,
        "A_lower_diagonal_blocks_comm": _A_lower_diagonal_blocks_comm,
    }

    return pobtrs


def map_ppobtx_to_pobtrs(
    A_diagonal_blocks: ArrayLike,
    A_lower_diagonal_blocks: ArrayLike,
    _A_diagonal_blocks: ArrayLike,
    _A_lower_diagonal_blocks: ArrayLike,
    comm: MPI.Comm,
    buffer: ArrayLike,
    strategy: str = "allgather",
) -> None:
    """Map the the boundary blocks of the PpobtX algorithm to the reduced system.

    Parameters
    ----------
    A_diagonal_blocks : ArrayLike,
        The diagonal blocks of the original system.
    A_lower_diagonal_blocks : ArrayLike,
        The lower diagonal blocks of the original system.
    _A_diagonal_blocks : ArrayLike,
        The diagonal blocks of the reduced system.
    _A_lower_diagonal_blocks : ArrayLike,
        The lower diagonal blocks of the reduced system.
    buffer : ArrayLike,
        The permutation buffer used in the PpobtX algorithm.
    strategy : str, optional
        Communication strategy to use. (default: "allgather")
    """
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    if strategy == "allgather":
        if comm_rank == 0:
            _A_diagonal_blocks[1] = A_diagonal_blocks[-1]
            _A_lower_diagonal_blocks[1] = A_lower_diagonal_blocks[-1]
        else:
            _A_diagonal_blocks[2 * comm_rank] = A_diagonal_blocks[0]
            _A_diagonal_blocks[2 * comm_rank + 1] = A_diagonal_blocks[-1]

            _A_lower_diagonal_blocks[2 * comm_rank] = buffer[-1].conj().T
            if comm_rank < comm_size - 1:
                _A_lower_diagonal_blocks[2 * comm_rank + 1] = A_lower_diagonal_blocks[
                    -1
                ]
    elif strategy == "gather-scatter":
        if comm_rank == 0:
            _A_diagonal_blocks[1] = A_diagonal_blocks[-1]
            _A_lower_diagonal_blocks[1] = A_lower_diagonal_blocks[-1]
        else:
            _A_diagonal_blocks[2 * comm_rank] = A_diagonal_blocks[0]
            _A_diagonal_blocks[2 * comm_rank + 1] = A_diagonal_blocks[-1]

            _A_lower_diagonal_blocks[2 * comm_rank] = buffer[-1].conj().T
            if comm_rank < comm_size - 1:
                _A_lower_diagonal_blocks[2 * comm_rank + 1] = A_lower_diagonal_blocks[
                    -1
                ]
    else:
        raise ValueError("Unknown communication strategy.")


def aggregate_pobtrs(
    pobtrs: dict,
    comm: MPI.Comm,
    strategy: str = "allgather",
    **kwargs,
):
    """Aggregate the reduced system.

    Parameters
    ----------
    pobtrs : dict
        Dictionary containing the reduced system arrays.
    strategy : str, optional
        Communication strategy to use. (default: "allgather")
    """
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    _A_diagonal_blocks: ArrayLike = pobtrs.get("A_diagonal_blocks", None)
    _A_lower_diagonal_blocks: ArrayLike = pobtrs.get("A_lower_diagonal_blocks", None)

    _A_diagonal_blocks_comm: ArrayLike = pobtrs.get("A_diagonal_blocks_comm", None)
    _A_lower_diagonal_blocks_comm: ArrayLike = pobtrs.get(
        "A_lower_diagonal_blocks_comm", None
    )
    if any(
        x is None
        for x in [
            _A_diagonal_blocks,
            _A_lower_diagonal_blocks,
            _A_diagonal_blocks_comm,
            _A_lower_diagonal_blocks_comm,
        ]
    ):
        raise ValueError(
            "The reduced system `pobtrs` doesn't contain the required arrays."
        )

    xp, _ = _get_module_from_array(arr=_A_diagonal_blocks)
    if xp.__name__ == "cupy":
        # We need to move the data of the reduced system from the GPU to the HOST pinned arrays.
        _A_diagonal_blocks.get(out=_A_diagonal_blocks_comm)
        _A_lower_diagonal_blocks.get(out=_A_lower_diagonal_blocks_comm)

        cpx.cuda.Stream.null.synchronize()

    if strategy == "allgather":
        comm.Allgather(
            MPI.IN_PLACE,
            _A_diagonal_blocks_comm,
        )
        comm.Allgather(
            MPI.IN_PLACE,
            _A_lower_diagonal_blocks_comm,
        )

        pobtrs["A_diagonal_blocks_comm"] = _A_diagonal_blocks_comm[1:]
        pobtrs["A_lower_diagonal_blocks_comm"] = _A_lower_diagonal_blocks_comm[1:-1]

        pobtrs["A_diagonal_blocks"] = _A_diagonal_blocks[1:]
        pobtrs["A_lower_diagonal_blocks"] = _A_lower_diagonal_blocks[1:-1]

    elif strategy == "gather-scatter":
        root = kwargs.get("root", None)
        if root is None:
            raise ValueError(
                "The root rank must be given for gather-scatter communication strategy."
            )

        comm.Gather(
            sendbuf=(
                _A_diagonal_blocks_comm[2 * comm_rank : 2 * (comm_rank + 1)]
                if comm_rank != root
                else MPI.IN_PLACE
            ),
            recvbuf=_A_diagonal_blocks_comm if comm_rank == root else None,
            root=root,
        )
        comm.Gather(
            sendbuf=(
                _A_lower_diagonal_blocks_comm[2 * comm_rank : 2 * (comm_rank + 1)]
                if comm_rank != root
                else MPI.IN_PLACE
            ),
            recvbuf=_A_lower_diagonal_blocks_comm if comm_rank == root else None,
            root=root,
        )

        # Do not slice/view the array here in the gather-scatter strategy, otherwise
        # the scatter-back won't work.
        pobtrs["A_diagonal_blocks_comm"] = _A_diagonal_blocks_comm
        pobtrs["A_lower_diagonal_blocks_comm"] = _A_lower_diagonal_blocks_comm

        pobtrs["A_diagonal_blocks"] = _A_diagonal_blocks
        pobtrs["A_lower_diagonal_blocks"] = _A_lower_diagonal_blocks
    else:
        raise ValueError("Unknown communication strategy.")

    if xp.__name__ == "cupy":
        # Need to put back the reduced system on the GPU
        _A_diagonal_blocks.set(arr=_A_diagonal_blocks_comm)
        _A_lower_diagonal_blocks.set(arr=_A_lower_diagonal_blocks_comm)

        cpx.cuda.Stream.null.synchronize()


def scatter_pobtrs(
    pobtrs: dict,
    comm: MPI.Comm,
    strategy: str = "allgather",
    **kwargs,
):
    """Scatter the reduced system."""
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    _A_diagonal_blocks: ArrayLike = pobtrs.get("A_diagonal_blocks", None)
    _A_lower_diagonal_blocks: ArrayLike = pobtrs.get("A_lower_diagonal_blocks", None)

    _A_diagonal_blocks_comm: ArrayLike = pobtrs.get("A_diagonal_blocks_comm", None)
    _A_lower_diagonal_blocks_comm: ArrayLike = pobtrs.get(
        "A_lower_diagonal_blocks_comm", None
    )
    if any(
        x is None
        for x in [
            _A_diagonal_blocks,
            _A_lower_diagonal_blocks,
            _A_diagonal_blocks_comm,
            _A_lower_diagonal_blocks_comm,
        ]
    ):
        raise ValueError(
            "The reduced system `pobtrs` doesn't contain the required arrays."
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

            cpx.cuda.Stream.null.synchronize()

        comm.Scatter(
            sendbuf=_A_diagonal_blocks_comm if comm_rank == root else None,
            recvbuf=(
                _A_diagonal_blocks_comm[2 * comm_rank : 2 * (comm_rank + 1)]
                if comm_rank != root
                else MPI.IN_PLACE
            ),
            root=root,
        )

        comm.Gather(
            sendbuf=(
                _A_diagonal_blocks_comm[2 * comm_rank : 2 * (comm_rank + 1)]
                if comm_rank != root
                else MPI.IN_PLACE
            ),
            recvbuf=_A_diagonal_blocks_comm if comm_rank == root else None,
            root=root,
        )

        comm.Scatter(
            sendbuf=_A_lower_diagonal_blocks_comm if comm_rank == root else None,
            recvbuf=(
                _A_lower_diagonal_blocks_comm[2 * comm_rank : 2 * (comm_rank + 1)]
                if comm_rank != root
                else MPI.IN_PLACE
            ),
            root=root,
        )

        if xp.__name__ == "cupy":
            # Need to put back the reduced system on the GPU
            _A_diagonal_blocks.set(arr=_A_diagonal_blocks_comm)
            _A_lower_diagonal_blocks.set(arr=_A_lower_diagonal_blocks_comm)

            cpx.cuda.Stream.null.synchronize()

    else:
        raise ValueError("Unknown communication strategy.")


def map_pobtrs_to_ppobtx(
    A_diagonal_blocks: ArrayLike,
    A_lower_diagonal_blocks: ArrayLike,
    _A_diagonal_blocks: ArrayLike,
    _A_lower_diagonal_blocks: ArrayLike,
    comm: MPI.Comm,
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
    _L_diagonal_blocks : ArrayLike
        The diagonal blocks of the reduced system.
    _L_lower_diagonal_blocks : ArrayLike
        The lower diagonal blocks of the reduced system.
    strategy : str, optional
        Communication strategy to use. (default: "allgather")
    """
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    buffer: dict = kwargs.get("buffer", None)

    if strategy == "allgather":
        if comm_rank == 0:
            A_diagonal_blocks[-1] = _A_diagonal_blocks[0]
            A_lower_diagonal_blocks[-1] = _A_lower_diagonal_blocks[0]
        else:
            A_diagonal_blocks[0] = _A_diagonal_blocks[2 * comm_rank - 1]
            A_diagonal_blocks[-1] = _A_diagonal_blocks[2 * comm_rank]

            buffer[-1] = _A_lower_diagonal_blocks[2 * comm_rank - 1].conj().T
            if comm_rank != comm_size - 1:
                A_lower_diagonal_blocks[-1] = _A_lower_diagonal_blocks[2 * comm_rank]
    elif strategy == "gather-scatter":
        if comm_rank == 0:
            A_diagonal_blocks[-1] = _A_diagonal_blocks[1]
            A_lower_diagonal_blocks[-1] = _A_lower_diagonal_blocks[1]
        else:
            A_diagonal_blocks[0] = _A_diagonal_blocks[2 * comm_rank]
            A_diagonal_blocks[-1] = _A_diagonal_blocks[2 * comm_rank + 1]

            buffer[-1] = _A_lower_diagonal_blocks[2 * comm_rank].conj().T
            if comm_rank < comm_size - 1:
                A_lower_diagonal_blocks[-1] = _A_lower_diagonal_blocks[
                    2 * comm_rank + 1
                ]
    else:
        raise ValueError("Unknown communication strategy.")
