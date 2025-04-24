# Copyright 2023-2025 ETH Zurich. All rights reserved.

from mpi4py import MPI

from serinv import (
    ArrayLike,
    backend_flags,
    _get_module_from_str,
    _get_module_from_array,
    _use_nccl,
    _get_nccl_parameters,
)

if backend_flags["cupy_avail"]:
    import cupyx as cpx
    import cupy as cp


def allocate_pobtars(
    A_diagonal_blocks: ArrayLike,
    A_lower_diagonal_blocks: ArrayLike,
    A_lower_arrow_blocks: ArrayLike,
    A_arrow_tip_block: ArrayLike,
    array_module: str,
    comm: MPI.Comm,
    B: ArrayLike = None,
    device_streaming: bool = False,
    strategy: str = "allgather",
    nccl_comm: object = None,
):
    """Allocate the buffers necessary for the reduced system of the PPOBTARX algorithms.

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
    pobtars : dict
        Dictionary containing the reduced system arrays.
    """
    communicator = None
    if nccl_comm is not None:
        communicator = nccl_comm
    else:
        communicator = comm

    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    xp, _ = _get_module_from_str(array_module)

    b = A_diagonal_blocks[0].shape[0]
    a = A_arrow_tip_block.shape[0]
    if B is not None:
        n_rhs = B.shape[1]
    else:
        _B = None
        _B_comm = None
    dtype = A_diagonal_blocks.dtype

    if device_streaming:
        zeros = cpx.zeros_pinned
        empty = cpx.empty_pinned
    else:
        zeros = xp.zeros
        empty = xp.empty

    if strategy == "allgather":
        _n: int = 2 * comm_size
        alloc = empty

        _A_diagonal_blocks = alloc((_n, b, b), dtype=dtype)
        _A_lower_diagonal_blocks = alloc((_n, b, b), dtype=dtype)
        _A_lower_arrow_blocks = alloc((_n, a, b), dtype=dtype)

        if B is not None:
            _B = alloc((_n * b + a, n_rhs), dtype=dtype)
    elif strategy == "gather-scatter":
        _n: int = 2 * comm_size
        alloc = zeros

        _A_diagonal_blocks = alloc((_n, b, b), dtype=dtype)
        _A_lower_diagonal_blocks = alloc((_n, b, b), dtype=dtype)
        _A_lower_arrow_blocks = alloc((_n, a, b), dtype=dtype)

        if B is not None:
            _B = alloc((_n * b + a, n_rhs), dtype=dtype)
    else:
        raise ValueError("Unknown communication strategy.")

    _A_arrow_tip_block = zeros((a, a), dtype=dtype)

    # If needed, allocate the reduced system for communication
    if (
        xp.__name__ == "cupy"
        and not backend_flags["mpi_cuda_aware"]
        and not _use_nccl(communicator)
    ):
        _A_diagonal_blocks_comm = cpx.empty_like_pinned(_A_diagonal_blocks)
        _A_lower_diagonal_blocks_comm = cpx.empty_like_pinned(_A_lower_diagonal_blocks)
        _A_lower_arrow_blocks_comm = cpx.empty_like_pinned(_A_lower_arrow_blocks)
        _A_arrow_tip_block_comm = cpx.empty_like_pinned(_A_arrow_tip_block)

        if B is not None:
            _B_comm = cpx.empty_like_pinned(_B)
    else:
        _A_diagonal_blocks_comm = _A_diagonal_blocks
        _A_lower_diagonal_blocks_comm = _A_lower_diagonal_blocks
        _A_lower_arrow_blocks_comm = _A_lower_arrow_blocks
        _A_arrow_tip_block_comm = _A_arrow_tip_block

        if B is not None:
            _B_comm = _B

    pobtars: dict = {
        "A_diagonal_blocks": _A_diagonal_blocks,
        "A_lower_diagonal_blocks": _A_lower_diagonal_blocks,
        "A_lower_arrow_blocks": _A_lower_arrow_blocks,
        "A_arrow_tip_block": _A_arrow_tip_block,
        "B": _B,
        "A_diagonal_blocks_comm": _A_diagonal_blocks_comm,
        "A_lower_diagonal_blocks_comm": _A_lower_diagonal_blocks_comm,
        "A_lower_arrow_blocks_comm": _A_lower_arrow_blocks_comm,
        "A_arrow_tip_block_comm": _A_arrow_tip_block_comm,
        "B_comm": _B_comm,
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
    comm: MPI.Comm,
    strategy: str = "allgather",
    nccl_comm: object = None,
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
    communicator = None
    if nccl_comm is not None:
        communicator = nccl_comm
    else:
        communicator = comm

    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

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


def map_ppobtas_to_pobtarss(
    A_diagonal_blocks: ArrayLike,
    A_arrow_tip_block: ArrayLike,
    B: ArrayLike,
    _B: ArrayLike,
    comm: MPI.Comm,
    strategy: str = "allgather",
    nccl_comm: object = None,
):
    """Map the right-hand side of the PPOBTAS algorithm to the right-hand-side
    of the reduced system."""
    communicator = None
    if nccl_comm is not None:
        communicator = nccl_comm
    else:
        communicator = comm

    comm_rank = comm.Get_rank()

    b = A_diagonal_blocks[0].shape[0]
    a = A_arrow_tip_block.shape[0]

    if strategy == "allgather":
        if comm_rank == 0:
            _B[b : 2 * b] = B[-b - a : -a]
            _B[-a:] = B[-a:]
        else:
            _B[2 * comm_rank * b : 2 * comm_rank * b + b] = B[:b]
            _B[2 * comm_rank * b + b : 2 * (comm_rank + 1) * b] = B[-b - a : -a]
            _B[-a:] = B[-a:]
    else:
        raise ValueError(f"Unknown communication strategy: {strategy}.")


def aggregate_pobtars(
    pobtars: dict,
    comm: MPI.Comm,
    strategy: str = "allgather",
    nccl_comm: object = None,
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
    communicator = None
    if nccl_comm is not None:
        communicator = nccl_comm
    else:
        communicator = comm

    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

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

    if (
        xp.__name__ == "cupy"
        and not backend_flags["mpi_cuda_aware"]
        and not _use_nccl(communicator)
    ):
        # We need to move the data of the reduced system from the GPU to the HOST pinned arrays.
        _A_diagonal_blocks.get(out=_A_diagonal_blocks_comm)
        _A_lower_diagonal_blocks.get(out=_A_lower_diagonal_blocks_comm)
        _A_lower_arrow_blocks.get(out=_A_lower_arrow_blocks_comm)
        _A_arrow_tip_block.get(out=_A_arrow_tip_block_comm)

        cp.cuda.runtime.deviceSynchronize()

    if strategy == "allgather":
        if _use_nccl(communicator):
            # --- Use NCCL ---
            cp.cuda.runtime.deviceSynchronize()
            count, displacement, datatype = _get_nccl_parameters(
                arr=_A_diagonal_blocks_comm, comm=communicator, rank=comm_rank, op="allgather"
            )
            communicator.allGather(
                sendbuf=_A_diagonal_blocks_comm.data.ptr + displacement,
                recvbuf=_A_diagonal_blocks_comm.data.ptr,
                count=count,
                datatype=datatype,
                stream=cp.cuda.Stream.null.ptr,
            )
            count, displacement, datatype = _get_nccl_parameters(
                arr=_A_lower_diagonal_blocks_comm, comm=communicator, rank=comm_rank, op="allgather"
            )
            communicator.allGather(
                sendbuf=_A_lower_diagonal_blocks_comm.data.ptr + displacement,
                recvbuf=_A_lower_diagonal_blocks_comm.data.ptr,
                count=count,
                datatype=datatype,
                stream=cp.cuda.Stream.null.ptr,
            )
            count, displacement, datatype = _get_nccl_parameters(
                arr=_A_lower_arrow_blocks_comm, comm=communicator, rank=comm_rank, op="allgather"
            )
            communicator.allGather(
                sendbuf=_A_lower_arrow_blocks_comm.data.ptr + displacement,
                recvbuf=_A_lower_arrow_blocks_comm.data.ptr,
                count=count,
                datatype=datatype,
                stream=cp.cuda.Stream.null.ptr,
            )
            count, displacement, datatype = _get_nccl_parameters(
                arr=_A_arrow_tip_block_comm, comm=communicator, rank=comm_rank, op="allreduce"
            )
            communicator.allReduce(
                sendbuf=_A_arrow_tip_block_comm.data.ptr,
                recvbuf=_A_arrow_tip_block_comm.data.ptr,
                count=count,
                datatype=datatype,
                op=cp.cuda.nccl.NCCL_SUM,
                stream=cp.cuda.Stream.null.ptr,
            )
            cp.cuda.runtime.deviceSynchronize()
            comm.Barrier()
        else:
            # --- Use MPI ---
            comm.Allgather(
                MPI.IN_PLACE,
                _A_diagonal_blocks_comm,
            )
            comm.Allgather(
                MPI.IN_PLACE,
                _A_lower_diagonal_blocks_comm,
            )
            comm.Allgather(
                MPI.IN_PLACE,
                _A_lower_arrow_blocks_comm,
            )
            comm.Allreduce(MPI.IN_PLACE, _A_arrow_tip_block_comm, op=MPI.SUM)
    elif strategy == "gather-scatter":
        if _use_nccl(communicator):
            raise ValueError(
                "NCCL is not supported for gather-scatter communication strategy."
            )

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
        comm.Gather(
            sendbuf=(
                _A_lower_arrow_blocks_comm[2 * comm_rank : 2 * (comm_rank + 1)]
                if comm_rank != root
                else MPI.IN_PLACE
            ),
            recvbuf=_A_lower_arrow_blocks_comm if comm_rank == root else None,
            root=root,
        )
        comm.Reduce(
            sendbuf=_A_arrow_tip_block_comm if comm_rank != root else MPI.IN_PLACE,
            recvbuf=_A_arrow_tip_block_comm if comm_rank == root else None,
            op=MPI.SUM,
            root=root,
        )
    else:
        raise ValueError("Unknown communication strategy.")

    pobtars["A_diagonal_blocks_comm"] = _A_diagonal_blocks_comm
    pobtars["A_lower_diagonal_blocks_comm"] = _A_lower_diagonal_blocks_comm
    pobtars["A_lower_arrow_blocks_comm"] = _A_lower_arrow_blocks_comm

    pobtars["A_diagonal_blocks"] = _A_diagonal_blocks
    pobtars["A_lower_diagonal_blocks"] = _A_lower_diagonal_blocks
    pobtars["A_lower_arrow_blocks"] = _A_lower_arrow_blocks

    if (
        xp.__name__ == "cupy"
        and not backend_flags["mpi_cuda_aware"]
        and not _use_nccl(communicator)
    ):
        # Need to put back the reduced system on the GPU
        _A_diagonal_blocks.set(arr=_A_diagonal_blocks_comm)
        _A_lower_diagonal_blocks.set(arr=_A_lower_diagonal_blocks_comm)
        _A_lower_arrow_blocks.set(arr=_A_lower_arrow_blocks_comm)
        _A_arrow_tip_block.set(arr=_A_arrow_tip_block_comm)

        cp.cuda.runtime.deviceSynchronize()


def aggregate_pobtarss(
    A_diagonal_blocks: ArrayLike,
    A_arrow_tip_block: ArrayLike,
    pobtars: dict,
    comm: MPI.Comm,
    strategy: str = "allgather",
    nccl_comm: object = None,
    **kwargs,
):
    communicator = None
    if nccl_comm is not None:
        communicator = nccl_comm
    else:
        communicator = comm

    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    b = A_diagonal_blocks[0].shape[0]
    a = A_arrow_tip_block.shape[0]

    _B: ArrayLike = pobtars.get("B", None)
    _B_comm: ArrayLike = pobtars.get("B_comm", None)
    if any(
        x is None
        for x in [
            _B,
            _B_comm,
        ]
    ):
        raise ValueError(
            "The reduced system `pobtars` doesn't contain the required arrays."
        )

    xp, _ = _get_module_from_array(arr=_B)
    if (
        xp.__name__ == "cupy"
        and not backend_flags["mpi_cuda_aware"]
        and not _use_nccl(communicator)
    ):
        # We need to move the data of the reduced system from the GPU to the
        # HOST pinned arrays.
        _B.get(out=_B_comm)

        cp.cuda.runtime.deviceSynchronize()

    if strategy == "allgather":
        if _use_nccl(comm):
            # --- Use NCCL ---
            count, displacement, datatype = _get_nccl_parameters(
                arr=_B_comm[:-a], comm=communicator, rank=comm_rank, op="allgather"
            )
            communicator.allGather(
                sendbuf=_B_comm[:-a].data.ptr + displacement,
                recvbuf=_B_comm[:-a].data.ptr,
                count=count,
                datatype=datatype,
                stream=cp.cuda.Stream.null.ptr,
            )
            count, displacement, datatype = _get_nccl_parameters(
                arr=_B_comm[-a:], comm=communicator, rank=comm_rank, op="allreduce"
            )
            communicator.allReduce(
                sendbuf=_B_comm[-a:].data.ptr,
                recvbuf=_B_comm[-a:].data.ptr,
                count=count,
                datatype=datatype,
                op=cp.cuda.nccl.NCCL_SUM,
                stream=cp.cuda.Stream.null.ptr,
            )
        else:
            # --- Use MPI ---
            comm.Allgather(
                MPI.IN_PLACE,
                _B_comm[:-a],
            )
            comm.Allreduce(MPI.IN_PLACE, _B_comm[-a:], op=MPI.SUM)
    elif strategy == "gather-scatter":
        if _use_nccl(communicator):
            raise ValueError(
                "NCCL is not supported for gather-scatter communication strategy."
            )

        root = kwargs.get("root", None)
        if root is None:
            raise ValueError(
                "The root rank must be given for gather-scatter communication strategy."
            )

        comm.Gather(
            sendbuf=(
                _B_comm[2 * comm_rank : 2 * (comm_rank + 1)]
                if comm_rank != 0
                else MPI.IN_PLACE
            ),
            recvbuf=_B_comm if comm_rank == 0 else None,
            root=0,
        )
        comm.Reduce(
            sendbuf=_B_comm[-a:] if comm_rank != 0 else MPI.IN_PLACE,
            recvbuf=_B_comm[-a:] if comm_rank == 0 else None,
            op=MPI.SUM,
            root=0,
        )
    else:
        raise ValueError("Unknown communication strategy.")

    pobtars["B_comm"] = _B_comm
    pobtars["B"] = _B

    if (
        xp.__name__ == "cupy"
        and not backend_flags["mpi_cuda_aware"]
        and not _use_nccl(communicator)
    ):
        # Need to put back the reduced system RHS on the GPU
        _B.set(arr=_B_comm)

        cp.cuda.runtime.deviceSynchronize()


def scatter_pobtars(
    pobtars: dict,
    comm: MPI.Comm,
    strategy: str = "allgather",
    nccl_comm: object = None,
    **kwargs,
):
    """Scatter the reduced system."""
    communicator = None
    if nccl_comm is not None:
        communicator = nccl_comm
    else:
        communicator = comm

    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

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
        if _use_nccl(communicator):
            raise ValueError(
                "NCCL is not supported for gather-scatter communication strategy."
            )
        
        root = kwargs.get("root", None)
        if root is None:
            raise ValueError(
                "The root rank must be given for gather-scatter communication strategy."
            )

        if (
            xp.__name__ == "cupy"
            and not backend_flags["mpi_cuda_aware"]
            and not _use_nccl(communicator)
        ):
            if comm_rank == root:
                # If cupy array, need to move the data to host before initiating the communications
                _A_diagonal_blocks.get(out=_A_diagonal_blocks_comm)
                _A_lower_diagonal_blocks.get(out=_A_lower_diagonal_blocks_comm)
                _A_lower_arrow_blocks.get(out=_A_lower_arrow_blocks_comm)
                _A_arrow_tip_block.get(out=_A_arrow_tip_block_comm)
            cp.cuda.runtime.deviceSynchronize()

        comm.Scatter(
            sendbuf=_A_diagonal_blocks_comm if comm_rank == root else None,
            recvbuf=(
                _A_diagonal_blocks_comm[2 * comm_rank : 2 * (comm_rank + 1)]
                if comm_rank != root
                else MPI.IN_PLACE
            ),
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
        comm.Scatter(
            sendbuf=_A_lower_arrow_blocks_comm if comm_rank == root else None,
            recvbuf=(
                _A_lower_arrow_blocks_comm[2 * comm_rank : 2 * (comm_rank + 1)]
                if comm_rank != root
                else MPI.IN_PLACE
            ),
            root=root,
        )
        comm.Bcast(
            buf=_A_arrow_tip_block_comm,
            root=root,
        )

        if (
            xp.__name__ == "cupy"
            and not backend_flags["mpi_cuda_aware"]
            and not _use_nccl(communicator)
        ):
            # Need to put back the reduced system on the GPU
            _A_diagonal_blocks.set(arr=_A_diagonal_blocks_comm)
            _A_lower_diagonal_blocks.set(arr=_A_lower_diagonal_blocks_comm)
            _A_lower_arrow_blocks.set(arr=_A_lower_arrow_blocks_comm)
            _A_arrow_tip_block.set(arr=_A_arrow_tip_block_comm)
            cp.cuda.runtime.deviceSynchronize()

    else:
        raise ValueError("Unknown communication strategy.")


def scatter_pobtarss(
    A_diagonal_blocks: ArrayLike,
    A_arrow_tip_block: ArrayLike,
    pobtars: dict,
    comm: MPI.Comm,
    strategy: str = "allgather",
    nccl_comm: object = None,
    **kwargs,
):
    communicator = None
    if nccl_comm is not None:
        communicator = nccl_comm
    else:
        communicator = comm

    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    b = A_diagonal_blocks[0].shape[0]
    a = A_arrow_tip_block.shape[0]

    _B: ArrayLike = pobtars.get("B", None)
    _B_comm: ArrayLike = pobtars.get("B_comm", None)
    if any(
        x is None
        for x in [
            _B,
            _B_comm,
        ]
    ):
        raise ValueError(
            "The reduced system `pobtars` doesn't contain the required arrays."
        )

    xp, _ = _get_module_from_array(arr=_B)
    if strategy == "allgather":
        ...
    elif strategy == "gather-scatter":
        if _use_nccl(communicator):
            raise ValueError(
                "NCCL is not supported for gather-scatter communication strategy."
            )

        root = kwargs.get("root", None)
        if root is None:
            raise ValueError(
                "The root rank must be given for gather-scatter communication strategy."
            )
        if (
            xp.__name__ == "cupy"
            and not backend_flags["mpi_cuda_aware"]
            and not _use_nccl(communicator)
        ):
            if comm_rank == root:
                # If cupy array, need to move the data to host before initiating the communications
                _B.get(out=_B_comm)
            cp.cuda.runtime.deviceSynchronize()

        comm.Scatter(
            sendbuf=_B_comm if comm_rank == root else None,
            recvbuf=(
                _B_comm[2 * comm_rank * b : 2 * (comm_rank + 1) * b]
                if comm_rank != root
                else MPI.IN_PLACE
            ),
            root=root,
        )
        comm.Bcast(
            buf=_B_comm[-a:],
            root=root,
        )

        if (
            xp.__name__ == "cupy"
            and not backend_flags["mpi_cuda_aware"]
            and not _use_nccl(communicator)
        ):
            # Need to put back the reduced system on the GPU
            _B.set(arr=_B_comm)
            cp.cuda.runtime.deviceSynchronize()
    else:
        raise ValueError("Unknown communication strategy.")


def map_pobtars_to_ppobtax(
    A_diagonal_blocks: ArrayLike,
    A_lower_diagonal_blocks: ArrayLike,
    A_lower_arrow_blocks: ArrayLike,
    A_arrow_tip_block: ArrayLike,
    _A_diagonal_blocks: ArrayLike,
    _A_lower_diagonal_blocks: ArrayLike,
    _A_lower_arrow_blocks: ArrayLike,
    _A_arrow_tip_block: ArrayLike,
    comm: MPI.Comm,
    strategy: str = "allgather",
    nccl_comm: object = None,
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
    communicator = None
    if nccl_comm is not None:
        communicator = nccl_comm
    else:
        communicator = comm

    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    buffer: dict = kwargs.get("buffer", None)

    if strategy == "allgather":
        if comm_rank == 0:
            A_diagonal_blocks[-1] = _A_diagonal_blocks[1]
            A_lower_diagonal_blocks[-1] = _A_lower_diagonal_blocks[1]
            A_lower_arrow_blocks[-1] = _A_lower_arrow_blocks[1]
        else:
            A_diagonal_blocks[0] = _A_diagonal_blocks[2 * comm_rank]
            A_diagonal_blocks[-1] = _A_diagonal_blocks[2 * comm_rank + 1]

            buffer[-1] = _A_lower_diagonal_blocks[2 * comm_rank].conj().T
            if comm_rank != comm_size - 1:
                A_lower_diagonal_blocks[-1] = _A_lower_diagonal_blocks[
                    2 * comm_rank + 1
                ]

            A_lower_arrow_blocks[0] = _A_lower_arrow_blocks[2 * comm_rank]
            A_lower_arrow_blocks[-1] = _A_lower_arrow_blocks[2 * comm_rank + 1]
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


def map_pobtarss_to_ppobtas(
    A_diagonal_blocks: ArrayLike,
    A_arrow_tip_block: ArrayLike,
    B: ArrayLike,
    _B: ArrayLike,
    comm: MPI.Comm,
    strategy: str = "allgather",
    nccl_comm: object = None,
):
    """Map the right-hand side of the PPOBTAS algorithm to the right-hand-side
    of the reduced system."""
    communicator = None
    if nccl_comm is not None:
        communicator = nccl_comm
    else:
        communicator = comm
        
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    b = A_diagonal_blocks[0].shape[0]
    a = A_arrow_tip_block.shape[0]

    if strategy == "allgather" or strategy == "gather-scatter":
        if comm_rank == 0:
            B[-b - a : -a] = _B[b : 2 * b]
            B[-a:] = _B[-a:]
        else:
            B[:b] = _B[2 * comm_rank * b : 2 * comm_rank * b + b]
            B[-b - a : -a] = _B[2 * comm_rank * b + b : 2 * (comm_rank + 1) * b]
            B[-a:] = _B[-a:]
    else:
        raise ValueError("Unknown communication strategy.")
