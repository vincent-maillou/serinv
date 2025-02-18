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


def allocate_ddbtrs(
    A_diagonal_blocks: ArrayLike,
    A_lower_diagonal_blocks: ArrayLike,
    A_upper_diagonal_blocks: ArrayLike,
    comm_size: int,
    array_module: str,
    strategy: str = "allgather",
    quadratic: bool = False,
):
    xp, _ = _get_module_from_str(array_module)

    if strategy == "allgather":
        _n: int = 2 * comm_size

        _A_diagonal_blocks = xp.empty(
            (_n, A_diagonal_blocks[0].shape[0], A_diagonal_blocks[0].shape[1]),
            dtype=A_diagonal_blocks.dtype,
        )
        _A_lower_diagonal_blocks = xp.empty(
            (
                _n,
                A_lower_diagonal_blocks[0].shape[0],
                A_lower_diagonal_blocks[0].shape[1],
            ),
            dtype=A_lower_diagonal_blocks.dtype,
        )
        _A_upper_diagonal_blocks = xp.empty(
            (
                _n,
                A_upper_diagonal_blocks[0].shape[0],
                A_upper_diagonal_blocks[0].shape[1],
            ),
            dtype=A_upper_diagonal_blocks.dtype,
        )

        if xp.__name__ == 'cupy':
            # In this case we also need to allocate a pinned-memory
            # reduced system on the host side.
            _A_diagonal_blocks_comm = cpx.empty_like_pinned(_A_diagonal_blocks)
            _A_lower_diagonal_blocks_comm = cpx.empty_like_pinned(_A_lower_diagonal_blocks)
            _A_upper_diagonal_blocks_comm = cpx.empty_like_pinned(_A_upper_diagonal_blocks)
        else:
            _A_diagonal_blocks_comm = _A_diagonal_blocks
            _A_lower_diagonal_blocks_comm = _A_lower_diagonal_blocks
            _A_upper_diagonal_blocks_comm = _A_upper_diagonal_blocks

        if quadratic:
            _B_diagonal_blocks = xp.empty(
                (_n, A_diagonal_blocks[0].shape[0], A_diagonal_blocks[0].shape[1]),
                dtype=A_diagonal_blocks.dtype,
            )
            _B_lower_diagonal_blocks = xp.empty(
                (
                    _n,
                    A_lower_diagonal_blocks[0].shape[0],
                    A_lower_diagonal_blocks[0].shape[1],
                ),
                dtype=A_lower_diagonal_blocks.dtype,
            )
            _B_upper_diagonal_blocks = xp.empty(
                (
                    _n,
                    A_upper_diagonal_blocks[0].shape[0],
                    A_upper_diagonal_blocks[0].shape[1],
                ),
                dtype=A_upper_diagonal_blocks.dtype,
            )

            if xp.__name__ == 'cupy':
                # In this case we also need to allocate a pinned-memory
                # reduced system on the host side.

                _B_diagonal_blocks_comm = cpx.empty_like_pinned(_B_diagonal_blocks)
                _B_lower_diagonal_blocks_comm = cpx.empty_like_pinned(_B_lower_diagonal_blocks)
                _B_upper_diagonal_blocks_comm = cpx.empty_like_pinned(_B_upper_diagonal_blocks)
            else:
                _B_diagonal_blocks_comm = _B_diagonal_blocks
                _B_lower_diagonal_blocks_comm = _B_lower_diagonal_blocks
                _B_upper_diagonal_blocks_comm = _B_upper_diagonal_blocks  

            _rhs = {
                "B_diagonal_blocks": _B_diagonal_blocks,
                "B_lower_diagonal_blocks": _B_lower_diagonal_blocks,
                "B_upper_diagonal_blocks": _B_upper_diagonal_blocks,
                "B_diagonal_blocks_comm": _B_diagonal_blocks_comm,
                "B_lower_diagonal_blocks_comm": _B_lower_diagonal_blocks_comm,
                "B_upper_diagonal_blocks_comm": _B_upper_diagonal_blocks_comm,
            }
        else:
            _rhs = None

        ddbtrs = {
            "A_diagonal_blocks": _A_diagonal_blocks,
            "A_lower_diagonal_blocks": _A_lower_diagonal_blocks,
            "A_upper_diagonal_blocks": _A_upper_diagonal_blocks,
            "A_diagonal_blocks_comm": _A_diagonal_blocks_comm,
            "A_lower_diagonal_blocks_comm": _A_lower_diagonal_blocks_comm,
            "A_upper_diagonal_blocks_comm": _A_upper_diagonal_blocks_comm,
            "_rhs": _rhs,
        }
    else:
        raise ValueError("Unknown communication strategy.")

    return ddbtrs


def map_ddbtsc_to_ddbtrs(
    A_diagonal_blocks: ArrayLike,
    A_lower_diagonal_blocks: ArrayLike,
    A_upper_diagonal_blocks: ArrayLike,
    _A_diagonal_blocks: ArrayLike,
    _A_lower_diagonal_blocks: ArrayLike,
    _A_upper_diagonal_blocks: ArrayLike,
    strategy: str = "allgather",
    **kwargs,
):
    rhs: dict = kwargs.get("rhs", None)
    quadratic: bool = kwargs.get("quadratic", False)
    buffers: dict = kwargs.get("buffers", None)
    _rhs: dict = kwargs.get("_rhs", None)

    A_lower_buffer_blocks: ArrayLike = buffers.get("A_lower_buffer_blocks", None)
    A_upper_buffer_blocks: ArrayLike = buffers.get("A_upper_buffer_blocks", None)

    if quadratic:
        # Get the RHS arrays
        B_diagonal_blocks: ArrayLike = rhs.get("B_diagonal_blocks", None)
        B_lower_diagonal_blocks: ArrayLike = rhs.get("B_lower_diagonal_blocks", None)
        B_upper_diagonal_blocks: ArrayLike = rhs.get("B_upper_diagonal_blocks", None)
        if any(
            x is None
            for x in [
                B_diagonal_blocks,
                B_lower_diagonal_blocks,
                B_upper_diagonal_blocks,
            ]
        ):
            raise ValueError("rhs does not contain the correct arrays")
        B_lower_buffer_blocks = buffers.get("B_lower_buffer_blocks", None)
        B_upper_buffer_blocks = buffers.get("B_upper_buffer_blocks", None)

        # Then check for the reduced system of the RHS
        _B_diagonal_blocks: ArrayLike = _rhs.get("B_diagonal_blocks", None)
        _B_lower_diagonal_blocks: ArrayLike = _rhs.get("B_lower_diagonal_blocks", None)
        _B_upper_diagonal_blocks: ArrayLike = _rhs.get("B_upper_diagonal_blocks", None)
        if any(
            x is None
            for x in [
                _B_diagonal_blocks,
                _B_lower_diagonal_blocks,
                _B_upper_diagonal_blocks,
            ]
        ):
            raise ValueError("_rhs does not contain the correct arrays")

    if strategy == "allgather":
        if comm_rank == 0:
            _A_diagonal_blocks[1] = A_diagonal_blocks[-1]
            _A_lower_diagonal_blocks[1] = A_lower_diagonal_blocks[-1]
            _A_upper_diagonal_blocks[1] = A_upper_diagonal_blocks[-1]

            if quadratic:
                _B_diagonal_blocks[1] = B_diagonal_blocks[-1]
                _B_lower_diagonal_blocks[1] = B_lower_diagonal_blocks[-1]
                _B_upper_diagonal_blocks[1] = B_upper_diagonal_blocks[-1]
        elif comm_rank == comm_size - 1:
            _A_diagonal_blocks[-2] = A_diagonal_blocks[0]

            if quadratic:
                _B_diagonal_blocks[-2] = B_diagonal_blocks[0]
        else:
            _A_diagonal_blocks[2 * comm_rank] = A_diagonal_blocks[0]
            _A_diagonal_blocks[2 * comm_rank + 1] = A_diagonal_blocks[-1]

            _A_lower_diagonal_blocks[2 * comm_rank] = A_upper_buffer_blocks[-2]
            _A_upper_diagonal_blocks[2 * comm_rank] = A_lower_buffer_blocks[-2]

            _A_lower_diagonal_blocks[2 * comm_rank + 1] = A_lower_diagonal_blocks[-1]
            _A_upper_diagonal_blocks[2 * comm_rank + 1] = A_upper_diagonal_blocks[-1]

            if quadratic:
                _B_diagonal_blocks[2 * comm_rank] = B_diagonal_blocks[0]
                _B_diagonal_blocks[2 * comm_rank + 1] = B_diagonal_blocks[-1]

                _B_lower_diagonal_blocks[2 * comm_rank] = B_upper_buffer_blocks[-2]
                _B_upper_diagonal_blocks[2 * comm_rank] = B_lower_buffer_blocks[-2]

                _B_lower_diagonal_blocks[2 * comm_rank + 1] = B_lower_diagonal_blocks[
                    -1
                ]
                _B_upper_diagonal_blocks[2 * comm_rank + 1] = B_upper_diagonal_blocks[
                    -1
                ]
    else:
        raise ValueError("Unknown communication strategy.")


def aggregate_ddbtrs(
    ddbtrs: dict,
    quadratic: bool = False,
    strategy: str = "allgather",
):
    _A_diagonal_blocks: ArrayLike = ddbtrs.get("A_diagonal_blocks", None)
    _A_lower_diagonal_blocks: ArrayLike = ddbtrs.get("A_lower_diagonal_blocks", None)
    _A_upper_diagonal_blocks: ArrayLike = ddbtrs.get("A_upper_diagonal_blocks", None)
    
    _A_diagonal_blocks_comm: ArrayLike = ddbtrs.get("A_diagonal_blocks_comm", None)
    _A_lower_diagonal_blocks_comm: ArrayLike = ddbtrs.get("A_lower_diagonal_blocks_comm", None)
    _A_upper_diagonal_blocks_comm: ArrayLike = ddbtrs.get("A_upper_diagonal_blocks_comm", None)
    if any(
        x is None
        for x in [
            _A_diagonal_blocks,
            _A_lower_diagonal_blocks,
            _A_upper_diagonal_blocks,
            _A_diagonal_blocks_comm,
            _A_lower_diagonal_blocks_comm,
            _A_upper_diagonal_blocks_comm,
        ]
    ):
        raise ValueError(
            "The reduced system `ddbtrs` doesn't contain the required arrays."
        )

    if quadratic:
        _rhs: dict = ddbtrs.get("_rhs", None)
        _B_diagonal_blocks: ArrayLike = _rhs.get("B_diagonal_blocks", None)
        _B_lower_diagonal_blocks: ArrayLike = _rhs.get("B_lower_diagonal_blocks", None)
        _B_upper_diagonal_blocks: ArrayLike = _rhs.get("B_upper_diagonal_blocks", None)
        
        _B_diagonal_blocks_comm: ArrayLike = _rhs.get("B_diagonal_blocks_comm", None)
        _B_lower_diagonal_blocks_comm: ArrayLike = _rhs.get("B_lower_diagonal_blocks_comm", None)
        _B_upper_diagonal_blocks_comm: ArrayLike = _rhs.get("B_upper_diagonal_blocks_comm", None)
        if any(
            x is None
            for x in [
                _B_diagonal_blocks,
                _B_lower_diagonal_blocks,
                _B_upper_diagonal_blocks,
                _B_diagonal_blocks_comm,
                _B_lower_diagonal_blocks_comm,
                _B_upper_diagonal_blocks_comm,
            ]
        ):
            raise ValueError(
                "The reduced system `ddbtrs` doesn't contain the required arrays for the quadratic equation."
            )

    xp, _ = _get_module_from_array(arr=_A_diagonal_blocks)

    if xp.__name__ == 'cupy':
        # We need to move the data of the reduced system from the GPU to the HOST
        # pinned arrays.
        _A_diagonal_blocks.get(out=_A_diagonal_blocks_comm)
        _A_lower_diagonal_blocks.get(out=_A_lower_diagonal_blocks_comm)
        _A_upper_diagonal_blocks.get(out=_A_upper_diagonal_blocks_comm)

        if quadratic:
            _B_diagonal_blocks.get(out=_B_diagonal_blocks_comm)
            _B_lower_diagonal_blocks.get(out=_B_lower_diagonal_blocks_comm)
            _B_upper_diagonal_blocks.get(out=_B_upper_diagonal_blocks_comm)

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
            _A_upper_diagonal_blocks_comm,
        )

        ddbtrs["A_diagonal_blocks_comm"] = _A_diagonal_blocks_comm[1:-1]
        ddbtrs["A_lower_diagonal_blocks_comm"] = _A_lower_diagonal_blocks_comm[1:-2]
        ddbtrs["A_upper_diagonal_blocks_comm"] = _A_upper_diagonal_blocks_comm[1:-2]

        ddbtrs["A_diagonal_blocks"] = _A_diagonal_blocks[1:-1]
        ddbtrs["A_lower_diagonal_blocks"] = _A_lower_diagonal_blocks[1:-2]
        ddbtrs["A_upper_diagonal_blocks"] = _A_upper_diagonal_blocks[1:-2]

        if quadratic:
            MPI.COMM_WORLD.Allgather(
                MPI.IN_PLACE,
                _B_diagonal_blocks_comm,
            )
            MPI.COMM_WORLD.Allgather(
                MPI.IN_PLACE,
                _B_lower_diagonal_blocks_comm,
            )
            MPI.COMM_WORLD.Allgather(
                MPI.IN_PLACE,
                _B_upper_diagonal_blocks_comm,
            )

            _rhs["B_diagonal_blocks_comm"] = _B_diagonal_blocks_comm[1:-1]
            _rhs["B_lower_diagonal_blocks_comm"] = _B_lower_diagonal_blocks_comm[1:-2]
            _rhs["B_upper_diagonal_blocks_comm"] = _B_upper_diagonal_blocks_comm[1:-2]
            
            _rhs["B_diagonal_blocks"] = _B_diagonal_blocks[1:-1]
            _rhs["B_lower_diagonal_blocks"] = _B_lower_diagonal_blocks[1:-2]
            _rhs["B_upper_diagonal_blocks"] = _B_upper_diagonal_blocks[1:-2]
            ddbtrs["_rhs"] = _rhs
    else:
        raise ValueError("Unknown communication strategy.")

    MPI.COMM_WORLD.Barrier()

    if xp.__name__ == 'cupy':
        # Need to put back the reduced system on the GPU
        _A_diagonal_blocks.set(arr=_A_diagonal_blocks_comm)
        _A_lower_diagonal_blocks.set(arr=_A_lower_diagonal_blocks_comm)
        _A_upper_diagonal_blocks.set(arr=_A_upper_diagonal_blocks_comm)

        if quadratic:   
            _B_diagonal_blocks.set(arr=_B_diagonal_blocks_comm)
            _B_lower_diagonal_blocks.set(arr=_B_lower_diagonal_blocks_comm)
            _B_upper_diagonal_blocks.set(arr=_B_upper_diagonal_blocks_comm)


def scatter_ddbtrs(
    ddbtrs: dict,
    quadratic: bool = False,
    strategy: str = "allgather",
):
    _A_diagonal_blocks: ArrayLike = ddbtrs.get("A_diagonal_blocks", None)
    _A_lower_diagonal_blocks: ArrayLike = ddbtrs.get("A_lower_diagonal_blocks", None)
    _A_upper_diagonal_blocks: ArrayLike = ddbtrs.get("A_upper_diagonal_blocks", None)
    if any(
        x is None
        for x in [
            _A_diagonal_blocks,
            _A_lower_diagonal_blocks,
            _A_upper_diagonal_blocks,
        ]
    ):
        raise ValueError(
            "The reduced system `ddbtrs` doesn't contain the required arrays."
        )

    if quadratic:
        _rhs: dict = ddbtrs.get("_rhs", None)
        _B_diagonal_blocks: ArrayLike = _rhs.get("B_diagonal_blocks", None)
        _B_lower_diagonal_blocks: ArrayLike = _rhs.get("B_lower_diagonal_blocks", None)
        _B_upper_diagonal_blocks: ArrayLike = _rhs.get("B_upper_diagonal_blocks", None)
        if any(
            x is None
            for x in [
                _B_diagonal_blocks,
                _B_lower_diagonal_blocks,
                _B_upper_diagonal_blocks,
            ]
        ):
            raise ValueError(
                "The reduced system `ddbtrs` doesn't contain the required arrays for the quadratic equation."
            )

    if strategy == "allgather":
        # In the case of the allgather strategy, nothing to be done.
        # > The solution of the reduced system is already distributed across
        #   all MPI processes.
        ...
    else:
        raise ValueError("Unknown communication strategy.")


def map_ddbtrs_to_ddbtsci(
    A_diagonal_blocks: ArrayLike,
    A_lower_diagonal_blocks: ArrayLike,
    A_upper_diagonal_blocks: ArrayLike,
    _A_diagonal_blocks: ArrayLike,
    _A_lower_diagonal_blocks: ArrayLike,
    _A_upper_diagonal_blocks: ArrayLike,
    strategy: str = "allgather",
    **kwargs,
):
    rhs: dict = kwargs.get("rhs", None)
    quadratic: bool = kwargs.get("quadratic", False)
    buffers: dict = kwargs.get("buffers", None)
    _rhs: dict = kwargs.get("_rhs", None)

    A_lower_buffer_blocks: ArrayLike = buffers.get("A_lower_buffer_blocks", None)
    A_upper_buffer_blocks: ArrayLike = buffers.get("A_upper_buffer_blocks", None)

    if quadratic:
        # Get the RHS arrays
        B_diagonal_blocks: ArrayLike = rhs.get("B_diagonal_blocks", None)
        B_lower_diagonal_blocks: ArrayLike = rhs.get("B_lower_diagonal_blocks", None)
        B_upper_diagonal_blocks: ArrayLike = rhs.get("B_upper_diagonal_blocks", None)
        if any(
            x is None
            for x in [
                B_diagonal_blocks,
                B_lower_diagonal_blocks,
                B_upper_diagonal_blocks,
            ]
        ):
            raise ValueError("rhs does not contain the correct arrays")
        B_lower_buffer_blocks = buffers.get("B_lower_buffer_blocks", None)
        B_upper_buffer_blocks = buffers.get("B_upper_buffer_blocks", None)

        # Then check for the reduced system of the RHS
        _B_diagonal_blocks: ArrayLike = _rhs.get("B_diagonal_blocks", None)
        _B_lower_diagonal_blocks: ArrayLike = _rhs.get("B_lower_diagonal_blocks", None)
        _B_upper_diagonal_blocks: ArrayLike = _rhs.get("B_upper_diagonal_blocks", None)
        if any(
            x is None
            for x in [
                _B_diagonal_blocks,
                _B_lower_diagonal_blocks,
                _B_upper_diagonal_blocks,
            ]
        ):
            raise ValueError("_rhs does not contain the correct arrays")

    if strategy == "allgather":
        if comm_rank == 0:
            A_diagonal_blocks[-1] = _A_diagonal_blocks[0]
            A_lower_diagonal_blocks[-1] = _A_lower_diagonal_blocks[0]
            A_upper_diagonal_blocks[-1] = _A_upper_diagonal_blocks[0]

            if quadratic:
                B_diagonal_blocks[-1] = _B_diagonal_blocks[0]
                B_lower_diagonal_blocks[-1] = _B_lower_diagonal_blocks[0]
                B_upper_diagonal_blocks[-1] = _B_upper_diagonal_blocks[0]
        elif comm_rank == comm_size - 1:
            A_diagonal_blocks[0] = _A_diagonal_blocks[-1]

            if quadratic:
                B_diagonal_blocks[0] = _B_diagonal_blocks[-1]
        else:
            A_diagonal_blocks[0] = _A_diagonal_blocks[2 * comm_rank - 1]
            A_diagonal_blocks[-1] = _A_diagonal_blocks[2 * comm_rank]

            A_upper_buffer_blocks[-2] = _A_lower_diagonal_blocks[2 * comm_rank - 1]
            A_lower_buffer_blocks[-2] = _A_upper_diagonal_blocks[2 * comm_rank - 1]

            A_lower_diagonal_blocks[-1] = _A_lower_diagonal_blocks[2 * comm_rank]
            A_upper_diagonal_blocks[-1] = _A_upper_diagonal_blocks[2 * comm_rank]

            if quadratic:
                B_diagonal_blocks[0] = _B_diagonal_blocks[2 * comm_rank - 1]
                B_diagonal_blocks[-1] = _B_diagonal_blocks[2 * comm_rank]

                B_upper_buffer_blocks[-2] = _B_lower_diagonal_blocks[2 * comm_rank - 1]
                B_lower_buffer_blocks[-2] = _B_upper_diagonal_blocks[2 * comm_rank - 1]

                B_lower_diagonal_blocks[-1] = _B_lower_diagonal_blocks[2 * comm_rank]
                B_upper_diagonal_blocks[-1] = _B_upper_diagonal_blocks[2 * comm_rank]
    else:
        raise ValueError("Unknown communication strategy.")
