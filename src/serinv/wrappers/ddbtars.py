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


def allocate_ddbtars(
    A_diagonal_blocks: ArrayLike,
    A_lower_diagonal_blocks: ArrayLike,
    A_upper_diagonal_blocks: ArrayLike,
    A_lower_arrow_blocks: ArrayLike,
    A_upper_arrow_blocks: ArrayLike,
    A_arrow_tip_block: ArrayLike,
    array_module: str,
    comm: MPI.Comm,
    strategy: str = "allgather",
    quadratic: bool = False,
) -> dict:
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

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
        _A_lower_arrow_blocks = xp.empty(
            (
                _n,
                A_lower_arrow_blocks[0].shape[0],
                A_lower_arrow_blocks[0].shape[1],
            ),
            dtype=A_lower_arrow_blocks.dtype,
        )
        _A_upper_arrow_blocks = xp.empty(
            (
                _n,
                A_upper_arrow_blocks[0].shape[0],
                A_upper_arrow_blocks[0].shape[1],
            ),
            dtype=A_upper_arrow_blocks.dtype,
        )
        _A_arrow_tip_block = xp.zeros_like(A_arrow_tip_block)

        if xp.__name__ == "cupy":
            # In this case we also need to allocate a pinned-memory
            # reduced system on the host side.
            _A_diagonal_blocks_comm = cpx.empty_like_pinned(_A_diagonal_blocks)
            _A_lower_diagonal_blocks_comm = cpx.empty_like_pinned(
                _A_lower_diagonal_blocks
            )
            _A_upper_diagonal_blocks_comm = cpx.empty_like_pinned(
                _A_upper_diagonal_blocks
            )
            _A_lower_arrow_blocks_comm = cpx.empty_like_pinned(_A_lower_arrow_blocks)
            _A_upper_arrow_blocks_comm = cpx.empty_like_pinned(_A_upper_arrow_blocks)
            _A_arrow_tip_block_comm = cpx.empty_like_pinned(_A_arrow_tip_block)
        else:
            _A_diagonal_blocks_comm = _A_diagonal_blocks
            _A_lower_diagonal_blocks_comm = _A_lower_diagonal_blocks
            _A_upper_diagonal_blocks_comm = _A_upper_diagonal_blocks
            _A_lower_arrow_blocks_comm = _A_lower_arrow_blocks
            _A_upper_arrow_blocks_comm = _A_upper_arrow_blocks
            _A_arrow_tip_block_comm = _A_arrow_tip_block

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
            _B_lower_arrow_blocks = xp.empty(
                (
                    _n,
                    A_lower_arrow_blocks[0].shape[0],
                    A_lower_arrow_blocks[0].shape[1],
                ),
                dtype=A_lower_arrow_blocks.dtype,
            )
            _B_upper_arrow_blocks = xp.empty(
                (
                    _n,
                    A_upper_arrow_blocks[0].shape[0],
                    A_upper_arrow_blocks[0].shape[1],
                ),
                dtype=A_upper_arrow_blocks.dtype,
            )
            _B_arrow_tip_block = xp.zeros_like(A_arrow_tip_block)

            if xp.__name__ == "cupy":
                # In this case we also need to allocate a pinned-memory
                # reduced system on the host side.

                _B_diagonal_blocks_comm = cpx.empty_like_pinned(_B_diagonal_blocks)
                _B_lower_diagonal_blocks_comm = cpx.empty_like_pinned(
                    _B_lower_diagonal_blocks
                )
                _B_upper_diagonal_blocks_comm = cpx.empty_like_pinned(
                    _B_upper_diagonal_blocks
                )
                _B_lower_arrow_blocks_comm = cpx.empty_like_pinned(
                    _B_lower_arrow_blocks
                )
                _B_upper_arrow_blocks_comm = cpx.empty_like_pinned(
                    _B_upper_arrow_blocks
                )
                _B_arrow_tip_block_comm = cpx.empty_like_pinned(_B_arrow_tip_block)
            else:
                _B_diagonal_blocks_comm = _B_diagonal_blocks
                _B_lower_diagonal_blocks_comm = _B_lower_diagonal_blocks
                _B_upper_diagonal_blocks_comm = _B_upper_diagonal_blocks
                _B_lower_arrow_blocks_comm = _B_lower_arrow_blocks
                _B_upper_arrow_blocks_comm = _B_upper_arrow_blocks
                _B_arrow_tip_block_comm = _B_arrow_tip_block

            _rhs = {
                "B_diagonal_blocks": _B_diagonal_blocks,
                "B_lower_diagonal_blocks": _B_lower_diagonal_blocks,
                "B_upper_diagonal_blocks": _B_upper_diagonal_blocks,
                "B_lower_arrow_blocks": _B_lower_arrow_blocks,
                "B_upper_arrow_blocks": _B_upper_arrow_blocks,
                "B_arrow_tip_block": _B_arrow_tip_block,
                "B_diagonal_blocks_comm": _B_diagonal_blocks_comm,
                "B_lower_diagonal_blocks_comm": _B_lower_diagonal_blocks_comm,
                "B_upper_diagonal_blocks_comm": _B_upper_diagonal_blocks_comm,
                "B_lower_arrow_blocks_comm": _B_lower_arrow_blocks_comm,
                "B_upper_arrow_blocks_comm": _B_upper_arrow_blocks_comm,
                "B_arrow_tip_block_comm": _B_arrow_tip_block_comm,
            }
        else:
            _rhs = None

        ddbtars = {
            "A_diagonal_blocks": _A_diagonal_blocks,
            "A_lower_diagonal_blocks": _A_lower_diagonal_blocks,
            "A_upper_diagonal_blocks": _A_upper_diagonal_blocks,
            "A_lower_arrow_blocks": _A_lower_arrow_blocks,
            "A_upper_arrow_blocks": _A_upper_arrow_blocks,
            "A_arrow_tip_block": _A_arrow_tip_block,
            "A_diagonal_blocks_comm": _A_diagonal_blocks_comm,
            "A_lower_diagonal_blocks_comm": _A_lower_diagonal_blocks_comm,
            "A_upper_diagonal_blocks_comm": _A_upper_diagonal_blocks_comm,
            "A_lower_arrow_blocks_comm": _A_lower_arrow_blocks_comm,
            "A_upper_arrow_blocks_comm": _A_upper_arrow_blocks_comm,
            "A_arrow_tip_block_comm": _A_arrow_tip_block_comm,
            "_rhs": _rhs,
        }
    else:
        raise ValueError("Unknown communication strategy.")

    return ddbtars


def map_ddbtasc_to_ddbtars(
    A_diagonal_blocks: ArrayLike,
    A_lower_diagonal_blocks: ArrayLike,
    A_upper_diagonal_blocks: ArrayLike,
    A_lower_arrow_blocks: ArrayLike,
    A_upper_arrow_blocks: ArrayLike,
    A_arrow_tip_block: ArrayLike,
    _A_diagonal_blocks: ArrayLike,
    _A_lower_diagonal_blocks: ArrayLike,
    _A_upper_diagonal_blocks: ArrayLike,
    _A_lower_arrow_blocks: ArrayLike,
    _A_upper_arrow_blocks: ArrayLike,
    _A_arrow_tip_block: ArrayLike,
    comm: MPI.Comm,
    strategy: str,
    **kwargs,
) -> None:
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

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
        B_lower_arrow_blocks: ArrayLike = rhs.get("B_lower_arrow_blocks", None)
        B_upper_arrow_blocks: ArrayLike = rhs.get("B_upper_arrow_blocks", None)
        B_arrow_tip_block: ArrayLike = rhs.get("B_arrow_tip_block", None)
        if any(
            x is None
            for x in [
                B_diagonal_blocks,
                B_lower_diagonal_blocks,
                B_upper_diagonal_blocks,
                B_lower_arrow_blocks,
                B_upper_arrow_blocks,
                B_arrow_tip_block,
            ]
        ):
            raise ValueError("rhs does not contain the correct arrays")
        B_lower_buffer_blocks = buffers.get("B_lower_buffer_blocks", None)
        B_upper_buffer_blocks = buffers.get("B_upper_buffer_blocks", None)

        # Then check for the reduced system of the RHS
        _B_diagonal_blocks: ArrayLike = _rhs.get("B_diagonal_blocks", None)
        _B_lower_diagonal_blocks: ArrayLike = _rhs.get("B_lower_diagonal_blocks", None)
        _B_upper_diagonal_blocks: ArrayLike = _rhs.get("B_upper_diagonal_blocks", None)
        _B_lower_arrow_blocks: ArrayLike = _rhs.get("B_lower_arrow_blocks", None)
        _B_upper_arrow_blocks: ArrayLike = _rhs.get("B_upper_arrow_blocks", None)
        _B_arrow_tip_block: ArrayLike = _rhs.get("B_arrow_tip_block", None)
        if any(
            x is None
            for x in [
                _B_diagonal_blocks,
                _B_lower_diagonal_blocks,
                _B_upper_diagonal_blocks,
                _B_lower_arrow_blocks,
                _B_upper_arrow_blocks,
                _B_arrow_tip_block,
            ]
        ):
            raise ValueError("_rhs does not contain the correct arrays")

    if strategy == "allgather":
        if comm_rank == 0:
            _A_diagonal_blocks[1] = A_diagonal_blocks[-1]
            _A_lower_diagonal_blocks[1] = A_lower_diagonal_blocks[-1]
            _A_upper_diagonal_blocks[1] = A_upper_diagonal_blocks[-1]
            _A_lower_arrow_blocks[1] = A_lower_arrow_blocks[-1]
            _A_upper_arrow_blocks[1] = A_upper_arrow_blocks[-1]

            if quadratic:
                _B_diagonal_blocks[1] = B_diagonal_blocks[-1]
                _B_lower_diagonal_blocks[1] = B_lower_diagonal_blocks[-1]
                _B_upper_diagonal_blocks[1] = B_upper_diagonal_blocks[-1]
                _B_lower_arrow_blocks[1] = B_lower_arrow_blocks[-1]
                _B_upper_arrow_blocks[1] = B_upper_arrow_blocks[-1]
        else:
            _A_diagonal_blocks[2 * comm_rank] = A_diagonal_blocks[0]
            _A_diagonal_blocks[2 * comm_rank + 1] = A_diagonal_blocks[-1]

            if comm_rank < comm_size - 1:
                # Warning: The size of the upper/lower buffer follow the shape
                # of the lower_diagonal_blocks slicing. That mean that the indexing
                # is different between the last and the "middle" processes.
                _A_lower_diagonal_blocks[2 * comm_rank] = A_upper_buffer_blocks[-2]
                _A_upper_diagonal_blocks[2 * comm_rank] = A_lower_buffer_blocks[-2]

                _A_lower_diagonal_blocks[2 * comm_rank + 1] = A_lower_diagonal_blocks[
                    -1
                ]
                _A_upper_diagonal_blocks[2 * comm_rank + 1] = A_upper_diagonal_blocks[
                    -1
                ]
            else:
                _A_lower_diagonal_blocks[2 * comm_rank] = A_upper_buffer_blocks[-1]
                _A_upper_diagonal_blocks[2 * comm_rank] = A_lower_buffer_blocks[-1]

            _A_lower_arrow_blocks[2 * comm_rank] = A_lower_arrow_blocks[0]
            _A_lower_arrow_blocks[2 * comm_rank + 1] = A_lower_arrow_blocks[-1]

            _A_upper_arrow_blocks[2 * comm_rank] = A_upper_arrow_blocks[0]
            _A_upper_arrow_blocks[2 * comm_rank + 1] = A_upper_arrow_blocks[-1]

            if quadratic:
                _B_diagonal_blocks[2 * comm_rank] = B_diagonal_blocks[0]
                _B_diagonal_blocks[2 * comm_rank + 1] = B_diagonal_blocks[-1]

                if comm_rank < comm_size - 1:
                    # Warning: The size of the upper/lower buffer follow the shape
                    # of the lower_diagonal_blocks slicing. That mean that the indexing
                    # is different between the last and the "middle" processes.
                    _B_lower_diagonal_blocks[2 * comm_rank] = B_upper_buffer_blocks[-2]
                    _B_upper_diagonal_blocks[2 * comm_rank] = B_lower_buffer_blocks[-2]

                    _B_lower_diagonal_blocks[2 * comm_rank + 1] = (
                        B_lower_diagonal_blocks[-1]
                    )
                    _B_upper_diagonal_blocks[2 * comm_rank + 1] = (
                        B_upper_diagonal_blocks[-1]
                    )
                else:
                    _B_lower_diagonal_blocks[2 * comm_rank] = B_upper_buffer_blocks[-1]
                    _B_upper_diagonal_blocks[2 * comm_rank] = B_lower_buffer_blocks[-1]

                _B_lower_arrow_blocks[2 * comm_rank] = B_lower_arrow_blocks[0]
                _B_lower_arrow_blocks[2 * comm_rank + 1] = B_lower_arrow_blocks[-1]

                _B_upper_arrow_blocks[2 * comm_rank] = B_upper_arrow_blocks[0]
                _B_upper_arrow_blocks[2 * comm_rank + 1] = B_upper_arrow_blocks[-1]

        _A_arrow_tip_block[:] = A_arrow_tip_block[:]
        if quadratic:
            _B_arrow_tip_block[:] = B_arrow_tip_block[:]
    else:
        raise ValueError("Unknown communication strategy.")


def aggregate_ddbtars(
    ddbtars: dict,
    comm: MPI.Comm,
    quadratic: bool = False,
    strategy: str = "allgather",
) -> None:
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    _A_diagonal_blocks: ArrayLike = ddbtars.get("A_diagonal_blocks", None)
    _A_lower_diagonal_blocks: ArrayLike = ddbtars.get("A_lower_diagonal_blocks", None)
    _A_upper_diagonal_blocks: ArrayLike = ddbtars.get("A_upper_diagonal_blocks", None)
    _A_lower_arrow_blocks: ArrayLike = ddbtars.get("A_lower_arrow_blocks", None)
    _A_upper_arrow_blocks: ArrayLike = ddbtars.get("A_upper_arrow_blocks", None)
    _A_arrow_tip_block: ArrayLike = ddbtars.get("A_arrow_tip_block", None)

    _A_diagonal_blocks_comm: ArrayLike = ddbtars.get("A_diagonal_blocks_comm", None)
    _A_lower_diagonal_blocks_comm: ArrayLike = ddbtars.get(
        "A_lower_diagonal_blocks_comm", None
    )
    _A_upper_diagonal_blocks_comm: ArrayLike = ddbtars.get(
        "A_upper_diagonal_blocks_comm", None
    )
    _A_lower_arrow_blocks_comm: ArrayLike = ddbtars.get(
        "A_lower_arrow_blocks_comm", None
    )
    _A_upper_arrow_blocks_comm: ArrayLike = ddbtars.get(
        "A_upper_arrow_blocks_comm", None
    )
    _A_arrow_tip_block_comm: ArrayLike = ddbtars.get("A_arrow_tip_block_comm", None)
    if any(
        x is None
        for x in [
            _A_diagonal_blocks,
            _A_lower_diagonal_blocks,
            _A_upper_diagonal_blocks,
            _A_lower_arrow_blocks,
            _A_upper_arrow_blocks,
            _A_arrow_tip_block,
            _A_diagonal_blocks_comm,
            _A_lower_diagonal_blocks_comm,
            _A_upper_diagonal_blocks_comm,
            _A_lower_arrow_blocks_comm,
            _A_upper_arrow_blocks_comm,
            _A_arrow_tip_block_comm,
        ]
    ):
        raise ValueError(
            "The reduced system `ddbtars` doesn't contain the required arrays."
        )

    if quadratic:
        _rhs: dict = ddbtars.get("_rhs", None)
        _B_diagonal_blocks: ArrayLike = _rhs.get("B_diagonal_blocks", None)
        _B_lower_diagonal_blocks: ArrayLike = _rhs.get("B_lower_diagonal_blocks", None)
        _B_upper_diagonal_blocks: ArrayLike = _rhs.get("B_upper_diagonal_blocks", None)
        _B_lower_arrow_blocks: ArrayLike = _rhs.get("B_lower_arrow_blocks", None)
        _B_upper_arrow_blocks: ArrayLike = _rhs.get("B_upper_arrow_blocks", None)
        _B_arrow_tip_block: ArrayLike = _rhs.get("B_arrow_tip_block", None)

        _B_diagonal_blocks_comm: ArrayLike = _rhs.get("B_diagonal_blocks_comm", None)
        _B_lower_diagonal_blocks_comm: ArrayLike = _rhs.get(
            "B_lower_diagonal_blocks_comm", None
        )
        _B_upper_diagonal_blocks_comm: ArrayLike = _rhs.get(
            "B_upper_diagonal_blocks_comm", None
        )
        _B_lower_arrow_blocks_comm: ArrayLike = _rhs.get(
            "B_lower_arrow_blocks_comm", None
        )
        _B_upper_arrow_blocks_comm: ArrayLike = _rhs.get(
            "B_upper_arrow_blocks_comm", None
        )
        _B_arrow_tip_block_comm: ArrayLike = _rhs.get("B_arrow_tip_block_comm", None)
        if any(
            x is None
            for x in [
                _B_diagonal_blocks,
                _B_lower_diagonal_blocks,
                _B_upper_diagonal_blocks,
                _B_lower_arrow_blocks,
                _B_upper_arrow_blocks,
                _B_arrow_tip_block,
                _B_diagonal_blocks_comm,
                _B_lower_diagonal_blocks_comm,
                _B_upper_diagonal_blocks_comm,
                _B_lower_arrow_blocks_comm,
                _B_upper_arrow_blocks_comm,
                _B_arrow_tip_block_comm,
            ]
        ):
            raise ValueError(
                "The reduced system `ddbtars` doesn't contain the required arrays for the quadratic equation."
            )

    xp, _ = _get_module_from_array(arr=_A_diagonal_blocks)
    if strategy == "allgather":
        if xp.__name__ == "cupy":
            # We need to move the data of the reduced system from the GPU to the HOST pinned arrays.
            if comm_rank == 0:
                _A_diagonal_blocks[1].get(out=_A_diagonal_blocks_comm[1])
                _A_lower_diagonal_blocks[1].get(out=_A_lower_diagonal_blocks_comm[1])
                _A_upper_diagonal_blocks[1].get(out=_A_upper_diagonal_blocks_comm[1])
                _A_lower_arrow_blocks[1].get(out=_A_lower_arrow_blocks_comm[1])
                _A_upper_arrow_blocks[1].get(out=_A_upper_arrow_blocks_comm[1])
            else:
                _A_diagonal_blocks[2 * comm_rank].get(
                    out=_A_diagonal_blocks_comm[2 * comm_rank]
                )
                _A_diagonal_blocks[2 * comm_rank + 1].get(
                    out=_A_diagonal_blocks_comm[2 * comm_rank + 1]
                )

                if comm_rank < comm_size - 1:
                    _A_lower_diagonal_blocks[2 * comm_rank].get(
                        out=_A_lower_diagonal_blocks_comm[2 * comm_rank]
                    )
                    _A_upper_diagonal_blocks[2 * comm_rank].get(
                        out=_A_upper_diagonal_blocks_comm[2 * comm_rank]
                    )

                    _A_lower_diagonal_blocks[2 * comm_rank + 1].get(
                        out=_A_lower_diagonal_blocks_comm[2 * comm_rank + 1]
                    )
                    _A_upper_diagonal_blocks[2 * comm_rank + 1].get(
                        out=_A_upper_diagonal_blocks_comm[2 * comm_rank + 1]
                    )
                else:
                    _A_lower_diagonal_blocks[2 * comm_rank].get(
                        out=_A_lower_diagonal_blocks_comm[2 * comm_rank]
                    )
                    _A_upper_diagonal_blocks[2 * comm_rank].get(
                        out=_A_upper_diagonal_blocks_comm[2 * comm_rank]
                    )

                _A_lower_arrow_blocks[2 * comm_rank].get(
                    out=_A_lower_arrow_blocks_comm[2 * comm_rank]
                )
                _A_lower_arrow_blocks[2 * comm_rank + 1].get(
                    out=_A_lower_arrow_blocks_comm[2 * comm_rank + 1]
                )

                _A_upper_arrow_blocks[2 * comm_rank].get(
                    out=_A_upper_arrow_blocks_comm[2 * comm_rank]
                )
                _A_upper_arrow_blocks[2 * comm_rank + 1].get(
                    out=_A_upper_arrow_blocks_comm[2 * comm_rank + 1]
                )

            _A_arrow_tip_block.get(out=_A_arrow_tip_block_comm)

            cpx.cuda.Stream.null.synchronize()

        # Perform the allgather operation
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
            _A_upper_diagonal_blocks_comm,
        )
        comm.Allgather(
            MPI.IN_PLACE,
            _A_lower_arrow_blocks_comm,
        )
        comm.Allgather(
            MPI.IN_PLACE,
            _A_upper_arrow_blocks_comm,
        )
        comm.Allreduce(MPI.IN_PLACE, _A_arrow_tip_block_comm, op=MPI.SUM)

        ddbtars["A_diagonal_blocks_comm"] = _A_diagonal_blocks_comm[1:]
        ddbtars["A_lower_diagonal_blocks_comm"] = _A_lower_diagonal_blocks_comm[1:-1]
        ddbtars["A_upper_diagonal_blocks_comm"] = _A_upper_diagonal_blocks_comm[1:-1]
        ddbtars["A_lower_arrow_blocks_comm"] = _A_lower_arrow_blocks_comm[1:]
        ddbtars["A_upper_arrow_blocks_comm"] = _A_upper_arrow_blocks_comm[1:]

        ddbtars["A_diagonal_blocks"] = _A_diagonal_blocks[1:]
        ddbtars["A_lower_diagonal_blocks"] = _A_lower_diagonal_blocks[1:-1]
        ddbtars["A_upper_diagonal_blocks"] = _A_upper_diagonal_blocks[1:-1]
        ddbtars["A_lower_arrow_blocks"] = _A_lower_arrow_blocks[1:]
        ddbtars["A_upper_arrow_blocks"] = _A_upper_arrow_blocks[1:]

        if quadratic:
            if xp.__name__ == "cupy":
                # We need to move the data of the reduced system from the GPU to the HOST pinned arrays.
                if comm_rank == 0:
                    _B_diagonal_blocks[1].get(out=_B_diagonal_blocks_comm[1])
                    _B_lower_diagonal_blocks[1].get(
                        out=_B_lower_diagonal_blocks_comm[1]
                    )
                    _B_upper_diagonal_blocks[1].get(
                        out=_B_upper_diagonal_blocks_comm[1]
                    )
                    _B_lower_arrow_blocks[1].get(out=_B_lower_arrow_blocks_comm[1])
                    _B_upper_arrow_blocks[1].get(out=_B_upper_arrow_blocks_comm[1])
                else:
                    _B_diagonal_blocks[2 * comm_rank].get(
                        out=_B_diagonal_blocks_comm[2 * comm_rank]
                    )
                    _B_diagonal_blocks[2 * comm_rank + 1].get(
                        out=_B_diagonal_blocks_comm[2 * comm_rank + 1]
                    )

                    if comm_rank < comm_size - 1:
                        _B_lower_diagonal_blocks[2 * comm_rank].get(
                            out=_B_lower_diagonal_blocks_comm[2 * comm_rank]
                        )
                        _B_upper_diagonal_blocks[2 * comm_rank].get(
                            out=_B_upper_diagonal_blocks_comm[2 * comm_rank]
                        )

                        _B_lower_diagonal_blocks[2 * comm_rank + 1].get(
                            out=_B_lower_diagonal_blocks_comm[2 * comm_rank + 1]
                        )
                        _B_upper_diagonal_blocks[2 * comm_rank + 1].get(
                            out=_B_upper_diagonal_blocks_comm[2 * comm_rank + 1]
                        )
                    else:
                        _B_lower_diagonal_blocks[2 * comm_rank].get(
                            out=_B_lower_diagonal_blocks_comm[2 * comm_rank]
                        )
                        _B_upper_diagonal_blocks[2 * comm_rank].get(
                            out=_B_upper_diagonal_blocks_comm[2 * comm_rank]
                        )

                    _B_lower_arrow_blocks[2 * comm_rank].get(
                        out=_B_lower_arrow_blocks_comm[2 * comm_rank]
                    )
                    _B_lower_arrow_blocks[2 * comm_rank + 1].get(
                        out=_B_lower_arrow_blocks_comm[2 * comm_rank + 1]
                    )

                    _B_upper_arrow_blocks[2 * comm_rank].get(
                        out=_B_upper_arrow_blocks_comm[2 * comm_rank]
                    )
                    _B_upper_arrow_blocks[2 * comm_rank + 1].get(
                        out=_B_upper_arrow_blocks_comm[2 * comm_rank + 1]
                    )

                _B_arrow_tip_block.get(out=_B_arrow_tip_block_comm)

            # Perform the allgather operation
            comm.Allgather(
                MPI.IN_PLACE,
                _B_diagonal_blocks_comm,
            )
            comm.Allgather(
                MPI.IN_PLACE,
                _B_lower_diagonal_blocks_comm,
            )
            comm.Allgather(
                MPI.IN_PLACE,
                _B_upper_diagonal_blocks_comm,
            )
            comm.Allgather(
                MPI.IN_PLACE,
                _B_lower_arrow_blocks_comm,
            )
            comm.Allgather(
                MPI.IN_PLACE,
                _B_upper_arrow_blocks_comm,
            )
            comm.Allreduce(MPI.IN_PLACE, _B_arrow_tip_block_comm, op=MPI.SUM)

            _rhs["B_diagonal_blocks_comm"] = _B_diagonal_blocks_comm[1:]
            _rhs["B_lower_diagonal_blocks_comm"] = _B_lower_diagonal_blocks_comm[1:-1]
            _rhs["B_upper_diagonal_blocks_comm"] = _B_upper_diagonal_blocks_comm[1:-1]
            _rhs["B_lower_arrow_blocks_comm"] = _B_lower_arrow_blocks_comm[1:]
            _rhs["B_upper_arrow_blocks_comm"] = _B_upper_arrow_blocks_comm[1:]

            _rhs["B_diagonal_blocks"] = _B_diagonal_blocks[1:]
            _rhs["B_lower_diagonal_blocks"] = _B_lower_diagonal_blocks[1:-1]
            _rhs["B_upper_diagonal_blocks"] = _B_upper_diagonal_blocks[1:-1]
            _rhs["B_lower_arrow_blocks"] = _B_lower_arrow_blocks[1:]
            _rhs["B_upper_arrow_blocks"] = _B_upper_arrow_blocks[1:]
            ddbtars["_rhs"] = _rhs
    else:
        raise ValueError("Unknown communication strategy.")

    comm.Barrier()

    if xp.__name__ == "cupy":
        # Need to put back the reduced system on the GPU
        _A_diagonal_blocks.set(arr=_A_diagonal_blocks_comm)
        _A_lower_diagonal_blocks.set(arr=_A_lower_diagonal_blocks_comm)
        _A_upper_diagonal_blocks.set(arr=_A_upper_diagonal_blocks_comm)
        _A_lower_arrow_blocks.set(arr=_A_lower_arrow_blocks_comm)
        _A_upper_arrow_blocks.set(arr=_A_upper_arrow_blocks_comm)
        _A_arrow_tip_block.set(arr=_A_arrow_tip_block_comm)

        if quadratic:
            _B_diagonal_blocks.set(arr=_B_diagonal_blocks_comm)
            _B_lower_diagonal_blocks.set(arr=_B_lower_diagonal_blocks_comm)
            _B_upper_diagonal_blocks.set(arr=_B_upper_diagonal_blocks_comm)
            _B_lower_arrow_blocks.set(arr=_B_lower_arrow_blocks_comm)
            _B_upper_arrow_blocks.set(arr=_B_upper_arrow_blocks_comm)
            _B_arrow_tip_block.set(arr=_B_arrow_tip_block_comm)


def scatter_ddbtars(
    ddbtars: dict,
    comm: MPI.Comm,
    quadratic: bool = False,
    strategy: str = "allgather",
    **kwargs,
):
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    _A_diagonal_blocks: ArrayLike = ddbtars.get("A_diagonal_blocks", None)
    _A_lower_diagonal_blocks: ArrayLike = ddbtars.get("A_lower_diagonal_blocks", None)
    _A_upper_diagonal_blocks: ArrayLike = ddbtars.get("A_upper_diagonal_blocks", None)
    _A_lower_arrow_blocks: ArrayLike = ddbtars.get("A_lower_arrow_blocks", None)
    _A_upper_arrow_blocks: ArrayLike = ddbtars.get("A_upper_arrow_blocks", None)
    _A_arrow_tip_block: ArrayLike = ddbtars.get("A_arrow_tip_block", None)
    if any(
        x is None
        for x in [
            _A_diagonal_blocks,
            _A_lower_diagonal_blocks,
            _A_upper_diagonal_blocks,
            _A_lower_arrow_blocks,
            _A_upper_arrow_blocks,
            _A_arrow_tip_block,
        ]
    ):
        raise ValueError(
            "The reduced system `ddbtars` doesn't contain the required arrays."
        )

    if quadratic:
        _rhs: dict = ddbtars.get("_rhs", None)
        _B_diagonal_blocks: ArrayLike = _rhs.get("B_diagonal_blocks", None)
        _B_lower_diagonal_blocks: ArrayLike = _rhs.get("B_lower_diagonal_blocks", None)
        _B_upper_diagonal_blocks: ArrayLike = _rhs.get("B_upper_diagonal_blocks", None)
        _B_lower_arrow_blocks: ArrayLike = _rhs.get("B_lower_arrow_blocks", None)
        _B_upper_arrow_blocks: ArrayLike = _rhs.get("B_upper_arrow_blocks", None)
        _B_arrow_tip_block: ArrayLike = _rhs.get("B_arrow_tip_block", None)
        if any(
            x is None
            for x in [
                _B_diagonal_blocks,
                _B_lower_diagonal_blocks,
                _B_upper_diagonal_blocks,
                _B_lower_arrow_blocks,
                _B_upper_arrow_blocks,
                _B_arrow_tip_block,
            ]
        ):
            raise ValueError(
                "The reduced system `ddbtars` doesn't contain the required arrays for the quadratic equation."
            )

    if strategy == "allgather":
        # In the case of the allgather strategy, nothing to be done.
        # > The solution of the reduced system is already distributed across
        #   all MPI processes.
        ...
    else:
        raise ValueError("Unknown communication strategy.")


def map_ddbtars_to_ddbtasci(
    A_diagonal_blocks: ArrayLike,
    A_lower_diagonal_blocks: ArrayLike,
    A_upper_diagonal_blocks: ArrayLike,
    A_lower_arrow_blocks: ArrayLike,
    A_upper_arrow_blocks: ArrayLike,
    A_arrow_tip_block: ArrayLike,
    _A_diagonal_blocks: ArrayLike,
    _A_lower_diagonal_blocks: ArrayLike,
    _A_upper_diagonal_blocks: ArrayLike,
    _A_lower_arrow_blocks: ArrayLike,
    _A_upper_arrow_blocks: ArrayLike,
    _A_arrow_tip_block: ArrayLike,
    comm: MPI.Comm,
    strategy: str = "allgather",
    **kwargs,
):
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

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
        B_lower_arrow_blocks: ArrayLike = rhs.get("B_lower_arrow_blocks", None)
        B_upper_arrow_blocks: ArrayLike = rhs.get("B_upper_arrow_blocks", None)
        B_arrow_tip_block: ArrayLike = rhs.get("B_arrow_tip_block", None)
        if any(
            x is None
            for x in [
                B_diagonal_blocks,
                B_lower_diagonal_blocks,
                B_upper_diagonal_blocks,
                B_lower_arrow_blocks,
                B_upper_arrow_blocks,
                B_arrow_tip_block,
            ]
        ):
            raise ValueError("rhs does not contain the correct arrays")
        B_lower_buffer_blocks = buffers.get("B_lower_buffer_blocks", None)
        B_upper_buffer_blocks = buffers.get("B_upper_buffer_blocks", None)

        # Then check for the reduced system of the RHS
        _B_diagonal_blocks: ArrayLike = _rhs.get("B_diagonal_blocks", None)
        _B_lower_diagonal_blocks: ArrayLike = _rhs.get("B_lower_diagonal_blocks", None)
        _B_upper_diagonal_blocks: ArrayLike = _rhs.get("B_upper_diagonal_blocks", None)
        _B_lower_arrow_blocks: ArrayLike = _rhs.get("B_lower_arrow_blocks", None)
        _B_upper_arrow_blocks: ArrayLike = _rhs.get("B_upper_arrow_blocks", None)
        _B_arrow_tip_block: ArrayLike = _rhs.get("B_arrow_tip_block", None)
        if any(
            x is None
            for x in [
                _B_diagonal_blocks,
                _B_lower_diagonal_blocks,
                _B_upper_diagonal_blocks,
                _B_lower_arrow_blocks,
                _B_upper_arrow_blocks,
                _B_arrow_tip_block,
            ]
        ):
            raise ValueError("_rhs does not contain the correct arrays")

    if strategy == "allgather":
        if comm_rank == 0:
            A_diagonal_blocks[-1] = _A_diagonal_blocks[0]
            A_lower_diagonal_blocks[-1] = _A_lower_diagonal_blocks[0]
            A_upper_diagonal_blocks[-1] = _A_upper_diagonal_blocks[0]
            A_lower_arrow_blocks[-1] = _A_lower_arrow_blocks[0]
            A_upper_arrow_blocks[-1] = _A_upper_arrow_blocks[0]

            if quadratic:
                B_diagonal_blocks[-1] = _B_diagonal_blocks[0]
                B_lower_diagonal_blocks[-1] = _B_lower_diagonal_blocks[0]
                B_upper_diagonal_blocks[-1] = _B_upper_diagonal_blocks[0]
                B_lower_arrow_blocks[-1] = _B_lower_arrow_blocks[0]
                B_upper_arrow_blocks[-1] = _B_upper_arrow_blocks[0]
        else:
            A_diagonal_blocks[0] = _A_diagonal_blocks[2 * comm_rank - 1]
            A_diagonal_blocks[-1] = _A_diagonal_blocks[2 * comm_rank]

            if comm_rank < comm_size - 1:
                # Warning: The size of the upper/lower buffer follow the shape
                # of the lower_diagonal_blocks slicing. That mean that the indexing
                # is different between the last and the "middle" processes.
                A_upper_buffer_blocks[-2] = _A_lower_diagonal_blocks[2 * comm_rank - 1]
                A_lower_buffer_blocks[-2] = _A_upper_diagonal_blocks[2 * comm_rank - 1]

                A_lower_diagonal_blocks[-1] = _A_lower_diagonal_blocks[2 * comm_rank]
                A_upper_diagonal_blocks[-1] = _A_upper_diagonal_blocks[2 * comm_rank]
            else:
                A_upper_buffer_blocks[-1] = _A_lower_diagonal_blocks[2 * comm_rank - 1]
                A_lower_buffer_blocks[-1] = _A_upper_diagonal_blocks[2 * comm_rank - 1]

            A_lower_arrow_blocks[0] = _A_lower_arrow_blocks[2 * comm_rank - 1]
            A_lower_arrow_blocks[-1] = _A_lower_arrow_blocks[2 * comm_rank]

            A_upper_arrow_blocks[0] = _A_upper_arrow_blocks[2 * comm_rank - 1]
            A_upper_arrow_blocks[-1] = _A_upper_arrow_blocks[2 * comm_rank]

            if quadratic:
                B_diagonal_blocks[0] = _B_diagonal_blocks[2 * comm_rank - 1]
                B_diagonal_blocks[-1] = _B_diagonal_blocks[2 * comm_rank]

                if comm_rank < comm_size - 1:
                    # Warning: The size of the upper/lower buffer follow the shape
                    # of the lower_diagonal_blocks slicing. That mean that the indexing
                    # is different between the last and the "middle" processes.
                    B_upper_buffer_blocks[-2] = _B_lower_diagonal_blocks[
                        2 * comm_rank - 1
                    ]
                    B_lower_buffer_blocks[-2] = _B_upper_diagonal_blocks[
                        2 * comm_rank - 1
                    ]

                    (B_lower_diagonal_blocks[-1]) = _B_lower_diagonal_blocks[
                        2 * comm_rank
                    ]
                    (B_upper_diagonal_blocks[-1]) = _B_upper_diagonal_blocks[
                        2 * comm_rank
                    ]
                else:
                    B_upper_buffer_blocks[-1] = _B_lower_diagonal_blocks[
                        2 * comm_rank - 1
                    ]
                    B_lower_buffer_blocks[-1] = _B_upper_diagonal_blocks[
                        2 * comm_rank - 1
                    ]

                B_lower_arrow_blocks[0] = _B_lower_arrow_blocks[2 * comm_rank - 1]
                B_lower_arrow_blocks[-1] = _B_lower_arrow_blocks[2 * comm_rank]

                B_upper_arrow_blocks[0] = _B_upper_arrow_blocks[2 * comm_rank - 1]
                B_upper_arrow_blocks[-1] = _B_upper_arrow_blocks[2 * comm_rank]

        A_arrow_tip_block[:] = _A_arrow_tip_block[:]
        if quadratic:
            B_arrow_tip_block[:] = _B_arrow_tip_block[:]
    else:
        raise ValueError("Unknown communication strategy.")
