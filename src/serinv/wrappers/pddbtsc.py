# Copyright 2023-2025 ETH Zurich. All rights reserved.

import time

from mpi4py import MPI

from serinv import (
    ArrayLike,
    _get_module_from_array,
)

from serinv.algs import ddbtsc
from .ddbtrs import (
    map_ddbtsc_to_ddbtrs,
    aggregate_ddbtrs,
)


def pddbtsc(
    A_diagonal_blocks: ArrayLike,
    A_lower_diagonal_blocks: ArrayLike,
    A_upper_diagonal_blocks: ArrayLike,
    comm: MPI.Comm = MPI.COMM_WORLD,
    nccl_comm: object = None,
    **kwargs,
) -> ArrayLike:
    """Perform the parallel Schur-complement of a block tridiagonal matrix.

    Parameters
    ----------
    A_diagonal_blocks : ArrayLike
        The diagonal blocks of the block tridiagonal with arrowhead matrix.
    A_lower_diagonal_blocks : ArrayLike
        The lower diagonal blocks of the block tridiagonal with arrowhead matrix.
    A_upper_diagonal_blocks : ArrayLike
        The upper diagonal blocks of the block tridiagonal with arrowhead matrix.
    comm : MPI.Comm
        The MPI communicator. Default is MPI.COMM_WORLD.

    Keyword Arguments
    -----------------
    rhs : dict
        The right-hand side of the equation to solve. If given, the rhs dictionary
        must contain the following arrays:
        - B_diagonal_blocks : ArrayLike
            The diagonal blocks of the right-hand side.
        - B_lower_diagonal_blocks : ArrayLike
            The lower diagonal blocks of the right-hand side.
        - B_upper_diagonal_blocks : ArrayLike
            The upper diagonal blocks of the right-hand side.
    quadratic : bool
        If True, and a rhs is given, the Schur-complement is performed for the equation AXA^T=B.
        If False, and a rhs is given, the Schur-complement is performed for the equation AX=B.
    buffers : dict
        The buffers to use in the permuted Schur-complement algorithm. If buffers are given, the
        Schur-complement is performed using the permuted Schur-complement algorithm.
        In the case of the `AX=I` equation the following buffers are required:
        - A_lower_buffer_blocks : ArrayLike
            The lower buffer blocks of the matrix A.
        - A_upper_buffer_blocks : ArrayLike
            The upper buffer blocks of the matrix A.
        In the case of `AXA^T=B` equation the following buffers are required:
        - B_lower_buffer_blocks : ArrayLike
            The lower buffer blocks of the matrix B.
        - B_upper_buffer_blocks : ArrayLike
            The upper buffer blocks of the matrix B.
    ddbtrs : dict
        The reduced system to use in the parallel implementation of the Schur-complement
        algorithm.
        The ddbtrs dictionary must contain the following arrays:
        - _A_diagonal_blocks : ArrayLike
            The diagonal blocks of the reduced system.
        - _A_lower_diagonal_blocks : ArrayLike
            The lower diagonal blocks of the reduced system.
        - _A_upper_diagonal_blocks : ArrayLike
            The upper diagonal blocks of the reduced system.
        In the case of the quadratic equation, the ddbtrs dictionary must also contain the reduced system for
        the right-hand side:
        - _rhs : dict
            The right-hand side of the reduced system. The _rhs dictionary must contain the following arrays:
            - _B_diagonal_blocks : ArrayLike
                The diagonal blocks of the reduced system.
            - _B_lower_diagonal_blocks : ArrayLike
                The lower diagonal blocks of the reduced system.
            - _B_upper_diagonal_blocks : ArrayLike
                The upper diagonal blocks of the reduced system.
    """

    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    if comm_size == 1:
        raise ValueError("The number of MPI processes must be greater than 1.")

    xp, _ = _get_module_from_array(arr=A_diagonal_blocks)

    rhs: dict = kwargs.get("rhs", None)
    quadratic: bool = kwargs.get("quadratic", False)
    buffers: dict = kwargs.get("buffers", None)
    ddbtrs: dict = kwargs.get("ddbtrs", None)
    strategy: str = kwargs.get("strategy", "allgather")

    # Check that the reduced system contains the required arrays
    ddbtrs: dict = kwargs.get("ddbtrs", None)
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
            "To run the distributed solvers, the reduced system `ddbtrs` need to contain the required arrays."
        )

    # Perform distributed Schur complement
    if comm_rank == 0:
        # Perform Schur-downward
        ddbtsc(
            A_diagonal_blocks=A_diagonal_blocks,
            A_lower_diagonal_blocks=A_lower_diagonal_blocks,
            A_upper_diagonal_blocks=A_upper_diagonal_blocks,
            rhs=rhs,
            quadratic=quadratic,
            invert_last_block=False,
            direction="downward",
        )
    elif comm_rank == comm_size - 1:
        # Perform Schur-upward
        ddbtsc(
            A_diagonal_blocks=A_diagonal_blocks,
            A_lower_diagonal_blocks=A_lower_diagonal_blocks,
            A_upper_diagonal_blocks=A_upper_diagonal_blocks,
            rhs=rhs,
            quadratic=quadratic,
            invert_last_block=False,
            direction="upward",
        )
    else:
        # Perform Schur on permuted partition
        ddbtsc(
            A_diagonal_blocks=A_diagonal_blocks,
            A_lower_diagonal_blocks=A_lower_diagonal_blocks,
            A_upper_diagonal_blocks=A_upper_diagonal_blocks,
            rhs=rhs,
            quadratic=quadratic,
            buffers=buffers,
            invert_last_block=False,
        )

    map_ddbtsc_to_ddbtrs(
        A_diagonal_blocks=A_diagonal_blocks,
        A_lower_diagonal_blocks=A_lower_diagonal_blocks,
        A_upper_diagonal_blocks=A_upper_diagonal_blocks,
        _A_diagonal_blocks=_A_diagonal_blocks,
        _A_lower_diagonal_blocks=_A_lower_diagonal_blocks,
        _A_upper_diagonal_blocks=_A_upper_diagonal_blocks,
        comm=comm,
        strategy=strategy,
        rhs=rhs,
        quadratic=quadratic,
        buffers=buffers,
        _rhs=ddbtrs.get("_rhs", None),
        nccl_comm=nccl_comm,
    )

    comm.Barrier()
    tic = time.perf_counter()
    aggregate_ddbtrs(
        ddbtrs=ddbtrs,
        quadratic=quadratic,
        comm=comm,
        strategy=strategy,
        nccl_comm=nccl_comm,
    )
    if xp.__name__ == "cupy":
        xp.cuda.runtime.deviceSynchronize()
    comm.Barrier()
    toc = time.perf_counter()
    elapsed = toc - tic

    # Perform Schur complement on the reduced system
    ddbtsc(
        A_diagonal_blocks=ddbtrs["A_diagonal_blocks"],
        A_lower_diagonal_blocks=ddbtrs["A_lower_diagonal_blocks"],
        A_upper_diagonal_blocks=ddbtrs["A_upper_diagonal_blocks"],
        rhs=ddbtrs.get("_rhs", None),
        quadratic=quadratic,
    )

    comm.Barrier()


    return elapsed