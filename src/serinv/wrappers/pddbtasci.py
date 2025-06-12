# Copyright 2023-2025 ETH Zurich. All rights reserved.

from mpi4py import MPI

from serinv import (
    ArrayLike,
    _get_module_from_array,
)

from serinv.algs import ddbtasci
from .ddbtars import (
    map_ddbtars_to_ddbtasci,
    scatter_ddbtars,
)


def pddbtasci(
    A_diagonal_blocks: ArrayLike,
    A_lower_diagonal_blocks: ArrayLike,
    A_upper_diagonal_blocks: ArrayLike,
    A_lower_arrow_blocks: ArrayLike,
    A_upper_arrow_blocks: ArrayLike,
    A_arrow_tip_block: ArrayLike,
    comm: MPI.Comm = MPI.COMM_WORLD,
    nccl_comm: object = None,
    **kwargs,
) -> ArrayLike:
    """Perform the parallel selected-inversion of the Schur-complement of a block tridiagonal matrix.
    This routine is refered to as Schur-complement Selected Inversion (SCI).

    Parameters
    ----------
    A_diagonal_blocks : ArrayLike
        The diagonal blocks of the block tridiagonal with arrowhead matrix.
    A_lower_diagonal_blocks : ArrayLike
        The lower diagonal blocks of the block tridiagonal with arrowhead matrix.
    A_upper_diagonal_blocks : ArrayLike
        The upper diagonal blocks of the block tridiagonal with arrowhead matrix.
    A_lower_arrow_blocks : ArrayLike
        The arrow bottom blocks of the block tridiagonal with arrowhead matrix.
    A_upper_arrow_blocks : ArrayLike
        The arrow top blocks of the block tridiagonal with arrowhead matrix.
    A_arrow_tip_block : ArrayLike
        The arrow tip block of the block tridiagonal with arrowhead matrix.
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
        - B_lower_arrow_blocks : ArrayLike
            The arrow bottom blocks of the right-hand side.
        - B_upper_arrow_blocks : ArrayLike
            The arrow top blocks of the right-hand side.
        - B_arrow_tip_block : ArrayLike
            The arrow tip block of the right-hand side.
    quadratic : bool
        If True, and a rhs is given, the SCI is performed for the equation AXA^T=B.
        If False, and a rhs is given, the SCI is performed for the equation AX=B.
    buffers : dict
        The buffers to use in the permuted SCI algorithm. If buffers are given, the
        SCI is performed using the permuted SCI algorithm.
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
    ddbtars : dict
        The reduced system to use in the parallel implementation of the SCI
        algorithm.
        The ddbtars dictionary must contain the following arrays:
        - _A_diagonal_blocks : ArrayLike
            The diagonal blocks of the reduced system.
        - _A_lower_diagonal_blocks : ArrayLike
            The lower diagonal blocks of the reduced system.
        - _A_upper_diagonal_blocks : ArrayLike
            The upper diagonal blocks of the reduced system.
        - _A_lower_arrow_blocks : ArrayLike
            The arrow bottom blocks of the reduced system.
        - _A_upper_arrow_blocks : ArrayLike
            The arrow top blocks of the reduced system.
        - _A_arrow_tip_block : ArrayLike
            The arrow tip block of the reduced system.
        In the case of the quadratic equation, the ddbtars dictionary must also contain the reduced system for
        the right-hand side:
        - _rhs : dict
            The right-hand side of the reduced system. The _rhs dictionary must contain the following arrays:
            - _B_diagonal_blocks : ArrayLike
                The diagonal blocks of the reduced system.
            - _B_lower_diagonal_blocks : ArrayLike
                The lower diagonal blocks of the reduced system.
            - _B_upper_diagonal_blocks : ArrayLike
                The upper diagonal blocks of the reduced system.
            - _B_lower_arrow_blocks : ArrayLike
                The arrow bottom blocks of the reduced system.
            - _B_upper_arrow_blocks : ArrayLike
                The arrow top blocks of the reduced system.
            - _B_arrow_tip_block : ArrayLike
                The arrow tip block of the reduced system.
    """
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    if comm_size == 1:
        raise ValueError("The number of MPI processes must be greater than 1.")

    rhs: dict = kwargs.get("rhs", None)
    quadratic: bool = kwargs.get("quadratic", False)
    buffers: dict = kwargs.get("buffers", None)
    ddbtars: dict = kwargs.get("ddbtars", None)

    # Check that the reduced system contains the required arrays
    ddbtars: dict = kwargs.get("ddbtars", None)
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
            "To run the distributed solvers, the reduced system `ddbtars` need to contain the required arrays."
        )

    ddbtasci(
        A_diagonal_blocks=ddbtars["A_diagonal_blocks"],
        A_lower_diagonal_blocks=ddbtars["A_lower_diagonal_blocks"],
        A_upper_diagonal_blocks=ddbtars["A_upper_diagonal_blocks"],
        A_lower_arrow_blocks=ddbtars["A_lower_arrow_blocks"],
        A_upper_arrow_blocks=ddbtars["A_upper_arrow_blocks"],
        A_arrow_tip_block=ddbtars["A_arrow_tip_block"],
        rhs=ddbtars.get("_rhs", None),
        quadratic=quadratic,
    )

    scatter_ddbtars(
        ddbtars=ddbtars,
        comm=comm,
        quadratic=quadratic,
        nccl_comm=nccl_comm,
    )

    map_ddbtars_to_ddbtasci(
        A_diagonal_blocks=A_diagonal_blocks,
        A_lower_diagonal_blocks=A_lower_diagonal_blocks,
        A_upper_diagonal_blocks=A_upper_diagonal_blocks,
        A_lower_arrow_blocks=A_lower_arrow_blocks,
        A_upper_arrow_blocks=A_upper_arrow_blocks,
        A_arrow_tip_block=A_arrow_tip_block,
        _A_diagonal_blocks=ddbtars["A_diagonal_blocks"],
        _A_lower_diagonal_blocks=ddbtars["A_lower_diagonal_blocks"],
        _A_upper_diagonal_blocks=ddbtars["A_upper_diagonal_blocks"],
        _A_lower_arrow_blocks=ddbtars["A_lower_arrow_blocks"],
        _A_upper_arrow_blocks=ddbtars["A_upper_arrow_blocks"],
        _A_arrow_tip_block=ddbtars["A_arrow_tip_block"],
        comm=comm,
        rhs=rhs,
        quadratic=quadratic,
        buffers=buffers,
        _rhs=ddbtars.get("_rhs", None),
        nccl_comm=nccl_comm,
    )

    # Perform distributed SCI
    if comm_rank == 0:
        # Perform SCI-downward
        ddbtasci(
            A_diagonal_blocks=A_diagonal_blocks,
            A_lower_diagonal_blocks=A_lower_diagonal_blocks,
            A_upper_diagonal_blocks=A_upper_diagonal_blocks,
            A_lower_arrow_blocks=A_lower_arrow_blocks,
            A_upper_arrow_blocks=A_upper_arrow_blocks,
            A_arrow_tip_block=A_arrow_tip_block,
            rhs=rhs,
            quadratic=quadratic,
            invert_last_block=False,
        )
    else:
        # Perform SCI on permuted partition
        ddbtasci(
            A_diagonal_blocks=A_diagonal_blocks,
            A_lower_diagonal_blocks=A_lower_diagonal_blocks,
            A_upper_diagonal_blocks=A_upper_diagonal_blocks,
            A_lower_arrow_blocks=A_lower_arrow_blocks,
            A_upper_arrow_blocks=A_upper_arrow_blocks,
            A_arrow_tip_block=A_arrow_tip_block,
            rhs=rhs,
            quadratic=quadratic,
            buffers=buffers,
        )
