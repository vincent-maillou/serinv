# Copyright 2023-2025 ETH Zurich. All rights reserved.

from mpi4py import MPI

from serinv import (
    ArrayLike,
    _get_module_from_array,
)

from serinv.algs import ddbtsci
from .ddbtrs import (
    map_ddbtrs_to_ddbtsci,
    scatter_ddbtrs,
)

comm_rank = MPI.COMM_WORLD.Get_rank()
comm_size = MPI.COMM_WORLD.Get_size()


def pddbtsci(
    A_diagonal_blocks: ArrayLike,
    A_lower_diagonal_blocks: ArrayLike,
    A_upper_diagonal_blocks: ArrayLike,
    **kwargs,
) -> ArrayLike:
    """Perform the parallel selected-inversion of the Schur-complement of a block tridiagonal matrix.
    This routine is refered to as Schur-complement Selected Inversion (SCI).

    Parameters
    ----------
    A_diagonal_blocks : ArrayLike
        The diagonal blocks of the block tridiagonal matrix.
    A_lower_diagonal_blocks : ArrayLike
        The lower diagonal blocks of the block tridiagonal matrix.
    A_upper_diagonal_blocks : ArrayLike
        The upper diagonal blocks of the block tridiagonal matrix.

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
    if comm_size == 1:
        raise ValueError("The number of MPI processes must be greater than 1.")

    xp, _ = _get_module_from_array(arr=A_diagonal_blocks)

    rhs: dict = kwargs.get("rhs", None)
    quadratic: bool = kwargs.get("quadratic", False)
    buffers: dict = kwargs.get("buffers", None)
    ddbtrs: dict = kwargs.get("ddbtrs", None)

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

    ddbtsci(
        A_diagonal_blocks=ddbtrs["A_diagonal_blocks"],
        A_lower_diagonal_blocks=ddbtrs["A_lower_diagonal_blocks"],
        A_upper_diagonal_blocks=ddbtrs["A_upper_diagonal_blocks"],
        rhs=ddbtrs.get("_rhs", None),
        quadratic=quadratic,
    )

    scatter_ddbtrs(
        ddbtrs=ddbtrs,
        quadratic=quadratic,
    )

    map_ddbtrs_to_ddbtsci(
        A_diagonal_blocks=A_diagonal_blocks,
        A_lower_diagonal_blocks=A_lower_diagonal_blocks,
        A_upper_diagonal_blocks=A_upper_diagonal_blocks,
        _A_diagonal_blocks=_A_diagonal_blocks,
        _A_lower_diagonal_blocks=_A_lower_diagonal_blocks,
        _A_upper_diagonal_blocks=_A_upper_diagonal_blocks,
        rhs=rhs,
        quadratic=quadratic,
        buffers=buffers,
        _rhs=ddbtrs.get("_rhs", None),
    )

    # Perform distributed SCI
    if comm_rank == 0:
        # Perform SCI-downward
        ddbtsci(
            A_diagonal_blocks=A_diagonal_blocks,
            A_lower_diagonal_blocks=A_lower_diagonal_blocks,
            A_upper_diagonal_blocks=A_upper_diagonal_blocks,
            rhs=rhs,
            quadratic=quadratic,
            invert_last_block=False,
            direction="downward",
        )
    elif comm_rank == comm_size - 1:
        # Perform SCI-upward
        ddbtsci(
            A_diagonal_blocks=A_diagonal_blocks,
            A_lower_diagonal_blocks=A_lower_diagonal_blocks,
            A_upper_diagonal_blocks=A_upper_diagonal_blocks,
            rhs=rhs,
            quadratic=quadratic,
            invert_last_block=False,
            direction="upward",
        )
    else:
        # Perform SCI on permuted partition
        ddbtsci(
            A_diagonal_blocks=A_diagonal_blocks,
            A_lower_diagonal_blocks=A_lower_diagonal_blocks,
            A_upper_diagonal_blocks=A_upper_diagonal_blocks,
            rhs=rhs,
            quadratic=quadratic,
            buffers=buffers,
            invert_last_block=False,
        )
