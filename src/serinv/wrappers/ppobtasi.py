# Copyright 2023-2025 ETH Zurich. All rights reserved.

from mpi4py import MPI

from serinv import (
    ArrayLike,
)

from serinv.algs import pobtasi
from serinv.wrappers.pobtars import (
    map_pobtars_to_ppobtax,
    scatter_pobtars,
)


def ppobtasi(
    L_diagonal_blocks: ArrayLike,
    L_lower_diagonal_blocks: ArrayLike,
    L_lower_arrow_blocks: ArrayLike,
    L_arrow_tip_block: ArrayLike,
    comm: MPI.Comm = MPI.COMM_WORLD,
    nccl_comm: object = None,
    **kwargs,
):
    """Perform a selected inversion of a block tridiagonal with arrowhead matrix (pointing downward by convention).

    Parameters
    ----------
    L_diagonal_blocks : ArrayLike
        Diagonal blocks of the Cholesky factor of the matrix.
    L_lower_diagonal_blocks : ArrayLike
        Lower diagonal blocks of the Cholesky factor of the matrix.
    L_lower_arrow_blocks : ArrayLike
        Arrow bottom blocks of the Cholesky factor of the matrix.
    L_arrow_tip_block : ArrayLike
        Arrow tip block of the Cholesky factor of the matrix.

    Keyword Arguments
    -----------------
    pobtars : dict
        The reduced system arrays, given as dictionary format.
    buffer : ArrayLike
        The permutation buffer for the permuted-partition algorithms
    device_streaming : bool, optional
        If True, the algorithm will run on the GPU. (default: False)
    strategy : str, optional
        The communication strategy to use. (default: "allgather")
    root : int, optional
        The root rank for the communication strategy. (default: 0)

    Note:
    -----
    - Will overwrite the inputs and store the results in them (in-place).
    - If a device array is given, the algorithm will run on the GPU.

    """
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    if comm_size == 1:
        raise ValueError("The number of MPI processes must be greater than 1.")

    # Check for optional parameters
    device_streaming: bool = kwargs.get("device_streaming", False)
    strategy: str = kwargs.get("strategy", "allgather")
    root: int = kwargs.get("root", 0)

    # Check for given permutation buffer
    buffer: ArrayLike = kwargs.get("buffer", None)
    if comm_rank != 0:
        assert buffer is not None

    pobtars: dict = kwargs.get("pobtars", None)

    # Check for given reduced system buffers
    _A_diagonal_blocks: ArrayLike = pobtars.get("A_diagonal_blocks", None)
    _A_lower_diagonal_blocks: ArrayLike = pobtars.get("A_lower_diagonal_blocks", None)
    _A_lower_arrow_blocks: ArrayLike = pobtars.get("A_lower_arrow_blocks", None)
    _A_arrow_tip_block: ArrayLike = pobtars.get("A_arrow_tip_block", None)
    if any(
        x is None
        for x in [
            _A_diagonal_blocks,
            _A_lower_diagonal_blocks,
            _A_lower_arrow_blocks,
            _A_arrow_tip_block,
        ]
    ):
        raise ValueError(
            "To run the distributed solvers, the reduced system `pobtars` need to contain the required arrays."
        )

    # Selected-inversion of the reduced system
    if strategy == "gather-scatter":
        if comm_rank == root:
            pobtasi(
                L_diagonal_blocks=pobtars["A_diagonal_blocks"][1:],
                L_lower_diagonal_blocks=pobtars["A_lower_diagonal_blocks"][1:-1],
                L_lower_arrow_blocks=pobtars["A_lower_arrow_blocks"][1:],
                L_arrow_tip_block=pobtars["A_arrow_tip_block"],
            )
        comm.Barrier()

    elif strategy == "allgather":
        pobtasi(
            L_diagonal_blocks=pobtars["A_diagonal_blocks"][1:],
            L_lower_diagonal_blocks=pobtars["A_lower_diagonal_blocks"][1:-1],
            L_lower_arrow_blocks=pobtars["A_lower_arrow_blocks"][1:],
            L_arrow_tip_block=pobtars["A_arrow_tip_block"],
        )

    scatter_pobtars(
        pobtars=pobtars,
        comm=comm,
        strategy=strategy,
        root=root,
        nccl_comm=nccl_comm,
    )

    # Map result of the reduced system back to the original system
    map_pobtars_to_ppobtax(
        A_diagonal_blocks=L_diagonal_blocks,
        A_lower_diagonal_blocks=L_lower_diagonal_blocks,
        A_lower_arrow_blocks=L_lower_arrow_blocks,
        A_arrow_tip_block=L_arrow_tip_block,
        _A_diagonal_blocks=pobtars["A_diagonal_blocks"],
        _A_lower_diagonal_blocks=pobtars["A_lower_diagonal_blocks"],
        _A_lower_arrow_blocks=pobtars["A_lower_arrow_blocks"],
        _A_arrow_tip_block=pobtars["A_arrow_tip_block"],
        comm=comm,
        buffer=buffer,
        strategy=strategy,
        nccl_comm=nccl_comm,
    )

    # Parallel selected inversion of the original system
    if comm_rank == root:
        pobtasi(
            L_diagonal_blocks=L_diagonal_blocks,
            L_lower_diagonal_blocks=L_lower_diagonal_blocks,
            L_lower_arrow_blocks=L_lower_arrow_blocks,
            L_arrow_tip_block=L_arrow_tip_block,
            invert_last_block=False,
        )
    else:
        pobtasi(
            L_diagonal_blocks=L_diagonal_blocks,
            L_lower_diagonal_blocks=L_lower_diagonal_blocks,
            L_lower_arrow_blocks=L_lower_arrow_blocks,
            L_arrow_tip_block=L_arrow_tip_block,
            device_streaming=device_streaming,
            buffer=buffer,
        )
