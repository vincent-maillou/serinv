# Copyright 2023-2025 ETH Zurich. All rights reserved.

import time

from mpi4py import MPI

from serinv import (
    ArrayLike,
    _get_module_from_array,
)

from serinv.algs import pobtaf
from .pobtars import (
    map_ppobtax_to_pobtars,
    aggregate_pobtars,
)


def ppobtaf(
    A_diagonal_blocks: ArrayLike,
    A_lower_diagonal_blocks: ArrayLike,
    A_lower_arrow_blocks: ArrayLike,
    A_arrow_tip_block: ArrayLike,
    comm: MPI.Comm = MPI.COMM_WORLD,
    nccl_comm: object = None,
    **kwargs,
) -> ArrayLike:
    """Perform the parallel factorization of a block tridiagonal with arrowhead matrix
    (pointing downward by convention).

    Parameters
    ----------
    A_diagonal_blocks : ArrayLike
        The diagonal blocks of the block tridiagonal with arrowhead matrix.
    A_lower_diagonal_blocks : ArrayLike
        The lower diagonal blocks of the block tridiagonal with arrowhead matrix.
    A_lower_arrow_blocks : ArrayLike
        The arrow bottom blocks of the block tridiagonal with arrowhead matrix.
    A_arrow_tip_block : ArrayLike
        The arrow tip block of the block tridiagonal with arrowhead matrix.

    Keyword Arguments
    -----------------
    pobtars : dict
        The reduced system arrays, given as dictionary format.
    buffer : ArrayLike
        The permutation buffer for the permuted-partition algorithms
    device_streaming : bool, optional
        If True, the algorithm will perform host-device streaming. (default: False)
    strategy : str, optional
        The communication strategy to use. (default: "allgather")
    root : int, optional
        The root rank for the communication strategy. (default: 0)

    Note:
    -----
    - The matrix, A, is assumed to be symmetric positive definite.
    - Will overwrite the inputs and store the results in them (in-place).
    - If a device array is given, the algorithm will run on the GPU.

    """
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    if comm_size == 1:
        raise ValueError("The number of MPI processes must be greater than 1.")

    xp, _ = _get_module_from_array(arr=A_diagonal_blocks)

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
            "To run the distributed solvers, the reduced system `ddbtars` need to contain the required arrays."
        )

    # Reset the arrow tip block in case of reccurent call using the same buffers
    pobtars["A_arrow_tip_block"][:, :] = 0.0
    pobtars["A_arrow_tip_block_comm"][:, :] = 0.0

    # Store the value of the tip of the arrow and reset the local arrow tip block to zero
    # in order to correctly accumulate the updates from the distributed Schur complement.
    A_arrow_tip_initial = A_arrow_tip_block.copy()
    A_arrow_tip_block[:] = 0.0

    # Perform the parallel factorization
    if comm_rank == root:
        pobtaf(
            A_diagonal_blocks=A_diagonal_blocks,
            A_lower_diagonal_blocks=A_lower_diagonal_blocks,
            A_lower_arrow_blocks=A_lower_arrow_blocks,
            A_arrow_tip_block=A_arrow_tip_block,
            device_streaming=device_streaming,
            factorize_last_block=False,
        )
    else:
        pobtaf(
            A_diagonal_blocks=A_diagonal_blocks,
            A_lower_diagonal_blocks=A_lower_diagonal_blocks,
            A_lower_arrow_blocks=A_lower_arrow_blocks,
            A_arrow_tip_block=A_arrow_tip_block,
            device_streaming=device_streaming,
            buffer=buffer,
        )

    map_ppobtax_to_pobtars(
        A_diagonal_blocks=A_diagonal_blocks,
        A_lower_diagonal_blocks=A_lower_diagonal_blocks,
        A_lower_arrow_blocks=A_lower_arrow_blocks,
        A_arrow_tip_block=A_arrow_tip_block,
        _A_diagonal_blocks=pobtars["A_diagonal_blocks"],
        _A_lower_diagonal_blocks=pobtars["A_lower_diagonal_blocks"],
        _A_lower_arrow_blocks=pobtars["A_lower_arrow_blocks"],
        _A_arrow_tip_block=pobtars["A_arrow_tip_block"],
        buffer=buffer,
        strategy=strategy,
        comm=comm,
        nccl_comm=nccl_comm,
    )

    MPI.COMM_WORLD.Barrier()
    tic = time.perf_counter()
    aggregate_pobtars(
        pobtars=pobtars,
        comm=comm,
        strategy=strategy,
        root=root,
        nccl_comm=nccl_comm,
    )
    if xp.__name__ == "cupy":
        xp.cuda.runtime.deviceSynchronize()
    MPI.COMM_WORLD.Barrier()
    toc = time.perf_counter()
    elapsed = toc - tic

    # --- Factorize the reduced system ---
    pobtars["A_arrow_tip_block"][:] += A_arrow_tip_initial

    if strategy == "gather-scatter":
        if comm_rank == root:
            pobtaf(
                A_diagonal_blocks=pobtars["A_diagonal_blocks"][1:],
                A_lower_diagonal_blocks=pobtars["A_lower_diagonal_blocks"][1:-1],
                A_lower_arrow_blocks=pobtars["A_lower_arrow_blocks"][1:],
                A_arrow_tip_block=pobtars["A_arrow_tip_block"],
            )
        else:
            # Do nothing.
            ...
    elif strategy == "allgather":
        pobtaf(
            A_diagonal_blocks=pobtars["A_diagonal_blocks"][1:],
            A_lower_diagonal_blocks=pobtars["A_lower_diagonal_blocks"][1:-1],
            A_lower_arrow_blocks=pobtars["A_lower_arrow_blocks"][1:],
            A_arrow_tip_block=pobtars["A_arrow_tip_block"],
        )

    comm.Barrier()

    return elapsed