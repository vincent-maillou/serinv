# Copyright 2023-2025 ETH Zurich. All rights reserved.

from mpi4py import MPI

from serinv import (
    ArrayLike,
    _get_module_from_array,
)

from serinv.algs import pobtf
from .pobtrs import (
    map_ppobtx_to_pobtrs,
    aggregate_pobtrs,
)


def ppobtf(
    A_diagonal_blocks: ArrayLike,
    A_lower_diagonal_blocks: ArrayLike,
    comm: MPI.Comm = MPI.COMM_WORLD,
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

    Keyword Arguments
    -----------------
    pobtrs : dict
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

    # Check for optional parameters
    device_streaming: bool = kwargs.get("device_streaming", False)
    strategy: str = kwargs.get("strategy", "allgather")
    root: int = kwargs.get("root", 0)

    # Check for given permutation buffer
    buffer: ArrayLike = kwargs.get("buffer", None)
    if comm_rank != 0:
        assert buffer is not None

    pobtrs: dict = kwargs.get("pobtrs", None)

    # Check for given reduced system buffers
    _A_diagonal_blocks: ArrayLike = pobtrs.get("A_diagonal_blocks", None)
    _A_lower_diagonal_blocks: ArrayLike = pobtrs.get("A_lower_diagonal_blocks", None)
    if any(
        x is None
        for x in [
            _A_diagonal_blocks,
            _A_lower_diagonal_blocks,
        ]
    ):
        raise ValueError(
            "To run the distributed solvers, the reduced system `pobtrs` need to contain the required arrays."
        )

    # Perform the parallel factorization
    if comm_rank == root:
        pobtf(
            A_diagonal_blocks=A_diagonal_blocks,
            A_lower_diagonal_blocks=A_lower_diagonal_blocks,
            factorize_last_block=False,
            device_streaming=device_streaming,
        )
    else:
        pobtf(
            A_diagonal_blocks=A_diagonal_blocks,
            A_lower_diagonal_blocks=A_lower_diagonal_blocks,
            buffer=buffer,
            device_streaming=device_streaming,
        )

    map_ppobtx_to_pobtrs(
        A_diagonal_blocks=A_diagonal_blocks,
        A_lower_diagonal_blocks=A_lower_diagonal_blocks,
        _A_diagonal_blocks=pobtrs["A_diagonal_blocks"],
        _A_lower_diagonal_blocks=pobtrs["A_lower_diagonal_blocks"],
        comm=comm,
        buffer=buffer,
        strategy=strategy,
    )

    aggregate_pobtrs(
        pobtrs=pobtrs,
        comm=comm,
        strategy=strategy,
        root=root,
    )

    # --- Factorize the reduced system ---
    if strategy == "gather-scatter":
        if comm_rank == root:
            pobtf(
                A_diagonal_blocks=pobtrs["A_diagonal_blocks"][1:],
                A_lower_diagonal_blocks=pobtrs["A_lower_diagonal_blocks"][1:-1],
                device_streaming=device_streaming,
            )
        else:
            # Do nothing.
            ...
    elif strategy == "allgather":
        pobtf(
            A_diagonal_blocks=pobtrs["A_diagonal_blocks"],
            A_lower_diagonal_blocks=pobtrs["A_lower_diagonal_blocks"],
            device_streaming=device_streaming,
        )

    comm.Barrier()
