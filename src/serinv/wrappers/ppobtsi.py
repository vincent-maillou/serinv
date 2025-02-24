# Copyright 2023-2025 ETH Zurich. All rights reserved.

from mpi4py import MPI

from serinv import (
    ArrayLike,
    _get_module_from_array,
)

from serinv.algs import pobtsi
from serinv.wrappers.pobtrs import (
    map_pobtrs_to_ppobtx,
    scatter_pobtrs,
)

comm_rank = MPI.COMM_WORLD.Get_rank()
comm_size = MPI.COMM_WORLD.Get_size()


def ppobtsi(
    L_diagonal_blocks: ArrayLike,
    L_lower_diagonal_blocks: ArrayLike,
    **kwargs,
):
    """Perform a selected inversion of a block tridiagonal with arrowhead matrix (pointing downward by convention).

    Parameters
    ----------
    L_diagonal_blocks : ArrayLike
        Diagonal blocks of the Cholesky factor of the matrix.
    L_lower_diagonal_blocks : ArrayLike
        Lower diagonal blocks of the Cholesky factor of the matrix.

    Keyword Arguments
    -----------------
    pobtrs : dict
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
    if comm_size == 1:
        raise ValueError("The number of MPI processes must be greater than 1.")

    xp, _ = _get_module_from_array(arr=L_diagonal_blocks)

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

    # Selected-inversion of the reduced system
    if strategy == "gather-scatter":
        if comm_rank == root:
            pobtsi(
                L_diagonal_blocks=pobtrs["A_diagonal_blocks"][1:],
                L_lower_diagonal_blocks=pobtrs["A_lower_diagonal_blocks"][1:-1],
            )
        MPI.COMM_WORLD.Barrier()

    elif strategy == "allgather":
        pobtsi(
            L_diagonal_blocks=pobtrs["A_diagonal_blocks"],
            L_lower_diagonal_blocks=pobtrs["A_lower_diagonal_blocks"],
        )

    scatter_pobtrs(
        pobtrs=pobtrs,
        strategy=strategy,
        root=root,
    )

    # Map result of the reduced system back to the original system
    map_pobtrs_to_ppobtx(
        A_diagonal_blocks=L_diagonal_blocks,
        A_lower_diagonal_blocks=L_lower_diagonal_blocks,
        _A_diagonal_blocks=pobtrs["A_diagonal_blocks"],
        _A_lower_diagonal_blocks=pobtrs["A_lower_diagonal_blocks"],
        buffer=buffer,
        strategy=strategy,
    )

    # Parallel selected inversion of the original system
    if comm_rank == 0:
        pobtsi(
            L_diagonal_blocks=L_diagonal_blocks,
            L_lower_diagonal_blocks=L_lower_diagonal_blocks,
            invert_last_block=False,
        )
    else:
        pobtsi(
            L_diagonal_blocks=L_diagonal_blocks,
            L_lower_diagonal_blocks=L_lower_diagonal_blocks,
            device_streaming=device_streaming,
            buffer=buffer,
        )
