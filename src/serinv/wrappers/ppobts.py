# Copyright 2023-2025 ETH Zurich. All rights reserved.

import time

from mpi4py import MPI

from serinv import (
    ArrayLike,
    _get_module_from_array,
)

from serinv.algs import pobts
from serinv.wrappers.pobtrs import (
    map_ppobts_to_pobtrss,
    aggregate_pobtrss,
    scatter_pobtrss,
    map_pobtrss_to_ppobts,
)


def ppobts(
    L_diagonal_blocks: ArrayLike,
    L_lower_diagonal_blocks: ArrayLike,
    B: ArrayLike,
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
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

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

    _B: ArrayLike = pobtrs.get("B", None)
    if _B is None:
        raise ValueError(
            "To run the distributed rhs-solve, the reduced system `pobtrs` need to contain the required arrays: 'B'."
        )

    # Isolate the tip block of the RHS
    b = L_diagonal_blocks.shape[1]

    # Parallel forward solve
    if comm_rank == root:
        pobts(
            L_diagonal_blocks=L_diagonal_blocks,
            L_lower_diagonal_blocks=L_lower_diagonal_blocks,
            B=B,
            trans="N",
            partial=True,
        )
    else:
        pobts(
            L_diagonal_blocks=L_diagonal_blocks,
            L_lower_diagonal_blocks=L_lower_diagonal_blocks,
            B=B,
            buffer=buffer,
            trans="N",
        )

    # Map RHS to reduced RHS
    map_ppobts_to_pobtrss(
        A_diagonal_blocks=L_diagonal_blocks,
        B=B,
        _B=_B,
        comm=comm,
        strategy=strategy,
        nccl_comm=nccl_comm,
    )

    # Agregate reduced RHS
    comm.Barrier()
    tic = time.perf_counter()
    aggregate_pobtrss(
        A_diagonal_blocks=L_diagonal_blocks,
        pobtrs=pobtrs,
        comm=comm,
        strategy=strategy,
        nccl_comm=nccl_comm,
    )
    if xp.__name__ == "cupy":
        xp.cuda.runtime.deviceSynchronize()
    comm.Barrier()
    toc = time.perf_counter()
    elapsed = toc - tic

    # Solve RHS FWD/BWD
    if strategy == "allgather":
        pobts(
            L_diagonal_blocks=_A_diagonal_blocks[1:],
            L_lower_diagonal_blocks=_A_lower_diagonal_blocks[1:-1],
            B=_B[b:],
            trans="N",
        )
        pobts(
            L_diagonal_blocks=_A_diagonal_blocks[1:],
            L_lower_diagonal_blocks=_A_lower_diagonal_blocks[1:-1],
            B=_B[b:],
            trans="C",
        )
    else:
        raise NotImplementedError(f"The strategy {strategy} is not yet implemented.")

    # Scatter solution of reduced RHS
    scatter_pobtrss(
        A_diagonal_blocks=L_diagonal_blocks,
        pobtrs=pobtrs,
        comm=comm,
        strategy=strategy,
        nccl_comm=nccl_comm,
    )

    # Map solution of reduced RHS to RHS
    map_pobtrss_to_ppobts(
        A_diagonal_blocks=L_diagonal_blocks,
        B=B,
        _B=_B,
        comm=comm,
        strategy=strategy,
        nccl_comm=nccl_comm,
    )

    # Parallel backward solve
    if comm_rank == root:
        pobts(
            L_diagonal_blocks=L_diagonal_blocks,
            L_lower_diagonal_blocks=L_lower_diagonal_blocks,
            B=B,
            trans="C",
            partial=True,
        )
    else:
        pobts(
            L_diagonal_blocks=L_diagonal_blocks,
            L_lower_diagonal_blocks=L_lower_diagonal_blocks,
            B=B,
            buffer=buffer,
            trans="C",
        )

    return elapsed