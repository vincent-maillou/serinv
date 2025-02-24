# Copyright 2023-2025 ETH Zurich. All rights reserved.

from mpi4py import MPI

from serinv import (
    ArrayLike,
    _get_module_from_array,
)

from serinv.algs import pobtas


comm_rank = MPI.COMM_WORLD.Get_rank()
comm_size = MPI.COMM_WORLD.Get_size()


def ppobtas(
    L_diagonal_blocks: ArrayLike,
    L_lower_diagonal_blocks: ArrayLike,
    L_lower_arrow_blocks: ArrayLike,
    L_arrow_tip_block: ArrayLike,
    B: ArrayLike,
    _L_diagonal_blocks: ArrayLike,
    _L_lower_diagonal_blocks: ArrayLike,
    _L_lower_arrow_blocks: ArrayLike,
    L_permutation_buffer: ArrayLike,
    **kwargs,
):
    if comm_size == 1:
        raise ValueError("The number of MPI processes must be greater than 1.")

    xp, _ = _get_module_from_array(arr=L_diagonal_blocks)

    # Check for optional parameters
    # device_streaming: bool = kwargs.get("device_streaming", False)
    # strategy: str = kwargs.get("strategy", "allgather")
    # root: int = kwargs.get("root", 0)

    """ pobtas(
        L_diagonal_blocks=L_diagonal_blocks,
        L_lower_diagonal_blocks=L_lower_diagonal_blocks,
        L_lower_arrow_blocks=L_lower_arrow_blocks,
        L_arrow_tip_block=L_arrow_tip_block,
        B=B,
    )
        
    return B """

    # 1. Map local RHS to the reduced system RHS
    n_rhs = B.shape[1]
    _n: int = 2 * comm_size - 1
    b = L_diagonal_blocks.shape[0]
    a = L_arrow_tip_block.shape[0]

    _B = xp.empty((_n * b + a, n_rhs), dtype=B.dtype)
    if comm_rank == 0:
        # Rank 0 map only the last block
        _B[:b] = B[-b - a : -a]
    else:
        # Other ranks map their first and last blocks
        _B[(2 * comm_rank - 1) * b : 2 * comm_rank * b] = B[:b]
        _B[2 * comm_rank * b : (2 * comm_rank + 1) * b] = B[-b - a : -a]

    # 2. Communicate the reduced RHS
    MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, _B, op=MPI.SUM)
    MPI.COMM_WORLD.Barrier()

    # Don't forget to add the tip of the arrow-RHS
    _B[-a:] = B[-a:]

    # 3. Solve the reduced RHS
    pobtas(
        L_diagonal_blocks=_L_diagonal_blocks,
        L_lower_diagonal_blocks=_L_lower_diagonal_blocks,
        L_lower_arrow_blocks=_L_lower_arrow_blocks,
        L_arrow_tip_block=L_arrow_tip_block,
        B=_B,
    )

    # 4. Map back the reduced solution to the local solution
    if comm_rank == 0:
        # Rank 0 map only the last block
        B[-b - a : -a] = _B[:b]
    else:
        # Other ranks map their first and last blocks
        B[:b] = _B[(2 * comm_rank - 1) * b : 2 * comm_rank * b]
        B[-b - a : -a] = _B[2 * comm_rank * b : (2 * comm_rank + 1) * b]
    B[-a:] = _B[-a:]

    return _B
