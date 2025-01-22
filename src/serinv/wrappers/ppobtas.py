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
    L_arrow_bottom_blocks: ArrayLike,
    L_arrow_tip_block: ArrayLike,
    _L_diagonal_blocks: ArrayLike,
    _L_lower_diagonal_blocks: ArrayLike,
    _L_lower_arrow_blocks: ArrayLike,
    L_permutation_buffer: ArrayLike,
    **kwargs,
):
    ...