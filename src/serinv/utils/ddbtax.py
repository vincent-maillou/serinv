# Copyright 2023-2025 ETH Zurich. All rights reserved.

from serinv import (
    ArrayLike,
)

from .ddbtx import allocate_ddbtx_permutation_buffers

def allocate_ddbtax_permutation_buffers(
    A_lower_diagonal_blocks: ArrayLike,
    quadratic: bool = False,
):

    return allocate_ddbtx_permutation_buffers(A_lower_diagonal_blocks, quadratic)
