# Copyright 2023-2025 ETH Zurich. All rights reserved.

from serinv import (
    ArrayLike,
    _get_module_from_array,
)

def allocate_ddbtx_permutation_buffers(
    A_lower_diagonal_blocks: ArrayLike,
    quadratic: bool = False,
):
    xp, _ = _get_module_from_array(arr=A_lower_diagonal_blocks)

    if quadratic:
        A_lower_buffer_blocks = xp.zeros_like(A_lower_diagonal_blocks)
        A_upper_buffer_blocks = xp.zeros_like(A_lower_diagonal_blocks)

        B_lower_buffer_blocks = xp.zeros_like(A_lower_diagonal_blocks)
        B_upper_buffer_blocks = xp.zeros_like(A_lower_diagonal_blocks)
        
        buffers = {
            "A_lower_buffer_blocks": A_lower_buffer_blocks,
            "A_upper_buffer_blocks": A_upper_buffer_blocks,
            "B_lower_buffer_blocks": B_lower_buffer_blocks,
            "B_upper_buffer_blocks": B_upper_buffer_blocks,
        }
    else:
        A_lower_buffer_blocks = xp.zeros_like(A_lower_diagonal_blocks)
        A_upper_buffer_blocks = xp.zeros_like(A_lower_diagonal_blocks)
        
        buffers = {
            "A_lower_buffer_blocks": A_lower_buffer_blocks,
            "A_upper_buffer_blocks": A_upper_buffer_blocks,
        }

    return buffers
