# Copyright 2023-2025 ETH Zurich. All rights reserved.

from serinv import (
    ArrayLike,
)

from .pobtx import allocate_pobtx_permutation_buffers


def allocate_pobtax_permutation_buffers(
    A_diagonal_blocks: ArrayLike,
    device_streaming: bool,
):

    return allocate_pobtx_permutation_buffers(A_diagonal_blocks, device_streaming)
