# Copyright 2023-2025 ETH Zurich. All rights reserved.

from serinv import (
    ArrayLike,
)

from .pobtx import allocate_pobtx_permutation_buffers


def allocate_pobtax_permutation_buffers(
    A_diagonal_blocks: ArrayLike,
    device_streaming: bool,
):
    """Allocate the (permutation) buffers necessary for the parallel BTA algorithms.

    Parameters
    ----------
    A_diagonal_blocks : ArrayLike
        The diagonal blocks of the original system.
    device_streaming : bool
        If True, pinned host-arrays will be allocated

    Returns
    -------
    buffer : ArrayLike
        The permutation buffer needed for the parallel BTA algorithms.
    """

    return allocate_pobtx_permutation_buffers(A_diagonal_blocks, device_streaming)
