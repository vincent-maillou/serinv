# Copyright 2023-2025 ETH Zurich. All rights reserved.

from serinv import (
    ArrayLike,
    backend_flags,
    _get_module_from_array,
)

if backend_flags["cupy_avail"]:
    import cupyx as cpx


def allocate_pobtx_permutation_buffers(
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
    xp, _ = _get_module_from_array(arr=A_diagonal_blocks)

    if device_streaming:
        empty_like = cpx.empty_like_pinned
    else:
        empty_like = xp.empty_like

    buffer = empty_like(A_diagonal_blocks)

    return buffer
