# Copyright 2023-2024 ETH Zurich. All rights reserved.

import numpy as np

try:
    import cupy as cp
except:
    ...

from serinv import ArrayLike, CUPY_AVAIL, _get_array_module


def diagonally_dominant_bt(
    n: int, b: int, device_arr: bool = False, symmetric: bool = False, seed: int = 63
) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    if device_arr and CUPY_AVAIL:
        xp = cp
    elif device_arr:
        raise ValueError("Device array have been requested but CuPy is not available.")
    else:
        xp = np

    xp.random.seed(seed)

    A_diagonal_blocks = xp.random.rand(n, b, b)
    A_lower_diagonal_blocks = xp.random.rand(n - 1, b, b)
    A_upper_diagonal_blocks = xp.random.rand(n - 1, b, b)

    if symmetric:
        for n_i in range(n):
            A_diagonal_blocks[n_i] = A_diagonal_blocks[n_i] + A_diagonal_blocks[n_i].T

            if n_i < n - 1:
                A_upper_diagonal_blocks[n_i] = A_lower_diagonal_blocks[n_i].T

    for n_i in range(n):
        A_diagonal_blocks[n_i] += xp.diag(
            xp.sum(xp.abs(A_diagonal_blocks[n_i]), axis=1)
        )
        if n_i > 0:
            A_diagonal_blocks[n_i] += xp.diag(
                xp.sum(xp.abs(A_lower_diagonal_blocks[n_i - 1]), axis=1)
            )
        if n_i < n - 1:
            A_diagonal_blocks[n_i] += xp.diag(
                xp.sum(xp.abs(A_upper_diagonal_blocks[n_i]), axis=1)
            )

    return (
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_upper_diagonal_blocks,
    )


def bt_to_dense(
    A_diagonal_blocks,
    A_lower_diagonal_blocks,
    A_upper_diagonal_blocks,
) -> ArrayLike:
    xp, _ = _get_array_module(A_diagonal_blocks)

    n = A_diagonal_blocks.shape[0]
    b = A_diagonal_blocks.shape[1]

    A = xp.zeros((n * b, n * b), dtype=A_diagonal_blocks.dtype)

    for i in range(n):
        A[i * b : (i + 1) * b, i * b : (i + 1) * b] = A_diagonal_blocks[i]
        if i > 0:
            A[i * b : (i + 1) * b, (i - 1) * b : i * b] = A_lower_diagonal_blocks[i - 1]
        if i < n - 1:
            A[i * b : (i + 1) * b, (i + 1) * b : (i + 2) * b] = A_upper_diagonal_blocks[
                i
            ]

    return A
