# Copyright 2023-2024 ETH Zurich. All rights reserved.

import numpy as np

try:
    import cupy as cp
except:
    ...

from serinv import ArrayLike, CUPY_AVAIL, _get_array_module


def diagonally_dominant_bta(
    n: int,
    b: int,
    a: int,
    direction: str = "downward",
    device_arr: bool = False,
    symmetric: bool = False,
    seed: int = 63,
) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
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
    A_arrow_tip_block = xp.random.rand(a, a)

    if direction == "downward" or direction == "down-middleward":
        A_lower_arrow_blocks = xp.random.rand(n, a, b)
        A_upper_arrow_blocks = xp.random.rand(n, b, a)
    elif direction == "upward" or direction == "up-middleward":
        A_lower_arrow_blocks = xp.random.rand(n, b, a)
        A_upper_arrow_blocks = xp.random.rand(n, a, b)

    if symmetric:
        for n_i in range(n):
            A_diagonal_blocks[n_i] = A_diagonal_blocks[n_i] + A_diagonal_blocks[n_i].T

            if n_i < n - 1:
                A_upper_diagonal_blocks[n_i] = A_lower_diagonal_blocks[n_i].T

            A_upper_arrow_blocks[n_i] = A_lower_arrow_blocks[n_i].T

        A_arrow_tip_block[:] = A_arrow_tip_block[:] + A_arrow_tip_block[:].T

    A_arrow_tip_block[:] += xp.diag(xp.sum(xp.abs(A_arrow_tip_block[:]), axis=1))

    if direction == "downward" or direction == "down-middleward":
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

            A_diagonal_blocks[n_i] += xp.diag(
                xp.sum(xp.abs(A_upper_arrow_blocks[n_i]), axis=1)
            )

            A_arrow_tip_block[:] += xp.diag(
                xp.sum(xp.abs(A_lower_arrow_blocks[n_i]), axis=1)
            )

    elif direction == "upward" or direction == "up-middleward":
        for n_i in range(n - 1, -1, -1):
            A_diagonal_blocks[n_i] += xp.diag(
                xp.sum(xp.abs(A_diagonal_blocks[n_i]), axis=1)
            )

            if n_i < n - 1:
                A_diagonal_blocks[n_i] += xp.diag(
                    xp.sum(xp.abs(A_upper_diagonal_blocks[n_i]), axis=1)
                )

            if n_i > 0:
                A_diagonal_blocks[n_i] += xp.diag(
                    xp.sum(xp.abs(A_lower_diagonal_blocks[n_i - 1]), axis=1)
                )

            A_diagonal_blocks[n_i] += xp.diag(
                xp.sum(xp.abs(A_lower_arrow_blocks[n_i]), axis=1)
            )

            A_arrow_tip_block[:] += xp.diag(
                xp.sum(xp.abs(A_upper_arrow_blocks[n_i]), axis=1)
            )

    return (
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_lower_arrow_blocks,
        A_upper_diagonal_blocks,
        A_upper_arrow_blocks,
        A_arrow_tip_block,
    )


def bta_to_dense(
    A_diagonal_blocks: ArrayLike,
    A_lower_diagonal_blocks: ArrayLike,
    A_lower_arrow_blocks: ArrayLike,
    A_upper_diagonal_blocks: ArrayLike,
    A_upper_arrow_blocks: ArrayLike,
    A_arrow_tip_block: ArrayLike,
    direction: str = "downward",
) -> ArrayLike:
    xp = _get_array_module(A_diagonal_blocks)

    n = A_diagonal_blocks.shape[0]
    b = A_diagonal_blocks.shape[1]
    a = A_arrow_tip_block.shape[0]

    A = xp.zeros((n * b + a, n * b + a), dtype=A_diagonal_blocks.dtype)

    if direction == "downward" or direction == "down-middleward":
        for n_i in range(n):
            A[n_i * b : (n_i + 1) * b, n_i * b : (n_i + 1) * b] = A_diagonal_blocks[n_i]
            if n_i > 0:
                A[n_i * b : (n_i + 1) * b, (n_i - 1) * b : n_i * b] = (
                    A_lower_diagonal_blocks[n_i - 1]
                )
            if n_i < n - 1:
                A[n_i * b : (n_i + 1) * b, (n_i + 1) * b : (n_i + 2) * b] = (
                    A_upper_diagonal_blocks[n_i]
                )
            A[n_i * b : (n_i + 1) * b, -a:] = A_upper_arrow_blocks[n_i]
            A[-a:, n_i * b : (n_i + 1) * b] = A_lower_arrow_blocks[n_i]
        A[-a:, -a:] = A_arrow_tip_block[:]

    if direction == "upward" or direction == "up-middleward":
        for n_i in range(n - 1, -1, -1):
            A[n_i * b + a : (n_i + 1) * b + a, n_i * b + a : (n_i + 1) * b + a] = (
                A_diagonal_blocks[n_i]
            )
            if n_i > 0:
                A[n_i * b + a : (n_i + 1) * b + a, (n_i - 1) * b + a : n_i * b + a] = (
                    A_lower_diagonal_blocks[n_i - 1]
                )
            if n_i < n - 1:
                A[
                    n_i * b + a : (n_i + 1) * b + a,
                    (n_i + 1) * b + a : (n_i + 2) * b + a,
                ] = A_upper_diagonal_blocks[n_i]
            A[n_i * b + a : (n_i + 1) * b + a, :a] = A_lower_arrow_blocks[n_i]
            A[:a, n_i * b + a : (n_i + 1) * b + a] = A_upper_arrow_blocks[n_i]
        A[:a, :a] = A_arrow_tip_block[:]

    return A
