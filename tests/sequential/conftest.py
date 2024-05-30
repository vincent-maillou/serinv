# Copyright 2023-2024 ETH Zurich. All rights reserved.

SEED = 63

try:
    import cupy as cp

    CUPY_AVAIL = True
    cp.random.seed(SEED)

except:
    CUPY_AVAIL = False

import numpy as np

np.random.seed(SEED)
import pytest


@pytest.fixture(scope="function", autouse=True)
def bta_dense_to_arrays():
    def _bta_dense_to_arrays(
        bta: np.ndarray | cp.ndarray,
        diagonal_blocksize: int,
        arrowhead_blocksize: int,
        n_diag_blocks: int,
    ):
        """Converts a block tridiagonal arrowhead matrix from a dense representation to arrays of blocks."""
        if CUPY_AVAIL:
            xp = cp.get_array_module(bta)
        else:
            xp = np

        A_diagonal_blocks = xp.zeros(
            (n_diag_blocks, diagonal_blocksize, diagonal_blocksize),
            dtype=xp.complex128,
        )

        A_lower_diagonal_blocks = xp.zeros(
            (n_diag_blocks - 1, diagonal_blocksize, diagonal_blocksize),
            dtype=xp.complex128,
        )
        A_upper_diagonal_blocks = xp.zeros(
            (n_diag_blocks - 1, diagonal_blocksize, diagonal_blocksize),
            dtype=xp.complex128,
        )

        A_arrow_bottom_blocks = xp.zeros(
            (n_diag_blocks, arrowhead_blocksize, diagonal_blocksize),
            dtype=xp.complex128,
        )

        A_arrow_right_blocks = xp.zeros(
            (n_diag_blocks, diagonal_blocksize, arrowhead_blocksize),
            dtype=xp.complex128,
        )

        for i in range(n_diag_blocks):
            A_diagonal_blocks[i, :, :] = bta[
                i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
                i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
            ]
            if i > 0:
                A_lower_diagonal_blocks[i - 1, :, :] = bta[
                    i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
                    (i - 1) * diagonal_blocksize : i * diagonal_blocksize,
                ]
            if i < n_diag_blocks - 1:
                A_upper_diagonal_blocks[i, :, :] = bta[
                    i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
                    (i + 1) * diagonal_blocksize : (i + 2) * diagonal_blocksize,
                ]

            A_arrow_bottom_blocks[i, :, :] = bta[
                -arrowhead_blocksize:,
                i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
            ]

            A_arrow_right_blocks[i, :, :] = bta[
                i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
                -arrowhead_blocksize:,
            ]

        A_arrow_tip_block = bta[-arrowhead_blocksize:, -arrowhead_blocksize:]

        return (
            A_diagonal_blocks,
            A_lower_diagonal_blocks,
            A_upper_diagonal_blocks,
            A_arrow_bottom_blocks,
            A_arrow_right_blocks,
            A_arrow_tip_block,
        )

    return _bta_dense_to_arrays


@pytest.fixture(scope="function", autouse=True)
def bta_arrays_to_dense():
    def _bta_arrays_to_dense(
        A_diagonal_blocks: np.ndarray | cp.ndarray,
        A_lower_diagonal_blocks: np.ndarray | cp.ndarray,
        A_upper_diagonal_blocks: np.ndarray | cp.ndarray,
        A_arrow_bottom_blocks: np.ndarray | cp.ndarray,
        A_arrow_right_blocks: np.ndarray | cp.ndarray,
        A_arrow_tip_block: np.ndarray | cp.ndarray,
    ):
        """Converts arrays of blocks to a block tridiagonal arrowhead matrix in a dense representation."""
        if CUPY_AVAIL:
            xp = cp.get_array_module(A_diagonal_blocks)
        else:
            xp = np

        diagonal_blocksize = A_diagonal_blocks.shape[1]
        arrowhead_blocksize = A_arrow_bottom_blocks.shape[1]
        n_diag_blocks = A_diagonal_blocks.shape[0]

        bta = xp.zeros(
            (
                diagonal_blocksize * n_diag_blocks + arrowhead_blocksize,
                diagonal_blocksize * n_diag_blocks + arrowhead_blocksize,
            ),
            dtype=xp.complex128,
        )

        for i in range(n_diag_blocks):
            bta[
                i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
                i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
            ] = A_diagonal_blocks[i, :, :]
            if i > 0:
                bta[
                    i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
                    (i - 1) * diagonal_blocksize : i * diagonal_blocksize,
                ] = A_lower_diagonal_blocks[i - 1, :, :]
            if i < n_diag_blocks - 1:
                bta[
                    i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
                    (i + 1) * diagonal_blocksize : (i + 2) * diagonal_blocksize,
                ] = A_upper_diagonal_blocks[i, :, :]

            bta[
                -arrowhead_blocksize:,
                i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
            ] = A_arrow_bottom_blocks[i, :, :]

            bta[
                i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
                -arrowhead_blocksize:,
            ] = A_arrow_right_blocks[i, :, :]

        bta[-arrowhead_blocksize:, -arrowhead_blocksize:] = A_arrow_tip_block

        return bta

    return _bta_arrays_to_dense


@pytest.fixture(scope="function", autouse=True)
def dd_bta(
    diagonal_blocksize: int,
    arrowhead_blocksize: int,
    n_diag_blocks: int,
    device_array: bool,
):
    """Returns a random, diagonaly dominant general, block tridiagonal arrowhead matrix."""
    xp = cp if device_array and CUPY_AVAIL else np

    DD_BTA = xp.zeros(
        (
            diagonal_blocksize * n_diag_blocks + arrowhead_blocksize,
            diagonal_blocksize * n_diag_blocks + arrowhead_blocksize,
        ),
        dtype=xp.complex128,
    )

    # Fill the lower arrowhead blocks
    DD_BTA[-arrowhead_blocksize:, :-arrowhead_blocksize] = (1 + 1.0j) * xp.random.rand(
        arrowhead_blocksize, n_diag_blocks * diagonal_blocksize
    )
    # Fill the right arrowhead blocks
    DD_BTA[:-arrowhead_blocksize, -arrowhead_blocksize:] = (1 + 1.0j) * xp.random.rand(
        n_diag_blocks * diagonal_blocksize, arrowhead_blocksize
    )

    # Fill the tip of the arrowhead
    DD_BTA[-arrowhead_blocksize:, -arrowhead_blocksize:] = (1 + 1.0j) * xp.random.rand(
        arrowhead_blocksize, arrowhead_blocksize
    )

    # Fill the diagonal blocks
    for i in range(n_diag_blocks):
        DD_BTA[
            i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
            i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
        ] = (1 + 1.0j) * xp.random.rand(diagonal_blocksize, diagonal_blocksize) + (
            1 + 1.0j
        ) * xp.eye(
            diagonal_blocksize
        )

        # Fill the off-diagonal blocks
        if i > 0:
            DD_BTA[
                i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
                (i - 1) * diagonal_blocksize : i * diagonal_blocksize,
            ] = (1 + 1.0j) * xp.random.rand(diagonal_blocksize, diagonal_blocksize)

        if i < n_diag_blocks - 1:
            DD_BTA[
                i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
                (i + 1) * diagonal_blocksize : (i + 2) * diagonal_blocksize,
            ] = (1 + 1.0j) * xp.random.rand(diagonal_blocksize, diagonal_blocksize)

    # Make the matrix diagonally dominant
    for i in range(DD_BTA.shape[0]):
        DD_BTA[i, i] = 1 + xp.sum(DD_BTA[i, :])

    return DD_BTA
