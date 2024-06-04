# Copyright 2023-2024 ETH Zurich. All rights reserved.

try:
    import cupy as cp

    CUPY_AVAIL = True

except:
    CUPY_AVAIL = False

import numpy as np


def check_block_dd(
    A_diagonal_blocks: np.ndarray | cp.ndarray,
) -> np.ndarray | cp.ndarray:
    """Check if the diagonal blocks of a block tridiagonal arrowhead matrix are diagonally dominant.

    Note:
    -----
    - If device array is given, the check will be performed on the GPU.

    Parameters
    ----------
    A_diagonal_blocks : np.ndarray | cp.ndarray
        The blocks on the diagonal of the matrix.

    Returns
    -------
    np.ndarray | cp.ndarray
        Array of booleans indicating if the diagonal blocks are diagonally dominant.
    """
    if CUPY_AVAIL:
        xp = cp.get_array_module(A_diagonal_blocks)
    else:
        xp = np

    block_dd = xp.zeros(A_diagonal_blocks.shape[0], dtype=bool)

    for i in range(A_diagonal_blocks.shape[0]):
        block_dd[i] = xp.all(
            xp.abs(xp.diag(A_diagonal_blocks[i, :, :]))
            > xp.abs(
                xp.sum(
                    A_diagonal_blocks[i, :, :]
                    - xp.diag(xp.diag(A_diagonal_blocks[i, :, :])),
                    axis=1,
                )
            )
        )

    return block_dd


def check_ddbta(
    A_diagonal_blocks: np.ndarray | cp.ndarray,
    A_lower_diagonal_blocks: np.ndarray | cp.ndarray,
    A_upper_diagonal_blocks: np.ndarray | cp.ndarray,
    A_arrow_bottom_blocks: np.ndarray | cp.ndarray,
    A_arrow_right_blocks: np.ndarray | cp.ndarray,
    A_arrow_tip_block: np.ndarray | cp.ndarray,
) -> np.ndarray | cp.ndarray:
    """Check if the given block tridiagonal arrowhead matrix is diagonally dominant.

    Note:
    -----
    - If device arrays are given, the check will be performed on the GPU.

    Parameters
    ----------
    A_diagonal_blocks : np.ndarray | cp.ndarray
        The blocks on the diagonal of the matrix.
    A_lower_diagonal_blocks : np.ndarray | cp.ndarray
        The blocks on the lower diagonal of the matrix.
    A_upper_diagonal_blocks : np.ndarray | cp.ndarray
        The blocks on the upper diagonal of the matrix.
    A_arrow_bottom_blocks : np.ndarray | cp.ndarray
        The blocks on the bottom arrow of the matrix.
    A_arrow_right_blocks : np.ndarray | cp.ndarray
        The blocks on the right arrow of the matrix.
    A_arrow_tip_block : np.ndarray | cp.ndarray
        The block at the tip of the arrowhead.

    Returns
    -------
    np.ndarray | cp.ndarray
        Array of booleans indicating if the corresponding row is diagonally dominant.
    """
    if CUPY_AVAIL:
        xp = cp.get_array_module(A_diagonal_blocks)
    else:
        xp = np

    diagonal_blocksize = A_diagonal_blocks.shape[1]
    arrowhead_blocksize = A_arrow_bottom_blocks.shape[1]
    n_diag_blocks = A_diagonal_blocks.shape[0]

    print("n_diag_blocks", n_diag_blocks, flush=True)
    print("diagonal_blocksize", diagonal_blocksize, flush=True)
    print("arrowhead_blocksize", arrowhead_blocksize, flush=True)

    matrix_size = n_diag_blocks * diagonal_blocksize + arrowhead_blocksize

    ddbta = xp.zeros(matrix_size, dtype=bool)

    arrow_colsum = xp.zeros((arrowhead_blocksize), dtype=A_diagonal_blocks.dtype)
    for i in range(A_diagonal_blocks.shape[0]):
        diag = xp.abs(xp.diag(A_diagonal_blocks[i, :, :]))
        colsum = (
            xp.sum(A_diagonal_blocks[i, :, :], axis=1)
            - xp.diag(A_diagonal_blocks[i, :, :][:, :])
            + xp.sum(A_arrow_right_blocks[i, :, :], axis=1)
        )
        if i > 0:
            colsum += xp.sum(A_lower_diagonal_blocks[i - 1, :, :], axis=1)
        if i < A_diagonal_blocks.shape[0] - 1:
            colsum += xp.sum(A_upper_diagonal_blocks[i, :, :], axis=1)

        ddbta[i * diagonal_blocksize : (i + 1) * diagonal_blocksize] = diag > colsum

        arrow_colsum[:] += xp.sum(A_arrow_bottom_blocks[i, :, :], axis=1)

    ddbta[-arrowhead_blocksize:] = xp.abs(xp.diag(A_arrow_tip_block[:, :])) > xp.abs(
        xp.sum(A_arrow_tip_block, axis=1)
        - xp.diag(A_arrow_tip_block[:, :])
        + arrow_colsum
    )
    return ddbta
