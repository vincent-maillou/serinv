# Copyright 2023-2024 ETH Zurich. All rights reserved.

try:
    import cupy as cp
    import cupyx.scipy.linalg as cu_la

    CUPY_AVAIL = True

except ImportError:
    CUPY_AVAIL = False

import numpy as np
import scipy.linalg as np_la
from numpy.typing import ArrayLike


def ddbtasi(
    LU_diagonal_blocks: ArrayLike,
    LU_lower_diagonal_blocks: ArrayLike,
    LU_upper_diagonal_blocks: ArrayLike,
    LU_arrow_bottom_blocks: ArrayLike,
    LU_arrow_right_blocks: ArrayLike,
    LU_arrow_tip_block: ArrayLike,
) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike,]:
    """Perform a selected inversion from a lu factorized matrix using
    a sequential block algorithm.

    Note:
    -----
    - If a device array is given, the algorithm will run on the GPU.

    Parameters
    ----------
    LU_diagonal_blocks : ArrayLike
        LU factors of the diagonal blocks.
    LU_lower_diagonal_blocks : ArrayLike
        LU factors of the lower diagonal blocks.
    LU_upper_diagonal_blocks : ArrayLike
        LU factors of the upper diagonal blocks.
    LU_arrow_bottom_blocks : ArrayLike
        LU factors of the bottom arrow blocks.
    LU_arrow_right_blocks : ArrayLike
        LU factors of the right arrow blocks.
    LU_arrow_tip_block : ArrayLike
        LU factors of the tip block of the arrowhead.

    Returns
    -------
    X_diagonal_blocks : ArrayLike
        Diagonal blocks of the selected inversion of the matrix.
    X_lower_diagonal_blocks : ArrayLike
        Lower diagonal blocks of the selected inversion of the matrix.
    X_upper_diagonal_blocks : ArrayLike
        Upper diagonal blocks of the selected inversion of the matrix.
    X_arrow_bottom_blocks : ArrayLike
        Bottom arrow blocks of the selected inversion of the matrix.
    X_arrow_right_blocks : ArrayLike
        Right arrow blocks of the selected inversion of the matrix.
    X_arrow_tip_block : ArrayLike
        Tip arrow block of the selected inversion of the matrix.
    """

    la = np_la
    if CUPY_AVAIL:
        xp = cp.get_array_module(LU_diagonal_blocks)
        if xp == cp:
            la = cu_la
    else:
        xp = np

    diag_blocksize = LU_diagonal_blocks.shape[1]
    arrow_blocksize = LU_arrow_bottom_blocks.shape[1]
    n_diag_blocks = LU_diagonal_blocks.shape[0]

    X_diagonal_blocks = xp.empty_like(LU_diagonal_blocks)
    X_lower_diagonal_blocks = xp.empty_like(LU_lower_diagonal_blocks)
    X_upper_diagonal_blocks = xp.empty_like(LU_upper_diagonal_blocks)
    X_arrow_bottom_blocks = xp.empty_like(LU_arrow_bottom_blocks)
    X_arrow_right_blocks = xp.empty_like(LU_arrow_right_blocks)
    X_arrow_tip_block = xp.empty_like(LU_arrow_tip_block)
    L_last_blk_inv = xp.empty_like(LU_arrow_tip_block)
    U_last_blk_inv = xp.empty_like(L_last_blk_inv)
    L_blk_inv = xp.empty_like(LU_diagonal_blocks[-1, :, :])
    U_blk_inv = xp.empty_like(LU_diagonal_blocks[-1, :, :])

    # Solve for the last block
    L_last_blk_inv = la.solve_triangular(
        LU_arrow_tip_block[:, :],
        xp.eye(arrow_blocksize),
        lower=True,
        unit_diagonal=True,
    )
    U_last_blk_inv = la.solve_triangular(
        LU_arrow_tip_block[:, :], xp.eye(arrow_blocksize), lower=False
    )

    X_arrow_tip_block[:, :] = U_last_blk_inv @ L_last_blk_inv

    # Solve for the last diagonal block
    L_blk_inv = la.solve_triangular(
        LU_diagonal_blocks[-1, :, :],
        xp.eye(diag_blocksize),
        lower=True,
        unit_diagonal=True,
    )
    U_blk_inv = la.solve_triangular(
        LU_diagonal_blocks[-1, :, :],
        xp.eye(diag_blocksize),
        lower=False,
    )

    # X_{ndb+1, ndb} = -X_{ndb+1, ndb+1} L_{ndb+1, ndb} L_{ndb, ndb}^{-1}
    X_arrow_bottom_blocks[-1, :, :] = (
        -X_arrow_tip_block[:, :] @ LU_arrow_bottom_blocks[-1, :, :] @ L_blk_inv
    )

    # X_{ndb, ndb+1} = -U_{ndb, ndb}^{-1} U_{ndb, ndb+1} X_{ndb+1, ndb+1}
    X_arrow_right_blocks[-1, :, :] = (
        -U_blk_inv @ LU_arrow_right_blocks[-1, :, :] @ X_arrow_tip_block[:, :]
    )

    # X_{ndb, ndb} = (U_{ndb, ndb}^{-1} - X_{ndb, ndb+1} L_{ndb+1, ndb}) L_{ndb, ndb}^{-1}
    X_diagonal_blocks[-1, :, :] = (
        U_blk_inv - X_arrow_right_blocks[-1, :, :] @ LU_arrow_bottom_blocks[-1, :, :]
    ) @ L_blk_inv

    for i in range(n_diag_blocks - 2, -1, -1):
        L_blk_inv = la.solve_triangular(
            LU_diagonal_blocks[i, :, :],
            xp.eye(diag_blocksize),
            lower=True,
            unit_diagonal=True,
        )

        U_blk_inv = la.solve_triangular(
            LU_diagonal_blocks[i, :, :],
            xp.eye(diag_blocksize),
            lower=False,
        )

        # --- Off-diagonal block part ---
        # X_{i+1, i} = (-X_{i+1, i+1} L_{i+1, i} - X_{i+1, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1}
        X_lower_diagonal_blocks[i, :, :] = (
            -X_diagonal_blocks[i + 1, :, :] @ LU_lower_diagonal_blocks[i, :, :]
            - X_arrow_right_blocks[i + 1, :, :] @ LU_arrow_bottom_blocks[i, :, :]
        ) @ L_blk_inv

        # X_{i, i+1} = U_{i, i}^{-1} (- U_{i, i+1} X_{i+1, i+1} - U_{i, ndb+1} X_{ndb+1, i+1})
        X_upper_diagonal_blocks[i, :, :] = U_blk_inv @ (
            -LU_upper_diagonal_blocks[i, :, :] @ X_diagonal_blocks[i + 1, :, :]
            - LU_arrow_right_blocks[i, :, :] @ X_arrow_bottom_blocks[i + 1, :, :]
        )

        # --- Arrowhead part ---
        # X_{ndb+1, i} = (- X_{ndb+1, i+1} L_{i+1, i} - X_{ndb+1, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1}
        X_arrow_bottom_blocks[i, :, :] = (
            -X_arrow_bottom_blocks[i + 1, :, :] @ LU_lower_diagonal_blocks[i, :, :]
            - X_arrow_tip_block[:, :] @ LU_arrow_bottom_blocks[i, :, :]
        ) @ L_blk_inv

        # X_{i, ndb+1} = U_{i, i}^{-1} (- U_{i, i+1} X_{i+1, ndb+1} - U_{i, ndb+1} X_{ndb+1, ndb+1})
        X_arrow_right_blocks[i, :, :] = U_blk_inv @ (
            -LU_upper_diagonal_blocks[i, :, :] @ X_arrow_right_blocks[i + 1, :, :]
            - LU_arrow_right_blocks[i, :, :] @ X_arrow_tip_block[:, :]
        )

        # --- Diagonal block part ---
        # X_{i, i} = (U_{i, i}^{-1} - X_{i, i+1} L_{i+1, i} - X_{i, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1}
        X_diagonal_blocks[i, :, :] = (
            U_blk_inv
            - X_upper_diagonal_blocks[i, :, :] @ LU_lower_diagonal_blocks[i, :, :]
            - X_arrow_right_blocks[i, :, :] @ LU_arrow_bottom_blocks[i, :, :]
        ) @ L_blk_inv

    return (
        X_diagonal_blocks,
        X_lower_diagonal_blocks,
        X_upper_diagonal_blocks,
        X_arrow_bottom_blocks,
        X_arrow_right_blocks,
        X_arrow_tip_block,
    )
