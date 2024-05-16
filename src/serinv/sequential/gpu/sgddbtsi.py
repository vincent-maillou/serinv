# Copyright 2023-2024 ETH Zurich and USI. All rights reserved.

import cupy as cp
import cupyx as cpx
import cupyx.scipy.linalg as cpla
import numpy as np


def sgddbtsi(
    L_diagonal_blocks: np.ndarray,
    L_lower_diagonal_blocks: np.ndarray,
    U_diagonal_blocks: np.ndarray,
    U_upper_diagonal_blocks: np.ndarray,
) -> np.ndarray:
    """Perform a selected inversion from a lu factorized matrix using
    a sequential algorithm on a GPU backend.

    Parameters
    ----------
    L_diagonal_blocks : np.ndarray
        Diagonal blocks of the lower factor of the lu factorization of the matrix.
    L_lower_diagonal_blocks : np.ndarray
        Lower diagonal blocks of the lower factor of the lu factorization of the matrix.
    U_diagonal_blocks : np.ndarray
        Diagonal blocks of the upper factor of the lu factorization of the matrix.
    U_upper_diagonal_blocks : np.ndarray
        Upper diagonal blocks of the upper factor of the lu factorization of the matrix.

    Returns
    -------
    X_diagonal_blocks : np.ndarray
        Diagonal blocks of the selected inversion of the matrix.
    X_lower_diagonal_blocks : np.ndarray
        Lower diagonal blocks of the selected inversion of the matrix.
    X_upper_diagonal_blocks : np.ndarray
        Upper diagonal blocks of the selected inversion of the matrix.
    X_arrow_bottom_blocks : np.ndarray
        Bottom arrow blocks of the selected inversion of the matrix.
    X_arrow_right_blocks : np.ndarray
        Right arrow blocks of the selected inversion of the matrix.
    X_arrow_tip_block : np.ndarray
        Tip arrow block of the selected inversion of the matrix.
    """

    blocksize = L_diagonal_blocks.shape[0]
    nblocks = L_diagonal_blocks.shape[1] // blocksize

    L_diagonal_blocks_gpu: cp.ndarray = cp.asarray(L_diagonal_blocks)
    L_lower_diagonal_blocks_gpu: cp.ndarray = cp.asarray(L_lower_diagonal_blocks)
    U_diagonal_blocks_gpu: cp.ndarray = cp.asarray(U_diagonal_blocks)
    U_upper_diagonal_blocks_gpu: cp.ndarray = cp.asarray(U_upper_diagonal_blocks)

    # Host side arrays
    X_diagonal_blocks: cp.ndarray = cpx.empty_like_pinned(L_diagonal_blocks_gpu)
    X_lower_diagonal_blocks: cp.ndarray = cpx.empty_like_pinned(
        L_lower_diagonal_blocks_gpu
    )
    X_upper_diagonal_blocks: cp.ndarray = cpx.empty_like_pinned(
        U_upper_diagonal_blocks_gpu
    )

    # Device side arrays
    X_diagonal_blocks_gpu: cp.ndarray = cp.empty_like(X_diagonal_blocks)
    X_lower_diagonal_blocks_gpu: cp.ndarray = cp.empty_like(X_lower_diagonal_blocks)
    X_upper_diagonal_blocks_gpu: cp.ndarray = cp.empty_like(X_upper_diagonal_blocks)

    L_blk_inv_gpu: cp.ndarray = cp.empty(
        (blocksize, blocksize), dtype=L_diagonal_blocks.dtype
    )
    U_blk_inv_gpu: cp.ndarray = cp.empty(
        (blocksize, blocksize), dtype=L_diagonal_blocks.dtype
    )

    L_blk_inv_gpu = cpla.solve_triangular(
        L_diagonal_blocks_gpu[:, -blocksize:], cp.eye(blocksize), lower=True
    )

    U_blk_inv_gpu = cpla.solve_triangular(
        U_diagonal_blocks_gpu[:, -blocksize:], cp.eye(blocksize), lower=False
    )

    X_diagonal_blocks_gpu[:, -blocksize:] = U_blk_inv_gpu @ L_blk_inv_gpu

    for i in range(nblocks - 2, -1, -1):
        L_blk_inv_gpu = cpla.solve_triangular(
            L_diagonal_blocks_gpu[:, i * blocksize : (i + 1) * blocksize],
            cp.eye(blocksize),
            lower=True,
        )

        U_blk_inv_gpu = cpla.solve_triangular(
            U_diagonal_blocks_gpu[:, i * blocksize : (i + 1) * blocksize],
            cp.eye(blocksize),
            lower=False,
        )

        # X_{i+1, i} = -X_{i+1, i+1} L_{i+1, i} L_{i, i}^{-1}
        X_lower_diagonal_blocks_gpu[:, i * blocksize : (i + 1) * blocksize] = (
            -X_diagonal_blocks_gpu[:, (i + 1) * blocksize : (i + 2) * blocksize]
            @ L_lower_diagonal_blocks_gpu[:, i * blocksize : (i + 1) * blocksize]
            @ L_blk_inv_gpu
        )

        # X_{i, i+1} = -U_{i, i}^{-1} U_{i, i+1} X_{i+1, i+1}
        X_upper_diagonal_blocks_gpu[:, i * blocksize : (i + 1) * blocksize] = (
            -U_blk_inv_gpu
            @ U_upper_diagonal_blocks_gpu[:, i * blocksize : (i + 1) * blocksize]
            @ X_diagonal_blocks_gpu[:, (i + 1) * blocksize : (i + 2) * blocksize]
        )

        # X_{i, i} = (U_{i, i}^{-1} - X_{i, i+1} L_{i+1, i}) L_{i, i}^{-1}
        X_diagonal_blocks_gpu[:, i * blocksize : (i + 1) * blocksize] = (
            U_blk_inv_gpu
            - X_upper_diagonal_blocks_gpu[:, i * blocksize : (i + 1) * blocksize]
            @ L_lower_diagonal_blocks_gpu[:, i * blocksize : (i + 1) * blocksize]
        ) @ L_blk_inv_gpu

    X_diagonal_blocks_gpu.get(out=X_diagonal_blocks)
    X_lower_diagonal_blocks_gpu.get(out=X_lower_diagonal_blocks)
    X_upper_diagonal_blocks_gpu.get(out=X_upper_diagonal_blocks)

    return (X_diagonal_blocks, X_lower_diagonal_blocks, X_upper_diagonal_blocks)
