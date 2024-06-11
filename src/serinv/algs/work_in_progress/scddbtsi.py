# Copyright 2023-2024 ETH Zurich. All rights reserved.

import numpy as np
import scipy.linalg as la


def scddbtsi(
    L_diagonal_blocks: np.ndarray,
    L_lower_diagonal_blocks: np.ndarray,
    U_diagonal_blocks: np.ndarray,
    U_upper_diagonal_blocks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform a selected inversion from a lu factorized matrix using
    a sequential algorithm on a CPU backend.

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

    X_diagonal_blocks = np.empty(
        (blocksize, nblocks * blocksize), dtype=L_diagonal_blocks.dtype
    )
    X_lower_diagonal_blocks = np.empty(
        (blocksize, (nblocks - 1) * blocksize), dtype=L_diagonal_blocks.dtype
    )
    X_upper_diagonal_blocks = np.empty(
        (blocksize, (nblocks - 1) * blocksize), dtype=L_diagonal_blocks.dtype
    )

    L_blk_inv = np.empty((blocksize, blocksize), dtype=L_diagonal_blocks.dtype)
    U_blk_inv = np.empty((blocksize, blocksize), dtype=L_diagonal_blocks.dtype)

    L_blk_inv = la.solve_triangular(
        L_diagonal_blocks[:, -blocksize:], np.eye(blocksize), lower=True
    )

    U_blk_inv = la.solve_triangular(
        U_diagonal_blocks[:, -blocksize:], np.eye(blocksize), lower=False
    )

    X_diagonal_blocks[:, -blocksize:] = U_blk_inv @ L_blk_inv

    for i in range(nblocks - 2, -1, -1):
        L_blk_inv = la.solve_triangular(
            L_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize],
            np.eye(blocksize),
            lower=True,
        )

        U_blk_inv = la.solve_triangular(
            U_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize],
            np.eye(blocksize),
            lower=False,
        )

        # X_{i+1, i} = -X_{i+1, i+1} L_{i+1, i} L_{i, i}^{-1}
        X_lower_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize] = (
            -X_diagonal_blocks[:, (i + 1) * blocksize : (i + 2) * blocksize]
            @ L_lower_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize]
            @ L_blk_inv
        )

        # X_{i, i+1} = -U_{i, i}^{-1} U_{i, i+1} X_{i+1, i+1}
        X_upper_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize] = (
            -U_blk_inv
            @ U_upper_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize]
            @ X_diagonal_blocks[:, (i + 1) * blocksize : (i + 2) * blocksize]
        )

        # X_{i, i} = (U_{i, i}^{-1} - X_{i, i+1} L_{i+1, i}) L_{i, i}^{-1}
        X_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize] = (
            U_blk_inv
            - X_upper_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize]
            @ L_lower_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize]
        ) @ L_blk_inv

    return (X_diagonal_blocks, X_lower_diagonal_blocks, X_upper_diagonal_blocks)
