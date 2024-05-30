# Copyright 2023-2024 ETH Zurich. All rights reserved.

import numpy as np
import scipy.linalg as la


def scpobtsi(
    L_diagonal_blocks: np.ndarray,
    L_lower_diagonal_blocks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Perform a selected inversion of a block tridiagonal matrix using a
    sequential algorithm on CPU backend.

    Parameters
    ----------
    L_diagonal_blocks : np.ndarray
        Diagonal blocks of the cholesky factorization.
    L_lower_diagonal_blocks : np.ndarray
        Lower diagonal blocks of the cholesky factorization.

    Returns
    -------
    X_diagonal_blocks : np.ndarray
        Diagonal blocks of the selected inverse.
    X_lower_diagonal_blocks : np.ndarray
        Lower diagonal blocks of the selected inverse.
    """

    blocksize = L_diagonal_blocks.shape[0]
    nblocks = L_diagonal_blocks.shape[1] // blocksize

    X_diagonal_blocks = np.zeros_like(L_diagonal_blocks)
    X_lower_diagonal_blocks = np.zeros_like(L_lower_diagonal_blocks)
    L_blk_inv = np.zeros((blocksize, blocksize), dtype=L_diagonal_blocks.dtype)

    L_blk_inv = la.solve_triangular(
        L_diagonal_blocks[:, -blocksize:], np.eye(blocksize), lower=True
    )
    X_diagonal_blocks[:, -blocksize:] = L_blk_inv.T @ L_blk_inv

    for i in range(nblocks - 2, -1, -1):
        L_blk_inv = la.solve_triangular(
            L_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize],
            np.eye(blocksize),
            lower=True,
        )

        # X_{i+1, i} = -X_{i+1, i+1} L_{i+1, i} L_{i, i}^{-1}
        X_lower_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize] = (
            -X_diagonal_blocks[:, (i + 1) * blocksize : (i + 2) * blocksize]
            @ L_lower_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize]
            @ L_blk_inv
        )

        # X_{i, i} = (L_{i, i}^{-T} - X_{i+1, i}^{T} L_{i+1, i}) L_{i, i}^{-1}
        X_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize] = (
            L_blk_inv.T
            - X_lower_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize].T
            @ L_lower_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize]
        ) @ L_blk_inv

    return (
        X_diagonal_blocks,
        X_lower_diagonal_blocks,
    )
