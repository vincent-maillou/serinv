# Copyright 2023-2024 ETH Zurich. All rights reserved.

import numpy as np
import scipy.linalg as la


def pobtasi(
    L_diagonal_blocks: np.ndarray,
    L_lower_diagonal_blocks: np.ndarray,
    L_arrow_bottom_blocks: np.ndarray,
    L_arrow_tip_block: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Perform a selected inversion of a block tridiagonal matrix using a
    sequential algorithm on CPU backend.

    Parameters
    ----------
    L_diagonal_blocks : np.ndarray
        Diagonal blocks of the cholesky factorization.
    L_lower_diagonal_blocks : np.ndarray
        Lower diagonal blocks of the cholesky factorization.
    L_arrow_bottom_blocks : np.ndarray
        Arrow bottom blocks of the cholesky factorization.
    L_arrow_tip_block : np.ndarray
        Arrow tip block of the cholesky factorization.

    Returns
    -------
    X_diagonal_blocks : np.ndarray
        Diagonal blocks of the selected inverse.
    X_lower_diagonal_blocks : np.ndarray
        Lower diagonal blocks of the selected inverse.
    X_arrow_bottom_blocks : np.ndarray
        Arrow bottom blocks of the selected inverse.
    X_arrow_tip_block : np.ndarray
        Arrow tip block of the selected inverse.
    """

    diag_blocksize = L_diagonal_blocks.shape[0]
    arrow_blocksize = L_arrow_tip_block.shape[0]
    n_diag_blocks = L_diagonal_blocks.shape[1] // diag_blocksize

    X_diagonal_blocks = np.zeros_like(L_diagonal_blocks)
    X_lower_diagonal_blocks = np.zeros_like(L_lower_diagonal_blocks)
    X_arrow_bottom_blocks = np.zeros_like(L_arrow_bottom_blocks)
    X_arrow_tip_block = np.zeros_like(L_arrow_tip_block)

    L_last_blk_inv = np.zeros(
        (arrow_blocksize, arrow_blocksize), dtype=L_diagonal_blocks.dtype
    )
    L_blk_inv = np.zeros(
        (diag_blocksize, diag_blocksize), dtype=L_diagonal_blocks.dtype
    )

    L_last_blk_inv = la.solve_triangular(
        L_arrow_tip_block[:, :], np.eye(arrow_blocksize), lower=True
    )

    X_arrow_tip_block[:, :] = L_last_blk_inv.T @ L_last_blk_inv

    L_blk_inv = la.solve_triangular(
        L_diagonal_blocks[:, -diag_blocksize:],
        np.eye(diag_blocksize),
        lower=True,
    )

    # X_{ndb+1, ndb} = -X_{ndb+1, ndb+1} L_{ndb+1, ndb} L_{ndb, ndb}^{-1}
    X_arrow_bottom_blocks[:, -diag_blocksize:] = (
        -X_arrow_tip_block[:, :]
        @ L_arrow_bottom_blocks[:, -diag_blocksize:]
        @ L_blk_inv
    )

    # X_{ndb, ndb} = (L_{ndb, ndb}^{-T} - X_{ndb+1, ndb}^{T} L_{ndb+1, ndb}) L_{ndb, ndb}^{-1}
    X_diagonal_blocks[:, -diag_blocksize:] = (
        L_blk_inv.T
        - X_arrow_bottom_blocks[:, -diag_blocksize:].T
        @ L_arrow_bottom_blocks[:, -diag_blocksize:]
    ) @ L_blk_inv

    for i in range(n_diag_blocks - 2, -1, -1):
        L_blk_inv = la.solve_triangular(
            L_diagonal_blocks[:, i * diag_blocksize : (i + 1) * diag_blocksize],
            np.eye(diag_blocksize),
            lower=True,
        )

        # --- Off-diagonal block part ---
        # X_{i+1, i} = (-X_{i+1, i+1} L_{i+1, i} - X_{ndb+1, i+1}^{T} L_{ndb+1, i}) L_{i, i}^{-1}
        X_lower_diagonal_blocks[:, i * diag_blocksize : (i + 1) * diag_blocksize] = (
            -X_diagonal_blocks[:, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize]
            @ L_lower_diagonal_blocks[:, i * diag_blocksize : (i + 1) * diag_blocksize]
            - X_arrow_bottom_blocks[
                :, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize
            ].T
            @ L_arrow_bottom_blocks[:, i * diag_blocksize : (i + 1) * diag_blocksize]
        ) @ L_blk_inv

        # --- Arrowhead part ---
        # X_{ndb+1, i} = (- X_{ndb+1, i+1} L_{i+1, i} - X_{ndb+1, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1}
        X_arrow_bottom_blocks[:, i * diag_blocksize : (i + 1) * diag_blocksize] = (
            -X_arrow_bottom_blocks[
                :, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize
            ]
            @ L_lower_diagonal_blocks[:, i * diag_blocksize : (i + 1) * diag_blocksize]
            - X_arrow_tip_block[:, :]
            @ L_arrow_bottom_blocks[:, i * diag_blocksize : (i + 1) * diag_blocksize]
        ) @ L_blk_inv

        # --- Diagonal block part ---
        # X_{i, i} = (L_{i, i}^{-T} - X_{i+1, i}^{T} L_{i+1, i} - X_{ndb+1, i}.T L_{ndb+1, i}) L_{i, i}^{-1}
        X_diagonal_blocks[:, i * diag_blocksize : (i + 1) * diag_blocksize] = (
            L_blk_inv.T
            - X_lower_diagonal_blocks[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ].T
            @ L_lower_diagonal_blocks[:, i * diag_blocksize : (i + 1) * diag_blocksize]
            - X_arrow_bottom_blocks[:, i * diag_blocksize : (i + 1) * diag_blocksize].T
            @ L_arrow_bottom_blocks[:, i * diag_blocksize : (i + 1) * diag_blocksize]
        ) @ L_blk_inv

    return (
        X_diagonal_blocks,
        X_lower_diagonal_blocks,
        X_arrow_bottom_blocks,
        X_arrow_tip_block,
    )
