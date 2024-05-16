# Copyright 2023-2024 ETH Zurich and USI. All rights reserved.

import numpy as np
import scipy.linalg as la


def scddbtasi(
    L_diagonal_blocks: np.ndarray,
    L_lower_diagonal_blocks: np.ndarray,
    L_arrow_bottom_blocks: np.ndarray,
    L_arrow_tip_block: np.ndarray,
    U_diagonal_blocks: np.ndarray,
    U_upper_diagonal_blocks: np.ndarray,
    U_arrow_right_blocks: np.ndarray,
    U_arrow_tip_block: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Perform a selected inversion from a lu factorized matrix using
    a sequential algorithm on a CPU backend.

    Parameters
    ----------
    L_diagonal_blocks : np.ndarray
        Diagonal blocks of the lower factor of the lu factorization of the matrix.
    L_lower_diagonal_blocks : np.ndarray
        Lower diagonal blocks of the lower factor of the lu factorization of the matrix.
    L_arrow_bottom_blocks : np.ndarray
        Bottom arrow blocks of the lower factor of the lu factorization of the matrix.
    L_arrow_tip_block : np.ndarray
        Tip arrow block of the lower factor of the lu factorization of the matrix.
    U_diagonal_blocks : np.ndarray
        Diagonal blocks of the upper factor of the lu factorization of the matrix.
    U_upper_diagonal_blocks : np.ndarray
        Upper diagonal blocks of the upper factor of the lu factorization of the matrix.
    U_arrow_right_blocks : np.ndarray
        Right arrow blocks of the upper factor of the lu factorization of the matrix.
    U_arrow_tip_block : np.ndarray
        Tip arrow block of the upper factor of the lu factorization of the matrix.

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

    diag_blocksize = L_diagonal_blocks.shape[1]
    arrow_blocksize = L_arrow_bottom_blocks.shape[1]
    n_diag_blocks = L_diagonal_blocks.shape[0]

    X_diagonal_blocks = np.empty(
        (n_diag_blocks, diag_blocksize, diag_blocksize), dtype=L_diagonal_blocks.dtype
    )
    X_lower_diagonal_blocks = np.empty(
        (n_diag_blocks - 1, diag_blocksize, diag_blocksize),
        dtype=L_diagonal_blocks.dtype,
    )
    X_upper_diagonal_blocks = np.empty(
        (n_diag_blocks - 1, diag_blocksize, diag_blocksize),
        dtype=L_diagonal_blocks.dtype,
    )
    X_arrow_bottom_blocks = np.empty(
        (n_diag_blocks, arrow_blocksize, diag_blocksize), dtype=L_diagonal_blocks.dtype
    )
    X_arrow_right_blocks = np.empty(
        (n_diag_blocks, diag_blocksize, arrow_blocksize), dtype=L_diagonal_blocks.dtype
    )
    X_arrow_tip_block = np.empty(
        (arrow_blocksize, arrow_blocksize), dtype=L_diagonal_blocks.dtype
    )

    L_last_blk_inv = np.empty(
        (arrow_blocksize, arrow_blocksize), dtype=L_diagonal_blocks.dtype
    )
    U_last_blk_inv = np.empty(
        (arrow_blocksize, arrow_blocksize), dtype=L_diagonal_blocks.dtype
    )

    L_last_blk_inv = la.solve_triangular(
        L_arrow_tip_block[:, :], np.eye(arrow_blocksize), lower=True
    )
    U_last_blk_inv = la.solve_triangular(
        U_arrow_tip_block[:, :], np.eye(arrow_blocksize), lower=False
    )

    X_arrow_tip_block[:, :] = U_last_blk_inv @ L_last_blk_inv

    L_blk_inv = la.solve_triangular(
        L_diagonal_blocks[-1, :, :],
        np.eye(diag_blocksize),
        lower=True,
    )
    U_blk_inv = la.solve_triangular(
        U_diagonal_blocks[-1, :, :],
        np.eye(diag_blocksize),
        lower=False,
    )

    # X_{ndb+1, ndb} = -X_{ndb+1, ndb+1} L_{ndb+1, ndb} L_{ndb, ndb}^{-1}
    X_arrow_bottom_blocks[-1, :, :] = (
        -X_arrow_tip_block[:, :] @ L_arrow_bottom_blocks[-1, :, :] @ L_blk_inv
    )

    # X_{ndb, ndb+1} = -U_{ndb, ndb}^{-1} U_{ndb, ndb+1} X_{ndb+1, ndb+1}
    X_arrow_right_blocks[-diag_blocksize:, :] = (
        -U_blk_inv @ U_arrow_right_blocks[-1, :, :] @ X_arrow_tip_block[:, :]
    )

    # X_{ndb, ndb} = (U_{ndb, ndb}^{-1} - X_{ndb, ndb+1} L_{ndb+1, ndb}) L_{ndb, ndb}^{-1}
    X_diagonal_blocks[-diag_blocksize:, -diag_blocksize:] = (
        U_blk_inv
        - X_arrow_right_blocks[-diag_blocksize:, :] @ L_arrow_bottom_blocks[-1, :, :]
    ) @ L_blk_inv

    for i in range(n_diag_blocks - 2, -1, -1):
        L_blk_inv = la.solve_triangular(
            L_diagonal_blocks[i, :, :],
            np.eye(diag_blocksize),
            lower=True,
        )

        U_blk_inv = la.solve_triangular(
            U_diagonal_blocks[i, :, :],
            np.eye(diag_blocksize),
            lower=False,
        )

        # --- Off-diagonal block part ---
        # X_{i+1, i} = (-X_{i+1, i+1} L_{i+1, i} - X_{i+1, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1}
        X_lower_diagonal_blocks[i, :, :] = (
            -X_diagonal_blocks[i + 1, :, :] @ L_lower_diagonal_blocks[i, :, :]
            - X_arrow_right_blocks[i + 1, :, :] @ L_arrow_bottom_blocks[i, :, :]
        ) @ L_blk_inv

        # X_{i, i+1} = U_{i, i}^{-1} (- U_{i, i+1} X_{i+1, i+1} - U_{i, ndb+1} X_{ndb+1, i+1})
        X_upper_diagonal_blocks[i, :, :] = U_blk_inv @ (
            -U_upper_diagonal_blocks[i, :, :] @ X_diagonal_blocks[i + 1, :, :]
            - U_arrow_right_blocks[i, :, :] @ X_arrow_bottom_blocks[i + 1, :, :]
        )

        # --- Arrowhead part ---
        # X_{ndb+1, i} = (- X_{ndb+1, i+1} L_{i+1, i} - X_{ndb+1, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1}
        X_arrow_bottom_blocks[i, :, :] = (
            -X_arrow_bottom_blocks[i + 1, :, :] @ L_lower_diagonal_blocks[i, :, :]
            - X_arrow_tip_block[:, :] @ L_arrow_bottom_blocks[i, :, :]
        ) @ L_blk_inv

        # X_{i, ndb+1} = U_{i, i}^{-1} (- U_{i, i+1} X_{i+1, ndb+1} - U_{i, ndb+1} X_{ndb+1, ndb+1})
        X_arrow_right_blocks[i, :, :] = U_blk_inv @ (
            -U_upper_diagonal_blocks[i, :, :] @ X_arrow_right_blocks[i + 1, :, :]
            - U_arrow_right_blocks[i, :, :] @ X_arrow_tip_block[:, :]
        )

        # --- Diagonal block part ---
        # X_{i, i} = (U_{i, i}^{-1} - X_{i, i+1} L_{i+1, i} - X_{i, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1}
        X_diagonal_blocks[i, :, :] = (
            U_blk_inv
            - X_upper_diagonal_blocks[i, :, :] @ L_lower_diagonal_blocks[i, :, :]
            - X_arrow_right_blocks[i, :, :] @ L_arrow_bottom_blocks[i, :, :]
        ) @ L_blk_inv

    return (
        X_diagonal_blocks,
        X_lower_diagonal_blocks,
        X_upper_diagonal_blocks,
        X_arrow_bottom_blocks,
        X_arrow_right_blocks,
        X_arrow_tip_block,
    )
