# Copyright 2023-2024 ETH Zurich and USI. All rights reserved.

import numpy as np
import scipy.linalg as la


def scddbtaf(
    A_diagonal_blocks: np.ndarray,
    A_lower_diagonal_blocks: np.ndarray,
    A_upper_diagonal_blocks: np.ndarray,
    A_arrow_bottom_blocks: np.ndarray,
    A_arrow_right_blocks: np.ndarray,
    A_arrow_tip_block: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Perform the LU factorization of a block tridiagonal matrix using
    a sequential algotithm on a CPU backend.

    The matrix is assumed to be block-diagonally dominant.

    The tip of the arrowhead is returned as the last block of the lower and upper
    arrows vectors.

    Parameters
    ----------
    A_diagonal_blocks : np.ndarray
        The blocks on the diagonal of the matrix.
    A_lower_diagonal_blocks : np.ndarray
        The blocks on the lower diagonal of the matrix.
    A_upper_diagonal_blocks : np.ndarray
        The blocks on the upper diagonal of the matrix.
    A_arrow_bottom_blocks : np.ndarray
        The blocks on the bottom arrow of the matrix.
    A_arrow_right_blocks : np.ndarray
        The blocks on the right arrow of the matrix.
    A_arrow_tip_block : np.ndarray
        The block at the tip of the arrowhead.

    Returns
    -------
    L_diagonal_blocks : np.ndarray
        Diagonal blocks of the lower factor.
    L_lower_diagonal_blocks : np.ndarray
        Lower diagonal blocks of the lower factor.
    L_arrow_bottom_blocks : np.ndarray
        Bottom arrow blocks of the lower factor.
    U_diagonal_blocks : np.ndarray
        Diagonal blocks of the upper factor.
    U_upper_diagonal_blocks : np.ndarray
        Upper diagonal blocks of the upper factor
    U_arrow_right_blocks : np.ndarray
        Right arrow blocks of the upper factor
    """

    diag_blocksize = A_diagonal_blocks.shape[0]
    arrow_blocksize = A_arrow_bottom_blocks.shape[0]

    n_diag_blocks = A_diagonal_blocks.shape[1] // diag_blocksize

    L_diagonal_blocks = np.empty(
        (diag_blocksize, n_diag_blocks * diag_blocksize), dtype=A_diagonal_blocks.dtype
    )
    L_lower_diagonal_blocks = np.empty(
        (diag_blocksize, (n_diag_blocks - 1) * diag_blocksize),
        dtype=A_diagonal_blocks.dtype,
    )
    L_arrow_bottom_blocks = np.empty(
        (arrow_blocksize, n_diag_blocks * diag_blocksize + arrow_blocksize),
        dtype=A_diagonal_blocks.dtype,
    )

    U_diagonal_blocks = np.empty(
        (diag_blocksize, n_diag_blocks * diag_blocksize), dtype=A_diagonal_blocks.dtype
    )
    U_upper_diagonal_blocks = np.empty(
        (diag_blocksize, (n_diag_blocks - 1) * diag_blocksize),
        dtype=A_diagonal_blocks.dtype,
    )
    U_arrow_right_blocks = np.empty(
        (n_diag_blocks * diag_blocksize + arrow_blocksize, arrow_blocksize),
        dtype=A_diagonal_blocks.dtype,
    )

    L_inv_temp = np.empty(
        (diag_blocksize, diag_blocksize), dtype=A_diagonal_blocks.dtype
    )
    U_inv_temp = np.empty(
        (diag_blocksize, diag_blocksize), dtype=A_diagonal_blocks.dtype
    )

    for i in range(0, n_diag_blocks - 1):
        # L_{i, i}, U_{i, i} = lu_dcmp(A_{i, i})
        (
            L_diagonal_blocks[:, i * diag_blocksize : (i + 1) * diag_blocksize],
            U_diagonal_blocks[:, i * diag_blocksize : (i + 1) * diag_blocksize],
        ) = la.lu(
            A_diagonal_blocks[:, i * diag_blocksize : (i + 1) * diag_blocksize],
            permute_l=True,
        )

        # Compute lower factors
        U_inv_temp = la.solve_triangular(
            U_diagonal_blocks[:, i * diag_blocksize : (i + 1) * diag_blocksize],
            np.eye(diag_blocksize),
            lower=False,
        )

        # L_{i+1, i} = A_{i+1, i} @ U{i, i}^{-1}
        L_lower_diagonal_blocks[:, i * diag_blocksize : (i + 1) * diag_blocksize] = (
            A_lower_diagonal_blocks[:, i * diag_blocksize : (i + 1) * diag_blocksize]
            @ U_inv_temp
        )

        # L_{ndb+1, i} = A_{ndb+1, i} @ U{i, i}^{-1}
        L_arrow_bottom_blocks[:, i * diag_blocksize : (i + 1) * diag_blocksize] = (
            A_arrow_bottom_blocks[:, i * diag_blocksize : (i + 1) * diag_blocksize]
            @ U_inv_temp
        )

        # Compute upper factors
        L_inv_temp = la.solve_triangular(
            L_diagonal_blocks[:, i * diag_blocksize : (i + 1) * diag_blocksize],
            np.eye(diag_blocksize),
            lower=True,
        )

        # U_{i, i+1} = L{i, i}^{-1} @ A_{i, i+1}
        U_upper_diagonal_blocks[:, i * diag_blocksize : (i + 1) * diag_blocksize] = (
            L_inv_temp
            @ A_upper_diagonal_blocks[:, i * diag_blocksize : (i + 1) * diag_blocksize]
        )

        # U_{i, ndb+1} = L{i, i}^{-1} @ A_{i, ndb+1}
        U_arrow_right_blocks[i * diag_blocksize : (i + 1) * diag_blocksize, :] = (
            L_inv_temp
            @ A_arrow_right_blocks[i * diag_blocksize : (i + 1) * diag_blocksize, :]
        )

        # Update next diagonal block
        # A_{i+1, i+1} = A_{i+1, i+1} - L_{i+1, i} @ U_{i, i+1}
        A_diagonal_blocks[:, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize] = (
            A_diagonal_blocks[:, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize]
            - L_lower_diagonal_blocks[:, i * diag_blocksize : (i + 1) * diag_blocksize]
            @ U_upper_diagonal_blocks[:, i * diag_blocksize : (i + 1) * diag_blocksize]
        )

        # Update next upper/lower blocks of the arrowhead
        # A_{ndb+1, i+1} = A_{ndb+1, i+1} - L_{ndb+1, i} @ U_{i, i+1}
        A_arrow_bottom_blocks[
            :, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize
        ] = (
            A_arrow_bottom_blocks[
                :, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize
            ]
            - L_arrow_bottom_blocks[:, i * diag_blocksize : (i + 1) * diag_blocksize]
            @ U_upper_diagonal_blocks[:, i * diag_blocksize : (i + 1) * diag_blocksize]
        )

        # A_{i+1, ndb+1} = A_{i+1, ndb+1} - L_{i+1, i} @ U_{i, ndb+1}
        A_arrow_right_blocks[(i + 1) * diag_blocksize : (i + 2) * diag_blocksize, :] = (
            A_arrow_right_blocks[(i + 1) * diag_blocksize : (i + 2) * diag_blocksize, :]
            - L_lower_diagonal_blocks[:, i * diag_blocksize : (i + 1) * diag_blocksize]
            @ U_arrow_right_blocks[i * diag_blocksize : (i + 1) * diag_blocksize, :]
        )

        # Update the block at the tip of the arrowhead
        # A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, i} @ U_{i, ndb+1}
        A_arrow_tip_block[:, :] = (
            A_arrow_tip_block[:, :]
            - L_arrow_bottom_blocks[:, i * diag_blocksize : (i + 1) * diag_blocksize]
            @ U_arrow_right_blocks[i * diag_blocksize : (i + 1) * diag_blocksize, :]
        )

    # L_{ndb, ndb}, U_{ndb, ndb} = lu_dcmp(A_{ndb, ndb})
    (
        L_diagonal_blocks[:, -diag_blocksize:],
        U_diagonal_blocks[:, -diag_blocksize:],
    ) = la.lu(
        A_diagonal_blocks[:, -diag_blocksize:],
        permute_l=True,
    )

    # L_{ndb+1, ndb} = A_{ndb+1, ndb} @ U_{ndb, ndb}^{-1}
    L_arrow_bottom_blocks[
        :, -diag_blocksize - arrow_blocksize : -arrow_blocksize
    ] = A_arrow_bottom_blocks[:, -diag_blocksize:] @ la.solve_triangular(
        U_diagonal_blocks[:, -diag_blocksize:],
        np.eye(diag_blocksize),
        lower=False,
    )

    # U_{ndb, ndb+1} = L_{ndb, ndb}^{-1} @ A_{ndb, ndb+1}
    U_arrow_right_blocks[-diag_blocksize - arrow_blocksize : -arrow_blocksize, :] = (
        la.solve_triangular(
            L_diagonal_blocks[:, -diag_blocksize:],
            np.eye(diag_blocksize),
            lower=True,
        )
        @ A_arrow_right_blocks[-diag_blocksize:, :]
    )

    # A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, ndb} @ U_{ndb, ndb+1}
    A_arrow_tip_block[:, :] = (
        A_arrow_tip_block[:, :]
        - L_arrow_bottom_blocks[:, -diag_blocksize - arrow_blocksize : -arrow_blocksize]
        @ U_arrow_right_blocks[-diag_blocksize - arrow_blocksize : -arrow_blocksize, :]
    )

    # L_{ndb+1, ndb+1}, U_{ndb+1, ndb+1} = lu_dcmp(A_{ndb+1, ndb+1})
    (
        L_arrow_bottom_blocks[:, -arrow_blocksize:],
        U_arrow_right_blocks[-arrow_blocksize:, :],
    ) = la.lu(A_arrow_tip_block[:, :], permute_l=True)

    return (
        L_diagonal_blocks,
        L_lower_diagonal_blocks,
        L_arrow_bottom_blocks,
        U_diagonal_blocks,
        U_upper_diagonal_blocks,
        U_arrow_right_blocks,
    )