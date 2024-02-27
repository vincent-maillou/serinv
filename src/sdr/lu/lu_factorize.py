"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Lisa Gaedke-Merzhaeuser  (lisa.gaedke.merzhaeuser@usi.ch)
@date: 2024-02

Contains the lu selected factorization routines.

Copyright 2023 ETH Zurich and USI. All rights reserved.
"""

import numpy as np
import scipy.linalg as la


def lu_factorize_tridiag(
    A_diagonal_blocks: np.ndarray,
    A_lower_diagonal_blocks: np.ndarray,
    A_upper_diagonal_blocks: np.ndarray,
) -> np.ndarray:
    """Perform the non-pivoted LU factorization of a block tridiagonal matrix. 
    The matrix is assumed to be non-singular and blocks are assumed to be of the 
    same size given in a sequential array.

    Parameters
    ----------
    A_diagonal_blocks : np.ndarray
        The blocks on the diagonal of the matrix.
    A_lower_diagonal_blocks : np.ndarray
        The blocks on the lower diagonal of the matrix.
    A_upper_diagonal_blocks : np.ndarray
        The blocks on the upper diagonal of the matrix.

    Returns
    -------
    L : np.ndarray
        Lower factor of the LU factorization of the matrix.
    U : np.ndarray
        Upper factor of the LU factorization of the matrix.
    """
    blocksize = A_diagonal_blocks.shape[0]
    nblocks = A_diagonal_blocks.shape[1] // blocksize

    L_diagonal_blocks = np.empty((blocksize, nblocks*blocksize))
    L_lower_diagonal_blocks = np.empty((blocksize, (nblocks-1)*blocksize))
    
    U_diagonal_blocks = np.empty((blocksize, nblocks*blocksize))
    U_upper_diagonal_blocks = np.empty((blocksize, (nblocks-1)*blocksize))

    for i in range(0, nblocks - 1, 1):
        # L_{i, i}, U_{i, i} = lu_dcmp(A_{i, i})
        (
            L_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize],
            U_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize]
        ) = la.lu(
            A_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize],
            permute_l=True,
        )

        # L_{i+1, i} = A_{i+1, i} @ U{i, i}^{-1}
        L_lower_diagonal_blocks[
            :, i * blocksize : (i + 1) * blocksize,
        ] = A_lower_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize] @ la.solve_triangular(
            U_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize],
            np.eye(blocksize),
            lower=False,
        )

        # U_{i, i+1} = L{i, i}^{-1} @ A_{i, i+1}
        U_upper_diagonal_blocks[
            :, i * blocksize : (i + 1) * blocksize,
        ] = (
            la.solve_triangular(
                L_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize],
                np.eye(blocksize),
                lower=True,
            )
            @ A_upper_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize]
        )

        # A_{i+1, i+1} = A_{i+1, i+1} - L_{i+1, i} @ U_{i, i+1}
        A_diagonal_blocks[:, (i + 1) * blocksize : (i + 2) * blocksize] = (
            A_diagonal_blocks[:, (i + 1) * blocksize : (i + 2) * blocksize]
            - L_lower_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize]
            @ U_upper_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize]
        )

    # L_{nblocks, nblocks}, U_{nblocks, nblocks} = lu_dcmp(A_{nblocks, nblocks})
    (
        L_diagonal_blocks[:, -blocksize:],
        U_diagonal_blocks[:, -blocksize:],
    ) = la.lu(A_diagonal_blocks[:, -blocksize:], permute_l=True)

    return (
        L_diagonal_blocks,
        L_lower_diagonal_blocks,
        U_diagonal_blocks,
        U_upper_diagonal_blocks,
    )


def lu_factorize_tridiag_arrowhead(
    A_diagonal_blocks: np.ndarray,
    A_lower_diagonal_blocks: np.ndarray,
    A_upper_diagonal_blocks: np.ndarray,
    A_arrow_bottom_blocks: np.ndarray,
    A_arrow_right_blocks: np.ndarray,
    A_arrow_tip_block: np.ndarray,
) -> [np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Perform the lu factorization of a block tridiagonal arrowhead
    matrix. The matrix is assumed to be non singular.

    Parameters
    ----------
    TODO:docstring

    Returns
    -------
    TODO:docstring
    
    """

    diag_blocksize = A_diagonal_blocks.shape[0]
    arrow_blocksize = A_arrow_bottom_blocks.shape[0]

    n_diag_blocks = A_diagonal_blocks.shape[1] // diag_blocksize
    
    L_diagonal_blocks = np.empty((diag_blocksize, n_diag_blocks*diag_blocksize))
    L_lower_diagonal_blocks = np.empty((diag_blocksize, (n_diag_blocks-1)*diag_blocksize))
    L_arrow_bottom_blocks = np.empty((arrow_blocksize, n_diag_blocks*diag_blocksize + arrow_blocksize))
    
    U_diagonal_blocks = np.empty((diag_blocksize, n_diag_blocks*diag_blocksize))
    U_upper_diagonal_blocks = np.empty((diag_blocksize, (n_diag_blocks-1)*diag_blocksize))
    U_arrow_right_blocks = np.empty((n_diag_blocks*diag_blocksize + arrow_blocksize, arrow_blocksize))

    L_inv_temp = np.empty((diag_blocksize, diag_blocksize))
    U_inv_temp = np.empty((diag_blocksize, diag_blocksize))

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
        A_arrow_bottom_blocks[:, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize] = (
            A_arrow_bottom_blocks[:, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize]
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
    L_arrow_bottom_blocks[:, -diag_blocksize - arrow_blocksize : -arrow_blocksize] = (
        A_arrow_bottom_blocks[:, -diag_blocksize:] 
        @ la.solve_triangular(
            U_diagonal_blocks[:, -diag_blocksize:],
            np.eye(diag_blocksize),
            lower=False,
        )
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
        L_arrow_bottom_blocks[:, -arrow_blocksize :],
        U_arrow_right_blocks[-arrow_blocksize : , :],
    ) = la.lu(A_arrow_tip_block[:, :], permute_l=True)

    return (
        L_diagonal_blocks, 
        L_lower_diagonal_blocks, 
        L_arrow_bottom_blocks, 
        U_diagonal_blocks, 
        U_upper_diagonal_blocks, 
        U_arrow_right_blocks
    )
