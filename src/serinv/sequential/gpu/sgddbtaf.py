# Copyright 2023-2024 ETH Zurich and USI. All rights reserved.

import cupy as cp
import cupyx as cpx
import cupyx.scipy.linalg as cpla

import numpy as np


def sgddbtaf(
    A_diagonal_blocks: np.ndarray,
    A_lower_diagonal_blocks: np.ndarray,
    A_upper_diagonal_blocks: np.ndarray,
    A_arrow_bottom_blocks: np.ndarray,
    A_arrow_right_blocks: np.ndarray,
    A_arrow_tip_block: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Perform the LU factorization of a block tridiagonal matrix using
    a sequential algotithm on a GPU backend.

    The matrix is assumed to be block-diagonally dominant.

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

    A_diagonal_blocks_gpu: cp.ndarray = cp.asarray(A_diagonal_blocks)
    A_lower_diagonal_blocks_gpu: cp.ndarray = cp.asarray(A_lower_diagonal_blocks)
    A_upper_diagonal_blocks_gpu: cp.ndarray = cp.asarray(A_upper_diagonal_blocks)
    A_arrow_bottom_blocks_gpu: cp.ndarray = cp.asarray(A_arrow_bottom_blocks)
    A_arrow_right_blocks_gpu: cp.ndarray = cp.asarray(A_arrow_right_blocks)
    A_arrow_tip_block_gpu: cp.ndarray = cp.asarray(A_arrow_tip_block)

    # Host side arrays
    L_diagonal_blocks: cp.ndarray = cpx.empty_like_pinned(A_diagonal_blocks)
    L_lower_diagonal_blocks: cp.ndarray = cpx.empty_like_pinned(A_lower_diagonal_blocks)
    L_arrow_bottom_blocks: cpx.ndarray = cpx.empty_pinned(
        (arrow_blocksize, n_diag_blocks * diag_blocksize + arrow_blocksize),
        dtype=A_diagonal_blocks.dtype,
    )
    U_diagonal_blocks: cp.ndarray = cpx.empty_like_pinned(A_diagonal_blocks)
    U_upper_diagonal_blocks: cp.ndarray = cpx.empty_like_pinned(A_upper_diagonal_blocks)
    U_arrow_right_blocks: cp.ndarray = cpx.empty_pinned(
        (n_diag_blocks * diag_blocksize + arrow_blocksize, arrow_blocksize),
        dtype=A_diagonal_blocks.dtype,
    )

    # Device side arrays
    L_diagonal_blocks_gpu: cp.ndarray = cp.empty_like(L_diagonal_blocks)
    L_lower_diagonal_blocks_gpu: cp.ndarray = cp.empty_like(L_lower_diagonal_blocks)
    L_arrow_bottom_blocks_gpu: cp.ndarray = cp.empty_like(L_arrow_bottom_blocks)

    U_diagonal_blocks_gpu: cp.ndarray = cp.empty_like(U_diagonal_blocks)
    U_upper_diagonal_blocks_gpu: cp.ndarray = cp.empty_like(U_upper_diagonal_blocks)
    U_arrow_right_blocks_gpu: cp.ndarray = cp.empty_like(U_arrow_right_blocks)

    L_inv_temp_gpu: cp.ndarray = cp.empty(
        (diag_blocksize, diag_blocksize), dtype=A_diagonal_blocks.dtype
    )
    U_inv_temp_gpu: cp.ndarray = cp.empty(
        (diag_blocksize, diag_blocksize), dtype=A_diagonal_blocks.dtype
    )

    for i in range(0, n_diag_blocks - 1):
        # L_{i, i}, U_{i, i} = lu_dcmp(A_{i, i})
        (
            L_diagonal_blocks_gpu[:, i * diag_blocksize : (i + 1) * diag_blocksize],
            U_diagonal_blocks_gpu[:, i * diag_blocksize : (i + 1) * diag_blocksize],
        ) = cpla.lu(
            A_diagonal_blocks_gpu[:, i * diag_blocksize : (i + 1) * diag_blocksize],
            permute_l=True,
        )

        # Compute lower factors
        U_inv_temp_gpu = cpla.solve_triangular(
            U_diagonal_blocks_gpu[:, i * diag_blocksize : (i + 1) * diag_blocksize],
            cp.eye(diag_blocksize),
            lower=False,
        )

        # L_{i+1, i} = A_{i+1, i} @ U{i, i}^{-1}
        L_lower_diagonal_blocks_gpu[
            :, i * diag_blocksize : (i + 1) * diag_blocksize
        ] = (
            A_lower_diagonal_blocks_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            @ U_inv_temp_gpu
        )

        # L_{ndb+1, i} = A_{ndb+1, i} @ U{i, i}^{-1}
        L_arrow_bottom_blocks_gpu[:, i * diag_blocksize : (i + 1) * diag_blocksize] = (
            A_arrow_bottom_blocks_gpu[:, i * diag_blocksize : (i + 1) * diag_blocksize]
            @ U_inv_temp_gpu
        )

        # Compute upper factors
        L_inv_temp_gpu = cpla.solve_triangular(
            L_diagonal_blocks_gpu[:, i * diag_blocksize : (i + 1) * diag_blocksize],
            cp.eye(diag_blocksize),
            lower=True,
        )

        # U_{i, i+1} = L{i, i}^{-1} @ A_{i, i+1}
        U_upper_diagonal_blocks_gpu[
            :, i * diag_blocksize : (i + 1) * diag_blocksize
        ] = (
            L_inv_temp_gpu
            @ A_upper_diagonal_blocks_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
        )

        # U_{i, ndb+1} = L{i, i}^{-1} @ A_{i, ndb+1}
        U_arrow_right_blocks_gpu[i * diag_blocksize : (i + 1) * diag_blocksize, :] = (
            L_inv_temp_gpu
            @ A_arrow_right_blocks_gpu[i * diag_blocksize : (i + 1) * diag_blocksize, :]
        )

        # Update next diagonal block
        # A_{i+1, i+1} = A_{i+1, i+1} - L_{i+1, i} @ U_{i, i+1}
        A_diagonal_blocks_gpu[
            :, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize
        ] = (
            A_diagonal_blocks_gpu[
                :, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize
            ]
            - L_lower_diagonal_blocks_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            @ U_upper_diagonal_blocks_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
        )

        # Update next upper/lower blocks of the arrowhead
        # A_{ndb+1, i+1} = A_{ndb+1, i+1} - L_{ndb+1, i} @ U_{i, i+1}
        A_arrow_bottom_blocks_gpu[
            :, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize
        ] = (
            A_arrow_bottom_blocks_gpu[
                :, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize
            ]
            - L_arrow_bottom_blocks_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            @ U_upper_diagonal_blocks_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
        )

        # A_{i+1, ndb+1} = A_{i+1, ndb+1} - L_{i+1, i} @ U_{i, ndb+1}
        A_arrow_right_blocks_gpu[
            (i + 1) * diag_blocksize : (i + 2) * diag_blocksize, :
        ] = (
            A_arrow_right_blocks_gpu[
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize, :
            ]
            - L_lower_diagonal_blocks_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            @ U_arrow_right_blocks_gpu[i * diag_blocksize : (i + 1) * diag_blocksize, :]
        )

        # Update the block at the tip of the arrowhead
        # A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, i} @ U_{i, ndb+1}
        A_arrow_tip_block_gpu[:, :] = (
            A_arrow_tip_block_gpu[:, :]
            - L_arrow_bottom_blocks_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            @ U_arrow_right_blocks_gpu[i * diag_blocksize : (i + 1) * diag_blocksize, :]
        )

    # L_{ndb, ndb}, U_{ndb, ndb} = lu_dcmp(A_{ndb, ndb})
    (
        L_diagonal_blocks_gpu[:, -diag_blocksize:],
        U_diagonal_blocks_gpu[:, -diag_blocksize:],
    ) = cpla.lu(
        A_diagonal_blocks_gpu[:, -diag_blocksize:],
        permute_l=True,
    )

    # L_{ndb+1, ndb} = A_{ndb+1, ndb} @ U_{ndb, ndb}^{-1}
    L_arrow_bottom_blocks_gpu[
        :, -diag_blocksize - arrow_blocksize : -arrow_blocksize
    ] = A_arrow_bottom_blocks_gpu[:, -diag_blocksize:] @ cpla.solve_triangular(
        U_diagonal_blocks_gpu[:, -diag_blocksize:],
        cp.eye(diag_blocksize),
        lower=False,
    )

    # U_{ndb, ndb+1} = L_{ndb, ndb}^{-1} @ A_{ndb, ndb+1}
    U_arrow_right_blocks_gpu[
        -diag_blocksize - arrow_blocksize : -arrow_blocksize, :
    ] = (
        cpla.solve_triangular(
            L_diagonal_blocks_gpu[:, -diag_blocksize:],
            cp.eye(diag_blocksize),
            lower=True,
        )
        @ A_arrow_right_blocks_gpu[-diag_blocksize:, :]
    )

    # A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, ndb} @ U_{ndb, ndb+1}
    A_arrow_tip_block_gpu[:, :] = (
        A_arrow_tip_block_gpu[:, :]
        - L_arrow_bottom_blocks_gpu[
            :, -diag_blocksize - arrow_blocksize : -arrow_blocksize
        ]
        @ U_arrow_right_blocks_gpu[
            -diag_blocksize - arrow_blocksize : -arrow_blocksize, :
        ]
    )

    # L_{ndb+1, ndb+1}, U_{ndb+1, ndb+1} = lu_dcmp(A_{ndb+1, ndb+1})
    (
        L_arrow_bottom_blocks_gpu[:, -arrow_blocksize:],
        U_arrow_right_blocks_gpu[-arrow_blocksize:, :],
    ) = cpla.lu(A_arrow_tip_block_gpu[:, :], permute_l=True)

    L_diagonal_blocks_gpu.get(out=L_diagonal_blocks)
    L_lower_diagonal_blocks_gpu.get(out=L_lower_diagonal_blocks)
    L_arrow_bottom_blocks_gpu.get(out=L_arrow_bottom_blocks)
    U_diagonal_blocks_gpu.get(out=U_diagonal_blocks)
    U_upper_diagonal_blocks_gpu.get(out=U_upper_diagonal_blocks)
    U_arrow_right_blocks_gpu.get(out=U_arrow_right_blocks)

    return (
        L_diagonal_blocks,
        L_lower_diagonal_blocks,
        L_arrow_bottom_blocks,
        U_diagonal_blocks,
        U_upper_diagonal_blocks,
        U_arrow_right_blocks,
    )
