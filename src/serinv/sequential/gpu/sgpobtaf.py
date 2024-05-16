# Copyright 2023-2024 ETH Zurich and USI. All rights reserved.

import cupy as cp
import cupyx as cpx
import cupyx.scipy.linalg as cpla
import numpy as np
from cupy.linalg import cholesky


def sgpobtaf(
    A_diagonal_blocks: np.ndarray,
    A_lower_diagonal_blocks: np.ndarray,
    A_arrow_bottom_blocks: np.ndarray,
    A_arrow_tip_block: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Perform the Cholesky factorization of a block tridiagonal matrix using
    a sequential algorithm on GPU backend.

    Parameters
    ----------
    A_diagonal_blocks : np.ndarray
        Diagonal blocks of the matrix.
    A_lower_diagonal_blocks : np.ndarray
        Lower diagonal blocks of the matrix.
    A_arrow_bottom_blocks : np.ndarray
        Arrow bottom blocks of the matrix.
    A_arrow_tip_block : np.ndarray
        Arrow tip block of the matrix.

    Returns
    -------
    L_diagonal_blocks : np.ndarray
        Diagonal blocks of the cholesky factorization.
    L_lower_diagonal_blocks : np.ndarray
        Lower diagonal blocks of the cholesky factorization.
    L_arrow_bottom_blocks : np.ndarray
        Arrow bottom blocks of the cholesky factorization.
    L_arrow_tip_block : np.ndarray
        Arrow tip block of the cholesky factorization.
    """

    diag_blocksize = A_diagonal_blocks.shape[0]
    n_diag_blocks = A_diagonal_blocks.shape[1] // diag_blocksize

    A_diagonal_blocks_gpu: cp.ndarray = cp.asarray(A_diagonal_blocks)
    A_lower_diagonal_blocks_gpu: cp.ndarray = cp.asarray(A_lower_diagonal_blocks)
    A_arrow_bottom_blocks_gpu: cp.ndarray = cp.asarray(A_arrow_bottom_blocks)
    A_arrow_tip_block_gpu: cp.ndarray = cp.asarray(A_arrow_tip_block)

    # Host side arrays
    L_diagonal_blocks: cp.ndarray = cpx.empty_like_pinned(A_diagonal_blocks)
    L_lower_diagonal_blocks: cp.ndarray = cpx.empty_like_pinned(A_lower_diagonal_blocks)
    L_arrow_bottom_blocks: cpx.ndarray = cpx.empty_like_pinned(A_arrow_bottom_blocks)
    L_arrow_tip_block: cpx.ndarray = cpx.empty_like_pinned(A_arrow_tip_block)

    # Device side arrays
    L_diagonal_blocks_gpu: cp.ndarray = cp.empty_like(L_diagonal_blocks)
    L_lower_diagonal_blocks_gpu: cp.ndarray = cp.empty_like(L_lower_diagonal_blocks)
    L_arrow_bottom_blocks_gpu: cp.ndarray = cp.empty_like(L_arrow_bottom_blocks)
    L_arrow_tip_block_gpu: cp.ndarray = cp.empty_like(L_arrow_tip_block)

    L_inv_temp_gpu: cp.ndarray = cp.empty(
        (diag_blocksize, diag_blocksize), dtype=A_diagonal_blocks.dtype
    )

    for i in range(0, n_diag_blocks - 1):
        # L_{i, i} = chol(A_{i, i})
        L_diagonal_blocks_gpu[:, i * diag_blocksize : (i + 1) * diag_blocksize] = (
            cholesky(
                A_diagonal_blocks_gpu[:, i * diag_blocksize : (i + 1) * diag_blocksize]
            )
        )

        # Temporary storage of used twice lower triangular solving
        L_inv_temp_gpu[:, :] = cpla.solve_triangular(
            L_diagonal_blocks_gpu[:, i * diag_blocksize : (i + 1) * diag_blocksize],
            cp.eye(diag_blocksize),
            lower=True,
        ).T

        # L_{i+1, i} = A_{i+1, i} @ L_{i, i}^{-T}
        L_lower_diagonal_blocks_gpu[
            :, i * diag_blocksize : (i + 1) * diag_blocksize
        ] = (
            A_lower_diagonal_blocks_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            @ L_inv_temp_gpu
        )

        # L_{ndb+1, i} = A_{ndb+1, i} @ L_{i, i}^{-T}
        L_arrow_bottom_blocks_gpu[:, i * diag_blocksize : (i + 1) * diag_blocksize] = (
            A_arrow_bottom_blocks_gpu[:, i * diag_blocksize : (i + 1) * diag_blocksize]
            @ L_inv_temp_gpu
        )

        # Update next diagonal block
        # A_{i+1, i+1} = A_{i+1, i+1} - L_{i+1, i} @ L_{i+1, i}.T
        A_diagonal_blocks_gpu[
            :,
            (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
        ] = (
            A_diagonal_blocks_gpu[
                :,
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
            ]
            - L_lower_diagonal_blocks_gpu[
                :,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ]
            @ L_lower_diagonal_blocks_gpu[
                :,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ].T
        )

        # A_{ndb+1, i+1} = A_{ndb+1, i+1} - L_{ndb+1, i} @ L_{i+1, i}.T
        A_arrow_bottom_blocks_gpu[
            :,
            (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
        ] = (
            A_arrow_bottom_blocks_gpu[
                :,
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
            ]
            - L_arrow_bottom_blocks_gpu[
                :,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ]
            @ L_lower_diagonal_blocks_gpu[
                :,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ].T
        )

        # A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, i} @ L_{ndb+1, i}.T
        A_arrow_tip_block_gpu[:, :] = (
            A_arrow_tip_block_gpu[:, :]
            - L_arrow_bottom_blocks_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            @ L_arrow_bottom_blocks_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ].T
        )

    # L_{ndb, ndb} = chol(A_{ndb, ndb})
    L_diagonal_blocks_gpu[:, -diag_blocksize:] = cholesky(
        A_diagonal_blocks_gpu[:, -diag_blocksize:]
    )

    # L_{ndb+1, ndb} = A_{ndb+1, ndb} @ L_{ndb, ndb}^{-T}
    L_arrow_bottom_blocks_gpu[:, -diag_blocksize:] = (
        A_arrow_bottom_blocks_gpu[:, -diag_blocksize:]
        @ cpla.solve_triangular(
            L_diagonal_blocks_gpu[:, -diag_blocksize:],
            cp.eye(diag_blocksize),
            lower=True,
        ).T
    )

    # A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, ndb} @ L_{ndb+1, ndb}^{T}
    A_arrow_tip_block_gpu[:, :] = (
        A_arrow_tip_block_gpu[:, :]
        - L_arrow_bottom_blocks_gpu[:, -diag_blocksize:]
        @ L_arrow_bottom_blocks_gpu[:, -diag_blocksize:].T
    )

    # L_{ndb+1, ndb+1} = chol(A_{ndb+1, ndb+1})
    L_arrow_tip_block_gpu[:, :] = cholesky(A_arrow_tip_block_gpu[:, :])

    L_diagonal_blocks_gpu.get(out=L_diagonal_blocks)
    L_lower_diagonal_blocks_gpu.get(out=L_lower_diagonal_blocks)
    L_arrow_bottom_blocks_gpu.get(out=L_arrow_bottom_blocks)
    L_arrow_tip_block_gpu.get(out=L_arrow_tip_block)

    return (
        L_diagonal_blocks,
        L_lower_diagonal_blocks,
        L_arrow_bottom_blocks,
        L_arrow_tip_block,
    )
