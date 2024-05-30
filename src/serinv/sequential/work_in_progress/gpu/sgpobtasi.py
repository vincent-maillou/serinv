# Copyright 2023-2024 ETH Zurich. All rights reserved.


import cupy as cp
import cupyx as cpx
import cupyx.scipy.linalg as cpla
import numpy as np


def sgpobtasi(
    L_diagonal_blocks: np.ndarray,
    L_lower_diagonal_blocks: np.ndarray,
    L_arrow_bottom_blocks: np.ndarray,
    L_arrow_tip_block: np.ndarray,
) -> np.ndarray:
    """Perform a selected inversion of a block tridiagonal matrix using a
    sequential algorithm on GPU backend.

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

    L_diagonal_blocks_gpu: cp.ndarray = cp.asarray(L_diagonal_blocks)
    L_lower_diagonal_blocks_gpu: cp.ndarray = cp.asarray(L_lower_diagonal_blocks)
    L_arrow_bottom_blocks_gpu: cp.ndarray = cp.asarray(L_arrow_bottom_blocks)

    # Host side arrays
    X_diagonal_blocks: cpx.ndarray = cpx.empty_like_pinned(L_diagonal_blocks_gpu)
    X_lower_diagonal_blocks: cpx.ndarray = cpx.empty_like_pinned(
        L_lower_diagonal_blocks_gpu
    )
    X_arrow_bottom_blocks: cpx.ndarray = cpx.empty_like_pinned(
        L_arrow_bottom_blocks_gpu
    )
    X_arrow_tip_block: cpx.ndarray = cpx.empty_pinned(
        (arrow_blocksize, arrow_blocksize), dtype=L_diagonal_blocks.dtype
    )

    # Device side arrays
    X_diagonal_blocks_gpu: cp.ndarray = cp.empty_like(X_diagonal_blocks)
    X_lower_diagonal_blocks_gpu: cp.ndarray = cp.empty_like(X_lower_diagonal_blocks)
    X_arrow_bottom_blocks_gpu: cp.ndarray = cp.empty_like(X_arrow_bottom_blocks)
    X_arrow_tip_block_gpu: cp.ndarray = cp.empty_like(X_arrow_tip_block)

    L_last_blk_inv_gpu = cp.asarray(L_arrow_tip_block)
    L_blk_inv_gpu = cp.empty(
        (diag_blocksize, diag_blocksize), dtype=L_diagonal_blocks.dtype
    )

    L_last_blk_inv_gpu[:, :] = cpla.solve_triangular(
        L_last_blk_inv_gpu[:, :], cp.eye(arrow_blocksize), lower=True
    )

    X_arrow_tip_block_gpu[:, :] = L_last_blk_inv_gpu.T @ L_last_blk_inv_gpu

    L_blk_inv_gpu[:, :] = cpla.solve_triangular(
        L_diagonal_blocks_gpu[:, -diag_blocksize:],
        cp.eye(diag_blocksize),
        lower=True,
    )

    # X_{ndb+1, ndb} = -X_{ndb+1, ndb+1} L_{ndb+1, ndb} L_{ndb, ndb}^{-1}
    X_arrow_bottom_blocks_gpu[:, -diag_blocksize:] = (
        -X_arrow_tip_block_gpu[:, :]
        @ L_arrow_bottom_blocks_gpu[:, -diag_blocksize:]
        @ L_blk_inv_gpu
    )

    # X_{ndb, ndb} = (L_{ndb, ndb}^{-T} - X_{ndb+1, ndb}^{T} L_{ndb+1, ndb}) L_{ndb, ndb}^{-1}
    X_diagonal_blocks_gpu[:, -diag_blocksize:] = (
        L_blk_inv_gpu.T
        - X_arrow_bottom_blocks_gpu[:, -diag_blocksize:].T
        @ L_arrow_bottom_blocks_gpu[:, -diag_blocksize:]
    ) @ L_blk_inv_gpu

    for i in range(n_diag_blocks - 2, -1, -1):
        L_blk_inv_gpu = cpla.solve_triangular(
            L_diagonal_blocks_gpu[:, i * diag_blocksize : (i + 1) * diag_blocksize],
            cp.eye(diag_blocksize),
            lower=True,
        )

        # --- Off-diagonal block part ---
        # X_{i+1, i} = (-X_{i+1, i+1} L_{i+1, i} - X_{ndb+1, i+1}^{T} L_{ndb+1, i}) L_{i, i}^{-1}
        X_lower_diagonal_blocks_gpu[
            :, i * diag_blocksize : (i + 1) * diag_blocksize
        ] = (
            -X_diagonal_blocks_gpu[
                :, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize
            ]
            @ L_lower_diagonal_blocks_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            - X_arrow_bottom_blocks_gpu[
                :, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize
            ].T
            @ L_arrow_bottom_blocks_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
        ) @ L_blk_inv_gpu

        # --- Arrowhead part ---
        # X_{ndb+1, i} = (- X_{ndb+1, i+1} L_{i+1, i} - X_{ndb+1, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1}
        X_arrow_bottom_blocks_gpu[:, i * diag_blocksize : (i + 1) * diag_blocksize] = (
            -X_arrow_bottom_blocks_gpu[
                :, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize
            ]
            @ L_lower_diagonal_blocks_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            - X_arrow_tip_block_gpu[:, :]
            @ L_arrow_bottom_blocks_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
        ) @ L_blk_inv_gpu

        # --- Diagonal block part ---
        # X_{i, i} = (L_{i, i}^{-T} - X_{i+1, i}^{T} L_{i+1, i} - X_{ndb+1, i}.T L_{ndb+1, i}) L_{i, i}^{-1}
        X_diagonal_blocks_gpu[:, i * diag_blocksize : (i + 1) * diag_blocksize] = (
            L_blk_inv_gpu.T
            - X_lower_diagonal_blocks_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ].T
            @ L_lower_diagonal_blocks_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            - X_arrow_bottom_blocks_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ].T
            @ L_arrow_bottom_blocks_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
        ) @ L_blk_inv_gpu

    X_diagonal_blocks_gpu.get(out=X_diagonal_blocks)
    X_lower_diagonal_blocks_gpu.get(out=X_lower_diagonal_blocks)
    X_arrow_bottom_blocks_gpu.get(out=X_arrow_bottom_blocks)
    X_arrow_tip_block_gpu.get(out=X_arrow_tip_block)

    return (
        X_diagonal_blocks,
        X_lower_diagonal_blocks,
        X_arrow_bottom_blocks,
        X_arrow_tip_block,
    )
