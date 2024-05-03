# Copyright 2023-2024 ETH Zurich and USI. All rights reserved.


try:
    import cupy as cp
    import cupyx as cpx
    import cupyx.scipy.linalg as cpla
except ImportError:
    pass

import numpy as np


def cholesky_sinv_block_tridiagonal_arrowhead_gpu(
    L_diagonal_blocks: np.ndarray,
    L_lower_diagonal_blocks: np.ndarray,
    L_arrow_bottom_blocks: np.ndarray,
    L_arrow_tip_block: np.ndarray,
) -> np.ndarray:
    """Perform a selected inversion from a cholesky decomposed matrix with a
    block tridiagonal arrowhead structure.

    Parameters
    ----------
    L : np.ndarray
        The cholesky factorization of the matrix.
    diag_blocksize : int
        Blocksize of the diagonals blocks of the matrix.
    arrow_blocksize : int
        Blocksize of the blocks composing the arrowhead.

    Returns
    -------
    X : np.ndarray
        Selected inversion of the matrix.
    """

    diag_blocksize = L_diagonal_blocks.shape[1]
    arrow_blocksize = L_arrow_tip_block.shape[1]
    n_diag_blocks = L_diagonal_blocks.shape[0]

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
        L_diagonal_blocks_gpu[-1, :, :],
        cp.eye(diag_blocksize),
        lower=True,
    )

    # X_{ndb+1, ndb} = -X_{ndb+1, ndb+1} L_{ndb+1, ndb} L_{ndb, ndb}^{-1}
    X_arrow_bottom_blocks_gpu[-1, :, :] = (
        -X_arrow_tip_block_gpu[:, :]
        @ L_arrow_bottom_blocks_gpu[-1, :, :]
        @ L_blk_inv_gpu
    )

    # X_{ndb, ndb} = (L_{ndb, ndb}^{-T} - X_{ndb+1, ndb}^{T} L_{ndb+1, ndb}) L_{ndb, ndb}^{-1}
    X_diagonal_blocks_gpu[-1, :, :] = (
        L_blk_inv_gpu.T
        - X_arrow_bottom_blocks_gpu[-1, :, :].T @ L_arrow_bottom_blocks_gpu[-1, :, :]
    ) @ L_blk_inv_gpu

    for i in range(n_diag_blocks - 2, -1, -1):
        L_blk_inv_gpu = cpla.solve_triangular(
            L_diagonal_blocks_gpu[i, :, :],
            cp.eye(diag_blocksize),
            lower=True,
        )

        # --- Off-diagonal block part ---
        # X_{i+1, i} = (-X_{i+1, i+1} L_{i+1, i} - X_{ndb+1, i+1}^{T} L_{ndb+1, i}) L_{i, i}^{-1}
        X_lower_diagonal_blocks_gpu[i, :, :] = (
            -X_diagonal_blocks_gpu[i + 1, :, :] @ L_lower_diagonal_blocks_gpu[i, :, :]
            - X_arrow_bottom_blocks_gpu[i + 1, :, :].T
            @ L_arrow_bottom_blocks_gpu[i, :, :]
        ) @ L_blk_inv_gpu

        # --- Arrowhead part ---
        # X_{ndb+1, i} = (- X_{ndb+1, i+1} L_{i+1, i} - X_{ndb+1, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1}
        X_arrow_bottom_blocks_gpu[i, :, :] = (
            -X_arrow_bottom_blocks_gpu[i + 1, :, :]
            @ L_lower_diagonal_blocks_gpu[i, :, :]
            - X_arrow_tip_block_gpu[:, :] @ L_arrow_bottom_blocks_gpu[i, :, :]
        ) @ L_blk_inv_gpu

        # --- Diagonal block part ---
        # X_{i, i} = (L_{i, i}^{-T} - X_{i+1, i}^{T} L_{i+1, i} - X_{ndb+1, i}.T L_{ndb+1, i}) L_{i, i}^{-1}
        X_diagonal_blocks_gpu[i, :, :] = (
            L_blk_inv_gpu.T
            - X_lower_diagonal_blocks_gpu[i, :, :].T
            @ L_lower_diagonal_blocks_gpu[i, :, :]
            - X_arrow_bottom_blocks_gpu[i, :, :].T @ L_arrow_bottom_blocks_gpu[i, :, :]
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
