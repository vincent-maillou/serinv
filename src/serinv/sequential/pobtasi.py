# Copyright 2023-2024 ETH Zurich. All rights reserved.

try:
    import cupy as cp
    import cupyx.scipy.linalg as cu_la

    CUPY_AVAIL = True

except:
    CUPY_AVAIL = False

import numpy as np
import scipy.linalg as np_la


def pobtasi(
    L_diagonal_blocks: np.ndarray,
    L_lower_diagonal_blocks: np.ndarray,
    L_arrow_bottom_blocks: np.ndarray,
    L_arrow_tip_block: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Perform a selected inversion of a block tridiagonal arrowhead matrix using a
    sequential block algorithm.

    Note:
    -----
    - If a device array is given, the algorithm will run on the GPU.

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

    la = np_la
    if CUPY_AVAIL:
        xp = cp.get_array_module(L_diagonal_blocks)
        if xp == cp:
            la = cu_la
    else:
        xp = np

    diag_blocksize = L_diagonal_blocks.shape[1]
    arrow_blocksize = L_arrow_tip_block.shape[0]
    n_diag_blocks = L_diagonal_blocks.shape[0]

    X_diagonal_blocks = xp.zeros_like(L_diagonal_blocks)
    X_lower_diagonal_blocks = xp.zeros_like(L_lower_diagonal_blocks)
    X_arrow_bottom_blocks = xp.zeros_like(L_arrow_bottom_blocks)
    X_arrow_tip_block = xp.zeros_like(L_arrow_tip_block)
    L_last_blk_inv = xp.zeros_like(L_arrow_tip_block)

    L_blk_inv = xp.zeros(
        (diag_blocksize, diag_blocksize), dtype=L_diagonal_blocks.dtype
    )

    L_last_blk_inv = la.solve_triangular(
        L_arrow_tip_block[:, :], xp.eye(arrow_blocksize), lower=True
    )

    X_arrow_tip_block[:, :] = L_last_blk_inv.conj().T @ L_last_blk_inv

    L_blk_inv = la.solve_triangular(
        L_diagonal_blocks[-1, :, :],
        xp.eye(diag_blocksize),
        lower=True,
    )

    # X_{ndb+1, ndb} = -X_{ndb+1, ndb+1} L_{ndb+1, ndb} L_{ndb, ndb}^{-1}
    X_arrow_bottom_blocks[-1, :, :] = (
        -X_arrow_tip_block[:, :] @ L_arrow_bottom_blocks[-1, :, :] @ L_blk_inv
    )

    # X_{ndb, ndb} = (L_{ndb, ndb}^{-T} - X_{ndb+1, ndb}^{T} L_{ndb+1, ndb}) L_{ndb, ndb}^{-1}
    X_diagonal_blocks[-1, :, :] = (
        L_blk_inv.conj().T
        - X_arrow_bottom_blocks[-1, :, :].conj().T @ L_arrow_bottom_blocks[-1, :, :]
    ) @ L_blk_inv

    for i in range(n_diag_blocks - 2, -1, -1):
        L_blk_inv = la.solve_triangular(
            L_diagonal_blocks[i, :, :],
            xp.eye(diag_blocksize),
            lower=True,
        )

        # --- Off-diagonal block part ---
        # X_{i+1, i} = (-X_{i+1, i+1} L_{i+1, i} - X_{ndb+1, i+1}^{T} L_{ndb+1, i}) L_{i, i}^{-1}
        X_lower_diagonal_blocks[i, :, :] = (
            -X_diagonal_blocks[i + 1, :, :] @ L_lower_diagonal_blocks[i, :, :]
            - X_arrow_bottom_blocks[i + 1, :, :].conj().T
            @ L_arrow_bottom_blocks[i, :, :]
        ) @ L_blk_inv

        # --- Arrowhead part ---
        # X_{ndb+1, i} = (- X_{ndb+1, i+1} L_{i+1, i} - X_{ndb+1, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1}
        X_arrow_bottom_blocks[i, :, :] = (
            -X_arrow_bottom_blocks[i + 1, :, :] @ L_lower_diagonal_blocks[i, :, :]
            - X_arrow_tip_block[:, :] @ L_arrow_bottom_blocks[i, :, :]
        ) @ L_blk_inv

        # --- Diagonal block part ---
        # X_{i, i} = (L_{i, i}^{-T} - X_{i+1, i}^{T} L_{i+1, i} - X_{ndb+1, i}.conj().T L_{ndb+1, i}) L_{i, i}^{-1}
        X_diagonal_blocks[i, :, :] = (
            L_blk_inv.conj().T
            - X_lower_diagonal_blocks[i, :, :].conj().T
            @ L_lower_diagonal_blocks[i, :, :]
            - X_arrow_bottom_blocks[i, :, :].conj().T @ L_arrow_bottom_blocks[i, :, :]
        ) @ L_blk_inv

    return (
        X_diagonal_blocks,
        X_lower_diagonal_blocks,
        X_arrow_bottom_blocks,
        X_arrow_tip_block,
    )
