# Copyright 2023-2024 ETH Zurich and USI. All rights reserved.

try:
    import cupy as cp
    import cupyx as cpx
    import cupyx.scipy.linalg as cpla
    from cupy.linalg import cholesky
except ImportError:
    pass

import numpy as np


def cholesky_factorize_block_tridiagonal_arrowhead_gpu(
    A_diagonal_blocks: np.ndarray,
    A_lower_diagonal_blocks: np.ndarray,
    A_arrow_bottom_blocks: np.ndarray,
    A_arrow_tip_block: np.ndarray,
) -> np.ndarray:
    """Perform the cholesky factorization of a block tridiagonal arrowhead
    matrix. The matrix is assumed to be symmetric positive definite.

    Parameters
    ----------
    A : np.ndarray
        Input matrix to decompose.
    diag_blocksize : int
        Blocksize of the diagonals blocks of the matrix.
    arrow_blocksize : int
        Blocksize of the blocks composing the arrowhead.
    overwrite : bool
        If True, the input matrix A is modified in place. Default is False.

    Returns
    -------
    L : np.ndarray
        The cholesky factorization of the matrix.
    """

    diag_blocksize = A_diagonal_blocks.shape[1]
    n_diag_blocks = A_diagonal_blocks.shape[0]

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
        L_diagonal_blocks_gpu[i, :, :] = cholesky(A_diagonal_blocks_gpu[i, :, :])

        # Temporary storage of used twice lower triangular solving
        L_inv_temp_gpu[:, :] = cpla.solve_triangular(
            L_diagonal_blocks_gpu[i, :, :],
            cp.eye(diag_blocksize),
            lower=True,
        ).T

        # L_{i+1, i} = A_{i+1, i} @ L_{i, i}^{-T}
        L_lower_diagonal_blocks_gpu[i, :, :] = (
            A_lower_diagonal_blocks_gpu[i, :, :] @ L_inv_temp_gpu
        )

        # L_{ndb+1, i} = A_{ndb+1, i} @ L_{i, i}^{-T}
        L_arrow_bottom_blocks_gpu[i, :, :] = (
            A_arrow_bottom_blocks_gpu[i, :, :] @ L_inv_temp_gpu
        )

        # Update next diagonal block
        # A_{i+1, i+1} = A_{i+1, i+1} - L_{i+1, i} @ L_{i+1, i}.T
        A_diagonal_blocks_gpu[i + 1, :, :] = (
            A_diagonal_blocks_gpu[i + 1, :, :]
            - L_lower_diagonal_blocks_gpu[i, :, :]
            @ L_lower_diagonal_blocks_gpu[i, :, :].T
        )

        # A_{ndb+1, i+1} = A_{ndb+1, i+1} - L_{ndb+1, i} @ L_{i+1, i}.T
        A_arrow_bottom_blocks_gpu[i + 1, :, :] = (
            A_arrow_bottom_blocks_gpu[i + 1, :, :]
            - L_arrow_bottom_blocks_gpu[i, :, :]
            @ L_lower_diagonal_blocks_gpu[i, :, :].T
        )

        # A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, i} @ L_{ndb+1, i}.T
        A_arrow_tip_block_gpu[:, :] = (
            A_arrow_tip_block_gpu[:, :]
            - L_arrow_bottom_blocks_gpu[i, :, :] @ L_arrow_bottom_blocks_gpu[i, :, :].T
        )

    # L_{ndb, ndb} = chol(A_{ndb, ndb})
    L_diagonal_blocks_gpu[-1, :, :] = cholesky(A_diagonal_blocks_gpu[-1, :, :])

    # L_{ndb+1, ndb} = A_{ndb+1, ndb} @ L_{ndb, ndb}^{-T}
    L_arrow_bottom_blocks_gpu[-1, :, :] = (
        A_arrow_bottom_blocks_gpu[-1, :, :]
        @ cpla.solve_triangular(
            L_diagonal_blocks_gpu[-1, :, :],
            cp.eye(diag_blocksize),
            lower=True,
        ).T
    )

    # A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, ndb} @ L_{ndb+1, ndb}^{T}
    A_arrow_tip_block_gpu[:, :] = (
        A_arrow_tip_block_gpu[:, :]
        - L_arrow_bottom_blocks_gpu[-1, :, :] @ L_arrow_bottom_blocks_gpu[-1, :, :].T
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
