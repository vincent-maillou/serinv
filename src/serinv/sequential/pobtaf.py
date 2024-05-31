# Copyright 2023-2024 ETH Zurich. All rights reserved.

try:
    import cupy as cp
    import cupyx.scipy.linalg as cu_la

    CUPY_AVAIL = True

except:
    CUPY_AVAIL = False

import numpy as np
import scipy.linalg as np_la


def pobtaf(
    A_diagonal_blocks: np.ndarray | cp.ndarray,
    A_lower_diagonal_blocks: np.ndarray | cp.ndarray,
    A_arrow_bottom_blocks: np.ndarray | cp.ndarray,
    A_arrow_tip_block: np.ndarray | cp.ndarray,
) -> tuple[
    np.ndarray | cp.ndarray,
    np.ndarray | cp.ndarray,
    np.ndarray | cp.ndarray,
    np.ndarray | cp.ndarray,
]:
    """Perform the Cholesky factorization of a block tridiagonal arrowhead matrix using
    a sequential block algorithm.

    Note:
    -----
    - If a device array is given, the algorithm will run on the GPU.

    Parameters
    ----------
    A_diagonal_blocks : np.ndarray | cp.ndarray
        Diagonal blocks of the matrix.
    A_lower_diagonal_blocks : np.ndarray | cp.ndarray
        Lower diagonal blocks of the matrix.
    A_arrow_bottom_blocks : np.ndarray | cp.ndarray
        Arrow bottom blocks of the matrix.
    A_arrow_tip_block : np.ndarray | cp.ndarray
        Arrow tip block of the matrix.

    Returns
    -------
    L_diagonal_blocks : np.ndarray | cp.ndarray
    L_lower_diagonal_blocks : np.ndarray | cp.ndarray
    L_arrow_bottom_blocks : np.ndarray | cp.ndarray
    L_arrow_tip_block : np.ndarray | cp.ndarray
    """

    la = np_la
    if CUPY_AVAIL:
        xp = cp.get_array_module(A_diagonal_blocks)
        if xp == cp:
            la = cu_la
    else:
        xp = np

    diag_blocksize = A_diagonal_blocks.shape[1]
    n_diag_blocks = A_diagonal_blocks.shape[0]

    L_diagonal_blocks = xp.zeros_like(A_diagonal_blocks)
    L_lower_diagonal_blocks = xp.zeros_like(A_lower_diagonal_blocks)
    L_arrow_bottom_blocks = xp.zeros_like(A_arrow_bottom_blocks)
    L_arrow_tip_block = xp.zeros_like(A_arrow_tip_block)

    L_inv_temp = xp.zeros(
        (diag_blocksize, diag_blocksize), dtype=A_diagonal_blocks.dtype
    )

    for i in range(0, n_diag_blocks - 1):
        # L_{i, i} = chol(A_{i, i})
        L_diagonal_blocks[i, :, :] = xp.linalg.cholesky(A_diagonal_blocks[i, :, :])

        # Temporary storage of used twice lower triangular solving
        L_inv_temp[:, :] = (
            la.solve_triangular(
                L_diagonal_blocks[i, :, :],
                xp.eye(diag_blocksize),
                lower=True,
            )
            .conj()
            .T
        )

        # L_{i+1, i} = A_{i+1, i} @ L_{i, i}^{-T}
        L_lower_diagonal_blocks[i, :, :] = A_lower_diagonal_blocks[i, :, :] @ L_inv_temp

        # L_{ndb+1, i} = A_{ndb+1, i} @ L_{i, i}^{-T}
        L_arrow_bottom_blocks[i, :, :] = A_arrow_bottom_blocks[i, :, :] @ L_inv_temp

        # Update next diagonal block
        # A_{i+1, i+1} = A_{i+1, i+1} - L_{i+1, i} @ L_{i+1, i}.conj().T
        A_diagonal_blocks[i + 1, :, :] = (
            A_diagonal_blocks[i + 1, :, :]
            - L_lower_diagonal_blocks[i, :, :]
            @ L_lower_diagonal_blocks[i, :, :].conj().T
        )

        # A_{ndb+1, i+1} = A_{ndb+1, i+1} - L_{ndb+1, i} @ L_{i+1, i}.conj().T
        A_arrow_bottom_blocks[i + 1, :, :] = (
            A_arrow_bottom_blocks[i + 1, :, :]
            - L_arrow_bottom_blocks[i, :, :] @ L_lower_diagonal_blocks[i, :, :].conj().T
        )

        # A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, i} @ L_{ndb+1, i}.conj().T
        A_arrow_tip_block[:, :] = (
            A_arrow_tip_block[:, :]
            - L_arrow_bottom_blocks[i, :, :] @ L_arrow_bottom_blocks[i, :, :].conj().T
        )

    # L_{ndb, ndb} = chol(A_{ndb, ndb})
    L_diagonal_blocks[-1, :, :] = xp.linalg.cholesky(A_diagonal_blocks[-1, :, :])

    # L_{ndb+1, ndb} = A_{ndb+1, ndb} @ L_{ndb, ndb}^{-T}
    L_arrow_bottom_blocks[-1, :, :] = (
        A_arrow_bottom_blocks[-1, :, :]
        @ la.solve_triangular(
            L_diagonal_blocks[-1, :, :],
            xp.eye(diag_blocksize),
            lower=True,
        )
        .conj()
        .T
    )

    # A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, ndb} @ L_{ndb+1, ndb}^{T}
    A_arrow_tip_block[:, :] = (
        A_arrow_tip_block[:, :]
        - L_arrow_bottom_blocks[-1, :, :] @ L_arrow_bottom_blocks[-1, :, :].conj().T
    )

    # L_{ndb+1, ndb+1} = chol(A_{ndb+1, ndb+1})
    L_arrow_tip_block[:, :] = xp.linalg.cholesky(A_arrow_tip_block[:, :])

    return (
        L_diagonal_blocks,
        L_lower_diagonal_blocks,
        L_arrow_bottom_blocks,
        L_arrow_tip_block,
    )
