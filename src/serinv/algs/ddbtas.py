# Copyright 2023-2024 ETH Zurich. All rights reserved.

try:
    import cupy as cp
    import cupyx.scipy.linalg as cu_la

    CUPY_AVAIL = True

except:
    CUPY_AVAIL = False

import numpy as np
import scipy.linalg as np_la


def ddbtas(
    L_diagonal_blocks: np.ndarray | cp.ndarray,
    L_lower_diagonal_blocks: np.ndarray | cp.ndarray,
    L_arrow_bottom_blocks: np.ndarray | cp.ndarray,
    L_arrow_tip_block: np.ndarray | cp.ndarray,
    U_diagonal_blocks: np.ndarray | cp.ndarray,
    U_upper_diagonal_blocks: np.ndarray | cp.ndarray,
    U_arrow_right_blocks: np.ndarray | cp.ndarray,
    U_arrow_tip_block: np.ndarray | cp.ndarray,
    B: np.ndarray | cp.ndarray,
) -> np.ndarray | cp.ndarray:
    """Solve a block tridiagonal arrowhead linear system given its LU factorization
    using a sequential block algorithm.

    Note:
    -----
    - If a device array is given, the algorithm will run on the GPU.

    Parameters
    ----------
    L_diagonal_blocks : np.ndarray | cp.ndarray
        Diagonal blocks of the lower factor.
    L_lower_diagonal_blocks : np.ndarray | cp.ndarray
        Lower diagonal blocks of the lower factor.
    L_arrow_bottom_blocks : np.ndarray | cp.ndarray
        Bottom arrow blocks of the lower factor.
    L_arrow_tip_block : np.ndarray | cp.ndarray
        Tip arrow block of the lower factor.
    U_diagonal_blocks : np.ndarray | cp.ndarray
        Diagonal blocks of the upper factor.
    U_upper_diagonal_blocks : np.ndarray | cp.ndarray
        Upper diagonal blocks of the upper factor.
    U_arrow_right_blocks : np.ndarray | cp.ndarray
        Right arrow blocks of the upper factor.
    U_arrow_tip_block : np.ndarray | cp.ndarray
        Tip arrow block of the upper factor.
    B : np.ndarray | cp.ndarray
        The right hand side.

    Returns
    -------
    X : np.ndarray | cp.ndarray
        The solution of the system.
    """

    la = np_la
    if CUPY_AVAIL:
        xp = cp.get_array_module(L_diagonal_blocks)
        if xp == cp:
            la = cu_la
    else:
        xp = np

    X = xp.zeros_like(B)

    # ----- Forward substitution -----
    diag_blocksize = L_diagonal_blocks.shape[1]
    arrow_blocksize = L_arrow_bottom_blocks.shape[1]
    n_diag_blocks = L_diagonal_blocks.shape[0]
    n_rhs = B.shape[1]

    X[0:diag_blocksize, :] = la.solve_triangular(
        L_diagonal_blocks[0, :, :],
        B[0:diag_blocksize, :],
        lower=True,
    )

    for i in range(1, n_diag_blocks):
        # Y_{i} = L_{i,i}^{-1} (B_{i} - L_{i,i-1} Y_{i-1})
        X[i * diag_blocksize : (i + 1) * diag_blocksize, :] = la.solve_triangular(
            L_diagonal_blocks[i, :, :],
            B[i * diag_blocksize : (i + 1) * diag_blocksize, :]
            - L_lower_diagonal_blocks[i - 1, :, :]
            @ X[(i - 1) * diag_blocksize : i * diag_blocksize, :],
            lower=True,
        )

    # Accumulation of the arrowhead blocks
    B_tip_rhs = B[-arrow_blocksize:, :]
    for i in range(0, n_diag_blocks):
        B_tip_rhs -= (
            L_arrow_bottom_blocks[i, :, :]
            @ X[i * diag_blocksize : (i + 1) * diag_blocksize, :]
        )

    X[-arrow_blocksize:, :] = la.solve_triangular(
        L_arrow_tip_block[:, :],
        B_tip_rhs[:, :],
        lower=True,
    )

    # ----- Backward substitution -----
    # X_{ndb+1} = U_{ndb+1,ndb+1}^{-1} (Y_{ndb+1})
    X[-arrow_blocksize:, :] = la.solve_triangular(
        U_arrow_tip_block[:, :],
        X[-arrow_blocksize:, :],
        lower=False,
    )

    # X_{ndb} = U_{ndb,ndb}^{-1} (Y_{ndb} - U_{ndb,ndb+1} X_{ndb+1})
    X[-arrow_blocksize - diag_blocksize : -arrow_blocksize, :] = la.solve_triangular(
        U_diagonal_blocks[-1, :, :],
        X[-arrow_blocksize - diag_blocksize : -arrow_blocksize, :]
        - U_arrow_right_blocks[-1, :, :] @ X[-arrow_blocksize:, :],
        lower=False,
    )

    for i in range(n_diag_blocks - 2, -1, -1):
        # X_{i} = U_{i,i}^{-1} (Y_{i} - U_{i,i+1} X_{i+1}) - U_{i,ndb+1} X_{ndb+1}
        X[i * diag_blocksize : (i + 1) * diag_blocksize] = la.solve_triangular(
            U_diagonal_blocks[i, :, :],
            X[i * diag_blocksize : (i + 1) * diag_blocksize, :]
            - U_upper_diagonal_blocks[i, :, :]
            @ X[(i + 1) * diag_blocksize : (i + 2) * diag_blocksize, :]
            - U_arrow_right_blocks[i, :, :] @ X[-arrow_blocksize:, :],
            lower=False,
        )

    return X
