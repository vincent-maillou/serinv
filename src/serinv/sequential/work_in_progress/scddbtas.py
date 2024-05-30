# Copyright 2023-2024 ETH Zurich. All rights reserved.

import numpy as np
import scipy.linalg as la


def scddbtas(
    L_diagonal_blocks: np.ndarray,
    L_lower_diagonal_blocks: np.ndarray,
    L_arrow_bottom_blocks: np.ndarray,
    U_diagonal_blocks: np.ndarray,
    U_upper_diagonal_blocks: np.ndarray,
    U_arrow_right_blocks: np.ndarray,
    B: np.ndarray,
) -> np.ndarray:
    """Solve a linear system given the LU factorization of a block tridiagonal arrowhead matrix using
    a sequential algotithm on a CPU backend.

    Parameters
    ----------
    L_diagonal_blocks : np.ndarray
        Diagonal blocks of the lower factor.
    L_lower_diagonal_blocks : np.ndarray
        Lower diagonal blocks of the lower factor.
    L_arrow_bottom_blocks : np.ndarray
        Bottom arrow blocks of the lower factor.
    U_diagonal_blocks : np.ndarray
        Diagonal blocks of the upper factor.
    U_upper_diagonal_blocks : np.ndarray
        Upper diagonal blocks of the upper factor.
    U_arrow_right_blocks : np.ndarray
        Right arrow blocks of the upper factor.
    B : np.ndarray
        The right hand side.

    Returns
    -------
    X : np.ndarray
        The solution of the system.
    """

    Y = np.zeros_like(B, dtype=B.dtype)
    X = np.zeros_like(B, dtype=B.dtype)

    # ----- Forward substitution -----
    diag_blocksize = L_diagonal_blocks.shape[0]
    arrow_blocksize = L_arrow_bottom_blocks.shape[0]
    n_rhs = B.shape[1]
    n_diag_blocks = L_diagonal_blocks.shape[1] // diag_blocksize

    Y[0:diag_blocksize, :] = la.solve_triangular(
        L_diagonal_blocks[:, 0:diag_blocksize],
        B[0:diag_blocksize, :],
        lower=True,
    )

    for i in range(1, n_diag_blocks):
        # Y_{i} = L_{i,i}^{-1} (B_{i} - L_{i,i-1} Y_{i-1})
        Y[i * diag_blocksize : (i + 1) * diag_blocksize, :] = la.solve_triangular(
            L_diagonal_blocks[:, i * diag_blocksize : (i + 1) * diag_blocksize],
            B[i * diag_blocksize : (i + 1) * diag_blocksize, :]
            - L_lower_diagonal_blocks[:, (i - 1) * diag_blocksize : i * diag_blocksize]
            @ Y[(i - 1) * diag_blocksize : i * diag_blocksize, :],
            lower=True,
        )

    # Accumulation of the arrowhead blocks
    # Y_{ndb+1} = L_{ndb+1,ndb+1}^{-1} (B_{ndb+1} - \Sigma_{i=1}^{ndb} L_{ndb+1,i} Y_{i))
    B_tip_rhs = (
        B[-arrow_blocksize:, :]
        - L_arrow_bottom_blocks[:, 0:-arrow_blocksize] @ Y[0:-arrow_blocksize, :]
    )

    Y[-arrow_blocksize:, :] = la.solve_triangular(
        L_arrow_bottom_blocks[:, -arrow_blocksize:],
        B_tip_rhs[:, :],
        lower=True,
    )

    # ----- Backward substitution -----
    # X_{ndb+1} = U_{ndb+1,ndb+1}^{-1} (Y_{ndb+1})
    X[-arrow_blocksize:, :] = la.solve_triangular(
        U_arrow_right_blocks[-arrow_blocksize:, :],
        Y[-arrow_blocksize:, :],
        lower=False,
    )

    # X_{ndb} = U_{ndb,ndb}^{-1} (Y_{ndb} - U_{ndb,ndb+1} X_{ndb+1})
    X[-arrow_blocksize - diag_blocksize : -arrow_blocksize, :] = la.solve_triangular(
        U_diagonal_blocks[:, -diag_blocksize:],
        Y[-arrow_blocksize - diag_blocksize : -arrow_blocksize, :]
        - U_arrow_right_blocks[-arrow_blocksize - diag_blocksize : -arrow_blocksize, :]
        @ X[-arrow_blocksize:, :],
        lower=False,
    )

    Y_temp = np.zeros((diag_blocksize, n_rhs), dtype=B.dtype)
    for i in range(n_diag_blocks - 2, -1, -1):
        # X_{i} = U_{i,i}^{-1} (Y_{i} - U_{i,i+1} X_{i+1}) - U_{i,ndb+1} X_{ndb+1}
        Y_temp = (
            Y[i * diag_blocksize : (i + 1) * diag_blocksize, :]
            - U_upper_diagonal_blocks[:, i * diag_blocksize : (i + 1) * diag_blocksize]
            @ X[(i + 1) * diag_blocksize : (i + 2) * diag_blocksize, :]
            - U_arrow_right_blocks[i * diag_blocksize : (i + 1) * diag_blocksize, :]
            @ X[-arrow_blocksize:, :]
        )
        X[i * diag_blocksize : (i + 1) * diag_blocksize] = la.solve_triangular(
            U_diagonal_blocks[:, i * diag_blocksize : (i + 1) * diag_blocksize],
            Y_temp,
            lower=False,
        )

    return X
