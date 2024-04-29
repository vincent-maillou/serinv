# Copyright 2023-2024 ETH Zurich and USI. All rights reserved.

import numpy as np
import scipy.linalg as la


def lu_solve_tridiag(
    L_diagonal_blocks: np.ndarray,
    L_lower_diagonal_blocks: np.ndarray,
    U_diagonal_blocks: np.ndarray,
    U_upper_diagonal_blocks: np.ndarray,
    B: np.ndarray,
) -> np.ndarray:
    """Solve a LU decomposed matrix with a block tridiagonal structure
    against the given right hand side.

    Parameters
    ----------
    L : np.ndarray
        The LU factorization of the matrix.
    U : np.ndarray
        The LU factorization of the matrix.
    B : np.ndarray
        The right hand side.
    blocksize : int
        The blocksize of the matrix.

    Returns
    -------
    X : np.ndarray
        The solution of the system.
    """

    Y = np.zeros_like(B, dtype=B.dtype)
    X = np.zeros_like(B, dtype=B.dtype)

    # ----- Forward substitution -----
    blocksize = L_diagonal_blocks.shape[0]
    n_blocks = L_diagonal_blocks.shape[1] // blocksize

    Y[0:blocksize] = la.solve_triangular(
        L_diagonal_blocks[:, 0:blocksize],
        B[0:blocksize],
        lower=True,
    )
    for i in range(1, n_blocks, 1):
        # Y_{i} = L_{i,i}^{-1} (B_{i} - L_{i,i-1} Y_{i-1})
        Y[i * blocksize : (i + 1) * blocksize] = la.solve_triangular(
            L_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize],
            B[i * blocksize : (i + 1) * blocksize, :]
            - L_lower_diagonal_blocks[
                :,
                (i - 1) * blocksize : i * blocksize,
            ]
            @ Y[(i - 1) * blocksize : (i) * blocksize, :],
            lower=True,
        )

    # ----- Backward substitution -----
    X[-blocksize:, :] = la.solve_triangular(
        U_diagonal_blocks[:, -blocksize:], Y[-blocksize:], lower=False
    )
    for i in range(n_blocks - 2, -1, -1):
        # X_{i} = U_{i,i}^{-1} (Y_{i} - U_{i,i+1} X_{i+1})
        X[i * blocksize : (i + 1) * blocksize, :] = la.solve_triangular(
            U_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize],
            Y[i * blocksize : (i + 1) * blocksize, :]
            - U_upper_diagonal_blocks[
                :,
                i * blocksize : (i + 1) * blocksize,
            ]
            @ X[(i + 1) * blocksize : (i + 2) * blocksize, :],
            lower=False,
        )

    return X


def lu_solve_tridiag_arrowhead(
    L_diagonal_blocks: np.ndarray,
    L_lower_diagonal_blocks: np.ndarray,
    L_arrow_bottom_blocks: np.ndarray,
    U_diagonal_blocks: np.ndarray,
    U_upper_diagonal_blocks: np.ndarray,
    U_arrow_right_blocks: np.ndarray,
    B: np.ndarray,
) -> np.ndarray:
    """Solve a lu decomposed matrix with a block tridiagonal arrowhead
    structure against the given right hand side.

    Parameters
    ----------
    L : np.ndarray
        The LU factorization of the matrix.
    U : np.ndarray
        The LU factorization of the matrix.
    B : np.ndarray
        The right hand side.
    diag_blocksize : int
        Blocksize of the diagonals blocks of the matrix.
    arrow_blocksize : int
        Blocksize of the blocks composing the arrowhead.

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
    B_tip_rhs = B[-arrow_blocksize:, :] - L_arrow_bottom_blocks[:, 0:-arrow_blocksize] @ Y[0:-arrow_blocksize, :]

    Y[-arrow_blocksize:, :] = la.solve_triangular(
        L_arrow_bottom_blocks[:, -arrow_blocksize:], 
        B_tip_rhs[:,:], 
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


def lu_slv_ndiags(
    L: np.ndarray,
    U: np.ndarray,
    B: np.ndarray,
    ndiags: int,
    blocksize: int,
) -> np.ndarray:
    """Solve a LU decomposed matrix with a block n-diagonals structure
    against the given right hand side.

    Parameters
    ----------
    L : np.ndarray
        The LU factorization of the matrix.
    U : np.ndarray
        The LU factorization of the matrix.
    B : np.ndarray
        The right hand side.
    ndiags : int
        Number of diagonals of the matrix.
    blocksize : int
        The blocksize of the matrix.

    Returns
    -------
    X : np.ndarray
        The solution of the system.
    """

    # number of lower (or upper) off-diagonal blocks
    n_offdiags = int((ndiags - 1) / 2)

    Y = np.zeros_like(B)
    X = np.zeros_like(B)

    # ----- Forward substitution -----
    n_blocks = L.shape[0] // blocksize
    Y[0:blocksize] = la.solve_triangular(
        L[0:blocksize, 0:blocksize], B[0:blocksize], lower=True
    )
    for i in range(1, n_blocks, 1):
        # Y_{i} = L_{i,i}^{-1} (B_{i} - sum_k=max(0,i-offdiags)^i-1 L_{i,k} Y_{k})
        B_temp = B[i * blocksize : (i + 1) * blocksize]
        for k in range(max(0, i - n_offdiags), i, 1):
            B_temp = (
                B_temp
                - L[
                    i * blocksize : (i + 1) * blocksize,
                    k * blocksize : (k + 1) * blocksize,
                ]
                @ Y[k * blocksize : (k + 1) * blocksize]
            )

        Y[i * blocksize : (i + 1) * blocksize] = la.solve_triangular(
            L[i * blocksize : (i + 1) * blocksize, i * blocksize : (i + 1) * blocksize],
            B_temp,
            lower=True,
        )

    # ----- Backward substitution -----
    X[-blocksize:] = la.solve_triangular(
        U[-blocksize:, -blocksize:], Y[-blocksize:], lower=False
    )
    for i in range(n_blocks - 2, -1, -1):
        # X_{i} = U_{i,i}^{-1} (Y_{i} - sum_k=i+1^i+n_offdiags U_{i,k} X_{k})
        Y_temp = Y[i * blocksize : (i + 1) * blocksize]
        for k in range(i + 1, min(i + 1 + n_offdiags, n_blocks)):
            Y_temp = (
                Y_temp
                - U[
                    i * blocksize : (i + 1) * blocksize,
                    k * blocksize : (k + 1) * blocksize,
                ]
                @ X[k * blocksize : (k + 1) * blocksize]
            )

        X[i * blocksize : (i + 1) * blocksize] = la.solve_triangular(
            U[i * blocksize : (i + 1) * blocksize, i * blocksize : (i + 1) * blocksize],
            Y_temp,
            lower=False,
        )

    return X


def lu_slv_ndiags_arrowhead(
    L: np.ndarray,
    U: np.ndarray,
    B: np.ndarray,
    ndiags: int,
    diag_blocksize: int,
    arrow_blocksize: int,
) -> np.ndarray:
    """Solve a LU decomposed matrix with a block n-diagonals arrowhead
    structure against the given right hand side. The matrix is assumed to be
    symmetric positive definite.

    Parameters
    ----------
    L : np.ndarray
        The LU factorization of the matrix.
    U : np.ndarray
        The LU factorization of the matrix.
    B : np.ndarray
        The right hand side.
    ndiags : int
        Number of diagonals of the matrix.
    diag_blocksize : int
        The blocksize of the matrix.
    arrow_blocksize : int
        Blocksize of the blocks composing the arrowhead.

    Returns
    -------
    X : np.ndarray
        The solution of the system.
    """

    # number of lower (or upper) off-diagonal blocks
    n_offdiags = int((ndiags - 1) / 2)

    Y = np.zeros_like(B)
    X = np.zeros_like(B)

    # ----- Forward substitution -----
    n_diag_blocks = (L.shape[0] - arrow_blocksize) // diag_blocksize

    Y[0:diag_blocksize] = la.solve_triangular(
        L[0:diag_blocksize, 0:diag_blocksize], B[0:diag_blocksize], lower=True
    )
    for i in range(1, n_diag_blocks, 1):
        # Y_{i} = L_{i,i}^{-1} (B_{i} - sum_k=max(0,i-offdiags)^i-1 L_{i,k} Y_{k})
        B_temp = B[i * diag_blocksize : (i + 1) * diag_blocksize]
        for k in range(max(0, i - n_offdiags), i, 1):
            B_temp = (
                B_temp
                - L[
                    i * diag_blocksize : (i + 1) * diag_blocksize,
                    k * diag_blocksize : (k + 1) * diag_blocksize,
                ]
                @ Y[k * diag_blocksize : (k + 1) * diag_blocksize]
            )

        Y[i * diag_blocksize : (i + 1) * diag_blocksize] = la.solve_triangular(
            L[
                i * diag_blocksize : (i + 1) * diag_blocksize,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ],
            B_temp,
            lower=True,
        )

    # Accumulation of the arrowhead blocks
    B_temp = B[-arrow_blocksize:]
    for i in range(n_diag_blocks):
        B_temp = (
            B_temp
            - L[-arrow_blocksize:, i * diag_blocksize : (i + 1) * diag_blocksize]
            @ Y[i * diag_blocksize : (i + 1) * diag_blocksize]
        )

    # Y_{ndb+1} = L_{ndb+1,ndb+1}^{-1} (B_{ndb+1} - \Sigma_{i=1}^{ndb} L_{ndb+1,i} Y_{i)
    Y[-arrow_blocksize:] = la.solve_triangular(
        L[-arrow_blocksize:, -arrow_blocksize:], B_temp, lower=True
    )

    # ----- Backward substitution -----
    # X_{ndb+1} = U_{ndb+1,ndb+1}^{-1} (Y_{ndb+1})
    X[-arrow_blocksize:] = la.solve_triangular(
        U[-arrow_blocksize:, -arrow_blocksize:], Y[-arrow_blocksize:], lower=False
    )

    # X_{ndb} = U_{ndb,ndb}^{-1} (Y_{ndb} - U_{ndb, ndb+1} X_{ndb+1})
    X[-arrow_blocksize - diag_blocksize : -arrow_blocksize] = la.solve_triangular(
        U[
            -arrow_blocksize - diag_blocksize : -arrow_blocksize,
            -arrow_blocksize - diag_blocksize : -arrow_blocksize,
        ],
        Y[-arrow_blocksize - diag_blocksize : -arrow_blocksize]
        - U[-arrow_blocksize - diag_blocksize : -arrow_blocksize, -arrow_blocksize:]
        @ X[-arrow_blocksize:],
        lower=False,
    )

    for i in range(n_diag_blocks - 2, -1, -1):
        # X_{i} = U_{i,i}^{-1} (Y_{i} - sum_k=i+1^i+n_offdiags U_{i,k} X_{k} - U_{i,ndb+1} X_{ndb+1})
        # Y_tmp = Y_{i} - sum_k=i+1^i+n_offdiags U_{i,k} X_{k}
        Y_temp = Y[i * diag_blocksize : (i + 1) * diag_blocksize]

        for k in range(i + 1, min(i + 1 + n_offdiags, n_diag_blocks)):
            Y_temp = (
                Y_temp
                - U[
                    i * diag_blocksize : (i + 1) * diag_blocksize,
                    k * diag_blocksize : (k + 1) * diag_blocksize,
                ]
                @ X[k * diag_blocksize : (k + 1) * diag_blocksize]
            )

        # Y_tmp = Y_tmp - U_{i,ndb+1} X_{ndb+1}
        Y_temp = (
            Y_temp
            - U[i * diag_blocksize : (i + 1) * diag_blocksize, -arrow_blocksize:]
            @ X[-arrow_blocksize:]
        )

        X[i * diag_blocksize : (i + 1) * diag_blocksize] = la.solve_triangular(
            U[
                i * diag_blocksize : (i + 1) * diag_blocksize,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ],
            Y_temp,
            lower=False,
        )

    return X
