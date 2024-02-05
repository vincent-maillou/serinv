"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Lisa Gaedke-Merzhaeuser  (lisa.gaedke.merzhaeuser@usi.ch)
@date: 2023-11

Contains the cholesky solve routines.

Copyright 2023 ETH Zurich and USI. All rights reserved.
"""


import numpy as np
import scipy.linalg as la


def chol_slv_tridiag(
    L: np.ndarray,
    B: np.ndarray,
    blocksize: int,
) -> np.ndarray:
    """Solve a cholesky decomposed matrix with a block tridiagonal structure
    against the given right hand side.

    Parameters
    ----------
    L : np.ndarray
        The cholesky factorization of the matrix.
    B : np.ndarray
        The right hand side.
    blocksize : int
        The blocksize of the matrix.

    Returns
    -------
    X : np.ndarray
        The solution of the system.
    """

    Y = np.zeros_like(B)
    X = np.zeros_like(B)

    # ----- Forward substitution -----
    n_blocks = L.shape[0] // blocksize
    Y[0:blocksize] = la.solve_triangular(
        L[0:blocksize, 0:blocksize], B[0:blocksize], lower=True
    )
    for i in range(1, n_blocks, 1):
        # Y_{i} = L_{i,i}^{-1} (B_{i} - L_{i,i-1} Y_{i-1})
        Y[i * blocksize : (i + 1) * blocksize] = la.solve_triangular(
            L[i * blocksize : (i + 1) * blocksize, i * blocksize : (i + 1) * blocksize],
            B[i * blocksize : (i + 1) * blocksize]
            - L[
                i * blocksize : (i + 1) * blocksize, (i - 1) * blocksize : i * blocksize
            ]
            @ Y[(i - 1) * blocksize : (i) * blocksize],
            lower=True,
        )

    # ----- Backward substitution -----
    X[-blocksize:] = la.solve_triangular(
        L[-blocksize:, -blocksize:], Y[-blocksize:], lower=True, trans="T"
    )
    for i in range(n_blocks - 2, -1, -1):
        # X_{i} = L_{i,i}^{-T} (Y_{i} - L_{i+1,i} X_{i+1})
        X[i * blocksize : (i + 1) * blocksize] = la.solve_triangular(
            L[i * blocksize : (i + 1) * blocksize, i * blocksize : (i + 1) * blocksize],
            Y[i * blocksize : (i + 1) * blocksize]
            - L[
                (i + 1) * blocksize : (i + 2) * blocksize,
                i * blocksize : (i + 1) * blocksize,
            ].T
            @ X[(i + 1) * blocksize : (i + 2) * blocksize],
            lower=True,
            trans="T",
        )

    return X


def chol_slv_tridiag_arrowhead(
    L: np.ndarray,
    B: np.ndarray,
    diag_blocksize: int,
    arrow_blocksize: int,
) -> np.ndarray:
    """Solve a cholesky decomposed matrix with a block tridiagonal arrowhead
    structure against the given right hand side.

    Parameters
    ----------
    L : np.ndarray
        The cholesky factorization of the matrix.
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

    Y = np.zeros_like(B)
    X = np.zeros_like(B)

    # ----- Forward substitution -----
    n_diag_blocks = (L.shape[0] - arrow_blocksize) // diag_blocksize
    Y[0:diag_blocksize] = la.solve_triangular(
        L[0:diag_blocksize, 0:diag_blocksize], B[0:diag_blocksize], lower=True
    )
    for i in range(0, n_diag_blocks):
        # Y_{i} = L_{i,i}^{-1} (B_{i} - L_{i,i-1} Y_{i-1})
        Y[i * diag_blocksize : (i + 1) * diag_blocksize] = la.solve_triangular(
            L[
                i * diag_blocksize : (i + 1) * diag_blocksize,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ],
            B[i * diag_blocksize : (i + 1) * diag_blocksize]
            - L[
                i * diag_blocksize : (i + 1) * diag_blocksize,
                (i - 1) * diag_blocksize : i * diag_blocksize,
            ]
            @ Y[(i - 1) * diag_blocksize : (i) * diag_blocksize],
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
    # X_{ndb+1} = L_{ndb+1,ndb+1}^{-T} (Y_{ndb+1})
    X[-arrow_blocksize:] = la.solve_triangular(
        L[-arrow_blocksize:, -arrow_blocksize:],
        Y[-arrow_blocksize:],
        lower=True,
        trans="T",
    )

    # X_{ndb} = L_{ndb,ndb}^{-T} (Y_{ndb} - L_{ndb+1,ndb}^{T} X_{ndb+1})
    X[-arrow_blocksize - diag_blocksize : -arrow_blocksize] = la.solve_triangular(
        L[
            -arrow_blocksize - diag_blocksize : -arrow_blocksize,
            -arrow_blocksize - diag_blocksize : -arrow_blocksize,
        ],
        Y[-arrow_blocksize - diag_blocksize : -arrow_blocksize]
        - L[-arrow_blocksize:, -arrow_blocksize - diag_blocksize : -arrow_blocksize].T
        @ X[-arrow_blocksize:],
        lower=True,
        trans="T",
    )

    Y_temp = np.ndarray(shape=(diag_blocksize, B.shape[1]))
    for i in range(n_diag_blocks - 2, -1, -1):
        # X_{i} = L_{i,i}^{-T} (Y_{i} - L_{i+1,i}^{T} X_{i+1}) - L_{ndb+1,i}^T X_{ndb+1}
        Y_temp = (
            Y[i * diag_blocksize : (i + 1) * diag_blocksize]
            - L[
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ].T
            @ X[(i + 1) * diag_blocksize : (i + 2) * diag_blocksize]
            - L[-arrow_blocksize:, i * diag_blocksize : (i + 1) * diag_blocksize].T
            @ X[-arrow_blocksize:]
        )
        X[i * diag_blocksize : (i + 1) * diag_blocksize] = la.solve_triangular(
            L[
                i * diag_blocksize : (i + 1) * diag_blocksize,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ],
            Y_temp,
            lower=True,
            trans="T",
        )

    return X


def chol_slv_ndiags(
    L: np.ndarray,
    B: np.ndarray,
    ndiags: int,
    blocksize: int,
) -> np.ndarray:
    """Solve a cholesky decomposed matrix with a block n-diagonals structure
    against the given right hand side. The matrix is assumed to be symmetric
    positive definite.

    Parameters
    ----------
    L : np.ndarray
        The cholesky factorization of the matrix.
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

    Y = np.zeros_like(B)
    X = np.zeros_like(B)

    n_offdiags_blk = ndiags // 2

    # ----- Forward substitution -----
    n_blocks = L.shape[0] // blocksize
    Y[0:blocksize] = la.solve_triangular(
        L[0:blocksize, 0:blocksize], B[0:blocksize], lower=True
    )
    for i in range(1, n_blocks, 1):
        # Y_{i} = L_{i,i}^{-1} (B_{i} - sum_{k=max(0,i-offdiags)}^{i-1} L_{i,k} Y_{k})
        B_temp = B[i * blocksize : (i + 1) * blocksize]
        for k in range(max(0, i - n_offdiags_blk), i, 1):
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
        L[-blocksize:, -blocksize:], Y[-blocksize:], lower=True, trans="T"
    )
    for i in range(n_blocks - 2, -1, -1):
        # X_{i} = L_{i,i}^{-T} (Y_{i} - sum_{k=i+1}^{i+n_offdiags_blk} L^T_{k,i} X_{k})
        Y_temp = Y[i * blocksize : (i + 1) * blocksize]
        for k in range(i + 1, min(i + 1 + n_offdiags_blk, n_blocks)):
            Y_temp = (
                Y_temp
                - L[
                    k * blocksize : (k + 1) * blocksize,
                    i * blocksize : (i + 1) * blocksize,
                ].T
                @ X[k * blocksize : (k + 1) * blocksize]
            )

        X[i * blocksize : (i + 1) * blocksize] = la.solve_triangular(
            L[i * blocksize : (i + 1) * blocksize, i * blocksize : (i + 1) * blocksize],
            Y_temp,
            lower=True,
            trans="T",
        )

    return X


def chol_slv_ndiags_arrowhead(
    L: np.ndarray,
    B: np.ndarray,
    ndiags: int,
    diag_blocksize: int,
    arrow_blocksize: int,
) -> np.ndarray:
    """Solve a cholesky decomposed matrix with a block n-diagonals arrowhead
    structure against the given right hand side. The matrix is assumed to be
    symmetric positive definite.

    Parameters
    ----------
    L : np.ndarray
        The cholesky factorization of the matrix.
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

    Y = np.zeros_like(B)
    X = np.zeros_like(B)

    n_offdiags_blk = ndiags // 2

    # ----- Forward substitution -----
    n_diag_blocks = (L.shape[0] - arrow_blocksize) // diag_blocksize

    Y[0:diag_blocksize] = la.solve_triangular(
        L[0:diag_blocksize, 0:diag_blocksize], B[0:diag_blocksize], lower=True
    )
    for i in range(1, n_diag_blocks, 1):
        # Y_{i} = L_{i,i}^{-1} (B_{i} - sum_{k=max(0,i-offdiags)}^{i-1} L_{i,k} Y_{k})
        B_temp = B[i * diag_blocksize : (i + 1) * diag_blocksize]
        for k in range(max(0, i - n_offdiags_blk), i, 1):
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

    # X_{ndb+1} = L_{ndb+1,ndb+1}^{-T} (Y_{ndb+1})
    X[-arrow_blocksize:] = la.solve_triangular(
        L[-arrow_blocksize:, -arrow_blocksize:],
        Y[-arrow_blocksize:],
        lower=True,
        trans="T",
    )

    # X_{ndb} = L_{ndb,ndb}^{-T} (Y_{ndb} - L_{ndb+1,ndb}^{T} X_{ndb+1})
    X[-arrow_blocksize - diag_blocksize : -arrow_blocksize] = la.solve_triangular(
        L[
            -arrow_blocksize - diag_blocksize : -arrow_blocksize,
            -arrow_blocksize - diag_blocksize : -arrow_blocksize,
        ],
        Y[-arrow_blocksize - diag_blocksize : -arrow_blocksize]
        - L[-arrow_blocksize:, -arrow_blocksize - diag_blocksize : -arrow_blocksize].T
        @ X[-arrow_blocksize:],
        lower=True,
        trans="T",
    )

    for i in range(n_diag_blocks - 2, -1, -1):
        # X_{i} = L_{i,i}^{-T} (Y_{i} - sum_k=i+1^i+n_offdiags_blk L^T_{k,i} X_{k} - L_{ndb+1,i}^T X_{ndb+1})
        # Y_tmp = Y_{i} - sum_k=i+1^i+n_offdiags_blk L^T_{k,i} X_{k}
        Y_temp = Y[i * diag_blocksize : (i + 1) * diag_blocksize]

        for k in range(i + 1, min(i + 1 + n_offdiags_blk, n_diag_blocks)):
            Y_temp = (
                Y_temp
                - L[
                    k * diag_blocksize : (k + 1) * diag_blocksize,
                    i * diag_blocksize : (i + 1) * diag_blocksize,
                ].T
                @ X[k * diag_blocksize : (k + 1) * diag_blocksize]
            )

        # Y_tmp = Y_tmp - L_{ndb+1,i}^T X_{ndb+1}
        Y_temp = (
            Y_temp
            - L[-arrow_blocksize:, i * diag_blocksize : (i + 1) * diag_blocksize].T
            @ X[-arrow_blocksize:]
        )

        X[i * diag_blocksize : (i + 1) * diag_blocksize] = la.solve_triangular(
            L[
                i * diag_blocksize : (i + 1) * diag_blocksize,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ],
            Y_temp,
            lower=True,
            trans="T",
        )

    return X
