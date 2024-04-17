"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Lisa Gaedke-Merzhaeuser  (lisa.gaedke.merzhaeuser@usi.ch)
@date: 2023-11

Contains the cholesky solve routines.

Copyright 2023-2024 ETH Zurich and USI. All rights reserved.
"""


import numpy as np
import scipy.linalg as la


def cholesky_solve_block_tridiagonal(
    L_diagonal_blocks: np.ndarray,
    L_lower_diagonal_blocks: np.ndarray,
    B: np.ndarray,
) -> np.ndarray:
    """Solve a cholesky decomposed matrix with a block tridiagonal structure
    against the given right hand side.

    Parameters
    ----------
    L_diagonal_blocks : np.ndarray
    L_lower_diagonal_blocks : np.ndarray

    Returns
    -------
    X : np.ndarray
        The solution of the system.
    """

    # ----- Forward substitution -----
    blocksize = L_diagonal_blocks.shape[0]
    nblocks = L_diagonal_blocks.shape[1] // blocksize

    X = np.zeros_like(B)

    X[0:blocksize] = la.solve_triangular(
        L_diagonal_blocks[:, 0:blocksize], B[0:blocksize], lower=True
    )

    for i in range(1, nblocks, 1):
        # Y_{i} = L_{i,i}^{-1} (B_{i} - L_{i,i-1} Y_{i-1})
        X[i * blocksize : (i + 1) * blocksize] = la.solve_triangular(
            L_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize],
            B[i * blocksize : (i + 1) * blocksize]
            - L_lower_diagonal_blocks[:, (i - 1) * blocksize : i * blocksize]
            @ X[(i - 1) * blocksize : (i) * blocksize],
            lower=True,
        )

    # ----- Backward substitution -----
    X[-blocksize:] = la.solve_triangular(
        L_diagonal_blocks[:, -blocksize:], X[-blocksize:], lower=True, trans="T"
    )

    for i in range(nblocks - 2, -1, -1):
        # X_{i} = L_{i,i}^{-T} (Y_{i} - L_{i+1,i} X_{i+1})
        X[i * blocksize : (i + 1) * blocksize] = la.solve_triangular(
            L_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize],
            X[i * blocksize : (i + 1) * blocksize]
            - L_lower_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize].T
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
    overwrite: bool = False,
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
    overwrite : bool
        If True, the rhs B is overwritten with the solution X. Default is False.

    Returns
    -------
    X : np.ndarray
        The solution of the system.
    """

    # Y = np.zeros_like(B)
    # X = np.zeros_like(B)

    if overwrite:
        X = B
    else:
        X = np.copy(B)

    # ----- Forward substitution -----
    n_diag_blocks = (L.shape[0] - arrow_blocksize) // diag_blocksize
    X[0:diag_blocksize] = la.solve_triangular(
        L[0:diag_blocksize, 0:diag_blocksize], X[0:diag_blocksize], lower=True
    )
    for i in range(1, n_diag_blocks):
        # Y_{i} = L_{i,i}^{-1} (B_{i} - L_{i,i-1} Y_{i-1})
        X[i * diag_blocksize : (i + 1) * diag_blocksize] = la.solve_triangular(
            L[
                i * diag_blocksize : (i + 1) * diag_blocksize,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ],
            B[i * diag_blocksize : (i + 1) * diag_blocksize]
            - L[
                i * diag_blocksize : (i + 1) * diag_blocksize,
                (i - 1) * diag_blocksize : i * diag_blocksize,
            ]
            @ X[(i - 1) * diag_blocksize : (i) * diag_blocksize],
            lower=True,
        )

    # Accumulation of the arrowhead blocks
    B_temp = B[-arrow_blocksize:]
    for i in range(n_diag_blocks):
        B_temp[:] = (
            B_temp
            - L[-arrow_blocksize:, i * diag_blocksize : (i + 1) * diag_blocksize]
            @ X[i * diag_blocksize : (i + 1) * diag_blocksize]
        )

    # Y_{ndb+1} = L_{ndb+1,ndb+1}^{-1} (B_{ndb+1} - \Sigma_{i=1}^{ndb} L_{ndb+1,i} Y_{i)
    X[-arrow_blocksize:] = la.solve_triangular(
        L[-arrow_blocksize:, -arrow_blocksize:], B_temp, lower=True
    )

    # ----- Backward substitution -----
    # X_{ndb+1} = L_{ndb+1,ndb+1}^{-T} (Y_{ndb+1})
    X[-arrow_blocksize:] = la.solve_triangular(
        L[-arrow_blocksize:, -arrow_blocksize:],
        X[-arrow_blocksize:],
        lower=True,
        trans="T",
    )

    # X_{ndb} = L_{ndb,ndb}^{-T} (Y_{ndb} - L_{ndb+1,ndb}^{T} X_{ndb+1})
    X[-arrow_blocksize - diag_blocksize : -arrow_blocksize] = la.solve_triangular(
        L[
            -arrow_blocksize - diag_blocksize : -arrow_blocksize,
            -arrow_blocksize - diag_blocksize : -arrow_blocksize,
        ],
        X[-arrow_blocksize - diag_blocksize : -arrow_blocksize]
        - L[-arrow_blocksize:, -arrow_blocksize - diag_blocksize : -arrow_blocksize].T
        @ X[-arrow_blocksize:],
        lower=True,
        trans="T",
    )

    Y_temp = np.ndarray(shape=(diag_blocksize, B.shape[1]))
    for i in range(n_diag_blocks - 2, -1, -1):
        # X_{i} = L_{i,i}^{-T} (Y_{i} - L_{i+1,i}^{T} X_{i+1}) - L_{ndb+1,i}^T X_{ndb+1}
        Y_temp = (
            X[i * diag_blocksize : (i + 1) * diag_blocksize]
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

    return X

    return X


def chol_slv_ndiags(
    L: np.ndarray,
    B: np.ndarray,
    ndiags: int,
    blocksize: int,
    overwrite: bool = False,
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
    overwrite: bool
        If True, the rhs B is overwritten with the solution X. Default is False.


    Returns
    -------
    X : np.ndarray
        The solution of the system.
    """

    if overwrite:
        X = B
    else:
        X = np.copy(B)

    # temporary buffer for sum
    B_temp = np.zeros_like(B[:blocksize])

    n_offdiags_blk = ndiags // 2

    # ----- Forward substitution -----
    n_blocks = L.shape[0] // blocksize
    X[0:blocksize] = la.solve_triangular(
        L[0:blocksize, 0:blocksize], B[0:blocksize], lower=True
    )
    for i in range(1, n_blocks, 1):
        # Y_{i} = L_{i,i}^{-1} (B_{i} - sum_{k=max(0,i-offdiags)}^{i-1} L_{i,k} Y_{k})
        B_temp[:blocksize] = B[i * blocksize : (i + 1) * blocksize]
        for k in range(max(0, i - n_offdiags_blk), i, 1):
            B_temp = (
                B_temp
                - L[
                    i * blocksize : (i + 1) * blocksize,
                    k * blocksize : (k + 1) * blocksize,
                ]
                @ X[k * blocksize : (k + 1) * blocksize]
            )

        X[i * blocksize : (i + 1) * blocksize] = la.solve_triangular(
            L[i * blocksize : (i + 1) * blocksize, i * blocksize : (i + 1) * blocksize],
            B_temp,
            lower=True,
        )

    # ----- Backward substitution -----

    X[-blocksize:] = la.solve_triangular(
        L[-blocksize:, -blocksize:], X[-blocksize:], lower=True, trans="T"
    )
    for i in range(n_blocks - 2, -1, -1):
        # X_{i} = L_{i,i}^{-T} (Y_{i} - sum_{k=i+1}^{i+n_offdiags_blk} L^T_{k,i} X_{k})
        B_temp[:blocksize] = X[i * blocksize : (i + 1) * blocksize]
        for k in range(i + 1, min(i + 1 + n_offdiags_blk, n_blocks)):
            B_temp = (
                B_temp
                - L[
                    k * blocksize : (k + 1) * blocksize,
                    i * blocksize : (i + 1) * blocksize,
                ].T
                @ X[k * blocksize : (k + 1) * blocksize]
            )

        X[i * blocksize : (i + 1) * blocksize] = la.solve_triangular(
            L[i * blocksize : (i + 1) * blocksize, i * blocksize : (i + 1) * blocksize],
            B_temp,
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
    overwrite: bool = False,
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
    overwrite : bool
        If True, the rhs B is overwritten with the solution X. Default is False.

    Returns
    -------
    X : np.ndarray
        The solution of the system.
    """

    if overwrite:
        X = B
    else:
        X = np.copy(B)

    n_offdiags_blk = ndiags // 2

    # ----- Forward substitution -----
    n_diag_blocks = (L.shape[0] - arrow_blocksize) // diag_blocksize

    B_temp = np.zeros_like(B[:diag_blocksize])

    X[0:diag_blocksize] = la.solve_triangular(
        L[0:diag_blocksize, 0:diag_blocksize], X[0:diag_blocksize], lower=True
    )
    for i in range(1, n_diag_blocks, 1):
        # Y_{i} = L_{i,i}^{-1} (B_{i} - sum_{k=max(0,i-offdiags)}^{i-1} L_{i,k} Y_{k})
        B_temp[:] = X[i * diag_blocksize : (i + 1) * diag_blocksize]
        for k in range(max(0, i - n_offdiags_blk), i, 1):
            B_temp[:] = (
                B_temp
                - L[
                    i * diag_blocksize : (i + 1) * diag_blocksize,
                    k * diag_blocksize : (k + 1) * diag_blocksize,
                ]
                @ X[k * diag_blocksize : (k + 1) * diag_blocksize]
            )

        X[i * diag_blocksize : (i + 1) * diag_blocksize] = la.solve_triangular(
            L[
                i * diag_blocksize : (i + 1) * diag_blocksize,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ],
            B_temp,
            lower=True,
        )

    # Accumulation of the arrowhead blocks
    B_temp = X[-arrow_blocksize:]
    for i in range(n_diag_blocks):
        B_temp[:] = (
            B_temp
            - L[-arrow_blocksize:, i * diag_blocksize : (i + 1) * diag_blocksize]
            @ X[i * diag_blocksize : (i + 1) * diag_blocksize]
        )

    # Y_{ndb+1} = L_{ndb+1,ndb+1}^{-1} (B_{ndb+1} - \Sigma_{i=1}^{ndb} L_{ndb+1,i} Y_{i)
    X[-arrow_blocksize:] = la.solve_triangular(
        L[-arrow_blocksize:, -arrow_blocksize:], B_temp, lower=True
    )

    # ----- Backward substitution -----

    # X_{ndb+1} = L_{ndb+1,ndb+1}^{-T} (Y_{ndb+1})
    X[-arrow_blocksize:] = la.solve_triangular(
        L[-arrow_blocksize:, -arrow_blocksize:],
        X[-arrow_blocksize:],
        lower=True,
        trans="T",
    )

    # X_{ndb} = L_{ndb,ndb}^{-T} (Y_{ndb} - L_{ndb+1,ndb}^{T} X_{ndb+1})
    X[-arrow_blocksize - diag_blocksize : -arrow_blocksize] = la.solve_triangular(
        L[
            -arrow_blocksize - diag_blocksize : -arrow_blocksize,
            -arrow_blocksize - diag_blocksize : -arrow_blocksize,
        ],
        X[-arrow_blocksize - diag_blocksize : -arrow_blocksize]
        - L[-arrow_blocksize:, -arrow_blocksize - diag_blocksize : -arrow_blocksize].T
        @ X[-arrow_blocksize:],
        lower=True,
        trans="T",
    )

    Y_temp = np.ndarray(shape=(diag_blocksize, B.shape[1]))
    for i in range(n_diag_blocks - 2, -1, -1):
        # X_{i} = L_{i,i}^{-T} (Y_{i} - sum_k=i+1^i+n_offdiags_blk L^T_{k,i} X_{k} - L_{ndb+1,i}^T X_{ndb+1})
        # Y_tmp = Y_{i} - sum_k=i+1^i+n_offdiags_blk L^T_{k,i} X_{k}
        Y_temp[:] = X[i * diag_blocksize : (i + 1) * diag_blocksize]

        for k in range(i + 1, min(i + 1 + n_offdiags_blk, n_diag_blocks)):
            Y_temp[:] = (
                Y_temp
                - L[
                    k * diag_blocksize : (k + 1) * diag_blocksize,
                    i * diag_blocksize : (i + 1) * diag_blocksize,
                ].T
                @ X[k * diag_blocksize : (k + 1) * diag_blocksize]
            )

        # Y_tmp = Y_tmp - L_{ndb+1,i}^T X_{ndb+1}
        Y_temp[:] = (
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
