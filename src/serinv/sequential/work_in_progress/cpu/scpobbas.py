# Copyright 2023-2024 ETH Zurich. All rights reserved.

import numpy as np
import scipy.linalg as la


def scpobbas(
    L: np.ndarray,
    B: np.ndarray,
    ndiags: int,
    diag_blocksize: int,
    arrow_blocksize: int,
    overwrite: bool = False,
) -> np.ndarray:
    """Solve a linear system using a cholesky factorization of a block bidiagonal matrix
    using a sequential algorithm on CPU backend.

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
