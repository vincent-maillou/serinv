# Copyright 2023-2024 ETH Zurich. All rights reserved.

import numpy as np
import scipy.linalg as la


def scpobtas(
    L_diagonal_blocks: np.ndarray,
    L_lower_diagonal_blocks: np.ndarray,
    L_arrow_bottom_blocks: np.ndarray,
    L_arrow_tip_block: np.ndarray,
    B: np.ndarray,
) -> np.ndarray:
    """Solve a linear system using a cholesky factorization of a block tridiagonal matrix
    using a sequential algorithm on CPU backend.

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
    B : np.ndarray
        Right-hand side of the linear system.

    Returns
    -------
    X : np.ndarray
        The solution of the system.
    """
    diag_blocksize = L_diagonal_blocks.shape[0]
    arrow_blocksize = L_arrow_bottom_blocks.shape[0]
    n_diag_blocks = L_diagonal_blocks.shape[1] // diag_blocksize

    X = np.zeros_like(B, dtype=B.dtype)

    # ----- Forward substitution -----
    X[0:diag_blocksize] = la.solve_triangular(
        L_diagonal_blocks[:, 0:diag_blocksize],
        B[0:diag_blocksize],
        lower=True,
    )

    for i in range(1, n_diag_blocks):
        # Y_{i} = L_{i,i}^{-1} (B_{i} - L_{i,i-1} Y_{i-1})
        X[i * diag_blocksize : (i + 1) * diag_blocksize] = la.solve_triangular(
            L_diagonal_blocks[:, i * diag_blocksize : (i + 1) * diag_blocksize],
            B[i * diag_blocksize : (i + 1) * diag_blocksize]
            - L_lower_diagonal_blocks[:, (i - 1) * diag_blocksize : i * diag_blocksize]
            @ X[(i - 1) * diag_blocksize : (i) * diag_blocksize],
            lower=True,
        )

    # Accumulation of the arrowhead blocks
    B_tip_rhs = (
        B[-arrow_blocksize:, :] - L_arrow_bottom_blocks[:, :] @ X[0:-arrow_blocksize, :]
    )

    # Y_{ndb+1} = L_{ndb+1,ndb+1}^{-1} (B_{ndb+1} - \Sigma_{i=1}^{ndb} L_{ndb+1,i} Y_{i)
    X[-arrow_blocksize:] = la.solve_triangular(
        L_arrow_tip_block[:, :], B_tip_rhs[:], lower=True
    )

    # ----- Backward substitution -----
    # X_{ndb+1} = L_{ndb+1,ndb+1}^{-T} (Y_{ndb+1})
    X[-arrow_blocksize:] = la.solve_triangular(
        L_arrow_tip_block[:, :],
        X[-arrow_blocksize:],
        lower=True,
        trans="T",
    )

    # X_{ndb} = L_{ndb,ndb}^{-T} (Y_{ndb} - L_{ndb+1,ndb}^{T} X_{ndb+1})
    X[-arrow_blocksize - diag_blocksize : -arrow_blocksize] = la.solve_triangular(
        L_diagonal_blocks[:, -diag_blocksize:],
        X[-arrow_blocksize - diag_blocksize : -arrow_blocksize]
        - L_arrow_bottom_blocks[:, -diag_blocksize:].T @ X[-arrow_blocksize:],
        lower=True,
        trans="T",
    )

    for i in range(n_diag_blocks - 2, -1, -1):
        # X_{i} = L_{i,i}^{-T} (Y_{i} - L_{i+1,i}^{T} X_{i+1}) - L_{ndb+1,i}^T X_{ndb+1}
        X[i * diag_blocksize : (i + 1) * diag_blocksize] = la.solve_triangular(
            L_diagonal_blocks[:, i * diag_blocksize : (i + 1) * diag_blocksize],
            X[i * diag_blocksize : (i + 1) * diag_blocksize]
            - L_lower_diagonal_blocks[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ].T
            @ X[(i + 1) * diag_blocksize : (i + 2) * diag_blocksize]
            - L_arrow_bottom_blocks[:, i * diag_blocksize : (i + 1) * diag_blocksize].T
            @ X[-arrow_blocksize:],
            lower=True,
            trans="T",
        )

    return X
