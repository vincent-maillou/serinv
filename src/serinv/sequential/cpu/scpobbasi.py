# Copyright 2023-2024 ETH Zurich and USI. All rights reserved.

import numpy as np
import scipy.linalg as la


def scpobbasi(
    L: np.ndarray,
    ndiags: int,
    diag_blocksize: int,
    arrow_blocksize: int,
) -> np.ndarray:
    """Perform a selected inversion of a block bidiagonal matrix using a
    sequential algorithm on CPU backend.

    Parameters
    ----------
    L : np.ndarray
        The cholesky factorization of the matrix.
    ndiags : int
        Number of diagonals.
    diag_blocksize : int
        Blocksize of the diagonals blocks of the matrix.
    arrow_blocksize : int
        Blocksize of the blocks composing the arrowhead.

    Returns
    -------
    X : np.ndarray
        Selected inversion of the matrix.
    """

    X = np.zeros(L.shape, dtype=L.dtype)
    L_last_blk_inv = np.zeros((arrow_blocksize, arrow_blocksize), dtype=L.dtype)

    L_last_blk_inv = la.solve_triangular(
        L[-arrow_blocksize:, -arrow_blocksize:], np.eye(arrow_blocksize), lower=True
    )
    X[-arrow_blocksize:, -arrow_blocksize:] = L_last_blk_inv.T @ L_last_blk_inv

    L_blk_inv = np.zeros((diag_blocksize, diag_blocksize), dtype=L.dtype)
    n_diag_blocks = (L.shape[0] - arrow_blocksize) // diag_blocksize
    n_offdiags_blk = ndiags // 2
    for i in range(n_diag_blocks - 1, -1, -1):
        # L_blk_inv = L_{i, i}^{-1}
        L_blk_inv = la.solve_triangular(
            L[
                i * diag_blocksize : (i + 1) * diag_blocksize,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ],
            np.eye(diag_blocksize),
            lower=True,
        )

        # Arrowhead part
        # X_{ndb+1, i} = - X_{ndb+1, ndb+1} L_{ndb+1, i}
        X[-arrow_blocksize:, i * diag_blocksize : (i + 1) * diag_blocksize] = (
            -X[-arrow_blocksize:, -arrow_blocksize:]
            @ L[-arrow_blocksize:, i * diag_blocksize : (i + 1) * diag_blocksize]
        )

        for k in range(i + 1, min(i + n_offdiags_blk + 1, n_diag_blocks), 1):
            # X_{ndb+1, i} = X_{ndb+1, i} - X_{ndb+1, k} L_{k, i}
            X[-arrow_blocksize:, i * diag_blocksize : (i + 1) * diag_blocksize] -= (
                X[-arrow_blocksize:, k * diag_blocksize : (k + 1) * diag_blocksize]
                @ L[
                    k * diag_blocksize : (k + 1) * diag_blocksize,
                    i * diag_blocksize : (i + 1) * diag_blocksize,
                ]
            )

        # X_{ndb+1, i} = X_{ndb+1, i} L_{i, i}^{-1}
        X[-arrow_blocksize:, i * diag_blocksize : (i + 1) * diag_blocksize] = (
            X[-arrow_blocksize:, i * diag_blocksize : (i + 1) * diag_blocksize]
            @ L_blk_inv
        )

        # X_{i, ndb+1} = X_{ndb+1, i}.T
        X[i * diag_blocksize : (i + 1) * diag_blocksize, -arrow_blocksize:] = X[
            -arrow_blocksize:, i * diag_blocksize : (i + 1) * diag_blocksize
        ].T

        # Off-diagonal block part
        for j in range(min(i + n_offdiags_blk, n_diag_blocks - 1), i, -1):
            # Take the effect of the arrowhead part into account
            # X_{j, i} = - X_{ndb+1, j}.T L_{ndb+1, i}
            X[
                j * diag_blocksize : (j + 1) * diag_blocksize,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ] = (
                -X[-arrow_blocksize:, j * diag_blocksize : (j + 1) * diag_blocksize].T
                @ L[-arrow_blocksize:, i * diag_blocksize : (i + 1) * diag_blocksize]
            )

            for k in range(i + 1, min(i + n_offdiags_blk + 1, n_diag_blocks), 1):
                # The following condition ensure to use the lower elements
                # produced duiring the selected inversion process. ie. the matrix
                # is symmetric.
                if k > j:
                    # X_{j, i} = X_{j, i} - X_{k, j}.T L_{k, i}
                    X[
                        j * diag_blocksize : (j + 1) * diag_blocksize,
                        i * diag_blocksize : (i + 1) * diag_blocksize,
                    ] -= (
                        X[
                            k * diag_blocksize : (k + 1) * diag_blocksize,
                            j * diag_blocksize : (j + 1) * diag_blocksize,
                        ].T
                        @ L[
                            k * diag_blocksize : (k + 1) * diag_blocksize,
                            i * diag_blocksize : (i + 1) * diag_blocksize,
                        ]
                    )
                else:
                    # X_{j, i} = X_{j, i} - X_{j, k} L_{k, i}
                    X[
                        j * diag_blocksize : (j + 1) * diag_blocksize,
                        i * diag_blocksize : (i + 1) * diag_blocksize,
                    ] -= (
                        X[
                            j * diag_blocksize : (j + 1) * diag_blocksize,
                            k * diag_blocksize : (k + 1) * diag_blocksize,
                        ]
                        @ L[
                            k * diag_blocksize : (k + 1) * diag_blocksize,
                            i * diag_blocksize : (i + 1) * diag_blocksize,
                        ]
                    )

            # X_{j, i} = X_{j, i} L_{i, i}^{-1}
            X[
                j * diag_blocksize : (j + 1) * diag_blocksize,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ] = (
                X[
                    j * diag_blocksize : (j + 1) * diag_blocksize,
                    i * diag_blocksize : (i + 1) * diag_blocksize,
                ]
                @ L_blk_inv
            )

            # X_{i, j} = X_{j, i}.T
            X[
                i * diag_blocksize : (i + 1) * diag_blocksize,
                j * diag_blocksize : (j + 1) * diag_blocksize,
            ] = X[
                j * diag_blocksize : (j + 1) * diag_blocksize,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ].T

        # Diagonal block part
        # X_{i, i} = (L_{i, i}^{-T} - X_{ndb+1, i}.T L_{ndb+1, i} - sum_{k=i+1}^{min(i+ndiags/2, n_diag_blocks)} X_{k, i}.T L_{k, i}) L_{i, i}^{-1}

        # X_{i, i} = L_{i, i}^{-T} - X_{ndb+1, i}.T L_{ndb+1, i}
        X[
            i * diag_blocksize : (i + 1) * diag_blocksize,
            i * diag_blocksize : (i + 1) * diag_blocksize,
        ] = (
            L_blk_inv.T
            - X[-arrow_blocksize:, i * diag_blocksize : (i + 1) * diag_blocksize].T
            @ L[-arrow_blocksize:, i * diag_blocksize : (i + 1) * diag_blocksize]
        )

        for k in range(i + 1, min(i + n_offdiags_blk + 1, n_diag_blocks), 1):
            # X_{i, i} = X_{i, i} - X_{k, i}.T L_{k, i}
            X[
                i * diag_blocksize : (i + 1) * diag_blocksize,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ] -= (
                X[
                    k * diag_blocksize : (k + 1) * diag_blocksize,
                    i * diag_blocksize : (i + 1) * diag_blocksize,
                ].T
                @ L[
                    k * diag_blocksize : (k + 1) * diag_blocksize,
                    i * diag_blocksize : (i + 1) * diag_blocksize,
                ]
            )

        # X_{i, i} = X_{i, i} L_{i, i}^{-1}
        X[
            i * diag_blocksize : (i + 1) * diag_blocksize,
            i * diag_blocksize : (i + 1) * diag_blocksize,
        ] = (
            X[
                i * diag_blocksize : (i + 1) * diag_blocksize,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ]
            @ L_blk_inv
        )

    return X
