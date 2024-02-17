"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Lisa Gaedke-Merzhaeuser  (lisa.gaedke.merzhaeuser@usi.ch)
@date: 2023-11

Contains the lu selected inversion routines.

Copyright 2023 ETH Zurich and USI. All rights reserved.
"""


import numpy as np
import scipy.linalg as la


def lu_sinv_tridiag(
    L: np.ndarray,
    U: np.ndarray,
    blocksize: int,
) -> np.ndarray:
    """Perform a selected inversion from a lu decomposed matrix with a
    block tridiagonal structure.

    Parameters
    ----------
    L : np.ndarray
        Lower factor of the lu factorization fo the matrix.
    U : np.ndarray
        Upper factor of the lu factorization fo the matrix.
    blocksize : int
        The blocksize of the matrix.

    Returns
    -------
    X : np.ndarray
        Selected inversion of the matrix.
    """

    X = np.zeros(L.shape, dtype=L.dtype)

    L_blk_inv = np.zeros((blocksize, blocksize), dtype=L.dtype)
    U_blk_inv = np.zeros((blocksize, blocksize), dtype=L.dtype)

    L_blk_inv = la.solve_triangular(
        L[-blocksize:, -blocksize:], np.eye(blocksize), lower=True
    )
    U_blk_inv = la.solve_triangular(
        U[-blocksize:, -blocksize:], np.eye(blocksize), lower=False
    )
    X[-blocksize:, -blocksize:] = U_blk_inv @ L_blk_inv

    nblocks = L.shape[0] // blocksize
    for i in range(nblocks - 2, -1, -1):
        L_blk_inv = la.solve_triangular(
            L[i * blocksize : (i + 1) * blocksize, i * blocksize : (i + 1) * blocksize],
            np.eye(blocksize),
            lower=True,
        )
        U_blk_inv = la.solve_triangular(
            U[i * blocksize : (i + 1) * blocksize, i * blocksize : (i + 1) * blocksize],
            np.eye(blocksize),
            lower=False,
        )

        # X_{i+1, i} = -X_{i+1, i+1} L_{i+1, i} L_{i, i}^{-1}
        X[
            (i + 1) * blocksize : (i + 2) * blocksize,
            i * blocksize : (i + 1) * blocksize,
        ] = (
            -X[
                (i + 1) * blocksize : (i + 2) * blocksize,
                (i + 1) * blocksize : (i + 2) * blocksize,
            ]
            @ L[
                (i + 1) * blocksize : (i + 2) * blocksize,
                i * blocksize : (i + 1) * blocksize,
            ]
            @ L_blk_inv
        )

        # X_{i, i+1} = -U_{i, i}^{-1} U_{i, i+1} X_{i+1, i+1}
        X[
            i * blocksize : (i + 1) * blocksize,
            (i + 1) * blocksize : (i + 2) * blocksize,
        ] = (
            -U_blk_inv
            @ U[
                i * blocksize : (i + 1) * blocksize,
                (i + 1) * blocksize : (i + 2) * blocksize,
            ]
            @ X[
                (i + 1) * blocksize : (i + 2) * blocksize,
                (i + 1) * blocksize : (i + 2) * blocksize,
            ]
        )

        # X_{i, i} = (U_{i, i}^{-1} - X_{i, i+1} L_{i+1, i}) L_{i, i}^{-1}
        X[i * blocksize : (i + 1) * blocksize, i * blocksize : (i + 1) * blocksize] = (
            U_blk_inv
            - X[
                i * blocksize : (i + 1) * blocksize,
                (i + 1) * blocksize : (i + 2) * blocksize,
            ]
            @ L[
                (i + 1) * blocksize : (i + 2) * blocksize,
                i * blocksize : (i + 1) * blocksize,
            ]
        ) @ L_blk_inv

    return X


def lu_sinv_tridiag_arrowhead(
    L: np.ndarray,
    U: np.ndarray,
    diag_blocksize: int,
    arrow_blocksize: int,
) -> np.ndarray:
    """Perform a selected inversion from a lu decomposed matrix with a
    block tridiagonal arrowhead structure.

    Parameters
    ----------
    L : np.ndarray
        Lower factor of the lu factorization fo the matrix.
    U : np.ndarray
        Upper factor of the lu factorization fo the matrix.
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
    U_last_blk_inv = np.zeros((arrow_blocksize, arrow_blocksize), dtype=L.dtype)

    L_last_blk_inv = la.solve_triangular(
        L[-arrow_blocksize:, -arrow_blocksize:], np.eye(arrow_blocksize), lower=True
    )
    U_last_blk_inv = la.solve_triangular(
        U[-arrow_blocksize:, -arrow_blocksize:], np.eye(arrow_blocksize), lower=False
    )

    X[-arrow_blocksize:, -arrow_blocksize:] = U_last_blk_inv @ L_last_blk_inv

    L_blk_inv = np.zeros((diag_blocksize, diag_blocksize), dtype=L.dtype)
    U_blk_inv = np.zeros((diag_blocksize, diag_blocksize), dtype=L.dtype)

    L_blk_inv = la.solve_triangular(
        L[
            -arrow_blocksize - diag_blocksize : -arrow_blocksize,
            -arrow_blocksize - diag_blocksize : -arrow_blocksize,
        ],
        np.eye(diag_blocksize),
        lower=True,
    )
    U_blk_inv = la.solve_triangular(
        U[
            -arrow_blocksize - diag_blocksize : -arrow_blocksize,
            -arrow_blocksize - diag_blocksize : -arrow_blocksize,
        ],
        np.eye(diag_blocksize),
        lower=False,
    )

    # X_{ndb+1, ndb} = -X_{ndb+1, ndb+1} L_{ndb+1, ndb} L_{ndb, ndb}^{-1}
    X[-arrow_blocksize:, -arrow_blocksize - diag_blocksize : -arrow_blocksize] = (
        -X[-arrow_blocksize:, -arrow_blocksize:]
        @ L[-arrow_blocksize:, -arrow_blocksize - diag_blocksize : -arrow_blocksize]
        @ L_blk_inv
    )

    # X_{ndb, ndb+1} = -U_{ndb, ndb}^{-1} U_{ndb, ndb+1} X_{ndb+1, ndb+1}
    X[-arrow_blocksize - diag_blocksize : -arrow_blocksize, -arrow_blocksize:] = (
        -U_blk_inv
        @ U[-arrow_blocksize - diag_blocksize : -arrow_blocksize, -arrow_blocksize:]
        @ X[-arrow_blocksize:, -arrow_blocksize:]
    )

    # X_{ndb, ndb} = (U_{ndb, ndb}^{-1} - X_{ndb, ndb+1} L_{ndb+1, ndb}) L_{ndb, ndb}^{-1}
    X[
        -arrow_blocksize - diag_blocksize : -arrow_blocksize,
        -arrow_blocksize - diag_blocksize : -arrow_blocksize,
    ] = (
        U_blk_inv
        - X[-arrow_blocksize - diag_blocksize : -arrow_blocksize, -arrow_blocksize:]
        @ L[-arrow_blocksize:, -arrow_blocksize - diag_blocksize : -arrow_blocksize]
    ) @ L_blk_inv

    n_diag_blocks = (L.shape[0] - arrow_blocksize) // diag_blocksize
    for i in range(n_diag_blocks - 2, -1, -1):
        L_blk_inv = la.solve_triangular(
            L[
                i * diag_blocksize : (i + 1) * diag_blocksize,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ],
            np.eye(diag_blocksize),
            lower=True,
        )
        U_blk_inv = la.solve_triangular(
            U[
                i * diag_blocksize : (i + 1) * diag_blocksize,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ],
            np.eye(diag_blocksize),
            lower=False,
        )

        # --- Off-diagonal block part ---
        # X_{i+1, i} = (-X_{i+1, i+1} L_{i+1, i} - X_{i+1, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1}
        X[
            (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
            i * diag_blocksize : (i + 1) * diag_blocksize,
        ] = (
            -X[
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
            ]
            @ L[
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ]
            - X[(i + 1) * diag_blocksize : (i + 2) * diag_blocksize, -arrow_blocksize:]
            @ L[-arrow_blocksize:, i * diag_blocksize : (i + 1) * diag_blocksize]
        ) @ L_blk_inv

        # X_{i, i+1} = U_{i, i}^{-1} (- U_{i, i+1} X_{i+1, i+1} - U_{i, ndb+1} X_{ndb+1, i+1})
        X[
            i * diag_blocksize : (i + 1) * diag_blocksize,
            (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
        ] = U_blk_inv @ (
            -U[
                i * diag_blocksize : (i + 1) * diag_blocksize,
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
            ]
            @ X[
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
            ]
            - U[i * diag_blocksize : (i + 1) * diag_blocksize, -arrow_blocksize:]
            @ X[-arrow_blocksize:, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize]
        )

        # --- Arrowhead part ---
        # X_{ndb+1, i} = (- X_{ndb+1, i+1} L_{i+1, i} - X_{ndb+1, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1}
        X[-arrow_blocksize:, i * diag_blocksize : (i + 1) * diag_blocksize] = (
            -X[-arrow_blocksize:, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize]
            @ L[
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ]
            - X[-arrow_blocksize:, -arrow_blocksize:]
            @ L[-arrow_blocksize:, i * diag_blocksize : (i + 1) * diag_blocksize]
        ) @ L_blk_inv

        # X_{i, ndb+1} = U_{i, i}^{-1} (- U_{i, i+1} X_{i+1, ndb+1} - U_{i, ndb+1} X_{ndb+1, ndb+1})
        X[
            i * diag_blocksize : (i + 1) * diag_blocksize, -arrow_blocksize:
        ] = U_blk_inv @ (
            -U[
                i * diag_blocksize : (i + 1) * diag_blocksize,
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
            ]
            @ X[(i + 1) * diag_blocksize : (i + 2) * diag_blocksize, -arrow_blocksize:]
            - U[i * diag_blocksize : (i + 1) * diag_blocksize, -arrow_blocksize:]
            @ X[-arrow_blocksize:, -arrow_blocksize:]
        )

        # --- Diagonal block part ---
        # X_{i, i} = (U_{i, i}^{-1} - X_{i, i+1} L_{i+1, i} - X_{i, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1}
        X[
            i * diag_blocksize : (i + 1) * diag_blocksize,
            i * diag_blocksize : (i + 1) * diag_blocksize,
        ] = (
            U_blk_inv
            - X[
                i * diag_blocksize : (i + 1) * diag_blocksize,
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
            ]
            @ L[
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ]
            - X[i * diag_blocksize : (i + 1) * diag_blocksize, -arrow_blocksize:]
            @ L[-arrow_blocksize:, i * diag_blocksize : (i + 1) * diag_blocksize]
        ) @ L_blk_inv

    return X


def lu_sinv_ndiags(
    L: np.ndarray,
    U: np.ndarray,
    ndiags: int,
    blocksize: int,
) -> np.ndarray:
    """Perform a selected inversion from a lu decomposed matrix with a
    block n-diagonals structure.

    Parameters
    ----------
    L : np.ndarray
        Lower factor of the lu factorization fo the matrix.
    U : np.ndarray
        Upper factor of the lu factorization fo the matrix.
    ndiags : int
        Number of diagonals.
    blocksize : int
        Size of the blocks.

    Returns
    -------
    X : np.ndarray
        Selected inversion of the matrix.
    """

    X = np.zeros(L.shape, dtype=L.dtype)

    L_blk_inv = np.zeros((blocksize, blocksize), dtype=L.dtype)
    U_blk_inv = np.zeros((blocksize, blocksize), dtype=L.dtype)

    nblocks = L.shape[0] // blocksize
    n_offdiags_blk = ndiags // 2
    for i in range(nblocks - 1, -1, -1):
        # L_blk_inv = L_{i, i}^{-1}
        L_blk_inv = la.solve_triangular(
            L[i * blocksize : (i + 1) * blocksize, i * blocksize : (i + 1) * blocksize],
            np.eye(blocksize),
            lower=True,
        )
        # U_blk_inv = U_{i, i}^{-1}
        U_blk_inv = la.solve_triangular(
            U[i * blocksize : (i + 1) * blocksize, i * blocksize : (i + 1) * blocksize],
            np.eye(blocksize),
            lower=False,
        )

        # Off-diagonal block part
        for j in range(min(i + n_offdiags_blk, nblocks - 1), i, -1):
            for k in range(i + 1, min(i + n_offdiags_blk + 1, nblocks), 1):
                # X_{j, i} = X_{j, i} - X_{j, k} L_{k, i}
                X[
                    j * blocksize : (j + 1) * blocksize,
                    i * blocksize : (i + 1) * blocksize,
                ] -= (
                    X[
                        j * blocksize : (j + 1) * blocksize,
                        k * blocksize : (k + 1) * blocksize,
                    ]
                    @ L[
                        k * blocksize : (k + 1) * blocksize,
                        i * blocksize : (i + 1) * blocksize,
                    ]
                )

                # X_{i, j} = X_{i, j} - U_{i, k} X_{k, j}
                X[
                    i * blocksize : (i + 1) * blocksize,
                    j * blocksize : (j + 1) * blocksize,
                ] -= (
                    U[
                        i * blocksize : (i + 1) * blocksize,
                        k * blocksize : (k + 1) * blocksize,
                    ]
                    @ X[
                        k * blocksize : (k + 1) * blocksize,
                        j * blocksize : (j + 1) * blocksize,
                    ]
                )

            # X_{j, i} = X_{j, i} L_{i, i}^{-1}
            X[
                j * blocksize : (j + 1) * blocksize, i * blocksize : (i + 1) * blocksize
            ] = (
                X[
                    j * blocksize : (j + 1) * blocksize,
                    i * blocksize : (i + 1) * blocksize,
                ]
                @ L_blk_inv
            )

            # X_{i, j} = U_{i, i}^{-1} X_{i, j}
            X[
                i * blocksize : (i + 1) * blocksize, j * blocksize : (j + 1) * blocksize
            ] = (
                U_blk_inv
                @ X[
                    i * blocksize : (i + 1) * blocksize,
                    j * blocksize : (j + 1) * blocksize,
                ]
            )

        # Diagonal block part
        # X_{i, i} = (U_{i, i}^{-1} - sum_{k=i+1}^{min(i+ndiags/2, nblocks)} X_{i, k} L_{k, i}) L_{i, i}^{-1}

        # X_{i, i} = U_{i, i}^{-1}
        X[
            i * blocksize : (i + 1) * blocksize, i * blocksize : (i + 1) * blocksize
        ] = U_blk_inv

        for k in range(i + 1, min(i + n_offdiags_blk + 1, nblocks), 1):
            # X_{i, i} = X_{i, i} - X_{i, k} L_{k, i}
            X[
                i * blocksize : (i + 1) * blocksize, i * blocksize : (i + 1) * blocksize
            ] -= (
                X[
                    i * blocksize : (i + 1) * blocksize,
                    k * blocksize : (k + 1) * blocksize,
                ]
                @ L[
                    k * blocksize : (k + 1) * blocksize,
                    i * blocksize : (i + 1) * blocksize,
                ]
            )

        # X_{i, i} = X_{i, i} L_{i, i}^{-1}
        X[i * blocksize : (i + 1) * blocksize, i * blocksize : (i + 1) * blocksize] = (
            X[i * blocksize : (i + 1) * blocksize, i * blocksize : (i + 1) * blocksize]
            @ L_blk_inv
        )

    return X


def lu_sinv_ndiags_arrowhead(
    L: np.ndarray,
    U: np.ndarray,
    ndiags: int,
    diag_blocksize: int,
    arrow_blocksize: int,
) -> np.ndarray:
    """Perform a selected inversion from a lu decomposed matrix with a
    block tridiagonal arrowhead structure.

    Parameters
    ----------
    L : np.ndarray
        Lower factor of the lu factorization fo the matrix.
    U : np.ndarray
        Upper factor of the lu factorization fo the matrix.
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
    U_last_blk_inv = np.zeros((arrow_blocksize, arrow_blocksize), dtype=L.dtype)

    L_last_blk_inv = la.solve_triangular(
        L[-arrow_blocksize:, -arrow_blocksize:], np.eye(arrow_blocksize), lower=True
    )
    U_last_blk_inv = la.solve_triangular(
        U[-arrow_blocksize:, -arrow_blocksize:], np.eye(arrow_blocksize), lower=False
    )

    X[-arrow_blocksize:, -arrow_blocksize:] = U_last_blk_inv @ L_last_blk_inv

    L_blk_inv = np.zeros((diag_blocksize, diag_blocksize), dtype=L.dtype)
    U_blk_inv = np.zeros((diag_blocksize, diag_blocksize), dtype=L.dtype)

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

        # U_blk_inv = U_{i, i}^{-1}
        U_blk_inv = la.solve_triangular(
            U[
                i * diag_blocksize : (i + 1) * diag_blocksize,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ],
            np.eye(diag_blocksize),
            lower=False,
        )

        # Arrowhead part
        # X_{ndb+1, i} = - X_{ndb+1, ndb+1} L_{ndb+1, i}
        X[-arrow_blocksize:, i * diag_blocksize : (i + 1) * diag_blocksize] = (
            -X[-arrow_blocksize:, -arrow_blocksize:]
            @ L[-arrow_blocksize:, i * diag_blocksize : (i + 1) * diag_blocksize]
        )

        # X_{i, ndb+1} = - U_{i, ndb+1} X_{ndb+1, ndb+1}
        X[i * diag_blocksize : (i + 1) * diag_blocksize, -arrow_blocksize:] = (
            -U[i * diag_blocksize : (i + 1) * diag_blocksize, -arrow_blocksize:]
            @ X[-arrow_blocksize:, -arrow_blocksize:]
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

            # X_{i, ndb+1} = X_{i, ndb+1} - U_{i, k} X_{k, ndb+1}
            X[i * diag_blocksize : (i + 1) * diag_blocksize, -arrow_blocksize:] -= (
                U[
                    i * diag_blocksize : (i + 1) * diag_blocksize,
                    k * diag_blocksize : (k + 1) * diag_blocksize,
                ]
                @ X[k * diag_blocksize : (k + 1) * diag_blocksize, -arrow_blocksize:]
            )

        # X_{ndb+1, i} = X_{ndb+1, i} L_{i, i}^{-1}
        X[-arrow_blocksize:, i * diag_blocksize : (i + 1) * diag_blocksize] = (
            X[-arrow_blocksize:, i * diag_blocksize : (i + 1) * diag_blocksize]
            @ L_blk_inv
        )

        # X_{i, ndb+1} = U_{i, i}^{-1} X_{i, ndb+1}
        X[i * diag_blocksize : (i + 1) * diag_blocksize, -arrow_blocksize:] = (
            U_blk_inv
            @ X[i * diag_blocksize : (i + 1) * diag_blocksize, -arrow_blocksize:]
        )

        # Off-diagonal block part
        for j in range(min(i + n_offdiags_blk, n_diag_blocks - 1), i, -1):
            # Take the effect of the arrowhead part into account
            # X_{j, i} = - X_{j, ndb+1} L_{ndb+1, i}
            X[
                j * diag_blocksize : (j + 1) * diag_blocksize,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ] = (
                -X[j * diag_blocksize : (j + 1) * diag_blocksize, -arrow_blocksize:]
                @ L[-arrow_blocksize:, i * diag_blocksize : (i + 1) * diag_blocksize]
            )

            # X_{i, j} = - U_{i, ndb+1} X_{ndb+1, j}
            X[
                i * diag_blocksize : (i + 1) * diag_blocksize,
                j * diag_blocksize : (j + 1) * diag_blocksize,
            ] = (
                -U[i * diag_blocksize : (i + 1) * diag_blocksize, -arrow_blocksize:]
                @ X[-arrow_blocksize:, j * diag_blocksize : (j + 1) * diag_blocksize]
            )

            for k in range(i + 1, min(i + n_offdiags_blk + 1, n_diag_blocks), 1):
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

                # X_{i, j} = X_{i, j} - U_{i, k} X_{k, j}
                X[
                    i * diag_blocksize : (i + 1) * diag_blocksize,
                    j * diag_blocksize : (j + 1) * diag_blocksize,
                ] -= (
                    U[
                        i * diag_blocksize : (i + 1) * diag_blocksize,
                        k * diag_blocksize : (k + 1) * diag_blocksize,
                    ]
                    @ X[
                        k * diag_blocksize : (k + 1) * diag_blocksize,
                        j * diag_blocksize : (j + 1) * diag_blocksize,
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

            # X_{i, j} = U_{i, i}^{-1} X_{i, j}
            X[
                i * diag_blocksize : (i + 1) * diag_blocksize,
                j * diag_blocksize : (j + 1) * diag_blocksize,
            ] = (
                U_blk_inv
                @ X[
                    i * diag_blocksize : (i + 1) * diag_blocksize,
                    j * diag_blocksize : (j + 1) * diag_blocksize,
                ]
            )

        # Diagonal block part
        # X_{i, i} = (U_{i, i}^{-1} - X_{i, ndb+1} L_{ndb+1, i} - sum_{k=i+1}^{min(i+ndiags/2, n_diag_blocks)} X_{i, k} L_{k, i}) L_{i, i}^{-1}

        # X_{i, i} = U_{i, i}^{-1} - X_{i, ndb+1} L_{ndb+1, i}
        X[
            i * diag_blocksize : (i + 1) * diag_blocksize,
            i * diag_blocksize : (i + 1) * diag_blocksize,
        ] = (
            U_blk_inv
            - X[i * diag_blocksize : (i + 1) * diag_blocksize, -arrow_blocksize:]
            @ L[-arrow_blocksize:, i * diag_blocksize : (i + 1) * diag_blocksize]
        )

        for k in range(i + 1, min(i + n_offdiags_blk + 1, n_diag_blocks), 1):
            # X_{i, i} = X_{i, i} - X_{i, k} L_{k, i}
            X[
                i * diag_blocksize : (i + 1) * diag_blocksize,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ] -= (
                X[
                    i * diag_blocksize : (i + 1) * diag_blocksize,
                    k * diag_blocksize : (k + 1) * diag_blocksize,
                ]
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
