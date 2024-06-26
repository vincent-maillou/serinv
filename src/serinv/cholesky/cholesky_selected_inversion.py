# Copyright 2023-2024 ETH Zurich and USI. All rights reserved.


import numpy as np
import scipy.linalg as la


def cholesky_sinv_block_tridiagonal(
    L_diagonal_blocks: np.ndarray,
    L_lower_diagonal_blocks: np.ndarray,
) -> np.ndarray:
    """Perform a selected inversion from a cholesky decomposed matrix with a
    block tridiagonal structure.

    Parameters
    ----------
    L_diagonal_blocks : np.ndarray
    L_lower_diagonal_blocks : np.ndarray

    Returns
    -------
    X : np.ndarray
        Selected inversion of the matrix.
    """

    blocksize = L_diagonal_blocks.shape[0]
    nblocks = L_diagonal_blocks.shape[1] // blocksize

    X_diagonal_blocks = np.zeros_like(L_diagonal_blocks)
    X_lower_diagonal_blocks = np.zeros_like(L_lower_diagonal_blocks)
    L_blk_inv = np.zeros((blocksize, blocksize), dtype=L_diagonal_blocks.dtype)

    L_blk_inv = la.solve_triangular(
        L_diagonal_blocks[:, -blocksize:], np.eye(blocksize), lower=True
    )
    X_diagonal_blocks[:, -blocksize:] = L_blk_inv.T @ L_blk_inv

    for i in range(nblocks - 2, -1, -1):
        L_blk_inv = la.solve_triangular(
            L_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize],
            np.eye(blocksize),
            lower=True,
        )

        # X_{i+1, i} = -X_{i+1, i+1} L_{i+1, i} L_{i, i}^{-1}
        X_lower_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize] = (
            -X_diagonal_blocks[:, (i + 1) * blocksize : (i + 2) * blocksize]
            @ L_lower_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize]
            @ L_blk_inv
        )

        # X_{i, i} = (L_{i, i}^{-T} - X_{i+1, i}^{T} L_{i+1, i}) L_{i, i}^{-1}
        X_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize] = (
            L_blk_inv.T
            - X_lower_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize].T
            @ L_lower_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize]
        ) @ L_blk_inv

    return (
        X_diagonal_blocks,
        X_lower_diagonal_blocks,
    )


def cholesky_sinv_block_tridiagonal_arrowhead(
    L_diagonal_blocks: np.ndarray,
    L_lower_diagonal_blocks: np.ndarray,
    L_arrow_bottom_blocks: np.ndarray,
    L_arrow_tip_block: np.ndarray,
) -> np.ndarray:
    """Perform a selected inversion from a cholesky decomposed matrix with a
    block tridiagonal arrowhead structure.

    Parameters
    ----------
    L : np.ndarray
        The cholesky factorization of the matrix.
    diag_blocksize : int
        Blocksize of the diagonals blocks of the matrix.
    arrow_blocksize : int
        Blocksize of the blocks composing the arrowhead.

    Returns
    -------
    X : np.ndarray
        Selected inversion of the matrix.
    """

    diag_blocksize = L_diagonal_blocks.shape[0]
    arrow_blocksize = L_arrow_tip_block.shape[0]
    n_diag_blocks = L_diagonal_blocks.shape[1] // diag_blocksize

    X_diagonal_blocks = np.zeros_like(L_diagonal_blocks)
    X_lower_diagonal_blocks = np.zeros_like(L_lower_diagonal_blocks)
    X_arrow_bottom_blocks = np.zeros_like(L_arrow_bottom_blocks)
    X_arrow_tip_block = np.zeros_like(L_arrow_tip_block)

    L_last_blk_inv = np.zeros(
        (arrow_blocksize, arrow_blocksize), dtype=L_diagonal_blocks.dtype
    )
    L_blk_inv = np.zeros(
        (diag_blocksize, diag_blocksize), dtype=L_diagonal_blocks.dtype
    )

    L_last_blk_inv = la.solve_triangular(
        L_arrow_tip_block[:, :], np.eye(arrow_blocksize), lower=True
    )

    X_arrow_tip_block[:, :] = L_last_blk_inv.T @ L_last_blk_inv

    L_blk_inv = la.solve_triangular(
        L_diagonal_blocks[:, -diag_blocksize:],
        np.eye(diag_blocksize),
        lower=True,
    )

    # X_{ndb+1, ndb} = -X_{ndb+1, ndb+1} L_{ndb+1, ndb} L_{ndb, ndb}^{-1}
    X_arrow_bottom_blocks[:, -diag_blocksize:] = (
        -X_arrow_tip_block[:, :]
        @ L_arrow_bottom_blocks[:, -diag_blocksize:]
        @ L_blk_inv
    )

    # X_{ndb, ndb} = (L_{ndb, ndb}^{-T} - X_{ndb+1, ndb}^{T} L_{ndb+1, ndb}) L_{ndb, ndb}^{-1}
    X_diagonal_blocks[:, -diag_blocksize:] = (
        L_blk_inv.T
        - X_arrow_bottom_blocks[:, -diag_blocksize:].T
        @ L_arrow_bottom_blocks[:, -diag_blocksize:]
    ) @ L_blk_inv

    for i in range(n_diag_blocks - 2, -1, -1):
        L_blk_inv = la.solve_triangular(
            L_diagonal_blocks[:, i * diag_blocksize : (i + 1) * diag_blocksize],
            np.eye(diag_blocksize),
            lower=True,
        )

        # --- Off-diagonal block part ---
        # X_{i+1, i} = (-X_{i+1, i+1} L_{i+1, i} - X_{ndb+1, i+1}^{T} L_{ndb+1, i}) L_{i, i}^{-1}
        X_lower_diagonal_blocks[:, i * diag_blocksize : (i + 1) * diag_blocksize] = (
            -X_diagonal_blocks[:, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize]
            @ L_lower_diagonal_blocks[:, i * diag_blocksize : (i + 1) * diag_blocksize]
            - X_arrow_bottom_blocks[
                :, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize
            ].T
            @ L_arrow_bottom_blocks[:, i * diag_blocksize : (i + 1) * diag_blocksize]
        ) @ L_blk_inv

        # --- Arrowhead part ---
        # X_{ndb+1, i} = (- X_{ndb+1, i+1} L_{i+1, i} - X_{ndb+1, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1}
        X_arrow_bottom_blocks[:, i * diag_blocksize : (i + 1) * diag_blocksize] = (
            -X_arrow_bottom_blocks[
                :, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize
            ]
            @ L_lower_diagonal_blocks[:, i * diag_blocksize : (i + 1) * diag_blocksize]
            - X_arrow_tip_block[:, :]
            @ L_arrow_bottom_blocks[:, i * diag_blocksize : (i + 1) * diag_blocksize]
        ) @ L_blk_inv

        # --- Diagonal block part ---
        # X_{i, i} = (L_{i, i}^{-T} - X_{i+1, i}^{T} L_{i+1, i} - X_{ndb+1, i}.T L_{ndb+1, i}) L_{i, i}^{-1}
        X_diagonal_blocks[:, i * diag_blocksize : (i + 1) * diag_blocksize] = (
            L_blk_inv.T
            - X_lower_diagonal_blocks[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ].T
            @ L_lower_diagonal_blocks[:, i * diag_blocksize : (i + 1) * diag_blocksize]
            - X_arrow_bottom_blocks[:, i * diag_blocksize : (i + 1) * diag_blocksize].T
            @ L_arrow_bottom_blocks[:, i * diag_blocksize : (i + 1) * diag_blocksize]
        ) @ L_blk_inv

    return (
        X_diagonal_blocks,
        X_lower_diagonal_blocks,
        X_arrow_bottom_blocks,
        X_arrow_tip_block,
    )


def chol_sinv_ndiags(
    L: np.ndarray,
    ndiags: int,
    blocksize: int,
) -> np.ndarray:
    """Perform a selected inversion from a cholesky decomposed matrix with a
    block n-diagonals structure.

    Parameters
    ----------
    L : np.ndarray
        The cholesky factorization of the matrix.
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

    nblocks = L.shape[0] // blocksize
    n_offdiags_blk = ndiags // 2
    for i in range(nblocks - 1, -1, -1):
        # L_blk_inv = L_{i, i}^{-1}
        L_blk_inv = la.solve_triangular(
            L[i * blocksize : (i + 1) * blocksize, i * blocksize : (i + 1) * blocksize],
            np.eye(blocksize),
            lower=True,
        )

        # Off-diagonal block part
        for j in range(min(i + n_offdiags_blk, nblocks - 1), i, -1):
            for k in range(i + 1, min(i + n_offdiags_blk + 1, nblocks), 1):
                # The following condition ensure to use the lower elements
                # produced duiring the selected inversion process. ie. the matrix
                # is symmetric.
                if k > j:
                    # X_{j, i} = X_{j, i} - X_{k, j}.T L_{k, i}
                    X[
                        j * blocksize : (j + 1) * blocksize,
                        i * blocksize : (i + 1) * blocksize,
                    ] -= (
                        X[
                            k * blocksize : (k + 1) * blocksize,
                            j * blocksize : (j + 1) * blocksize,
                        ].T
                        @ L[
                            k * blocksize : (k + 1) * blocksize,
                            i * blocksize : (i + 1) * blocksize,
                        ]
                    )
                else:
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

            # X_{i, j} = X_{j, i}.T
            X[
                i * blocksize : (i + 1) * blocksize, j * blocksize : (j + 1) * blocksize
            ] = X[
                j * blocksize : (j + 1) * blocksize, i * blocksize : (i + 1) * blocksize
            ].T

        # Diagonal block part
        # X_{i, i} = (L_{i, i}^{-T} - sum_{k=i+1}^{min(i+ndiags/2, nblocks)} X_{i, k} L_{k, i}) L_{i, i}^{-1}

        # X_{i, i} = L_{i, i}^{-T}
        X[
            i * blocksize : (i + 1) * blocksize, i * blocksize : (i + 1) * blocksize
        ] = L_blk_inv.T

        for k in range(i + 1, min(i + n_offdiags_blk + 1, nblocks), 1):
            # X_{i, i} = X_{i, i} - X_{k, i}.T L_{k, i}
            X[
                i * blocksize : (i + 1) * blocksize, i * blocksize : (i + 1) * blocksize
            ] -= (
                X[
                    k * blocksize : (k + 1) * blocksize,
                    i * blocksize : (i + 1) * blocksize,
                ].T
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


def chol_sinv_ndiags_arrowhead(
    L: np.ndarray,
    ndiags: int,
    diag_blocksize: int,
    arrow_blocksize: int,
) -> np.ndarray:
    """Perform a selected inversion from a cholesky decomposed matrix with a
    block tridiagonal arrowhead structure.

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
