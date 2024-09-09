# Copyright 2023-2024 ETH Zurich. All rights reserved.

import numpy as np
import scipy.linalg as la


def scpobbsi(
    L: np.ndarray,
    ndiags: int,
    blocksize: int,
) -> np.ndarray:
    """Perform a selected inversion of a block bidiagonal matrix using a
    sequential algorithm on CPU backend.

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
                # produced during the selected inversion process. ie. the matrix
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
