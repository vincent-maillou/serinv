# Copyright 2023-2024 ETH Zurich and USI. All rights reserved.

import numpy as np
import scipy.linalg as la


def lu_dcmp_ndiags(
    A: np.ndarray,
    ndiags: int,
    blocksize: int,
) -> np.ndarray:
    """Perform the LU factorization of a block ndiagonals matrix. The
    matrix is assumed to be non-singular.

    Parameters
    ----------
    A : np.ndarray
        Input matrix to decompose.
    ndiags : int
        Number of diagonals of the matrix.
    blocksize : int
        Size of the blocks.

    Returns
    -------
    L : np.ndarray
        Lower factor of the LU factorization of the matrix.
    U : np.ndarray
        Upper factor of the LU factorization of the matrix.
    """

    L = np.zeros_like(A)
    U = np.zeros_like(A)

    L_inv_temp = np.zeros((blocksize, blocksize))
    U_inv_temp = np.zeros((blocksize, blocksize))

    n_offdiags_blk = ndiags // 2

    nblocks = A.shape[0] // blocksize
    for i in range(0, nblocks - 1):
        # L_{i, i}, U_{i, i} = lu_dcmp(A_{i, i})
        (
            L[i * blocksize : (i + 1) * blocksize, i * blocksize : (i + 1) * blocksize],
            U[i * blocksize : (i + 1) * blocksize, i * blocksize : (i + 1) * blocksize],
        ) = la.lu(
            A[i * blocksize : (i + 1) * blocksize, i * blocksize : (i + 1) * blocksize],
            permute_l=True,
        )

        # Invert L_{i, i} and U_{i, i} for later use
        L_inv_temp = la.solve_triangular(
            L[i * blocksize : (i + 1) * blocksize, i * blocksize : (i + 1) * blocksize],
            np.eye(blocksize),
            lower=True,
        )
        U_inv_temp = la.solve_triangular(
            U[i * blocksize : (i + 1) * blocksize, i * blocksize : (i + 1) * blocksize],
            np.eye(blocksize),
            lower=False,
        )

        for j in range(1, min(n_offdiags_blk + 1, nblocks - i)):
            # L_{i+j, i} = A_{i+j, i} @ U{i, i}^{-1}
            L[
                (i + j) * blocksize : (i + j + 1) * blocksize,
                i * blocksize : (i + 1) * blocksize,
            ] = (
                A[
                    (i + j) * blocksize : (i + j + 1) * blocksize,
                    i * blocksize : (i + 1) * blocksize,
                ]
                @ U_inv_temp
            )

            # U_{i, i+j} = L{i, i}^{-1} @ A_{i, i+j}
            U[
                i * blocksize : (i + 1) * blocksize,
                (i + j) * blocksize : (i + j + 1) * blocksize,
            ] = (
                L_inv_temp
                @ A[
                    i * blocksize : (i + 1) * blocksize,
                    (i + j) * blocksize : (i + j + 1) * blocksize,
                ]
            )

            for k in range(1, j):
                # A_{i+j, i+k} = A_{i+j, i+k} - L_{i+j, i} @ U_{i, i+k}
                A[
                    (i + j) * blocksize : (i + j + 1) * blocksize,
                    (i + k) * blocksize : (i + k + 1) * blocksize,
                ] = (
                    A[
                        (i + j) * blocksize : (i + j + 1) * blocksize,
                        (i + k) * blocksize : (i + k + 1) * blocksize,
                    ]
                    - L[
                        (i + j) * blocksize : (i + j + 1) * blocksize,
                        i * blocksize : (i + 1) * blocksize,
                    ]
                    @ U[
                        i * blocksize : (i + 1) * blocksize,
                        (i + k) * blocksize : (i + k + 1) * blocksize,
                    ]
                )

                # A_{i+k, i+j} = A_{i+k, i+j} - L_{i+k, i} @ U_{i, i+j}
                A[
                    (i + k) * blocksize : (i + k + 1) * blocksize,
                    (i + j) * blocksize : (i + j + 1) * blocksize,
                ] = (
                    A[
                        (i + k) * blocksize : (i + k + 1) * blocksize,
                        (i + j) * blocksize : (i + j + 1) * blocksize,
                    ]
                    - L[
                        (i + k) * blocksize : (i + k + 1) * blocksize,
                        i * blocksize : (i + 1) * blocksize,
                    ]
                    @ U[
                        i * blocksize : (i + 1) * blocksize,
                        (i + j) * blocksize : (i + j + 1) * blocksize,
                    ]
                )

            # A_{i+j, i+j} = A_{i+j, i+j} - L_{i+j, i} @ U_{i, i+j}
            A[
                (i + j) * blocksize : (i + j + 1) * blocksize,
                (i + j) * blocksize : (i + j + 1) * blocksize,
            ] = (
                A[
                    (i + j) * blocksize : (i + j + 1) * blocksize,
                    (i + j) * blocksize : (i + j + 1) * blocksize,
                ]
                - L[
                    (i + j) * blocksize : (i + j + 1) * blocksize,
                    i * blocksize : (i + 1) * blocksize,
                ]
                @ U[
                    i * blocksize : (i + 1) * blocksize,
                    (i + j) * blocksize : (i + j + 1) * blocksize,
                ]
            )

    # L_{nblocks, nblocks}, U_{nblocks, nblocks} = lu_dcmp(A_{nblocks, nblocks})
    L[-blocksize:, -blocksize:], U[-blocksize:, -blocksize:] = la.lu(
        A[-blocksize:, -blocksize:], permute_l=True
    )

    return L, U


def lu_dcmp_ndiags_arrowhead(
    A: np.ndarray,
    ndiags: int,
    diag_blocksize: int,
    arrow_blocksize: int,
) -> np.ndarray:
    """Perform the LU factorization of a block n-diagonals arrowhead matrix.
    The matrix is assumed to be non-singular.

    Parameters
    ----------
    A : np.ndarray
        Input matrix to decompose.
    ndiags : int
        Number of diagonals of the matrix.
    diag_blocksize : int
        Blocksize of the diagonals blocks of the matrix.
    arrow_blocksize : int
        Blocksize of the blocks composing the arrowhead.

    Returns
    -------
    L : np.ndarray
        Lower factor of the LU factorization of the matrix.
    U : np.ndarray
        Upper factor of the LU factorization of the matrix.
    """

    L = np.zeros_like(A)
    U = np.zeros_like(A)

    L_inv_temp = np.zeros((diag_blocksize, diag_blocksize))
    U_inv_temp = np.zeros((diag_blocksize, diag_blocksize))

    n_offdiags_blk = ndiags // 2

    n_diag_blocks = (A.shape[0] - arrow_blocksize) // diag_blocksize
    for i in range(0, n_diag_blocks - 1):
        # L_{i, i}, U_{i, i} = lu_dcmp(A_{i, i})
        (
            L[
                i * diag_blocksize : (i + 1) * diag_blocksize,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ],
            U[
                i * diag_blocksize : (i + 1) * diag_blocksize,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ],
        ) = la.lu(
            A[
                i * diag_blocksize : (i + 1) * diag_blocksize,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ],
            permute_l=True,
        )

        # Invert L_{i, i} and U_{i, i} for later use
        L_inv_temp = la.solve_triangular(
            L[
                i * diag_blocksize : (i + 1) * diag_blocksize,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ],
            np.eye(diag_blocksize),
            lower=True,
        )
        U_inv_temp = la.solve_triangular(
            U[
                i * diag_blocksize : (i + 1) * diag_blocksize,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ],
            np.eye(diag_blocksize),
            lower=False,
        )

        for j in range(1, min(n_offdiags_blk + 1, n_diag_blocks - i)):
            # L_{i+j, i} = A_{i+j, i} @ U{i, i}^{-1}
            L[
                (i + j) * diag_blocksize : (i + j + 1) * diag_blocksize,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ] = (
                A[
                    (i + j) * diag_blocksize : (i + j + 1) * diag_blocksize,
                    i * diag_blocksize : (i + 1) * diag_blocksize,
                ]
                @ U_inv_temp
            )

            # U_{i, i+j} = L{i, i}^{-1} @ A_{i, i+j}
            U[
                i * diag_blocksize : (i + 1) * diag_blocksize,
                (i + j) * diag_blocksize : (i + j + 1) * diag_blocksize,
            ] = (
                L_inv_temp
                @ A[
                    i * diag_blocksize : (i + 1) * diag_blocksize,
                    (i + j) * diag_blocksize : (i + j + 1) * diag_blocksize,
                ]
            )

            for k in range(1, j):
                # A_{i+j, i+k} = A_{i+j, i+k} - L_{i+j, i} @ U_{i, i+k}
                A[
                    (i + j) * diag_blocksize : (i + j + 1) * diag_blocksize,
                    (i + k) * diag_blocksize : (i + k + 1) * diag_blocksize,
                ] = (
                    A[
                        (i + j) * diag_blocksize : (i + j + 1) * diag_blocksize,
                        (i + k) * diag_blocksize : (i + k + 1) * diag_blocksize,
                    ]
                    - L[
                        (i + j) * diag_blocksize : (i + j + 1) * diag_blocksize,
                        i * diag_blocksize : (i + 1) * diag_blocksize,
                    ]
                    @ U[
                        i * diag_blocksize : (i + 1) * diag_blocksize,
                        (i + k) * diag_blocksize : (i + k + 1) * diag_blocksize,
                    ]
                )

                # A_{i+k, i+j} = A_{i+k, i+j} - L_{i+k, i} @ U_{i, i+j}
                A[
                    (i + k) * diag_blocksize : (i + k + 1) * diag_blocksize,
                    (i + j) * diag_blocksize : (i + j + 1) * diag_blocksize,
                ] = (
                    A[
                        (i + k) * diag_blocksize : (i + k + 1) * diag_blocksize,
                        (i + j) * diag_blocksize : (i + j + 1) * diag_blocksize,
                    ]
                    - L[
                        (i + k) * diag_blocksize : (i + k + 1) * diag_blocksize,
                        i * diag_blocksize : (i + 1) * diag_blocksize,
                    ]
                    @ U[
                        i * diag_blocksize : (i + 1) * diag_blocksize,
                        (i + j) * diag_blocksize : (i + j + 1) * diag_blocksize,
                    ]
                )

            # A_{i+j, i+j} = A_{i+j, i+j} - L_{i+j, i} @ U_{i, i+j}
            A[
                (i + j) * diag_blocksize : (i + j + 1) * diag_blocksize,
                (i + j) * diag_blocksize : (i + j + 1) * diag_blocksize,
            ] = (
                A[
                    (i + j) * diag_blocksize : (i + j + 1) * diag_blocksize,
                    (i + j) * diag_blocksize : (i + j + 1) * diag_blocksize,
                ]
                - L[
                    (i + j) * diag_blocksize : (i + j + 1) * diag_blocksize,
                    i * diag_blocksize : (i + 1) * diag_blocksize,
                ]
                @ U[
                    i * diag_blocksize : (i + 1) * diag_blocksize,
                    (i + j) * diag_blocksize : (i + j + 1) * diag_blocksize,
                ]
            )

        # Part of the decomposition for the arrowhead structure
        # L_{ndb+1, i} = A_{ndb+1, i} @ U{i, i}^{-1}
        L[-arrow_blocksize:, i * diag_blocksize : (i + 1) * diag_blocksize] = (
            A[-arrow_blocksize:, i * diag_blocksize : (i + 1) * diag_blocksize]
            @ U_inv_temp
        )

        # U_{i, ndb+1} = L{i, i}^{-1} @ A_{i, ndb+1}
        U[i * diag_blocksize : (i + 1) * diag_blocksize, -arrow_blocksize:] = (
            L_inv_temp
            @ A[i * diag_blocksize : (i + 1) * diag_blocksize, -arrow_blocksize:]
        )

        for k in range(1, min(n_offdiags_blk + 1, n_diag_blocks - i)):
            # A_{ndb+1, i+k} = A_{ndb+1, i+k} - L_{ndb+1, i} @ U_{i, i+k}
            A[
                -arrow_blocksize:,
                (i + k) * diag_blocksize : (i + k + 1) * diag_blocksize,
            ] = (
                A[
                    -arrow_blocksize:,
                    (i + k) * diag_blocksize : (i + k + 1) * diag_blocksize,
                ]
                - L[-arrow_blocksize:, i * diag_blocksize : (i + 1) * diag_blocksize]
                @ U[
                    i * diag_blocksize : (i + 1) * diag_blocksize,
                    (i + k) * diag_blocksize : (i + k + 1) * diag_blocksize,
                ]
            )

            # A_{i+k, ndb+1} = A_{i+k, ndb+1} - L_{i+k, i} @ U_{i, ndb+1}
            A[
                (i + k) * diag_blocksize : (i + k + 1) * diag_blocksize,
                -arrow_blocksize:,
            ] = (
                A[
                    (i + k) * diag_blocksize : (i + k + 1) * diag_blocksize,
                    -arrow_blocksize:,
                ]
                - L[
                    (i + k) * diag_blocksize : (i + k + 1) * diag_blocksize,
                    i * diag_blocksize : (i + 1) * diag_blocksize,
                ]
                @ U[i * diag_blocksize : (i + 1) * diag_blocksize, -arrow_blocksize:]
            )

        # A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, i} @ U_{i, ndb+1}
        A[-arrow_blocksize:, -arrow_blocksize:] = (
            A[-arrow_blocksize:, -arrow_blocksize:]
            - L[-arrow_blocksize:, i * diag_blocksize : (i + 1) * diag_blocksize]
            @ U[i * diag_blocksize : (i + 1) * diag_blocksize, -arrow_blocksize:]
        )

    # L_{ndb, ndb}, U_{ndb, ndb} = lu_dcmp(A_{ndb, ndb})
    (
        L[
            -diag_blocksize - arrow_blocksize : -arrow_blocksize,
            -diag_blocksize - arrow_blocksize : -arrow_blocksize,
        ],
        U[
            -diag_blocksize - arrow_blocksize : -arrow_blocksize,
            -diag_blocksize - arrow_blocksize : -arrow_blocksize,
        ],
    ) = la.lu(
        A[
            -diag_blocksize - arrow_blocksize : -arrow_blocksize,
            -diag_blocksize - arrow_blocksize : -arrow_blocksize,
        ],
        permute_l=True,
    )

    # L_{ndb+1, nbd} = A_{ndb+1, nbd} @ U_{ndb, ndb}^{-1}
    L[-arrow_blocksize:, -diag_blocksize - arrow_blocksize : -arrow_blocksize] = A[
        -arrow_blocksize:, -diag_blocksize - arrow_blocksize : -arrow_blocksize
    ] @ la.solve_triangular(
        U[
            -diag_blocksize - arrow_blocksize : -arrow_blocksize,
            -diag_blocksize - arrow_blocksize : -arrow_blocksize,
        ],
        np.eye(diag_blocksize),
        lower=False,
    )

    # U_{ndb, nbd+1} = L_{ndb, ndb}^{-1} @ A_{ndb, nbd+1}
    U[-diag_blocksize - arrow_blocksize : -arrow_blocksize, -arrow_blocksize:] = (
        la.solve_triangular(
            L[
                -diag_blocksize - arrow_blocksize : -arrow_blocksize,
                -diag_blocksize - arrow_blocksize : -arrow_blocksize,
            ],
            np.eye(diag_blocksize),
            lower=True,
        )
        @ A[-diag_blocksize - arrow_blocksize : -arrow_blocksize, -arrow_blocksize:]
    )

    # A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, ndb} @ U_{ndb, ndb+1}
    A[-arrow_blocksize:, -arrow_blocksize:] = (
        A[-arrow_blocksize:, -arrow_blocksize:]
        - L[-arrow_blocksize:, -diag_blocksize - arrow_blocksize : -arrow_blocksize]
        @ U[-diag_blocksize - arrow_blocksize : -arrow_blocksize, -arrow_blocksize:]
    )

    # L_{ndb+1, ndb+1}, U_{ndb+1, ndb+1} = lu_dcmp(A_{ndb+1, ndb+1})
    (
        L[-arrow_blocksize:, -arrow_blocksize:],
        U[-arrow_blocksize:, -arrow_blocksize:],
    ) = la.lu(A[-arrow_blocksize:, -arrow_blocksize:], permute_l=True)

    return L, U
