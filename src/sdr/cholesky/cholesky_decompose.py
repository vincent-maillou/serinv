"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Lisa Gaedke-Merzhaeuser  (lisa.gaedke.merzhaeuser@usi.ch)
@date: 2023-11

Contains the cholesky selected decompositions routines.

Copyright 2023 ETH Zurich and USI. All rights reserved.
"""


import numpy as np
import scipy.linalg as la


def chol_dcmp_tridiag(
    A: np.ndarray,
    blocksize: int,
    overwrite: bool = False,
) -> np.ndarray:
    """Perform the cholesky factorization of a block tridiagonal matrix. The
    matrix is assumed to be symmetric positive definite.

    Current implementation doesn't modify the input matrix A.

    Parameters
    ----------
    A : np.ndarray
        Input matrix to decompose.
    blocksize : int
        Size of the blocks.
    overwrite : bool
        If True, the input matrix A is modified in place. Default is False.
    
    Returns
    -------
    L : np.ndarray
        The cholesky factorization of the matrix.
    """

    if overwrite:
        L = A
    else:   
        L = np.copy(A)

    nblocks = A.shape[0] // blocksize
    for i in range(0, nblocks - 1):
        # L_{i, i} = chol(A_{i, i})
        L[
            i * blocksize : (i + 1) * blocksize, i * blocksize : (i + 1) * blocksize
        ] = la.cholesky(
            L[i * blocksize : (i + 1) * blocksize, i * blocksize : (i + 1) * blocksize],
        ).T

        # L_{i+1, i} = A_{i+1, i} @ L_{i, i}^{-T}
        L[
            (i + 1) * blocksize : (i + 2) * blocksize,
            i * blocksize : (i + 1) * blocksize,
        ] = (
            L[
                (i + 1) * blocksize : (i + 2) * blocksize,
                i * blocksize : (i + 1) * blocksize,
            ]
            @ la.solve_triangular(
                L[
                    i * blocksize : (i + 1) * blocksize,
                    i * blocksize : (i + 1) * blocksize,
                ],
                np.eye(blocksize),
                lower=True,
            ).T
        )

        # A_{i+1, i+1} = A_{i+1, i+1} - L_{i+1, i} @ L_{i+1, i}^{T}
        L[
            (i + 1) * blocksize : (i + 2) * blocksize,
            (i + 1) * blocksize : (i + 2) * blocksize,
        ] = (
            L[
                (i + 1) * blocksize : (i + 2) * blocksize,
                (i + 1) * blocksize : (i + 2) * blocksize,
            ]
            - L[
                (i + 1) * blocksize : (i + 2) * blocksize,
                i * blocksize : (i + 1) * blocksize,
            ]
            @ L[
                (i + 1) * blocksize : (i + 2) * blocksize,
                i * blocksize : (i + 1) * blocksize,
            ].T
        )

    L[-blocksize:, -blocksize:] = la.cholesky(L[-blocksize:, -blocksize:]).T
    
    L[:] = L * np.tri(*L.shape, k=0)
    
    return L


def chol_dcmp_tridiag_arrowhead(
    A: np.ndarray,
    diag_blocksize: int,
    arrow_blocksize: int,
    overwrite: bool = False,
) -> np.ndarray:
    """Perform the cholesky factorization of a block tridiagonal arrowhead
    matrix. The matrix is assumed to be symmetric positive definite.

    Parameters
    ----------
    A : np.ndarray
        Input matrix to decompose.
    diag_blocksize : int
        Blocksize of the diagonals blocks of the matrix.
    arrow_blocksize : int
        Blocksize of the blocks composing the arrowhead.
    overwrite : bool
        If True, the input matrix A is modified in place. Default is False.

    Returns
    -------
    L : np.ndarray
        The cholesky factorization of the matrix.
    """
    
    if overwrite:
        L = A
    else:    
        L = np.copy(A)
        
    L_inv_temp = np.zeros((diag_blocksize, diag_blocksize))

    n_diag_blocks = (A.shape[0] - arrow_blocksize) // diag_blocksize
    for i in range(0, n_diag_blocks - 1):
        # L_{i, i} = chol(A_{i, i})
        L[
            i * diag_blocksize : (i + 1) * diag_blocksize,
            i * diag_blocksize : (i + 1) * diag_blocksize,
        ] = la.cholesky(
            L[
                i * diag_blocksize : (i + 1) * diag_blocksize,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ]
        ).T

        # Temporary storage of used twice lower triangular solving
        # (Lisa) TODO: double check if this is actually better than solving twice ... 
        # especially when number of rows in arrowhead is small
        L_inv_temp[:diag_blocksize, :diag_blocksize] = la.solve_triangular(
            L[
                i * diag_blocksize : (i + 1) * diag_blocksize,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ],
            np.eye(diag_blocksize),
            lower=True,
        ).T

        # L_{i+1, i} = A_{i+1, i} @ L_{i, i}^{-T}
        L[
            (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
            i * diag_blocksize : (i + 1) * diag_blocksize,
        ] = (
            L[
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ]
            @ L_inv_temp
        )

        # L_{ndb+1, i} = A_{ndb+1, i} @ L_{i, i}^{-T}
        L[-arrow_blocksize:, i * diag_blocksize : (i + 1) * diag_blocksize] = (
            L[-arrow_blocksize:, i * diag_blocksize : (i + 1) * diag_blocksize]
            @ L_inv_temp
        )

        # A_{i+1, i+1} = A_{i+1, i+1} - L_{i+1, i} @ L_{i+1, i}.T
        L[
            (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
            (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
        ] = (
            L[
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
            ]
            - L[
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ]
            @ L[
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ].T
        )

        # A_{ndb+1, i+1} = A_{ndb+1, i+1} - L_{ndb+1, i} @ L_{i+1, i}.T
        L[-arrow_blocksize:, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize] = (
            L[-arrow_blocksize:, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize]
            - L[-arrow_blocksize:, i * diag_blocksize : (i + 1) * diag_blocksize]
            @ L[
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ].T
        )

        # A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, i} @ L_{ndb+1, i}.T
        L[-arrow_blocksize:, -arrow_blocksize:] = (
            L[-arrow_blocksize:, -arrow_blocksize:]
            - L[-arrow_blocksize:, i * diag_blocksize : (i + 1) * diag_blocksize]
            @ L[-arrow_blocksize:, i * diag_blocksize : (i + 1) * diag_blocksize].T
        )

    # L_{ndb, ndb} = chol(A_{ndb, ndb})
    L[
        -(diag_blocksize + arrow_blocksize) : -arrow_blocksize,
        -(diag_blocksize + arrow_blocksize) : -arrow_blocksize,
    ] = la.cholesky(
        L[
            -(diag_blocksize + arrow_blocksize) : -arrow_blocksize,
            -(diag_blocksize + arrow_blocksize) : -arrow_blocksize,
        ]
    ).T

    # L_{ndb+1, ndb} = A_{ndb+1, ndb} @ L_{ndb, ndb}^{-T}
    L[-arrow_blocksize:, -(diag_blocksize + arrow_blocksize) : -arrow_blocksize] = (
        L[-arrow_blocksize:, -(diag_blocksize + arrow_blocksize) : -arrow_blocksize]
        @ la.solve_triangular(
            L[
                -(diag_blocksize + arrow_blocksize) : -arrow_blocksize,
                -(diag_blocksize + arrow_blocksize) : -arrow_blocksize,
            ],
            np.eye(diag_blocksize),
            lower=True,
        ).T
    )

    # A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, ndb} @ L_{ndb+1, ndb}^{T}
    L[-arrow_blocksize:, -arrow_blocksize:] = (
        L[-arrow_blocksize:, -arrow_blocksize:]
        - L[-arrow_blocksize:, -(diag_blocksize + arrow_blocksize) : -arrow_blocksize]
        @ L[-arrow_blocksize:, -(diag_blocksize + arrow_blocksize) : -arrow_blocksize].T
    )

    # L_{ndb+1, ndb+1} = chol(A_{ndb+1, ndb+1})
    L[-arrow_blocksize:, -arrow_blocksize:] = la.cholesky(
        L[-arrow_blocksize:, -arrow_blocksize:]
    ).T
    
    # zero out upper triangular part
    L[:] = L * np.tri(*L.shape, k=0)

    return L


def chol_dcmp_ndiags(
    A: np.ndarray,
    ndiags: int,
    blocksize: int,
    overwrite: bool = False,
) -> np.ndarray:
    """Perform the cholesky factorization of a block n-diagonals matrix. The
    matrix is assumed to be symmetric positive definite.

    Parameters
    ----------
    A : np.ndarray
        Input matrix to decompose.
    ndiags : int
        Number of diagonals of the matrix.
    blocksize : int
        Size of the blocks of the matrix.
    overwrite : bool
        If True, the input matrix A is modified in place. Default is False.

    Returns
    -------
    L : np.ndarray
        The cholesky factorization of the matrix.
    """
    
    if overwrite:
        L = A
    else:    
        L = np.copy(A)
        
    L_inv_temp = np.zeros((blocksize, blocksize))

    n_offdiags_blk = ndiags // 2

    nblocks = A.shape[0] // blocksize
    for i in range(0, nblocks - 1):
        # L_{i, i} = chol(A_{i, i})
        L[
            i * blocksize : (i + 1) * blocksize, i * blocksize : (i + 1) * blocksize
        ] = la.cholesky(
            L[i * blocksize : (i + 1) * blocksize, i * blocksize : (i + 1) * blocksize]
        ).T

        # Temporary storage of re-used triangular solving
        L_inv_temp = la.solve_triangular(
            L[i * blocksize : (i + 1) * blocksize, i * blocksize : (i + 1) * blocksize],
            np.eye(blocksize),
            lower=True,
        ).T

        for j in range(1, min(n_offdiags_blk + 1, nblocks - i)):
            # L_{i+j, i} = A_{i+j, i} @ L_{i, i}^{-T}
            L[
                (i + j) * blocksize : (i + j + 1) * blocksize,
                i * blocksize : (i + 1) * blocksize,
            ] = (
                L[
                    (i + j) * blocksize : (i + j + 1) * blocksize,
                    i * blocksize : (i + 1) * blocksize,
                ]
                @ L_inv_temp
            )

            for k in range(1, j + 1):
                # A_{i+j, i+k} = A_{i+j, i+k} - L_{i+j, i} @ L_{i+k, i}^{T}
                L[
                    (i + j) * blocksize : (i + j + 1) * blocksize,
                    (i + k) * blocksize : (i + k + 1) * blocksize,
                ] = (
                    L[
                        (i + j) * blocksize : (i + j + 1) * blocksize,
                        (i + k) * blocksize : (i + k + 1) * blocksize,
                    ]
                    - L[
                        (i + j) * blocksize : (i + j + 1) * blocksize,
                        i * blocksize : (i + 1) * blocksize,
                    ]
                    @ L[
                        (i + k) * blocksize : (i + k + 1) * blocksize,
                        i * blocksize : (i + 1) * blocksize,
                    ].T
                )

    # L_{ndb, ndb} = chol(A_{ndb, ndb})
    L[-blocksize:, -blocksize:] = la.cholesky(L[-blocksize:, -blocksize:]).T
    
    # zero out upper triangular part
    L[:] = L * np.tri(*L.shape, k=0)

    return L


def chol_dcmp_ndiags_arrowhead(
    A: np.ndarray,
    ndiags: int,
    diag_blocksize: int,
    arrow_blocksize: int,
    overwrite: bool = False,
) -> np.ndarray:
    """Perform the cholesky factorization of a block n-diagonals arrowhead
    matrix. The matrix is assumed to be symmetric positive definite.

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
    overwrite : bool
        If True, the input matrix A is modified in place. Default is False.

    Returns
    -------
    L : np.ndarray
        The cholesky factorization of the matrix.
    """

    if overwrite:
        L = A
    else:    
        L = np.copy(A)
        
    L_inv_temp = np.zeros((diag_blocksize, diag_blocksize))

    n_offdiags_blk = ndiags // 2

    n_diag_blocks = (A.shape[0] - arrow_blocksize) // diag_blocksize
    for i in range(0, n_diag_blocks - 1):
        # L_{i, i} = chol(A_{i, i})
        L[
            i * diag_blocksize : (i + 1) * diag_blocksize,
            i * diag_blocksize : (i + 1) * diag_blocksize,
        ] = la.cholesky(
            L[
                i * diag_blocksize : (i + 1) * diag_blocksize,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ]
        ).T

        # Temporary storage of re-used triangular solving
        L_inv_temp = la.solve_triangular(
            L[
                i * diag_blocksize : (i + 1) * diag_blocksize,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ],
            np.eye(diag_blocksize),
            lower=True,
        ).T

        for j in range(1, min(n_offdiags_blk + 1, n_diag_blocks - i)):
            # L_{i+j, i} = A_{i+j, i} @ L_{i, i}^{-T}
            L[
                (i + j) * diag_blocksize : (i + j + 1) * diag_blocksize,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ] = (
                L[
                    (i + j) * diag_blocksize : (i + j + 1) * diag_blocksize,
                    i * diag_blocksize : (i + 1) * diag_blocksize,
                ]
                @ L_inv_temp
            )

            for k in range(1, j + 1):
                # A_{i+j, i+k} = A_{i+j, i+k} - L_{i+j, i} @ L_{i+k, i}^{T}
                L[
                    (i + j) * diag_blocksize : (i + j + 1) * diag_blocksize,
                    (i + k) * diag_blocksize : (i + k + 1) * diag_blocksize,
                ] = (
                    L[
                        (i + j) * diag_blocksize : (i + j + 1) * diag_blocksize,
                        (i + k) * diag_blocksize : (i + k + 1) * diag_blocksize,
                    ]
                    - L[
                        (i + j) * diag_blocksize : (i + j + 1) * diag_blocksize,
                        i * diag_blocksize : (i + 1) * diag_blocksize,
                    ]
                    @ L[
                        (i + k) * diag_blocksize : (i + k + 1) * diag_blocksize,
                        i * diag_blocksize : (i + 1) * diag_blocksize,
                    ].T
                )

        # Part of the decomposition for the arrowhead structure
        # L_{ndb+1, i} = A_{ndb+1, i} @ L_{i, i}^{-T}
        L[-arrow_blocksize:, i * diag_blocksize : (i + 1) * diag_blocksize] = (
            L[-arrow_blocksize:, i * diag_blocksize : (i + 1) * diag_blocksize]
            @ L_inv_temp
        )

        for k in range(1, min(n_offdiags_blk + 1, n_diag_blocks - i)):
            # A_{ndb+1, i+k} = A_{ndb+1, i+k} - L_{ndb+1, i} @ L_{i+k, i}^{T}
            L[
                -arrow_blocksize:,
                (i + k) * diag_blocksize : (i + k + 1) * diag_blocksize,
            ] = (
                L[
                    -arrow_blocksize:,
                    (i + k) * diag_blocksize : (i + k + 1) * diag_blocksize,
                ]
                - L[-arrow_blocksize:, i * diag_blocksize : (i + 1) * diag_blocksize]
                @ L[
                    (i + k) * diag_blocksize : (i + k + 1) * diag_blocksize,
                    i * diag_blocksize : (i + 1) * diag_blocksize,
                ].T
            )

        # A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, i} @ L_{ndb+1, i}^{T}
        L[-arrow_blocksize:, -arrow_blocksize:] = (
            L[-arrow_blocksize:, -arrow_blocksize:]
            - L[-arrow_blocksize:, i * diag_blocksize : (i + 1) * diag_blocksize]
            @ L[-arrow_blocksize:, i * diag_blocksize : (i + 1) * diag_blocksize].T
        )

    # L_{ndb, ndb} = chol(A_{ndb, ndb})
    L[
        -diag_blocksize - arrow_blocksize : -arrow_blocksize,
        -diag_blocksize - arrow_blocksize : -arrow_blocksize,
    ] = la.cholesky(
        L[
            -diag_blocksize - arrow_blocksize : -arrow_blocksize,
            -diag_blocksize - arrow_blocksize : -arrow_blocksize,
        ]
    ).T

    # L_{ndb+1, nbd} = A_{ndb+1, nbd} @ L_{ndb, ndb}^{-T}
    L[-arrow_blocksize:, -diag_blocksize - arrow_blocksize : -arrow_blocksize] = (
        L[-arrow_blocksize:, -diag_blocksize - arrow_blocksize : -arrow_blocksize]
        @ la.solve_triangular(
            L[
                -diag_blocksize - arrow_blocksize : -arrow_blocksize,
                -diag_blocksize - arrow_blocksize : -arrow_blocksize,
            ],
            np.eye(diag_blocksize),
            lower=True,
        ).T
    )

    # A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, ndb} @ L_{ndb+1, ndb}^{T}
    L[-arrow_blocksize:, -arrow_blocksize:] = (
        L[-arrow_blocksize:, -arrow_blocksize:]
        - L[-arrow_blocksize:, -diag_blocksize - arrow_blocksize : -arrow_blocksize]
        @ L[-arrow_blocksize:, -diag_blocksize - arrow_blocksize : -arrow_blocksize].T
    )

    # L_{ndb+1, ndb+1} = chol(A_{ndb+1, ndb+1})
    L[-arrow_blocksize:, -arrow_blocksize:] = la.cholesky(
        L[-arrow_blocksize:, -arrow_blocksize:]
    ).T
    
    # zero out upper triangular part
    L[:] = L * np.tri(*L.shape, k=0)    

    return L
