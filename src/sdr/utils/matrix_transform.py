"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Lisa Gaedke-Merzhaeuser  (lisa.gaedke.merzhaeuser@usi.ch)
@date: 2023-11

Matrix transformations routines.

Copyright 2023-2024 ETH Zurich and USI. All rights reserved.
"""

import numpy as np


def cut_to_blocktridiag(
    A: np.ndarray,
    blocksize: int,
) -> np.ndarray:
    """Cut a dense matrix to a block tridiagonal matrix.

    Parameters
    ----------
    A : np.ndarray
        Matrix to cut.
    blocksize : int
        Size of the blocks.

    Returns
    -------
    A_cut : np.ndarray
        Block tridiagonal matrix.
    """

    matrice_size = A.shape[0]
    nblocks = matrice_size // blocksize

    A_cut = np.zeros((matrice_size, matrice_size))

    for i in range(nblocks):
        A_cut[
            i * blocksize : (i + 1) * blocksize, i * blocksize : (i + 1) * blocksize
        ] = A[i * blocksize : (i + 1) * blocksize, i * blocksize : (i + 1) * blocksize]
        if i > 0:
            A_cut[
                i * blocksize : (i + 1) * blocksize, (i - 1) * blocksize : i * blocksize
            ] = A[
                i * blocksize : (i + 1) * blocksize, (i - 1) * blocksize : i * blocksize
            ]
            A_cut[
                (i - 1) * blocksize : i * blocksize, i * blocksize : (i + 1) * blocksize
            ] = A[
                (i - 1) * blocksize : i * blocksize, i * blocksize : (i + 1) * blocksize
            ]

    return A_cut


def cut_to_blocktridiag_arrowhead(
    A: np.ndarray,
    diag_blocksize: int,
    arrow_blocksize: int,
) -> np.ndarray:
    """Cut a dense matrix to a block tridiagonal arrowhead matrix.

    Parameters
    ----------
    A : np.ndarray
        Matrix to cut.
    diag_blocksize : int
        Size of the diagonal blocks.
    arrow_blocksize : int
        Size of the arrow blocks.

    Returns
    -------
    A_cut : np.ndarray
        Block tridiagonal arrowhead matrix.
    """

    matrice_size = A.shape[0]
    nblocks = ((matrice_size - arrow_blocksize) // diag_blocksize) + 1

    A_cut = np.zeros((matrice_size, matrice_size))

    for i in range(nblocks):
        if i < nblocks - 1:
            A_cut[
                i * diag_blocksize : (i + 1) * diag_blocksize,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ] = A[
                i * diag_blocksize : (i + 1) * diag_blocksize,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ]
            if i > 0:
                A_cut[
                    i * diag_blocksize : (i + 1) * diag_blocksize,
                    (i - 1) * diag_blocksize : i * diag_blocksize,
                ] = A[
                    i * diag_blocksize : (i + 1) * diag_blocksize,
                    (i - 1) * diag_blocksize : i * diag_blocksize,
                ]
                A_cut[
                    (i - 1) * diag_blocksize : i * diag_blocksize,
                    i * diag_blocksize : (i + 1) * diag_blocksize,
                ] = A[
                    (i - 1) * diag_blocksize : i * diag_blocksize,
                    i * diag_blocksize : (i + 1) * diag_blocksize,
                ]

            A_cut[
                (nblocks - 1) * diag_blocksize : (nblocks - 1) * diag_blocksize
                + arrow_blocksize,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ] = A[
                (nblocks - 1) * diag_blocksize : (nblocks - 1) * diag_blocksize
                + arrow_blocksize,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ]
            A_cut[
                i * diag_blocksize : (i + 1) * diag_blocksize,
                (nblocks - 1) * diag_blocksize : (nblocks - 1) * diag_blocksize
                + arrow_blocksize,
            ] = A[
                i * diag_blocksize : (i + 1) * diag_blocksize,
                (nblocks - 1) * diag_blocksize : (nblocks - 1) * diag_blocksize
                + arrow_blocksize,
            ]

        else:
            A_cut[
                i * diag_blocksize : i * diag_blocksize + arrow_blocksize,
                i * diag_blocksize : i * diag_blocksize + arrow_blocksize,
            ] = A[
                i * diag_blocksize : i * diag_blocksize + arrow_blocksize,
                i * diag_blocksize : i * diag_blocksize + arrow_blocksize,
            ]

    return A_cut


def cut_to_block_ndiags(
    A: np.ndarray,
    ndiags: int,
    blocksize: int,
) -> np.ndarray:
    """Cut a dense matrix to a block ndiags matrix.

    Parameters
    ----------
    A : np.ndarray
        Matrix to cut.
    ndiags : int
        Number of diagonals.
    blocksize : int
        Size of the blocks.

    Returns
    -------
    A_cut : np.ndarray
        Block ndiags matrix.
    """

    matrice_size = A.shape[0]
    nblocks = matrice_size // blocksize

    A_cut = np.zeros((matrice_size, matrice_size))

    for i in range(nblocks):
        for j in range((ndiags // 2) + 1):
            if i + j < nblocks:
                A_cut[
                    i * blocksize : (i + 1) * blocksize,
                    (i + j) * blocksize : (i + j + 1) * blocksize,
                ] = A[
                    i * blocksize : (i + 1) * blocksize,
                    (i + j) * blocksize : (i + j + 1) * blocksize,
                ]
                A_cut[
                    (i + j) * blocksize : (i + j + 1) * blocksize,
                    i * blocksize : (i + 1) * blocksize,
                ] = A[
                    (i + j) * blocksize : (i + j + 1) * blocksize,
                    i * blocksize : (i + 1) * blocksize,
                ]

    return A_cut


def cut_to_blockndiags_arrowhead(
    A: np.ndarray,
    ndiags: int,
    diag_blocksize: int,
    arrow_blocksize: int,
) -> np.ndarray:
    """Cut a dense matrix to a block ndiags arrowhead matrix.

    Parameters
    ----------
    A : np.ndarray
        Matrix to cut.
    ndiags : int
        Number of diagonals.
    diag_blocksize : int
        Size of the diagonal blocks.
    arrow_blocksize : int
        Size of the arrow blocks.

    Returns
    -------
    A_cut : np.ndarray
        Block ndiags arrowhead matrix.
    """

    matrice_size = A.shape[0]
    n_diag_blocks = (matrice_size - arrow_blocksize) // diag_blocksize

    A_cut = np.zeros((matrice_size, matrice_size))

    for i in range(n_diag_blocks):
        for j in range((ndiags // 2) + 1):
            if i + j < n_diag_blocks:
                A_cut[
                    i * diag_blocksize : (i + 1) * diag_blocksize,
                    (i + j) * diag_blocksize : (i + j + 1) * diag_blocksize,
                ] = A[
                    i * diag_blocksize : (i + 1) * diag_blocksize,
                    (i + j) * diag_blocksize : (i + j + 1) * diag_blocksize,
                ]
                A_cut[
                    (i + j) * diag_blocksize : (i + j + 1) * diag_blocksize,
                    i * diag_blocksize : (i + 1) * diag_blocksize,
                ] = A[
                    (i + j) * diag_blocksize : (i + j + 1) * diag_blocksize,
                    i * diag_blocksize : (i + 1) * diag_blocksize,
                ]

        A_cut[-arrow_blocksize:, i * diag_blocksize : (i + 1) * diag_blocksize] = A[
            -arrow_blocksize:, i * diag_blocksize : (i + 1) * diag_blocksize
        ]
        A_cut[i * diag_blocksize : (i + 1) * diag_blocksize, -arrow_blocksize:] = A[
            i * diag_blocksize : (i + 1) * diag_blocksize, -arrow_blocksize:
        ]

    A_cut[-arrow_blocksize:, -arrow_blocksize:] = A[
        -arrow_blocksize:, -arrow_blocksize:
    ]

    return A_cut


def make_symmetric_tridiagonal_arrays(
    A_diagonal_blocks: np.ndarray,
    A_lower_diagonal_blocks: np.ndarray,
    A_upper_diagonal_blocks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Make a tridiagonal matrix, passed as arrays, symmetric.

    Parameters
    ----------
    A_diagonal_blocks: np.ndarray
        Diagonal blocks of the input matrix.
    A_upper_diagonal_blocks: np.ndarray
        Upper diagonal blocks of the input matrix.
    A_lower_diagonal_blocks: np.ndarray
        Lower diagonal blocks of the input matrix.

    Returns
    -------
    A_diagonal_blocks : np.ndarray
        Diagonal blocks made symmetric.
    A_lower_diagonal_blocks : np.ndarray
        Lower diagonal blocks made symmetric.
    A_upper_diagonal_blocks : np.ndarray
        Upper diagonal blocks made symmetric.
    """

    blocksize = A_diagonal_blocks.shape[0]
    nblocks = A_diagonal_blocks.shape[1] // blocksize

    for i in range(nblocks):
        A_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize] += A_diagonal_blocks[
            :, i * blocksize : (i + 1) * blocksize
        ].T
        if i < nblocks - 1:
            A_lower_diagonal_blocks[
                :, i * blocksize : (i + 1) * blocksize
            ] = A_upper_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize].T

    return (A_diagonal_blocks, A_lower_diagonal_blocks, A_upper_diagonal_blocks)


def make_diagonally_dominante_dense(
    A: np.ndarray,
) -> np.ndarray:
    """Make a dense matrix diagonally dominant.

    Parameters
    ----------
    A : np.ndarray
        Input matrix.

    Returns
    -------
    A : np.ndarray
        Diagonally dominant matrix.
    """

    A = A + np.diag(np.sum(np.abs(A), axis=1))

    return A


def make_diagonally_dominante_tridiagonal_arrays(
    A_diagonal_blocks: np.ndarray,
    A_upper_diagonal_blocks: np.ndarray,
    A_lower_diagonal_blocks: np.ndarray,
) -> np.ndarray:
    """Make a tridiagonal matrix, passed as arrays, diagonally dominant.

    Parameters
    ----------
    A_diagonal_blocks: np.ndarray
        Diagonal blocks of the input matrix.
    A_upper_diagonal_blocks: np.ndarray
        Upper diagonal blocks of the input matrix.
    A_lower_diagonal_blocks: np.ndarray
        Lower diagonal blocks of the input matrix.

    Returns
    -------
    A_diagonal_blocks : np.ndarray
        Diagonal blocks made diagonally dominant.
    """

    blocksize = A_diagonal_blocks.shape[0]
    nblocks = A_diagonal_blocks.shape[1] // blocksize

    for i in range(0, nblocks):
        A_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize] += np.diag(
            np.sum(
                np.abs(A_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize]),
                axis=1,
            )
        )
        if i > 0:
            A_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize] += np.diag(
                np.sum(
                    np.abs(
                        A_upper_diagonal_blocks[:, (i - 1) * blocksize : i * blocksize]
                    ),
                    axis=1,
                )
            )
        if i < nblocks - 1:
            A_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize] += np.diag(
                np.sum(
                    np.abs(
                        A_lower_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize]
                    ),
                    axis=1,
                )
            )

    return A_diagonal_blocks


def make_diagonally_dominante_tridiagonal_arrowhead_arrays(
    A_diagonal_blocks: np.ndarray,
    A_lower_diagonal_blocks: np.ndarray,
    A_upper_diagonal_blocks: np.ndarray,
    A_arrow_bottom_blocks: np.ndarray,
    A_arrow_right_blocks: np.ndarray,
    A_arrow_tip_block: np.ndarray,
) -> np.ndarray:
    """Make a tridiagonal arrowhead matrix, passed as arrays, diagonally dominant.

    Parameters
    ----------
    A_diagonal_blocks: np.ndarray
        Diagonal blocks of the input matrix.
    A_upper_diagonal_blocks: np.ndarray
        Upper diagonal blocks of the input matrix.
    A_lower_diagonal_blocks: np.ndarray
        Lower diagonal blocks of the input matrix.
    A_arrow_bottom_blocks: np.ndarray
        Arrow bottom blocks of the input matrix.
    A_arrow_right_blocks: np.ndarray
        Arrow right blocks of the input matrix.
    A_arrow_tip_block: np.ndarray
        Arrow tip block of the input matrix.

    Returns
    -------
    A_diagonal_blocks : np.ndarray
        Diagonal blocks made diagonally dominant.
    """

    diag_blocksize = A_diagonal_blocks.shape[0]

    n_diag_blocks = A_diagonal_blocks.shape[1] // diag_blocksize
    for i in range(0, n_diag_blocks):
        A_diagonal_blocks[:, i * diag_blocksize : (i + 1) * diag_blocksize] += np.diag(
            np.sum(
                np.abs(
                    A_diagonal_blocks[:, i * diag_blocksize : (i + 1) * diag_blocksize]
                ),
                axis=1,
            )
        )
        if i < n_diag_blocks - 1:
            A_diagonal_blocks[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ] += np.diag(
                np.sum(
                    np.abs(
                        A_upper_diagonal_blocks[
                            :, i * diag_blocksize : (i + 1) * diag_blocksize
                        ]
                    ),
                    axis=1,
                )
            )
            A_diagonal_blocks[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ] += np.diag(
                np.sum(
                    np.abs(
                        A_arrow_right_blocks[
                            :, i * diag_blocksize : (i + 1) * diag_blocksize
                        ]
                    ),
                    axis=1,
                )
            )
        if i > 0:
            A_diagonal_blocks[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ] += np.diag(
                np.sum(
                    np.abs(
                        A_lower_diagonal_blocks[
                            :, i * diag_blocksize : (i + 1) * diag_blocksize
                        ]
                    ),
                    axis=1,
                )
            )

        A_arrow_tip_block += np.diag(np.sum(np.abs(A_arrow_tip_block), axis=1))
        A_arrow_tip_block += np.diag(np.sum(np.abs(A_arrow_bottom_blocks), axis=1))

    return A_diagonal_blocks


def from_tridiagonal_arrays_to_dense(
    A_diagonal_blocks: np.ndarray,
    A_lower_diagonal_blocks: np.ndarray,
    A_upper_diagonal_blocks: np.ndarray,
) -> np.ndarray:
    """Convert a tridiagonal matrix, passed as arrays, to a dense matrix.

    Parameters
    ----------
    A_diagonal_blocks: np.ndarray
        Diagonal blocks of the input matrix.
    A_upper_diagonal_blocks: np.ndarray
        Upper diagonal blocks of the input matrix.
    A_lower_diagonal_blocks: np.ndarray
        Lower diagonal blocks of the input matrix.

    Returns
    -------
    A : np.ndarray
        Dense matrix.
    """

    blocksize = A_diagonal_blocks.shape[0]
    nblocks = A_diagonal_blocks.shape[1] // blocksize

    A = np.zeros((blocksize * nblocks, blocksize * nblocks))

    for i in range(nblocks):
        A[
            i * blocksize : (i + 1) * blocksize, i * blocksize : (i + 1) * blocksize
        ] = A_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize]
        if i < (nblocks - 1):
            A[
                i * blocksize : (i + 1) * blocksize,
                (i + 1) * blocksize : (i + 2) * blocksize,
            ] = A_upper_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize]
            A[
                (i + 1) * blocksize : (i + 2) * blocksize,
                i * blocksize : (i + 1) * blocksize,
            ] = A_lower_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize]

    return A


def from_dense_to_tridiagonal_arrays(
    A: np.ndarray,
    blocksize: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract the tridiagonal blocks from a dense matrix.

    Parameters
    ----------
    A : np.ndarray
        Dense matrix.
    blocksize : int
        Size of the blocks.

    Returns
    -------
    A_diagonal_blocks: np.ndarray
        Diagonal blocks of the tridiagonal matrix.
    A_upper_diagonal_blocks: np.ndarray
        Upper diagonal blocks of the tridiagonal matrix.
    A_lower_diagonal_blocks: np.ndarray
        Lower diagonal blocks of the tridiagonal matrix.
    """

    matrice_size = A.shape[0]
    nblocks = matrice_size // blocksize

    A_diagonal_blocks = np.empty((blocksize, nblocks * blocksize))
    A_upper_diagonal_blocks = np.empty((blocksize, (nblocks - 1) * blocksize))
    A_lower_diagonal_blocks = np.empty((blocksize, (nblocks - 1) * blocksize))

    for i in range(nblocks):
        A_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize] = A[
            i * blocksize : (i + 1) * blocksize, i * blocksize : (i + 1) * blocksize
        ]
        if i < (nblocks - 1):
            A_upper_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize] = A[
                i * blocksize : (i + 1) * blocksize,
                (i + 1) * blocksize : (i + 2) * blocksize,
            ]
            A_lower_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize] = A[
                (i + 1) * blocksize : (i + 2) * blocksize,
                i * blocksize : (i + 1) * blocksize,
            ]

    return (A_diagonal_blocks, A_lower_diagonal_blocks, A_upper_diagonal_blocks)


def from_arrowhead_arrays_to_dense(
    A_diagonal_blocks: np.ndarray,
    A_lower_diagonal_blocks: np.ndarray,
    A_upper_diagonal_blocks: np.ndarray,
    A_arrow_bottom_blocks: np.ndarray,
    A_arrow_right_blocks: np.ndarray,
    A_arrow_tip_block: np.ndarray,
) -> np.ndarray:
    """Convert a tridiagonal arrowhead matrix, passed as arrays, to a dense matrix.

    Parameters
    ----------
    A_diagonal_blocks: np.ndarray
        Diagonal blocks of the input matrix.
    A_upper_diagonal_blocks: np.ndarray
        Upper diagonal blocks of the input matrix.
    A_lower_diagonal_blocks: np.ndarray
        Lower diagonal blocks of the input matrix.
    A_arrow_bottom_blocks: np.ndarray
        Arrow bottom blocks of the input matrix.
    A_arrow_right_blocks: np.ndarray
        Arrow right blocks of the input matrix.
    A_arrow_tip_block: np.ndarray
        Arrow tip block of the input matrix.

    Returns
    -------
    A : np.ndarray
        Dense matrix.
    """

    diag_blocksize = A_diagonal_blocks.shape[0]
    arrowhead_blocksize = A_arrow_bottom_blocks.shape[0]

    n_diag_blocks = A_diagonal_blocks.shape[1] // diag_blocksize

    A = np.zeros(
        (
            n_diag_blocks * diag_blocksize + arrowhead_blocksize,
            n_diag_blocks * diag_blocksize + arrowhead_blocksize,
        )
    )

    for i in range(0, n_diag_blocks):
        A[
            i * diag_blocksize : (i + 1) * diag_blocksize,
            i * diag_blocksize : (i + 1) * diag_blocksize,
        ] = A_diagonal_blocks[:, i * diag_blocksize : (i + 1) * diag_blocksize]
        if i < n_diag_blocks - 1:
            A[
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ] = A_lower_diagonal_blocks[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            A[
                i * diag_blocksize : (i + 1) * diag_blocksize,
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
            ] = A_upper_diagonal_blocks[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]

    A[-arrowhead_blocksize:, :-arrowhead_blocksize] = A_arrow_bottom_blocks[:, :]
    A[:-arrowhead_blocksize, -arrowhead_blocksize:] = A_arrow_right_blocks[:, :]
    A[-arrowhead_blocksize:, -arrowhead_blocksize:] = A_arrow_tip_block[:, :]

    return A


def from_dense_to_arrowhead_arrays(
    A: np.ndarray,
    diag_blocksize: int,
    arrow_blocksize: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract the arrowhead blocks from a dense matrix.

    Parameters
    ----------
    A : np.ndarray
        Dense matrix.
    diag_blocksize : int
        Size of the diagonal blocks.
    arrow_blocksize : int
        Size of the arrow blocks.

    Returns
    -------
    A_diagonal_blocks: np.ndarray
        Diagonal blocks of the tridiagonal matrix.
    A_lower_diagonal_blocks: np.ndarray
        Lower diagonal blocks of the tridiagonal matrix.
    A_upper_diagonal_blocks: np.ndarray
        Upper diagonal blocks of the tridiagonal matrix.
    A_arrow_bottom_blocks: np.ndarray
        Arrow bottom blocks of the tridiagonal matrix.
    A_arrow_right_blocks: np.ndarray
        Arrow right blocks of the tridiagonal matrix.
    A_arrow_tip_block: np.ndarray
        Arrow tip block of the tridiagonal matrix.
    """

    n_diag_blocks = (A.shape[0] - arrow_blocksize) // diag_blocksize

    A_diagonal_blocks = np.empty((diag_blocksize, n_diag_blocks * diag_blocksize))
    A_lower_diagonal_blocks = np.empty(
        (diag_blocksize, (n_diag_blocks - 1) * diag_blocksize)
    )
    A_upper_diagonal_blocks = np.empty(
        (diag_blocksize, (n_diag_blocks - 1) * diag_blocksize)
    )
    A_arrow_bottom_blocks = np.empty((arrow_blocksize, n_diag_blocks * diag_blocksize))
    A_arrow_right_blocks = np.empty((arrow_blocksize, n_diag_blocks * diag_blocksize))
    A_arrow_tip_block = np.empty((arrow_blocksize, arrow_blocksize))

    for i in range(0, n_diag_blocks):
        A_diagonal_blocks[:, i * diag_blocksize : (i + 1) * diag_blocksize] = A[
            i * diag_blocksize : (i + 1) * diag_blocksize,
            i * diag_blocksize : (i + 1) * diag_blocksize,
        ]
        if i < n_diag_blocks - 1:
            A_lower_diagonal_blocks[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ] = A[
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ]
            A_upper_diagonal_blocks[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ] = A[
                i * diag_blocksize : (i + 1) * diag_blocksize,
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
            ]

    A_arrow_bottom_blocks = A[-arrow_blocksize:, :-arrow_blocksize]
    A_arrow_right_blocks = A[:-arrow_blocksize, -arrow_blocksize:]
    A_arrow_tip_block = A[-arrow_blocksize:, -arrow_blocksize:]

    return (
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_upper_diagonal_blocks,
        A_arrow_bottom_blocks,
        A_arrow_right_blocks,
        A_arrow_tip_block,
    )
