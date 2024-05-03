# Copyright 2023-2024 ETH Zurich and USI. All rights reserved.

import numpy as np


# -----------------------------------------------
# Section: Zeros to Shape
# -----------------------------------------------


def zeros_to_block_tridiagonal_shape(
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


def zeros_to_block_tridiagonal_arrowhead_shape(
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


def zeros_to_blocks_banded_shape(
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


def zeros_to_blocks_banded_arrowhead_shape(
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


# -----------------------------------------------
# Section: Make diagonally dominante
# -----------------------------------------------


def make_dense_matrix_diagonally_dominante(
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


# -----------------------------------------------
# Section: Make matrix symmetric
# -----------------------------------------------


def make_dense_matrix_symmetric(
    A: np.ndarray,
) -> np.ndarray:
    """Make a dense matrix symmetric.

    Parameters
    ----------
    A : np.ndarray
        Input matrix.

    Returns
    -------
    A : np.ndarray
        Symmetric matrix.
    """

    A = 0.5 * (A + A.T)

    return A


# -----------------------------------------------
# Section: Convert from arays to dense
# -----------------------------------------------


def convert_block_tridiagonal_arrays_to_dense(
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
        A[i * blocksize : (i + 1) * blocksize, i * blocksize : (i + 1) * blocksize] = (
            A_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize]
        )
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


def convert_block_tridiagonal_arrowhead_arrays_to_dense(
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
    A_lower_diagonal_blocks: np.ndarray
        Lower diagonal blocks of the input matrix.
    A_upper_diagonal_blocks: np.ndarray
        Upper diagonal blocks of the input matrix.
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


def convert_blocks_banded_arrays_to_dense():
    pass


def convert_blocks_banded_arrowhead_arrays_to_dense():
    pass
