"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Lisa Gaedke-Merzhaeuser  (lisa.gaedke.merzhaeuser@usi.ch)
@date: 2023-11

Matrix transformations routines.

Copyright 2023 ETH Zurich and USI. All rights reserved.
"""

import numpy as np


def cut_to_blocktridiag(
    A: np.ndarray,
    blocksize: int,
) -> np.ndarray:
    """Cut a matrix to a block tridiagonal matrix.

    Parameters
    ----------
    A : np.ndarray
        Matrix to cut.
    blocksize : int
        Size of the blocks.

    Returns
    -------
    A : np.ndarray
        Block tridiagonal matrix.
    """

    matrice_size = A.shape[0]
    nblocks = matrice_size // blocksize

    A_cut = np.zeros((matrice_size, matrice_size))

    for i in range(nblocks):
        A_cut[i * blocksize : (i + 1) * blocksize, i * blocksize : (i + 1) * blocksize] = A[
            i * blocksize : (i + 1) * blocksize, i * blocksize : (i + 1) * blocksize
        ]
        if i > 0:
            A_cut[i * blocksize : (i + 1) * blocksize, (i - 1) * blocksize : i * blocksize] = A[
                i * blocksize : (i + 1) * blocksize, (i - 1) * blocksize : i * blocksize
            ]
            A_cut[(i - 1) * blocksize : i * blocksize, i * blocksize : (i + 1) * blocksize] = A[
                (i - 1) * blocksize : i * blocksize, i * blocksize : (i + 1) * blocksize
            ]

    return A_cut


def cut_to_blocktridiag_arrowhead(
    A: np.ndarray,
    diag_blocksize: int,
    arrow_blocksize: int,
) -> np.ndarray:
    """Cut a matrix to a block tridiagonal arrowhead matrix.

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
    A : np.ndarray
        Block tridiagonal arrowhead matrix.
    """

    matrice_size = A.shape[0]
    nblocks = ((matrice_size - arrow_blocksize) // diag_blocksize) + 1

    A_cut = np.zeros((matrice_size, matrice_size))

    for i in range(nblocks):
        if i < nblocks - 1:
            A_cut[i * diag_blocksize : (i + 1) * diag_blocksize, i * diag_blocksize : (i + 1) * diag_blocksize] = A[
                i * diag_blocksize : (i + 1) * diag_blocksize, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            if i > 0:
                A_cut[i * diag_blocksize : (i + 1) * diag_blocksize, (i - 1) * diag_blocksize : i * diag_blocksize] = A[
                    i * diag_blocksize : (i + 1) * diag_blocksize, (i - 1) * diag_blocksize : i * diag_blocksize
                ]
                A_cut[(i - 1) * diag_blocksize : i * diag_blocksize, i * diag_blocksize : (i + 1) * diag_blocksize] = A[
                    (i - 1) * diag_blocksize : i * diag_blocksize, i * diag_blocksize : (i + 1) * diag_blocksize
                ]

            A_cut[
                (nblocks - 1) * diag_blocksize : (nblocks - 1) * diag_blocksize + arrow_blocksize,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ] = A[
                (nblocks - 1) * diag_blocksize : (nblocks - 1) * diag_blocksize + arrow_blocksize,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ]
            A_cut[
                i * diag_blocksize : (i + 1) * diag_blocksize,
                (nblocks - 1) * diag_blocksize : (nblocks - 1) * diag_blocksize + arrow_blocksize,
            ] = A[
                i * diag_blocksize : (i + 1) * diag_blocksize,
                (nblocks - 1) * diag_blocksize : (nblocks - 1) * diag_blocksize + arrow_blocksize,
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


def cut_to_blockndiags(
    A: np.ndarray,
    ndiags: int,
    blocksize: int,
):
    """Cut a matrix to a block ndiags matrix.

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
    A : np.ndarray
        Block ndiags matrix.
    """

    matrice_size = A.shape[0]
    nblocks = matrice_size // blocksize

    A_cut = np.zeros((matrice_size, matrice_size))

    for i in range(nblocks):
        for j in range((ndiags // 2) + 1):
            if i + j < nblocks:
                A_cut[i * blocksize : (i + 1) * blocksize, (i + j) * blocksize : (i + j + 1) * blocksize] = A[
                    i * blocksize : (i + 1) * blocksize, (i + j) * blocksize : (i + j + 1) * blocksize
                ]
                A_cut[(i + j) * blocksize : (i + j + 1) * blocksize, i * blocksize : (i + 1) * blocksize] = A[
                    (i + j) * blocksize : (i + j + 1) * blocksize, i * blocksize : (i + 1) * blocksize
                ]

    return A_cut


def cut_to_blockndiags_arrowhead(
    A: np.ndarray,
    ndiags: int,
    diag_blocksize: int,
    arrow_blocksize: int,
):
    """Cut a matrix to a block ndiags arrowhead matrix.

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
    A : np.ndarray
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

    A_cut[-arrow_blocksize:, -arrow_blocksize:] = A[-arrow_blocksize:, -arrow_blocksize:]

    return A_cut
