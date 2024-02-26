"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Lisa Gaedke-Merzhaeuser  (lisa.gaedke.merzhaeuser@usi.ch)
@date: 2023-11

Matrix generations routines.

Copyright 2023 ETH Zurich and USI. All rights reserved.
"""

from sdr.utils import matrix_transform as mt

import numpy as np


def generate_tridiag_dense(
    nblocks: int,
    blocksize: int,    
    symmetric: bool = False,  
    diagonal_dominant: bool = False,
    seed: int = None,
) -> np.ndarray:
    """ Generate a block tridiagonal matrix, returned as dense matrix.

    Parameters
    ----------
    nblocks : int
        Number of diagonal blocks.
    blocksize : int
        Size of the blocks.
    symmetric : bool, optional
        If True, the matrix will be symmetric.
    seed : int, optional
        Seed for the random number generator.
        
    Returns
    -------
    A : np.ndarray
        Block tridiagonal matrix.
    """

    if seed is not None:
        np.random.seed(seed)

    matrice_size = nblocks*blocksize

    A = np.zeros((matrice_size, matrice_size))

    for i in range(nblocks):
        A[i*blocksize:(i+1)*blocksize, i*blocksize:(i+1)*blocksize] = np.random.rand(blocksize, blocksize)
        if i > 0:
            A[i*blocksize:(i+1)*blocksize, (i-1)*blocksize:i*blocksize] = np.random.rand(blocksize, blocksize)
            A[(i-1)*blocksize:i*blocksize, i*blocksize:(i+1)*blocksize] = np.random.rand(blocksize, blocksize)

    if symmetric:
        A = A + A.T

    if diagonal_dominant:
        A = mt.make_diagonally_dominante_dense(A)

    return A  


def generate_tridiag_array(
    nblocks: int,
    blocksize: int,    
    symmetric: bool = False,  
    diagonal_dominant: bool = False,
    seed: int = None,
) -> [np.ndarray, np.ndarray, np.ndarray]:
    """ Generate a block tridiagonal matrix returned as three arrays.

    Parameters
    ----------
    nblocks : int
        Number of diagonal blocks.
    blocksize : int
        Size of the blocks.
    symmetric : bool, optional
        If True, the matrix will be symmetric.
    seed : int, optional
        Seed for the random number generator.
        
    Returns
    -------
    A : np.ndarray
        Block tridiagonal matrix.
    """

    if seed is not None:
        np.random.seed(seed)

    A_diagonal_blocks = np.empty((blocksize, nblocks*blocksize))
    A_upper_diagonal_blocks = np.empty((blocksize, (nblocks-1)*blocksize))
    A_lower_diagonal_blocks = np.empty((blocksize, (nblocks-1)*blocksize))

    for i in range(nblocks):
        A_diagonal_blocks[:, i*blocksize:(i+1)*blocksize] = np.random.rand(blocksize, blocksize)
        if i > 0:
            A_upper_diagonal_blocks[:, (i-1)*blocksize:i*blocksize] = np.random.rand(blocksize, blocksize)
        if i < nblocks-1:
            A_lower_diagonal_blocks[:, i*blocksize:(i+1)*blocksize] = np.random.rand(blocksize, blocksize)

    if symmetric:
        (
            A_diagonal_blocks, 
            A_lower_diagonal_blocks,
            A_upper_diagonal_blocks
        ) = mt.make_symmetric_tridiagonal_arrays(A_diagonal_blocks, A_lower_diagonal_blocks, A_upper_diagonal_blocks)

    if diagonal_dominant:
        A_diagonal_blocks = mt.make_diagonally_dominante_tridiagonal_arrays(A_diagonal_blocks, A_upper_diagonal_blocks, A_lower_diagonal_blocks)

    return A_diagonal_blocks, A_upper_diagonal_blocks, A_lower_diagonal_blocks   


def generate_block_ndiags(
    nblocks: int,
    ndiags: int,
    blocksize: int,    
    symmetric: bool = False,  
    diagonal_dominant: bool = False,
    seed: int = None,
) -> np.ndarray:
    """ Generate a block n-diagonals matrix.

    Parameters
    ----------
    nblocks : int
        Number of diagonal blocks.
    ndiags : int
        Number of diagonals.
    blocksize : int
        Size of the blocks.
    symmetric : bool, optional
        If True, the matrix will be symmetric.
    seed : int, optional
        Seed for the random number generator.
        
    Returns
    -------
    A : np.ndarray
        Block n-diagonals matrix.
    """

    if seed is not None:
        np.random.seed(seed)

    if (ndiags+1)/2 > nblocks:
        raise ValueError("(ndiags+1)/2 must be smaller or equal to nblocks")
    
    if ndiags % 2 == 0:
        raise ValueError("ndiags must be odd")


    matrice_size = nblocks*blocksize

    A = np.zeros((matrice_size, matrice_size))

    for i in range(nblocks):
        A[i*blocksize:(i+1)*blocksize, i*blocksize:(i+1)*blocksize] = np.random.rand(blocksize, blocksize)
        for j in range(1, (ndiags+1)//2):
            if i+j < nblocks:
                A[i*blocksize:(i+1)*blocksize, (i+j)*blocksize:(i+j+1)*blocksize] = np.random.rand(blocksize, blocksize)
                A[(i+j)*blocksize:(i+j+1)*blocksize, i*blocksize:(i+1)*blocksize] = np.random.rand(blocksize, blocksize)
            if i-j >= 0:
                A[i*blocksize:(i+1)*blocksize, (i-j)*blocksize:(i-j+1)*blocksize] = np.random.rand(blocksize, blocksize)
                A[(i-j)*blocksize:(i-j+1)*blocksize, i*blocksize:(i+1)*blocksize] = np.random.rand(blocksize, blocksize)

    if symmetric:
        A = A + A.T

    if diagonal_dominant:
        A = mt.make_diagonally_dominante_dense(A)

    return A        



def generate_tridiag_arrowhead_dense(
    nblocks: int,
    diag_blocksize: int,    
    arrow_blocksize: int,    
    symmetric: bool = False,
    diagonal_dominant: bool = False,  
    seed: int = None,
) -> np.ndarray:
    """ Generate a block tridiagonal arrowhead matrix.

    Parameters
    ----------
    nblocks : int
        Number of diagonal blocks.
    diag_blocksize : int
        Size of the diagonal blocks.
    arrow_blocksize : int
        Size of the arrowhead blocks. These blocks will be of sizes: 
        (arrow_blocksize*diag_blocksize).
    symmetric : bool, optional
        If True, the matrix will be symmetric.
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    A : np.ndarray
        Block tridiagonal arrowhead matrix.
    """

    if seed is not None:
        np.random.seed(seed)

    matrice_size = (nblocks-1)*diag_blocksize+arrow_blocksize

    A = np.zeros((matrice_size, matrice_size))

    for i in range(nblocks):
        if i < nblocks-1:
            A[i*diag_blocksize:(i+1)*diag_blocksize, i*diag_blocksize:(i+1)*diag_blocksize] = np.random.rand(diag_blocksize, diag_blocksize)
            if i > 0:
                A[i*diag_blocksize:(i+1)*diag_blocksize, (i-1)*diag_blocksize:i*diag_blocksize] = np.random.rand(diag_blocksize, diag_blocksize)
                A[(i-1)*diag_blocksize:i*diag_blocksize, i*diag_blocksize:(i+1)*diag_blocksize] = np.random.rand(diag_blocksize, diag_blocksize)
            
            A[(nblocks-1)*diag_blocksize:(nblocks-1)*diag_blocksize+arrow_blocksize, i*diag_blocksize:(i+1)*diag_blocksize] = np.random.rand(arrow_blocksize, diag_blocksize)
            A[i*diag_blocksize:(i+1)*diag_blocksize, (nblocks-1)*diag_blocksize:(nblocks-1)*diag_blocksize+arrow_blocksize] = np.random.rand(diag_blocksize, arrow_blocksize)
        
        else:
            A[i*diag_blocksize:i*diag_blocksize+arrow_blocksize, i*diag_blocksize:i*diag_blocksize+arrow_blocksize] = np.random.rand(arrow_blocksize, arrow_blocksize)
    

    if symmetric:
        A = A + A.T

    if diagonal_dominant:
        A = mt.make_diagonally_dominante_dense(A)
        
    return A


def generate_tridiag_arrowhead_arrays(
    nblocks: int,
    diag_blocksize: int,    
    arrow_blocksize: int,    
    symmetric: bool = False,
    diagonal_dominant: bool = False,  
    seed: int = None,
) -> [np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ Generate a block tridiagonal arrowhead matrix.

    Parameters
    ----------
    nblocks : int
        Number of diagonal blocks.
    diag_blocksize : int
        Size of the diagonal blocks.
    arrow_blocksize : int
        Size of the arrowhead blocks. These blocks will be of sizes: 
        (arrow_blocksize*diag_blocksize).
    symmetric : bool, optional
        If True, the matrix will be symmetric.
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    TODO:docstring
    """

    if seed is not None:
        np.random.seed(seed)

    n_diag_blocks = nblocks - 1

    (
        A_diagonal_blocks, 
        A_lower_diagonal_blocks,
        A_upper_diagonal_blocks
    ) = generate_tridiag_array(n_diag_blocks, diag_blocksize, symmetric, diagonal_dominant, seed)
    
    A_arrow_bottom_blocks  = np.random.rand(arrow_blocksize, n_diag_blocks*diag_blocksize)
    A_arrow_right_blocks = np.random.rand(arrow_blocksize, n_diag_blocks*diag_blocksize)
    
    A_arrow_tip_block = np.random.rand(arrow_blocksize, arrow_blocksize)

    if symmetric:
        (
            A_diagonal_blocks, 
            A_lower_diagonal_blocks,
            A_upper_diagonal_blocks, 
        ) = mt.make_symmetric_tridiagonal_arrays(A_diagonal_blocks, A_lower_diagonal_blocks, A_upper_diagonal_blocks)
        
        for i in range(n_diag_blocks):
            A_arrow_bottom_blocks[:, i*diag_blocksize:(i+1)*diag_blocksize] = A_arrow_right_blocks[:, i*diag_blocksize:(i+1)*diag_blocksize].T

        A_arrow_tip_block += A_arrow_tip_block.T

    if diagonal_dominant:
        (
            A_diagonal_blocks
        ) = mt.make_diagonally_dominante_tridiagonal_arrowhead_arrays(
            A_diagonal_blocks, 
            A_lower_diagonal_blocks, 
            A_upper_diagonal_blocks, 
            A_arrow_bottom_blocks, 
            A_arrow_right_blocks, 
            A_arrow_tip_block
        )
        
    return A_diagonal_blocks, A_lower_diagonal_blocks, A_upper_diagonal_blocks, A_arrow_bottom_blocks, A_arrow_right_blocks, A_arrow_tip_block


def generate_ndiags_arrowhead(
    nblocks: int,
    ndiags: int,
    diag_blocksize: int,    
    arrow_blocksize: int,    
    symmetric: bool = False,
    diagonal_dominant: bool = False,  
    seed: int = None,
) -> np.ndarray:
    """ Generate a block tridiagonal arrowhead matrix.

    Parameters
    ----------
    nblocks : int
        Number of diagonal blocks.
    ndiags : int
        Number of diagonals.
    diag_blocksize : int
        Size of the diagonal blocks.
    arrow_blocksize : int
        Size of the arrowhead blocks. These blocks will be of sizes: 
        (arrow_blocksize*diag_blocksize).
    symmetric : bool, optional
        If True, the matrix will be symmetric.
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    A : np.ndarray
        Block tridiagonal arrowhead matrix.
    """

    if seed is not None:
        np.random.seed(seed)

    matrice_size = (nblocks-1)*diag_blocksize+arrow_blocksize

    A = np.zeros((matrice_size, matrice_size))

    for i in range(nblocks):
        if i < nblocks-1:
            A[i*diag_blocksize:(i+1)*diag_blocksize, i*diag_blocksize:(i+1)*diag_blocksize] = np.random.rand(diag_blocksize, diag_blocksize)
            for j in range(1, (ndiags+1)//2):
                if i+j < nblocks-1:
                    A[i*diag_blocksize:(i+1)*diag_blocksize, (i+j)*diag_blocksize:(i+j+1)*diag_blocksize] = np.random.rand(diag_blocksize, diag_blocksize)
                    A[(i+j)*diag_blocksize:(i+j+1)*diag_blocksize, i*diag_blocksize:(i+1)*diag_blocksize] = np.random.rand(diag_blocksize, diag_blocksize)
                if i-j >= 0:
                    A[i*diag_blocksize:(i+1)*diag_blocksize, (i-j)*diag_blocksize:(i-j+1)*diag_blocksize] = np.random.rand(diag_blocksize, diag_blocksize)
                    A[(i-j)*diag_blocksize:(i-j+1)*diag_blocksize, i*diag_blocksize:(i+1)*diag_blocksize] = np.random.rand(diag_blocksize, diag_blocksize)


            A[(nblocks-1)*diag_blocksize:(nblocks-1)*diag_blocksize+arrow_blocksize, i*diag_blocksize:(i+1)*diag_blocksize] = np.random.rand(arrow_blocksize, diag_blocksize)
            A[i*diag_blocksize:(i+1)*diag_blocksize, (nblocks-1)*diag_blocksize:(nblocks-1)*diag_blocksize+arrow_blocksize] = np.random.rand(diag_blocksize, arrow_blocksize)
        
        else:
            A[i*diag_blocksize:i*diag_blocksize+arrow_blocksize, i*diag_blocksize:i*diag_blocksize+arrow_blocksize] = np.random.rand(arrow_blocksize, arrow_blocksize)
    

    if symmetric:
        A = A + A.T

    if diagonal_dominant:
        A = mt.make_diagonally_dominante_dense(A)
        
    return A


