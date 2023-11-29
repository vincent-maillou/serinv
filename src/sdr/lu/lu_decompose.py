"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Lisa Gaedke-Merzhaeuser  (lisa.gaedke.merzhaeuser@usi.ch)
@date: 2023-11

Contains the lu selected decompositions routines.

Copyright 2023 ETH Zurich and USI. All rights reserved.
"""


import numpy as np
import scipy.linalg as la



def lu_dcmp_tridiag(
    A: np.ndarray,
    blocksize: int,
) -> np.ndarray:
    """ Perform the LU factorization of a block tridiagonal matrix. The 
    matrix is assumed to be non-singular.

    Parameters
    ----------
    A : np.ndarray
        Input matrix to decompose.
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
    

    nblocks = A.shape[0] // blocksize
    for i in range(nblocks-1):
        # L_{i, i}, U_{i, i} = lu_dcmp(A_{i, i})
        L[i*blocksize:(i+1)*blocksize, i*blocksize:(i+1)*blocksize], U[i*blocksize:(i+1)*blocksize, i*blocksize:(i+1)*blocksize] = la.lu(A[i*blocksize:(i+1)*blocksize, i*blocksize:(i+1)*blocksize], permute_l=True)

        # L_{i+1, i} = A_{i+1, i} @ U{i, i+1}^{-1}
        L[(i+1)*blocksize:(i+2)*blocksize, i*blocksize:(i+1)*blocksize] = A[(i+1)*blocksize:(i+2)*blocksize, i*blocksize:(i+1)*blocksize] @ la.solve_triangular(U[i*blocksize:(i+1)*blocksize, i*blocksize:(i+1)*blocksize], np.eye(blocksize), lower=False)

        # L_{i, i+1} = L{i+1, i}^{-1} @ A_{i, i+1} 
        U[i*blocksize:(i+1)*blocksize, (i+1)*blocksize:(i+2)*blocksize] =  la.solve_triangular(L[i*blocksize:(i+1)*blocksize, i*blocksize:(i+1)*blocksize], np.eye(blocksize), lower=True) @ A[i*blocksize:(i+1)*blocksize, (i+1)*blocksize:(i+2)*blocksize]

        # A_{i+1, i+1} = A_{i+1, i+1} - L_{i+1, i} @ U_{i, i+1}
        A[(i+1)*blocksize:(i+2)*blocksize, (i+1)*blocksize:(i+2)*blocksize] = A[(i+1)*blocksize:(i+2)*blocksize, (i+1)*blocksize:(i+2)*blocksize] - L[(i+1)*blocksize:(i+2)*blocksize, i*blocksize:(i+1)*blocksize] @ U[i*blocksize:(i+1)*blocksize, (i+1)*blocksize:(i+2)*blocksize]

    # L_{nblocks, nblocks}, U_{nblocks, nblocks} = lu_dcmp(A_{nblocks, nblocks})
    L[-blocksize:, -blocksize:], U[-blocksize:, -blocksize:] = la.lu(A[-blocksize:, -blocksize:], permute_l=True)

    return L, U