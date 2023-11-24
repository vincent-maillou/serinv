"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Lisa Gaedke-Merzhaeuser  (lisa.gaedke.merzhaeuser@usi.ch)
@date: 2023-11

Contains the cholesky decompositions routines.

Copyright 2023 ETH Zurich and USI. All rights reserved.
"""


import numpy as np
import scipy.linalg as la



def chol_dcmp_tridiag(
    A: np.ndarray,
    blocksize: int,
) -> np.ndarray:
    """ Perform the cholesky factorization of a block tridiagonal matrix. The 
    matrix is assumed to be symmetric positive definite.
    
    Parameters
    ----------
    A : np.ndarray
        Input matrix to decompose.
    
    Returns
    -------
    L : np.ndarray
        The cholesky factorization of the matrix.
    """

    L = np.zeros_like(A)

    # Initialize the first diagonal block L_{0, 0}
    L[0:blocksize, 0:blocksize] = la.cholesky(A[0:blocksize, 0:blocksize], lower=True)

    nblocks = A.shape[0] // blocksize
    for i in range(1, nblocks):
        L[i*blocksize:(i+1)*blocksize, (i-1)*blocksize:i*blocksize] = A[i*blocksize:(i+1)*blocksize, (i-1)*blocksize:i*blocksize] @ la.solve_triangular(L[(i-1)*blocksize:i*blocksize, (i-1)*blocksize:i*blocksize], np.eye(blocksize), lower=True).T
        L[i*blocksize:(i+1)*blocksize, i*blocksize:(i+1)*blocksize] = A[i*blocksize:(i+1)*blocksize, i*blocksize:(i+1)*blocksize] - L[i*blocksize:(i+1)*blocksize, (i-1)*blocksize:i*blocksize] @ L[i*blocksize:(i+1)*blocksize, (i-1)*blocksize:i*blocksize].T
        L[i*blocksize:(i+1)*blocksize, i*blocksize:(i+1)*blocksize] = la.cholesky(L[i*blocksize:(i+1)*blocksize, i*blocksize:(i+1)*blocksize]).T

    return L



def chol_dcmp_tridia_arrowhead(
    A: np.ndarray,
    blocksize: int,
) -> np.ndarray:
    """ Perform the cholesky factorization of a block tridiagonal arrowhead 
    matrix. The matrix is assumed to be symmetric positive definite.
    
    Parameters
    ----------
    A : np.ndarray
        Input matrix to decompose.
    
    Returns
    -------
    L : np.ndarray
        The cholesky factorization of the matrix.
    """

    L = np.zeros_like(A)



    return L



def chol_dcmp_ndiags(
    A: np.ndarray,
    blocksize: int,
) -> np.ndarray:
    """ Perform the cholesky factorization of a block n-diagonals matrix. The 
    matrix is assumed to be symmetric positive definite.
    
    Parameters
    ----------
    A : np.ndarray
        Input matrix to decompose.
    
    Returns
    -------
    L : np.ndarray
        The cholesky factorization of the matrix.
    """

    L = np.zeros_like(A)



    return L



def chol_dcmp_ndiags_arrowhead(
    A: np.ndarray,
    blocksize: int,
) -> np.ndarray:
    """ Perform the cholesky factorization of a block n-diagonals matrix. The 
    matrix is assumed to be symmetric positive definite.
    
    Parameters
    ----------
    A : np.ndarray
        Input matrix to decompose.
    
    Returns
    -------
    L : np.ndarray
        The cholesky factorization of the matrix.
    """

    L = np.zeros_like(A)



    return L