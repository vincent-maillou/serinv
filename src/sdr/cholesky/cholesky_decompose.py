"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Lisa Gaedke-Merzhaeuser  (lisa.gaedke.merzhaeuser@usi.ch)
@date: 2023-11

Contains the cholesky decompositions routines.

Copyright 2023 ETH Zurich and USI. All rights reserved.
"""


import numpy as np
import scipy.linalg as la



""" def chol_dcmp_tridiag(
    A: np.ndarray,
    blocksize: int,
) -> np.ndarray:
    Perform the cholesky factorization of a block tridiagonal matrix. The 
    matrix is assumed to be symmetric positive definite.
    
    Parameters
    ----------
    A : np.ndarray
        Input matrix to decompose.
    
    Returns
    -------
    L : np.ndarray
        The cholesky factorization of the matrix.
   

    L = np.zeros_like(A)

    # Initialize the first diagonal block L_{0, 0}
    L[0:blocksize, 0:blocksize] = la.cholesky(A[0:blocksize, 0:blocksize], lower=True)

    nblocks = A.shape[0] // blocksize
    # Proceed to the decomposition downwards
    for i in range(1, nblocks):
        L[i*blocksize:(i+1)*blocksize, (i-1)*blocksize:i*blocksize] = A[i*blocksize:(i+1)*blocksize, (i-1)*blocksize:i*blocksize] @ la.solve_triangular(L[(i-1)*blocksize:i*blocksize, (i-1)*blocksize:i*blocksize], np.eye(blocksize), lower=True).T
        L[i*blocksize:(i+1)*blocksize, i*blocksize:(i+1)*blocksize] = A[i*blocksize:(i+1)*blocksize, i*blocksize:(i+1)*blocksize] - L[i*blocksize:(i+1)*blocksize, (i-1)*blocksize:i*blocksize] @ L[i*blocksize:(i+1)*blocksize, (i-1)*blocksize:i*blocksize].T
        L[i*blocksize:(i+1)*blocksize, i*blocksize:(i+1)*blocksize] = la.cholesky(L[i*blocksize:(i+1)*blocksize, i*blocksize:(i+1)*blocksize]).T

    return L """



def chol_dcmp_tridiag(
    A: np.ndarray,
    blocksize: int,
) -> np.ndarray:
    """ Perform the cholesky factorization of a block tridiagonal matrix. The 
    matrix is assumed to be symmetric positive definite.

    Current implementation doesn't modify the input matrix A.
    
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

    L[0:blocksize, 0:blocksize] = A[0:blocksize, 0:blocksize]

    nblocks = A.shape[0] // blocksize
    for i in range(0, nblocks-1):
        # L_{i, i} = chol(A_{i, i})
        L[i*blocksize:(i+1)*blocksize, i*blocksize:(i+1)*blocksize] = la.cholesky(L[i*blocksize:(i+1)*blocksize, i*blocksize:(i+1)*blocksize]).T

        # L_{i+1, i} = A_{i+1, i} @ L_{i, i}^{-T}
        L[(i+1)*blocksize:(i+2)*blocksize, i*blocksize:(i+1)*blocksize] = A[(i+1)*blocksize:(i+2)*blocksize, i*blocksize:(i+1)*blocksize] @ la.solve_triangular(L[i*blocksize:(i+1)*blocksize, i*blocksize:(i+1)*blocksize], np.eye(blocksize), lower=True).T
        
        # A_{i+1, i+1} = A_{i+1, i+1} - L_{i+1, i} @ L_{i+1, i}^{T}
        L[(i+1)*blocksize:(i+2)*blocksize, (i+1)*blocksize:(i+2)*blocksize] = A[(i+1)*blocksize:(i+2)*blocksize, (i+1)*blocksize:(i+2)*blocksize] - L[(i+1)*blocksize:(i+2)*blocksize, i*blocksize:(i+1)*blocksize] @ L[(i+1)*blocksize:(i+2)*blocksize, i*blocksize:(i+1)*blocksize].T
       
    L[-blocksize:, -blocksize:] = la.cholesky(L[-blocksize:, -blocksize:]).T

    return L



def chol_dcmp_tridia_arrowhead(
    A: np.ndarray,
    diag_blocksize: int,
    arrow_blocksize: int,
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
    
    # Initialize the first diagonal block L_{0, 0}
    L[0:diag_blocksize, 0:diag_blocksize] = la.cholesky(A[0:diag_blocksize, 0:diag_blocksize], lower=True)

    nblocks = (A.shape[0]-arrow_blocksize) // diag_blocksize + 1
    # Proceed to the decomposition downwards
    for i in range(1, nblocks):
        # L{i+1, i}
        L[i*diag_blocksize:(i+1)*diag_blocksize, (i-1)*diag_blocksize:i*diag_blocksize] = A[i*diag_blocksize:(i+1)*diag_blocksize, (i-1)*diag_blocksize:i*diag_blocksize] @ la.solve_triangular(L[(i-1)*diag_blocksize:i*diag_blocksize, (i-1)*diag_blocksize:i*diag_blocksize], np.eye(diag_blocksize), lower=True).T
        
        # L_{nb, i}
        L[-arrow_blocksize:, (i-1)*diag_blocksize:i*diag_blocksize] = A[-arrow_blocksize:, (i-1)*diag_blocksize:i*diag_blocksize] @ la.solve_triangular(L[(i-1)*diag_blocksize:i*diag_blocksize, (i-1)*diag_blocksize:i*diag_blocksize], np.eye(diag_blocksize), lower=True).T
        # A_{nb, i+1}
        A[-arrow_blocksize:, i*diag_blocksize:(i+1)*diag_blocksize] = A[-arrow_blocksize:, i*diag_blocksize:(i+1)*diag_blocksize] - L[-arrow_blocksize:, (i-1)*diag_blocksize:i*diag_blocksize] @ L[i*diag_blocksize:(i+1)*diag_blocksize, (i-1)*diag_blocksize:i*diag_blocksize].T
        # A_{nb, nb}
        A[-arrow_blocksize:, -arrow_blocksize:] = A[-arrow_blocksize:, -arrow_blocksize:] - L[-arrow_blocksize:, (i-1)*diag_blocksize:i*diag_blocksize] @ L[-arrow_blocksize:, (i-1)*diag_blocksize:i*diag_blocksize].T

        # L_{i+1, i+1}
        L[i*diag_blocksize:(i+1)*diag_blocksize, i*diag_blocksize:(i+1)*diag_blocksize] = A[i*diag_blocksize:(i+1)*diag_blocksize, i*diag_blocksize:(i+1)*diag_blocksize] - L[i*diag_blocksize:(i+1)*diag_blocksize, (i-1)*diag_blocksize:i*diag_blocksize] @ L[i*diag_blocksize:(i+1)*diag_blocksize, (i-1)*diag_blocksize:i*diag_blocksize].T
        L[i*diag_blocksize:(i+1)*diag_blocksize, i*diag_blocksize:(i+1)*diag_blocksize] = la.cholesky(L[i*diag_blocksize:(i+1)*diag_blocksize, i*diag_blocksize:(i+1)*diag_blocksize]).T

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