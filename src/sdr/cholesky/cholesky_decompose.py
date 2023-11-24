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
    L_inv_temp = np.zeros((diag_blocksize, diag_blocksize))

    n_diag_blocks = (A.shape[0]-arrow_blocksize) // diag_blocksize 
    for i in range(0, n_diag_blocks-1):
        # L_{i, i} = chol(A_{i, i})
        L[i*diag_blocksize:(i+1)*diag_blocksize, i*diag_blocksize:(i+1)*diag_blocksize] = la.cholesky(A[i*diag_blocksize:(i+1)*diag_blocksize, i*diag_blocksize:(i+1)*diag_blocksize]).T

        # Temporary storage of used twice lower triangular solving
        L_inv_temp = la.solve_triangular(L[i*diag_blocksize:(i+1)*diag_blocksize, i*diag_blocksize:(i+1)*diag_blocksize], np.eye(diag_blocksize), lower=True).T

        # L_{i+1, i} = A_{i+1, i} @ L_{i, i}^{-T}
        L[(i+1)*diag_blocksize:(i+2)*diag_blocksize, i*diag_blocksize:(i+1)*diag_blocksize] = A[(i+1)*diag_blocksize:(i+2)*diag_blocksize, i*diag_blocksize:(i+1)*diag_blocksize] @ L_inv_temp
        
        # L_{ndb+1, i} = A_{ndb+1, i} @ L_{i, i}^{-T}
        L[-arrow_blocksize:, i*diag_blocksize:(i+1)*diag_blocksize] = A[-arrow_blocksize:, i*diag_blocksize:(i+1)*diag_blocksize] @ L_inv_temp

        # A_{i+1, i+1} = A_{i+1, i+1} - L_{i+1, i} @ L_{i+1, i}.T
        A[(i+1)*diag_blocksize:(i+2)*diag_blocksize, (i+1)*diag_blocksize:(i+2)*diag_blocksize] = A[(i+1)*diag_blocksize:(i+2)*diag_blocksize, (i+1)*diag_blocksize:(i+2)*diag_blocksize] - L[(i+1)*diag_blocksize:(i+2)*diag_blocksize, i*diag_blocksize:(i+1)*diag_blocksize] @ L[(i+1)*diag_blocksize:(i+2)*diag_blocksize, i*diag_blocksize:(i+1)*diag_blocksize].T

        # A_{ndb+1, i+1} = A_{ndb+1, i+1} - L_{ndb+1, i} @ L_{i+1, i}.T
        A[-arrow_blocksize:, (i+1)*diag_blocksize:(i+2)*diag_blocksize] = A[-arrow_blocksize:, (i+1)*diag_blocksize:(i+2)*diag_blocksize] - L[-arrow_blocksize:, i*diag_blocksize:(i+1)*diag_blocksize] @ L[(i+1)*diag_blocksize:(i+2)*diag_blocksize, i*diag_blocksize:(i+1)*diag_blocksize].T
        
        # A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, i} @ L_{ndb+1, i}.T  
        A[-arrow_blocksize:, -arrow_blocksize:] = A[-arrow_blocksize:, -arrow_blocksize:] - L[-arrow_blocksize:, i*diag_blocksize:(i+1)*diag_blocksize] @ L[-arrow_blocksize:, i*diag_blocksize:(i+1)*diag_blocksize].T

    # L_{ndb, ndb} = chol(A_{ndb, ndb})
    L[-(diag_blocksize+arrow_blocksize):-arrow_blocksize, -(diag_blocksize+arrow_blocksize):-arrow_blocksize] = la.cholesky(A[-(diag_blocksize+arrow_blocksize):-arrow_blocksize, -(diag_blocksize+arrow_blocksize):-arrow_blocksize]).T

    # L_{ndb+1, ndb} = A_{ndb+1, ndb} @ L_{ndb, ndb}^{-T}
    L[-arrow_blocksize:, -(diag_blocksize+arrow_blocksize):-arrow_blocksize] = A[-arrow_blocksize:, -(diag_blocksize+arrow_blocksize):-arrow_blocksize] @ la.solve_triangular(L[-(diag_blocksize+arrow_blocksize):-arrow_blocksize, -(diag_blocksize+arrow_blocksize):-arrow_blocksize], np.eye(diag_blocksize), lower=True).T

    # A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, ndb} @ L_{ndb+1, ndb}^{T}
    A[-arrow_blocksize:, -arrow_blocksize:] = A[-arrow_blocksize:, -arrow_blocksize:] - L[-arrow_blocksize:, -(diag_blocksize+arrow_blocksize):-arrow_blocksize] @ L[-arrow_blocksize:, -(diag_blocksize+arrow_blocksize):-arrow_blocksize].T

    # L_{ndb+1, ndb+1} = chol(A_{ndb+1, ndb+1})
    L[-arrow_blocksize:, -arrow_blocksize:] = la.cholesky(A[-arrow_blocksize:, -arrow_blocksize:]).T

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