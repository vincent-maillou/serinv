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