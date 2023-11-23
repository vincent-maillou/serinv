"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Lisa Gaedke-Merzhaeuser  (lisa.gaedke.merzhaeuser@usi.ch)
@date: 2023-11

Contains the cholesky solve routines.

Copyright 2023 ETH Zurich and USI. All rights reserved.
"""


import numpy as np
import scipy.linalg as la



def chol_solve_tridiag(
    L: np.ndarray,
    B: np.ndarray,
    blocksize: int,
) -> np.ndarray:
    """ Solve a cholesky decomposed matrix with a block tridiagonal structure 
    against the given right hand side. The matrix is assumed to be symmetric 
    positive definite.
    
    Parameters
    ----------
    L : np.ndarray
        The cholesky factorization of the matrix.
    B : np.ndarray
        The right hand side.
    
    Returns
    -------
    X : np.ndarray
        The solution of the system.
    """

    X = np.zeros_like(B)



    return X



def chol_solve_tridia_arrowhead(
    L: np.ndarray,
    B: np.ndarray,
    blocksize: int,
) -> np.ndarray:
    """ Solve a cholesky decomposed matrix with a block tridiagonal arrowhead 
    structure against the given right hand side. The matrix is assumed to be 
    symmetric positive definite.
    
    Parameters
    ----------
    L : np.ndarray
        The cholesky factorization of the matrix.
    B : np.ndarray
        The right hand side.
    
    Returns
    -------
    X : np.ndarray
        The solution of the system.
    """

    X = np.zeros_like(B)



    return X



def chol_solve_ndiags(
    L: np.ndarray,
    B: np.ndarray,
    blocksize: int,
) -> np.ndarray:
    """ Solve a cholesky decomposed matrix with a block n-diagonals structure 
    against the given right hand side. The matrix is assumed to be symmetric 
    positive definite.
    
    Parameters
    ----------
    L : np.ndarray
        The cholesky factorization of the matrix.
    B : np.ndarray
        The right hand side.
    
    Returns
    -------
    X : np.ndarray
        The solution of the system.
    """

    X = np.zeros_like(B)



    return X



def chol_solve_ndiags_arrowhead(
    L: np.ndarray,
    B: np.ndarray,
    blocksize: int,
) -> np.ndarray:
    """ Solve a cholesky decomposed matrix with a block n-diagonals arrowhead 
    structure against the given right hand side. The matrix is assumed to be 
    symmetric positive definite.
    
    Parameters
    ----------
    L : np.ndarray
        The cholesky factorization of the matrix.
    B : np.ndarray
        The right hand side.
    
    Returns
    -------
    X : np.ndarray
        The solution of the system.
    """

    X = np.zeros_like(B)



    return X


