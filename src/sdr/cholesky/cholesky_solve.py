"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Lisa Gaedke-Merzhaeuser  (lisa.gaedke.merzhaeuser@usi.ch)
@date: 2023-11

Contains the cholesky solve routines.

Copyright 2023 ETH Zurich and USI. All rights reserved.
"""


import numpy as np
import scipy.linalg as la



def chol_slv_tridiag(
    L: np.ndarray,
    B: np.ndarray,
    blocksize: int,
) -> np.ndarray:
    """ Solve a cholesky decomposed matrix with a block tridiagonal structure 
    against the given right hand side.
    
    Parameters
    ----------
    L : np.ndarray
        The cholesky factorization of the matrix.
    B : np.ndarray
        The right hand side.
    blocksize : int
        The blocksize of the matrix.
    
    Returns
    -------
    X : np.ndarray
        The solution of the system.
    """

    Y = np.zeros_like(B)
    X = np.zeros_like(B)

    n_blocks = L.shape[0] // blocksize
    Y[0:blocksize] = la.solve_triangular(L[0:blocksize, 0:blocksize], B[0:blocksize], lower=True)
    for i in range(1, n_blocks, 1):
        # Y_{i} = L_{i,i}^{-1} (B_{i} - L_{i,i-1} Y_{i-1})
        Y[i*blocksize:(i+1)*blocksize] = la.solve_triangular(
            L[i*blocksize:(i+1)*blocksize, i*blocksize:(i+1)*blocksize], 
            B[i*blocksize:(i+1)*blocksize]-L[i*blocksize:(i+1)*blocksize, (i-1)*blocksize:i*blocksize] @ Y[(i-1)*blocksize:(i)*blocksize], 
            lower=True
        )

    X[-blocksize:] = la.solve_triangular(L[-blocksize:, -blocksize:], Y[-blocksize:], lower=True, trans='T')
    for i in range(n_blocks-2, -1, -1):
        # X_{i} = L_{i,i}^{-T} (Y_{i} - L_{i+1,i} X_{i+1})
        X[i*blocksize:(i+1)*blocksize] = la.solve_triangular(
            L[i*blocksize:(i+1)*blocksize, i*blocksize:(i+1)*blocksize], 
            Y[i*blocksize:(i+1)*blocksize]-L[(i+1)*blocksize:(i+2)*blocksize, i*blocksize:(i+1)*blocksize].T @ X[(i+1)*blocksize:(i+2)*blocksize], 
            lower=True, 
            trans='T'
        )

    return X



def chol_slv_tridiag_arrowhead(
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



def chol_slv_ndiags(
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



def chol_slv_ndiags_arrowhead(
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


