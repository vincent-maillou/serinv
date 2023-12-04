"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Lisa Gaedke-Merzhaeuser  (lisa.gaedke.merzhaeuser@usi.ch)
@date: 2023-11

Contains the lu solve routines.

Copyright 2023 ETH Zurich and USI. All rights reserved.
"""


import numpy as np
import scipy.linalg as la



def lu_slv_tridiag(
    L: np.ndarray,
    U: np.ndarray,
    B: np.ndarray,
    blocksize: int,
) -> np.ndarray:
    """ Solve a LU decomposed matrix with a block tridiagonal structure 
    against the given right hand side.
    
    Parameters
    ----------
    L : np.ndarray
        The LU factorization of the matrix.
    U : np.ndarray
        The LU factorization of the matrix.
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

    # ----- Forward substitution -----
    n_blocks = L.shape[0] // blocksize
    Y[0:blocksize] = la.solve_triangular(L[0:blocksize, 0:blocksize], B[0:blocksize], lower=True)
    for i in range(1, n_blocks, 1):
        # Y_{i} = L_{i,i}^{-1} (B_{i} - L_{i,i-1} Y_{i-1})
        Y[i*blocksize:(i+1)*blocksize] = la.solve_triangular(
            L[i*blocksize:(i+1)*blocksize, i*blocksize:(i+1)*blocksize], 
            B[i*blocksize:(i+1)*blocksize]-L[i*blocksize:(i+1)*blocksize, (i-1)*blocksize:i*blocksize] @ Y[(i-1)*blocksize:(i)*blocksize], 
            lower=True
        )

    # ----- Backward substitution -----
    X[-blocksize:] = la.solve_triangular(U[-blocksize:, -blocksize:], Y[-blocksize:], lower=False)
    for i in range(n_blocks-2, -1, -1):
        # X_{i} = U_{i,i}^{-T} (Y_{i} - U_{i,i+1} X_{i+1})
        X[i*blocksize:(i+1)*blocksize] = la.solve_triangular(
            U[i*blocksize:(i+1)*blocksize, i*blocksize:(i+1)*blocksize], 
            Y[i*blocksize:(i+1)*blocksize] - U[i*blocksize:(i+1)*blocksize, (i+1)*blocksize:(i+2)*blocksize] @ X[(i+1)*blocksize:(i+2)*blocksize], 
            lower=False, 
        )

    return X
