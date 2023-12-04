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
        # X_{i} = U_{i,i}^{-1} (Y_{i} - U_{i,i+1} X_{i+1})
        X[i*blocksize:(i+1)*blocksize] = la.solve_triangular(
            U[i*blocksize:(i+1)*blocksize, i*blocksize:(i+1)*blocksize], 
            Y[i*blocksize:(i+1)*blocksize] - U[i*blocksize:(i+1)*blocksize, (i+1)*blocksize:(i+2)*blocksize] @ X[(i+1)*blocksize:(i+2)*blocksize], 
            lower=False, 
        )

    return X



def lu_slv_tridiag_arrowhead(
    L: np.ndarray,
    U: np.ndarray,
    B: np.ndarray,
    diag_blocksize: int,
    arrow_blocksize: int,
) -> np.ndarray:
    """ Solve a lu decomposed matrix with a block tridiagonal arrowhead 
    structure against the given right hand side. 
    
    Parameters
    ----------
    L : np.ndarray
        The LU factorization of the matrix.
    U : np.ndarray
        The LU factorization of the matrix.
    B : np.ndarray
        The right hand side.
    diag_blocksize : int
        Blocksize of the diagonals blocks of the matrix.
    arrow_blocksize : int
        Blocksize of the blocks composing the arrowhead.
        
    Returns
    -------
    X : np.ndarray
        The solution of the system.
    """

    Y = np.zeros_like(B)
    X = np.zeros_like(B)

    # ----- Forward substitution -----
    n_diag_blocks = (L.shape[0]-arrow_blocksize) // diag_blocksize 
    Y[0:diag_blocksize] = la.solve_triangular(L[0:diag_blocksize, 0:diag_blocksize], B[0:diag_blocksize], lower=True)
    for i in range(0, n_diag_blocks):
        # Y_{i} = L_{i,i}^{-1} (B_{i} - L_{i,i-1} Y_{i-1})
        Y[i*diag_blocksize:(i+1)*diag_blocksize] = la.solve_triangular(
            L[i*diag_blocksize:(i+1)*diag_blocksize, i*diag_blocksize:(i+1)*diag_blocksize], 
            B[i*diag_blocksize:(i+1)*diag_blocksize]-L[i*diag_blocksize:(i+1)*diag_blocksize, (i-1)*diag_blocksize:i*diag_blocksize] @ Y[(i-1)*diag_blocksize:(i)*diag_blocksize], 
            lower=True
        )
    
    # Accumulation of the arrowhead blocks
    B_temp = B[-arrow_blocksize:]
    for i in range(n_diag_blocks):
        B_temp = B_temp - L[-arrow_blocksize:, i*diag_blocksize:(i+1)*diag_blocksize] @ Y[i*diag_blocksize:(i+1)*diag_blocksize]
    # Y_{ndb+1} = L_{ndb+1,ndb+1}^{-1} (B_{ndb+1} - \Sigma_{i=1}^{ndb} L_{ndb+1,i} Y_{i)
    Y[-arrow_blocksize:] = la.solve_triangular(
        L[-arrow_blocksize:, -arrow_blocksize:], 
        B_temp, 
        lower=True
    )

    # ----- Backward substitution -----
    # X_{ndb+1} = U_{ndb+1,ndb+1}^{-1} (Y_{ndb+1})
    X[-arrow_blocksize:] = la.solve_triangular(U[-arrow_blocksize:, -arrow_blocksize:], Y[-arrow_blocksize:], lower=False)

    # X_{ndb} = U_{ndb,ndb}^{-1} (Y_{ndb} - U_{ndb,ndb+1} X_{ndb+1})
    X[-arrow_blocksize-diag_blocksize:-arrow_blocksize] = la.solve_triangular(
        U[-arrow_blocksize-diag_blocksize:-arrow_blocksize, -arrow_blocksize-diag_blocksize:-arrow_blocksize], 
        Y[-arrow_blocksize-diag_blocksize:-arrow_blocksize] - U[-arrow_blocksize-diag_blocksize:-arrow_blocksize, -arrow_blocksize:] @ X[-arrow_blocksize:],
        lower=False 
    )

    Y_temp = np.ndarray(shape=(diag_blocksize, B.shape[1]))
    for i in range(n_diag_blocks-2, -1, -1):
        # X_{i} = U_{i,i}^{-1} (Y_{i} - U_{i,i+1} X_{i+1}) - U_{i,ndb+1} X_{ndb+1}
        Y_temp = Y[i*diag_blocksize:(i+1)*diag_blocksize] - U[i*diag_blocksize:(i+1)*diag_blocksize, (i+1)*diag_blocksize:(i+2)*diag_blocksize] @ X[(i+1)*diag_blocksize:(i+2)*diag_blocksize] - U[i*diag_blocksize:(i+1)*diag_blocksize, -arrow_blocksize:] @ X[-arrow_blocksize:]
        X[i*diag_blocksize:(i+1)*diag_blocksize] = la.solve_triangular(
            U[i*diag_blocksize:(i+1)*diag_blocksize, i*diag_blocksize:(i+1)*diag_blocksize], 
            Y_temp, 
            lower=False, 
        )
    
    return X