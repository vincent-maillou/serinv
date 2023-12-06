"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Lisa Gaedke-Merzhaeuser  (lisa.gaedke.merzhaeuser@usi.ch)
@date: 2023-11

Contains the cholesky selected inversion routines.

Copyright 2023 ETH Zurich and USI. All rights reserved.
"""


import numpy as np
import scipy.linalg as la



def chol_sinv_tridiag(
    L: np.ndarray,
    blocksize: int,
) -> np.ndarray:
    """ Perform a selected inversion from a cholesky decomposed matrix with a
    block tridiagonal structure.
    
    Parameters
    ----------
    L : np.ndarray
        The cholesky factorization of the matrix.
    blocksize : int
        The blocksize of the matrix.
    
    Returns
    -------
    X : np.ndarray
        Selected inversion of the matrix.
    """
    
    X = np.zeros(L.shape, dtype=L.dtype)

    nblocks = L.shape[0] // blocksize

    L_blk_inv = np.zeros((blocksize, blocksize), dtype=L.dtype)
    L_blk_inv = la.solve_triangular(L[-blocksize:, -blocksize:], np.eye(blocksize), lower=True)
    X[-blocksize:, -blocksize:] = L_blk_inv.T @ L_blk_inv

    for i in range(nblocks-2, -1, -1):
        L_blk_inv = la.solve_triangular(L[i*blocksize:(i+1)*blocksize, i*blocksize:(i+1)*blocksize], np.eye(blocksize), lower=True)
        
        # X_{i+1, i} = -X_{i+1, i+1} L_{i+1, i} L_{i, i}^{-1}
        X[(i+1)*blocksize:(i+2)*blocksize, i*blocksize:(i+1)*blocksize] = -X[(i+1)*blocksize:(i+2)*blocksize, (i+1)*blocksize:(i+2)*blocksize] @ L[(i+1)*blocksize:(i+2)*blocksize, i*blocksize:(i+1)*blocksize] @ L_blk_inv

        # X_{i, i+1} = X_{i+1, i}.T
        X[i*blocksize:(i+1)*blocksize, (i+1)*blocksize:(i+2)*blocksize] = X[(i+1)*blocksize:(i+2)*blocksize, i*blocksize:(i+1)*blocksize].T

        # X_{i, i} = (L_{i, i}^{-T} - X_{i+1, i}^{T} L_{i+1, i}) L_{i, i}^{-1}
        X[i*blocksize:(i+1)*blocksize, i*blocksize:(i+1)*blocksize] = (L_blk_inv.T - X[(i+1)*blocksize:(i+2)*blocksize, i*blocksize:(i+1)*blocksize].T @ L[(i+1)*blocksize:(i+2)*blocksize, i*blocksize:(i+1)*blocksize]) @ L_blk_inv

    return X

