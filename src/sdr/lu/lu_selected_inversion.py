"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Lisa Gaedke-Merzhaeuser  (lisa.gaedke.merzhaeuser@usi.ch)
@date: 2023-11

Contains the lu selected inversion routines.

Copyright 2023 ETH Zurich and USI. All rights reserved.
"""


import copy
import random
import numpy as np
import scipy.linalg as la
from sdr.lu.lu_decompose import lu_dcmp_tridiag_explicit, lu_dcmp_tridiag

from sdr.utils import matrix_generation
from tracing import CDAG, Vertex, Edge, sinv_cdag, invert_LU_explicit, lu_dcmp_explicit, MMM_explicit, subs_explicit, create_permutation_matrix


    # return U + np.tril(L, k=-1)
    # return U @ L




# def sinv_ndiags_greg(
#     in_A: np.ndarray,
#     ndiags: int,
# ) -> np.ndarray:
#     """ Perform the LU factorization of an n-diagonals matrix. The matrix is assumed to be non-singular.

#     Parameters
#     ----------
#     A : np.ndarray
#         Input matrix to decompose.
#     ndiags : int
#         Number of diagonals of the matrix above the main diagonal. The total number of diagonals is 2*ndiags+1.
#     blocksize : int
#         Size of the blocks.
    
#     Returns
#     -------
#     L : np.ndarray
#         Lower factor of the LU factorization of the matrix.
#     U : np.ndarray
#         Upper factor of the LU factorization of the matrix.
#     """


#     A_inv_str = np.zeros(in_A.shape, dtype=in_A.dtype)
#     A = copy.deepcopy(in_A)
#     # in-place LU decomposition without pivoting
#     n = A.shape[0]
#     for k in range(n):
#         A[k, k] = 1 / A[k, k]

#         for i in range(k+1,min(k+1 + ndiags, n)): #(n-1, k, -1)
#             A[i, k] = A[i,k] * A[k,k]
#             for j in range(k+1,min(k+1 + ndiags, n)): #range(n-1, k, -1): 
#                 A[i,j]  -= A[i,k]*A[k,j]      

#         A_inv_str[k,k] = A[k,k] 
#         for i in range(k): #range(max(0, k - ndiags), k): 
#             s1 = A[i, k] * A[i, i]
#             s2 = A[k, i]

#             for j in range(i+1, k):  
#                 s1 += A[j, k] * A[i, j]
#                 s2 += A[k, j] * A[j, i]
#             A[i,k] = - s1 * A[k, k]
#             A[k,i] = - s2


#             A_inv_str[i,k] = A[i,k] 
#             A_inv_str[k,i] = A[k,i] * A[k,k] 
#             for j in range(i):
#                 A_inv_str[i,j] +=  A[i,k]*A[k,j]
#                 A_inv_str[j,i] +=  A[j,k]*A[k,i]
#             A_inv_str[i,i] += A[i,k]*A[k,i] 

#     LU = copy.deepcopy(A)
    

#     U = np.triu(LU, k=0)
#     L = np.tril(LU, k=-1) + np.eye(n, dtype=A.dtype)
#     A_inv = U @ L

#     return U @ L


def cut_to_banded(
    A: np.ndarray,
    ndiags : int
):
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if abs(i-j) > ndiags:
                A[i, j] = 0
    return A


def sinv_tridiag_explicit(in_A: np.ndarray,
                     blocksize: int) -> np.ndarray:
    L, U = lu_dcmp_tridiag_explicit(in_A, blocksize=blocksize)
    LU_inv = lu_sinv_tridiag_explicit(L, U, blocksize=blocksize)
    # sinv_cdag.plot()
    return LU_inv




def sinv_ndiags_greg2(
    in_A: np.ndarray,
    ndiags: int
) -> np.ndarray:
    """ Perform the selected inversion on the given input matrix in_A. 

    Parameters
    ----------
    in_A : np.ndarray
        Input matrix to decompose.
    ndiags : int
        Number of diagonals of the matrix above the main diagonal. The total number of diagonals is 2*ndiags+1.
    
    Returns
    -------
    A_inv_str : np.ndarray
        A_inv_str = in_A^{-1} * nonzero_mask
    """
    # blocksize = ndiags
    # L, U = lu_dcmp_tridiag_explicit(in_A, blocksize=blocksize)
    # LU = L + U - np.eye(L.shape[0], dtype=L.dtype)
    # L_inv, U_inv = invert_LU_explicit(LU) 

    # p, l, u = la.lu(in_A)
    # ref_lu = u + np.tril(l, k=-1)
    # ref_inv_lu = np.linalg.inv(u) + np.tril(np.linalg.inv(l), k=-1)
    # ref_inv_A = la.inv(in_A)
    # orginal = lu_sinv_tridiag(l, u, ndiags)
    # oringal_l = cut_to_banded(np.tril(orginal, k=-1), ndiags)
    # another = lu_sinv_tridiag_explicit(l, u, ndiags)
    # another2 = lu_sinv_ndiags_explicit(l, u, 2*ndiags, 1)

    counter = 0
    try:
        A_inv = np.zeros(in_A.shape, dtype=in_A.dtype)
    except:
        from sympy import Matrix, MatrixSymbol
        A_inv = Matrix(np.zeros(in_A.shape))
    A = copy.deepcopy(in_A)
    n = A.shape[0]

    import scipy as sp
    p, l, u = sp.linalg.lu(in_A)
    ref_lu = u + np.tril(l, k=-1)
    ref_invL = np.linalg.inv(l)
    ref_invU = np.linalg.inv(u)
    ref_invA = np.linalg.inv(in_A)
    ref_invLU = ref_invU + np.tril(ref_invL, k=-1)
    invL = invert_L(l)
    invU = invert_U(u)
    invLU = invert_LU(ref_lu)
    np.allclose(invert_LU(ref_lu), ref_invLU)
    for k in range(n):
        # BEGIN in-place LU decomposition without pivoting
        sinv_cdag.add([Vertex("A", A, (k, k), output=False)], Vertex("A", A, (k, k), output=True))
        A[k, k] = 1 / A[k, k]  # <- this is NOT the part of LU! This is already a first step of INVERSION

        # for i in range(k+1,min(n, n)): #(n-1, k, -1)
        # rng_i = list(range(k+1,min(k+1 + ndiags, n)))
        # random.shuffle(rng_i)
        # for i in rng_i:
        for i in range(k+1,min(k+1 + ndiags, n)): #(n-1, k, -1)
            sinv_cdag.add([Vertex("A", A, (i, k), output=False), Vertex("A", A, (k, k), output=False)], Vertex("A", A, (i, k), output=True))
            A[i, k] = A[i,k] * A[k,k]
            counter += 1
            # for j in range(k+1,min(n, n)): #range(n-1, k, -1): 
            rng_j = list(range(k+1,min(k+1 + ndiags, n)))
            random.shuffle(rng_j)
            for j in rng_j:
            # for j in range(k+1,min(k+1 + ndiags, n)): #range(n-1, k, -1): 
                sinv_cdag.add([Vertex("A", A, (i, j), output=False), 
                               Vertex("A", A, (i, k), output=False), 
                               Vertex("A", A, (k, j), output=False)], 
                               Vertex("A", A, (i, j), output=True))
                A[i,j]  -= A[i,k]*A[k,j]      
                counter += 1
        # END in-place LU decomposition without pivoting

    for k in range(n-1, -1, -1):
        sinv_cdag.add([Vertex("A", A, (k, k), output=False)], Vertex("A_inv", A_inv, (k, k), output=True))
        A_inv[k, k] = A[k,k]
        # Off-diagonal block part
        rng_i = list(range(min(k+ndiags, n-1), k, -1))
        random.shuffle(rng_i)
        for i in rng_i:
        # for i in range(min(k+ndiags, n-1), k, -1):
            rng_j = list(range(k+1, min(k+ndiags+1, n)))
            random.shuffle(rng_j)
            for j in rng_j:
            # for j in range(k+1, min(k+ndiags+1, n)):
                if j == k+1:
                    sinv_cdag.add([Vertex("A", A, (j, k), output=False),
                                    Vertex("A_inv", A_inv, (i, j), output=False),
                                    Vertex("A_inv", A_inv, (i, k), output=False, is_zero=False)],
                                    Vertex("A_inv", A_inv, (i, k), output=True))
                    
                    sinv_cdag.add([Vertex("A", A, (k, j), output=False),
                                 Vertex("A_inv", A_inv, (j, i), output=False),
                                 Vertex("A_inv", A_inv, (k, i), output=False, is_zero=False)],
                                 Vertex("A_inv", A_inv, (k, i), output=True))
                else:
                    sinv_cdag.add([Vertex("A", A, (j, k), output=False),
                                    Vertex("A_inv", A_inv, (i, j), output=False),
                                    Vertex("A_inv", A_inv, (i, k), output=False)],
                                    Vertex("A_inv", A_inv, (i, k), output=True))
                    
                    sinv_cdag.add([Vertex("A", A, (k, j), output=False),
                                 Vertex("A_inv", A_inv, (j, i), output=False),
                                 Vertex("A_inv", A_inv, (k, i), output=False)],
                                 Vertex("A_inv", A_inv, (k, i), output=True))

                # X_{i, k} = X_{i, k} - X_{i, j} L_{j, k}
                counter += 1
                A_inv[i, k] -= A_inv[i, j] * A[j, k]
                # X_{k, i} = X_{k, i} - U_{k, j} X_{j, i}
                A_inv[k, i] -= A[k, j] * A_inv[j, i]
                counter += 1

                # A[i, k] -= A[i, j] * A[j, k]
                # # X_{k, i} = X_{k, i} - U_{k, j} X_{j, i}
                # A[k, i] -= A[k, j] * A[j, i]
        
            # X_{k, i} = U_{k, k}^{-1} X_{k, i}
            sinv_cdag.add([Vertex("A", A, (k, k), output=False),
                            Vertex("A_inv", A_inv, (k, i), output=False)],
                            Vertex("A_inv", A_inv, (k, i), output=True))                            
            A_inv[k, i] = A_inv[k, i]  * A[k,k]
            counter += 1

            sinv_cdag.add([Vertex("A", A, (i, k), output=False),
                            Vertex("A_inv", A_inv, (k, i), output=False),
                            Vertex("A_inv", A_inv, (k, k), output=False)],
                            Vertex("A_inv", A_inv, (k, k), output=True))
            A_inv[k, k] -= A_inv[k, i] * A[i, k]
            counter += 1

            # A[k, i] = A[k, i]  * A[k,k]
            # A[k, k] -= A[k, i] * A[i, k]



    sinv_cdag.plot()
    return A_inv


def sinv_ndiags_with_enough_parallelism(in_A: np.ndarray,
            ndiags: int,
            b: int = 1
) -> np.ndarray:
    """ 
    Perform the selected inversion on the given input matrix in_A. The sparsity structure is encoded in the loop ranges.
    Parameter b (block size) increases paralellism for GPU exectuion.
    
    Parameters
    ----------
    A : np.ndarray
        Input matrix to decompose.
    ndiags : int
        Number of diagonals of the matrix above the main diagonal. The total number of diagonals is 2*ndiags+1.
    b : int
        Blocking parameter for additional dimension of parallelism. a tunable block size, such that  
        ndiags * ndiags * b saturates GPU threads. Generally, b should be the 
        smallest possible number such that ndiags * ndiags * b >= # hardware_treads
    
    """
    A = copy.deepcopy(in_A)
    n = A.shape[0]
    A_inv = np.zeros(in_A.shape, dtype=in_A.dtype)

    # debug only
    import scipy as sp
    p, l, u = sp.linalg.lu(in_A)
    ref_lu = u + np.tril(l, k=-1)
    ref_invL = np.linalg.inv(l)
    ref_invU = np.linalg.inv(u)
    ref_invA = np.linalg.inv(in_A)
    ref_invLU = ref_invU + np.tril(ref_invL, k=-1)
    invL = invert_L(l)
    invU = invert_U(u)
    # invLU = invert_LU(ref_lu)

    # in the following, we denote an additional iteration variable kk that would loop over b (for k in range(b))
    # to expose additinal degree of paralellism.

    b = 2
    A = copy.deepcopy(in_A)
    for k in range(0, n, b):
       # the following can be done e.g., using numpy.linalg.inv, or cupy.linalg.inv
    #    A[k: k+b, k:k+b] = invert_matrix(A[k: k+b, k:k+b])
        A[k:k+b, k:k+b] = inplace_LU(A[k: k+b, k:k+b], b)

        start = k+b
        end = min(k+b+ndiags, n)
        # toblerone solve
        A[start:end, k:start], A[k:start, start:end] = \
            inplace_double_trsm(A[k:start, k:start],
                                A[start:end, k:start],
                                A[k:start, start:end])

        # oh nice oh nice oh nice, a sweet perfect fatty 3D parallel GEMM is coming!
        # Notice that the third argument (matrix C) is inverted, because we are doing
        # C = C - A @ B, so effectively we are doing C = -GEMM(A, B, -C)
        A[start:end, start:end] =  -inplace_GEMM(
            A[start:end, k:start],
            A[k:start, start:end],
            -A[start:end, start:end])
        # END in-place LU decomposition without pivoting

    for k in range(n-b, -b, -b):        
        a00_start = k
        a11_start = k+b
        a11_end=min(k+b+ndiags,n)

        A_inv[a00_start:a11_start, a00_start:a11_start] = \
                invert_LU(A[a00_start:a11_start, a00_start:a11_start])

        L_inv = np.tril(A_inv[a00_start:a11_start, a00_start:a11_start], k=-1) + np.eye(b)
        U_inv = np.triu(A_inv[a00_start:a11_start, a00_start:a11_start], k=0)
        
        # if k < n-b:
        # col = A @ col
        A_inv[a11_start:a11_end, a00_start:a11_start] =  \
                A_inv[a11_start:a11_end, a00_start:a11_start] - \
                A_inv[a11_start:a11_end, a11_start:a11_end] @ \
                    A[a11_start:a11_end, a00_start:a11_start]  @ L_inv
        
    
        # row = row @ A   <- the same A as above! Reuse?
        A_inv[a00_start:a11_start, a11_start:a11_end] = \
            A_inv[a00_start:a11_start, a11_start:a11_end] - \
                U_inv @ A[a00_start:a11_start, a11_start:a11_end] @ \
                A_inv[a11_start:a11_end, a11_start:a11_end]
            


        A_inv[a00_start:a11_start, a00_start:a11_start]  = \
            -(A_inv[a00_start:a11_start, a11_start:a11_end] @ \
            A[a11_start:a11_end, a00_start:a11_start] - \
            U_inv) @ \
            L_inv

        # A_inv[a00_start:a11_start, a00_start:a11_start] = \
        #     GEMM_UL( A_inv[a00_start:a11_start, a00_start:a11_start])
        
    return A_inv
        



def sinv_ndiags_cupy(in_A: np.ndarray,
            ndiags: int,
            b: int = 1
) -> np.ndarray:
    """ 
    Perform the selected inversion on the given input matrix in_A. The sparsity structure is encoded in the loop ranges.
    Parameter b (block size) increases paralellism for GPU exectuion.
    
    Parameters
    ----------
    A : np.ndarray
        Input matrix to decompose.
    ndiags : int
        Number of diagonals of the matrix above the main diagonal. The total number of diagonals is 2*ndiags+1.
    b : int
        Blocking parameter for additional dimension of parallelism. a tunable block size, such that  
        ndiags * ndiags * b saturates GPU threads. Generally, b should be the 
        smallest possible number such that ndiags * ndiags * b >= # hardware_treads
    
    """
    A = copy.deepcopy(in_A)
    n = A.shape[0]
    A_inv = np.zeros(in_A.shape, dtype=in_A.dtype)

    # debug only
    import scipy as sp
    p, l, u = sp.linalg.lu(in_A)
    ref_lu = u + np.tril(l, k=-1)
    ref_invL = np.linalg.inv(l)
    ref_invU = np.linalg.inv(u)
    ref_invA = np.linalg.inv(in_A)
    ref_invLU = ref_invU + np.tril(ref_invL, k=-1)
    invL = invert_L(l)
    invU = invert_U(u)
    # invLU = invert_LU(ref_lu)

    # in the following, we denote an additional iteration variable kk that would loop over b (for k in range(b))
    # to expose additinal degree of paralellism.

    b = 2
    A = copy.deepcopy(in_A)
    for k in range(0, n, b):
       # the following can be done e.g., using numpy.linalg.inv, or cupy.linalg.inv
    #    A[k: k+b, k:k+b] = invert_matrix(A[k: k+b, k:k+b])
        A[k:k+b, k:k+b] = sp.linalg.lu_factor(A[k: k+b, k:k+b])[0]

        start = k+b
        end = min(k+b+ndiags, n)
        # toblerone solve
        A[start:end, k:start], A[k:start, start:end] = \
            inplace_double_trsm(A[k:start, k:start],
                                A[start:end, k:start],
                                A[k:start, start:end])

        # oh nice oh nice oh nice, a sweet perfect fatty 3D parallel GEMM is coming!
        # Notice that the third argument (matrix C) is inverted, because we are doing
        # C = C - A @ B, so effectively we are doing C = -GEMM(A, B, -C)
        A[start:end, start:end] =  -inplace_GEMM(
            A[start:end, k:start],
            A[k:start, start:end],
            -A[start:end, start:end])
        # END in-place LU decomposition without pivoting

    for k in range(n-b, -b, -b):        
        a00_start = k
        a11_start = k+b
        a11_end=min(k+b+ndiags,n)

        A_inv[a00_start:a11_start, a00_start:a11_start] = \
                invert_LU(A[a00_start:a11_start, a00_start:a11_start])

        L_inv = np.tril(A_inv[a00_start:a11_start, a00_start:a11_start], k=-1) + np.eye(b)
        U_inv = np.triu(A_inv[a00_start:a11_start, a00_start:a11_start], k=0)
        
        # if k < n-b:
        # col = A @ col
        A_inv[a11_start:a11_end, a00_start:a11_start] =  \
                A_inv[a11_start:a11_end, a00_start:a11_start] - \
                A_inv[a11_start:a11_end, a11_start:a11_end] @ \
                    A[a11_start:a11_end, a00_start:a11_start]  @ L_inv
        
    
        # row = row @ A   <- the same A as above! Reuse?
        A_inv[a00_start:a11_start, a11_start:a11_end] = \
            A_inv[a00_start:a11_start, a11_start:a11_end] - \
                U_inv @ A[a00_start:a11_start, a11_start:a11_end] @ \
                A_inv[a11_start:a11_end, a11_start:a11_end]
            


        A_inv[a00_start:a11_start, a00_start:a11_start]  = \
            -(A_inv[a00_start:a11_start, a11_start:a11_end] @ \
            A[a11_start:a11_end, a00_start:a11_start] - \
            U_inv) @ \
            L_inv

        # A_inv[a00_start:a11_start, a00_start:a11_start] = \
        #     GEMM_UL( A_inv[a00_start:a11_start, a00_start:a11_start])
        
    return A_inv


def sinv_ndiags_greg(
    in_A: np.ndarray,
    ndiags: int
) -> np.ndarray:
    """ Perform the selected inversion on the given input matrix in_A. The sparsity structure is encoded in the loop ranges
    (lines 160, 162, 170, 174, 184)

    Parameters
    ----------
    in_A : np.ndarray
        Input matrix to decompose.
    ndiags : int
        Number of diagonals of the matrix above the main diagonal. The total number of diagonals is 2*ndiags+1.
    
    Returns
    -------
    A_inv_str : np.ndarray
        A_inv_str = in_A^{-1} * nonzero_mask
    """
    blocksize = ndiags
    L, U = lu_dcmp_tridiag_explicit(in_A, blocksize=blocksize)
    LU = L + U - np.eye(L.shape[0], dtype=L.dtype)
    L_inv, U_inv = invert_LU_explicit(LU) 

    p, l, u = la.lu(in_A)
    ref_lu = u + np.tril(l, k=-1)
    ref_inv_lu = np.linalg.inv(u) + np.tril(np.linalg.inv(l), k=-1)
    ref_inv_A = la.inv(in_A)
    orginal = lu_sinv_tridiag(l, u, ndiags)
    oringal_l = cut_to_banded(np.tril(orginal, k=-1), ndiags)
    another = lu_sinv_tridiag_explicit(l, u, ndiags)
    another2 = lu_sinv_ndiags_explicit(l, u, 2*ndiags, 1)

    A_inv_str = np.zeros(in_A.shape, dtype=in_A.dtype)
    A = copy.deepcopy(in_A)
    n = A.shape[0]
    for k in range(n):
        # BEGIN in-place LU decomposition without pivoting
        sinv_cdag.add([Vertex("A", A, (k, k), output=False)], Vertex("A", A, (k, k), output=True))
        A[k, k] = 1 / A[k, k]  # <- this is NOT the part of LU! This is already a first step of INVERSION

        # for i in range(k+1,min(n, n)): #(n-1, k, -1)
        for i in range(k+1,min(k+1 + ndiags, n)): #(n-1, k, -1)
            sinv_cdag.add([Vertex("A", A, (i, k), output=False), Vertex("A", A, (k, k), output=False)], Vertex("A", A, (i, k), output=True))
            A[i, k] = A[i,k] * A[k,k]
            # for j in range(k+1,min(n, n)): #range(n-1, k, -1): 
            for j in range(k+1,min(k+1 + ndiags, n)): #range(n-1, k, -1): 
                sinv_cdag.add([Vertex("A", A, (i, j), output=False), 
                               Vertex("A", A, (i, k), output=False), 
                               Vertex("A", A, (k, j), output=False)], 
                               Vertex("A", A, (i, j), output=True))
                A[i,j]  -= A[i,k]*A[k,j]      
        # END in-place LU decomposition without pivoting

        sinv_cdag.add([Vertex("A", A, (k, k), output=False)], Vertex("A_inv_str", A_inv_str,  (k, k), output=True))
        A_inv_str[k,k] = A[k,k]  # <- A will hold  LU inverted. la.tril(A) is L^(-1), 
                                 #  la.triu(A) is U^(-1), A_inv_str is  U^(-1) @ L^(-1)

        # BEGIN invert in-place LU decomposition (two triangular L and U matrices are stored in A)
        for i in range(max(0, k - ndiags), k):  # range(k): #
        # for i in range(max(0, 0), k):  # range(k): #
            sinv_cdag.add([Vertex("A", A, (i, k), output=False),
                        Vertex("A", A, (i, i), output=False)], 
                        Vertex("A", A, (i, k), output=True))
            A[i,k] = A[i, k] * A[i, i]
            sinv_cdag.add([Vertex("A", A, (k, i), output=False)],
                        Vertex("A", A, (k, i), output=True))
            A[k,i] = -A[k, i]

            for j in range(i+1, k):  
                sinv_cdag.add([Vertex("A", A, (j, k), output=False),
                            Vertex("A", A, (i, j), output=False),
                            Vertex("A", A, (i, k), output=False)],
                            Vertex("A", A, (i, k), output=True))
                A[i,k] += A[j, k] * A[i, j]
                sinv_cdag.add([Vertex("A", A, (k, j), output=False),
                            Vertex("A", A, (j, i), output=False),
                            Vertex("A", A, (k, i), output=False)],
                            Vertex("A", A, (k, i), output=True))
                A[k,i] -= A[k, j] * A[j, i]
            sinv_cdag.add([Vertex("A", A, (i, k), output=False),
                           Vertex("A", A, (k,k), output=False)],
                           Vertex("A", A, (i,k), output=True))
            A[i,k] = - A[i,k] * A[k, k]
            # A[k,i] = - A[k,i]
            # END invert in-place LU decomposition (two triangular L and U matrices are stored in A)

            # BEGIN matrix-matrix multiplcation of U^(k-1) L^(k-1) (stored in A)
            A_inv_str[i,k] = A[i,k] 

            # test
            A_inv_str[k,i] = A[k,i] * A[k,k] 
            # A[ k,i] = A[k,i] * A[k,k] 
            for j in range(i):
                # A[i,j] +=  A[i,k]*A[k,j]
                # A[j,i] +=  A[j,k]*A[k,i]
                sinv_cdag.add([Vertex("A", A, (i, k), output=False),
                               Vertex("A", A, (k, j), output=False),
                               Vertex("A_inv_str", A_inv_str, (i, j), output=False)],
                               Vertex("A_inv_str", A_inv_str, (i, j), output=True))
                A_inv_str[i,j] +=  A[i,k]*A[k,j]
                sinv_cdag.add([Vertex("A", A, (j, k), output=False),
                               Vertex("A", A, (k, i), output=False),
                               Vertex("A_inv_str", A_inv_str, (j, i), output=False)],
                               Vertex("A_inv_str", A_inv_str, (j, i), output=True))
                A_inv_str[j,i] +=  A[j,k]*A[k,i]
            # if i == 0:
            #     print(f"A[i,i] = {A_inv_str[i,i]}, A[i,k] = {A[i,k]}, A[k,i] = {A[k,i]}, res={A_inv_str[i,i] + A[i,k]*A[k,i]}, i={i}, k={k}")
            A_inv_str[i,i] += A[i,k]*A[k,i] 
            # A[i,i] += A[i,k]*A[k,i] 
            # END matrix-matrix multiplcation of U^(k-1) L^(k-1) (stored in A)

    greg_invA_l = np.tril(cut_to_banded(A_inv_str, ndiags), k = -1)
    greg_inv_u = np.triu(cut_to_banded(A_inv_str, ndiags), k = 0)

    greg_invL_l = np.tril(cut_to_banded(A, ndiags), k = -1)
    greg_invU_u = np.triu(cut_to_banded(A, ndiags), k = 0)

    # A_inv_str = np.triu(A, k=0) @ (np.tril(A, k=-1) + np.eye(n, dtype=A.dtype))
    sinv_cdag.plot()
    return A_inv_str
    # return A



# def invert_lu(
#     in_LU: np.ndarray,
#     ndiags: int = -1
# ) -> None:
#     """ Performs the inversion of a matrix given its LU decomposition. L and U parts
#     are inverted separately in-place in their respectful lower and upper triangular parts.

#     Parameters
#     ----------
#     LU : np.ndarray
#         Input LU matrix to invert
    
#     Returns
#     -------
#     None
#     """


#     import scipy as sp
#     # p, l, u = sp.linalg.lu(in_A)
#     # ref_lu = u + np.tril(l, k=-1)
#     if ndiags == -1:
#         ndiags = in_LU.shape[0]
#     u = np.triu(in_LU, k=0)
#     l = np.tril(in_LU, k=-1) + np.eye(in_LU.shape[0], dtype=in_LU.dtype)
#     ref_inv_lu = np.linalg.inv(u) + np.tril(np.linalg.inv(l), k=-1)
#     another = lu_sinv_tridiag(l, u, 1)

#     # A_inv_str = np.zeros(in_A.shape, dtype=in_A.dtype)
#     LU = copy.deepcopy(in_LU)
#     n = LU.shape[0]
#     for k in range(n):
#         # BEGIN in-place LU decomposition without pivoting
#         sinv_cdag.add([Vertex("A", LU, (k, k), output=False)], Vertex("A", LU, (k, k), output=True))
#         LU[k, k] = 1 / LU[k, k]  # <- this is NOT the part of LU! This is already a first step of INVERSION


#         # BEGIN invert in-place LU decomposition (two triangular L and U matrices are stored in A)
#         for i in range(max(0, k - ndiags), k):  # range(k): #
#         # for i in range(max(0, 0), k):  # range(k): #
#             sinv_cdag.add([Vertex("A", LU, (i, k), output=False),
#                         Vertex("A", LU, (i, i), output=False)], 
#                         Vertex("A", LU, (i, k), output=True))
#             LU[i,k] = LU[i, k] * LU[i, i]
#             sinv_cdag.add([Vertex("A", LU, (k, i), output=False)],
#                         Vertex("A", LU, (k, i), output=True))
#             LU[k,i] = -LU[k, i]

#             for j in range(i+1, k):  
#                 sinv_cdag.add([Vertex("A", LU, (j, k), output=False),
#                             Vertex("A", LU, (i, j), output=False),
#                             Vertex("A", LU, (i, k), output=False)],
#                             Vertex("A", LU, (i, k), output=True))
#                 LU[i,k] += LU[j, k] * LU[i, j]
#                 sinv_cdag.add([Vertex("A", LU, (k, j), output=False),
#                             Vertex("A", LU, (j, i), output=False),
#                             Vertex("A", LU, (k, i), output=False)],
#                             Vertex("A", LU, (k, i), output=True))
#                 LU[k,i] -= LU[k, j] * LU[j, i]
#             sinv_cdag.add([Vertex("A", LU, (i, k), output=False),
#                            Vertex("A", LU, (k,k), output=False)],
#                            Vertex("A", LU, (i,k), output=True))
#             LU[i,k] = - LU[i,k] * LU[k, k]
#             # A[k,i] = - A[k,i]
#             # END invert in-place LU decomposition (two triangular L and U matrices are stored in A)

#     # A_inv_str = np.triu(A, k=0) @ (np.tril(A, k=-1) + np.eye(n, dtype=A.dtype))
#     # sinv_cdag.plot()
#     in_LU = LU
#     return


def lu_sinv_tridiag(
    L: np.ndarray,
    U: np.ndarray,
    blocksize: int,
) -> np.ndarray:
    """Perform a selected inversion from a lu decomposed matrix with a
    block tridiagonal structure.

    Parameters
    ----------
    L : np.ndarray
        Lower factor of the lu factorization fo the matrix.
    U : np.ndarray
        Upper factor of the lu factorization fo the matrix.
    blocksize : int
        The blocksize of the matrix.

    Returns
    -------
    X : np.ndarray
        Selected inversion of the matrix.
    """
    X = np.zeros(L.shape, dtype=L.dtype)

    L_blk_inv = np.zeros((blocksize, blocksize), dtype=L.dtype)
    U_blk_inv = np.zeros((blocksize, blocksize), dtype=L.dtype)

    L_blk_inv = la.solve_triangular(L[-blocksize:, -blocksize:], np.eye(blocksize), lower=True)
    U_blk_inv = la.solve_triangular(U[-blocksize:, -blocksize:], np.eye(blocksize), lower=False)
    X[-blocksize:, -blocksize:] = U_blk_inv @ L_blk_inv

    nblocks = L.shape[0] // blocksize
    for i in range(nblocks - 2, -1, -1):
        L_blk_inv = la.solve_triangular(
            L[i * blocksize : (i + 1) * blocksize, i * blocksize : (i + 1) * blocksize], np.eye(blocksize), lower=True
        )
        U_blk_inv = la.solve_triangular(
            U[i * blocksize : (i + 1) * blocksize, i * blocksize : (i + 1) * blocksize], np.eye(blocksize), lower=False
        )

        # X_{i+1, i} = -X_{i+1, i+1} L_{i+1, i} L_{i, i}^{-1}
        X[(i + 1) * blocksize : (i + 2) * blocksize, i * blocksize : (i + 1) * blocksize] = (
            -X[(i + 1) * blocksize : (i + 2) * blocksize, (i + 1) * blocksize : (i + 2) * blocksize]
            @ L[(i + 1) * blocksize : (i + 2) * blocksize, i * blocksize : (i + 1) * blocksize]
            @ L_blk_inv
        )

        # X_{i, i+1} = -U_{i, i}^{-1} U_{i, i+1} X_{i+1, i+1}
        X[i * blocksize : (i + 1) * blocksize, (i + 1) * blocksize : (i + 2) * blocksize] = (
            -U_blk_inv
            @ U[i * blocksize : (i + 1) * blocksize, (i + 1) * blocksize : (i + 2) * blocksize]
            @ X[(i + 1) * blocksize : (i + 2) * blocksize, (i + 1) * blocksize : (i + 2) * blocksize]
        )

        # X_{i, i} = (U_{i, i}^{-1} - X_{i, i+1} L_{i+1, i}) L_{i, i}^{-1}
        # print(f"U_blk_inv:")
        # print(U_blk_inv)
        # print(f"X[i*blocksize:(i+1)*blocksize, (i+1)*blocksize:(i+2)*blocksize]:")
        X[i * blocksize : (i + 1) * blocksize, i * blocksize : (i + 1) * blocksize] = (
            U_blk_inv
            - X[i * blocksize : (i + 1) * blocksize, (i + 1) * blocksize : (i + 2) * blocksize]
            @ L[(i + 1) * blocksize : (i + 2) * blocksize, i * blocksize : (i + 1) * blocksize]
        ) @ L_blk_inv

    return X


def lu_sinv_tridiag_explicit(
    L: np.ndarray,
    U: np.ndarray,
    blocksize: int,
) -> np.ndarray:
    """Perform a selected inversion from a lu decomposed matrix with a
    block tridiagonal structure.

    Parameters
    ----------
    L : np.ndarray
        Lower factor of the lu factorization fo the matrix.
    U : np.ndarray
        Upper factor of the lu factorization fo the matrix.
    blocksize : int
        The blocksize of the matrix.

    Returns
    -------
    X : np.ndarray
        Selected inversion of the matrix.
    """
    LU = L + U - np.eye(L.shape[0], dtype=L.dtype)

    X = np.zeros(L.shape, dtype=L.dtype)

    L_blk_inv = np.zeros((blocksize, blocksize), dtype=L.dtype)
    U_blk_inv = np.zeros((blocksize, blocksize), dtype=L.dtype)

    # L_blk_inv = la.solve_triangular(L[-blocksize:, -blocksize:], np.eye(blocksize), lower=True)
    # U_blk_inv = la.solve_triangular(U[-blocksize:, -blocksize:], np.eye(blocksize), lower=False)


    L_blk_inv, U_blk_inv = invert_LU_explicit(LU[-blocksize:, -blocksize:])
    X[-blocksize:, -blocksize:] = U_blk_inv @ L_blk_inv

    # LU = L[-blocksize:, -blocksize:] + U[-blocksize:, -blocksize:] - np.eye(blocksize, dtype=L.dtype)

    nblocks = L.shape[0] // blocksize
    for i in range(nblocks-2, -1, -1):
        # L_blk_inv = la.solve_triangular(L[i*blocksize:(i+1)*blocksize, i*blocksize:(i+1)*blocksize], np.eye(blocksize), lower=True)
        # U_blk_inv = la.solve_triangular(U[i*blocksize:(i+1)*blocksize, i*blocksize:(i+1)*blocksize], np.eye(blocksize), lower=False)
        L_blk_inv, U_blk_inv = invert_LU_explicit(LU[i*blocksize:(i+1)*blocksize, i*blocksize:(i+1)*blocksize])

        # X_{i+1, i} = -X_{i+1, i+1} L_{i+1, i} L_{i, i}^{-1}
        X[(i+1)*blocksize:(i+2)*blocksize, i*blocksize:(i+1)*blocksize] = \
            MMM_explicit(
                MMM_explicit(-X[(i+1)*blocksize:(i+2)*blocksize, (i+1)*blocksize:(i+2)*blocksize],
                     L[(i+1)*blocksize:(i+2)*blocksize, i*blocksize:(i+1)*blocksize]),
                       L_blk_inv)

        # X_{i, i+1} = -U_{i, i}^{-1} U_{i, i+1} X_{i+1, i+1}
        X[i*blocksize:(i+1)*blocksize, (i+1)*blocksize:(i+2)*blocksize] = \
            MMM_explicit(
                MMM_explicit(-U_blk_inv,
                     U[i*blocksize:(i+1)*blocksize, (i+1)*blocksize:(i+2)*blocksize]),
                     X[(i+1)*blocksize:(i+2)*blocksize, (i+1)*blocksize:(i+2)*blocksize])

        # X_{i, i} = (U_{i, i}^{-1} - X_{i, i+1} L_{i+1, i}) L_{i, i}^{-1}
        X[i*blocksize:(i+1)*blocksize, i*blocksize:(i+1)*blocksize] = \
            MMM_explicit(
                subs_explicit(
                    U_blk_inv,
                    MMM_explicit(
                        X[i*blocksize:(i+1)*blocksize, (i+1)*blocksize:(i+2)*blocksize],
                        L[(i+1)*blocksize:(i+2)*blocksize, i*blocksize:(i+1)*blocksize])),
            L_blk_inv)

    return X



def lu_sinv_tridiag_arrowhead(
    L: np.ndarray,
    U: np.ndarray,
    diag_blocksize: int,
    arrow_blocksize: int,
) -> np.ndarray:
    """ Perform a selected inversion from a lu decomposed matrix with a
    block tridiagonal arrowhead structure.
    
    Parameters
    ----------
    L : np.ndarray
        Lower factor of the lu factorization fo the matrix.
    U : np.ndarray
        Upper factor of the lu factorization fo the matrix.
    diag_blocksize : int
        Blocksize of the diagonals blocks of the matrix.
    arrow_blocksize : int
        Blocksize of the blocks composing the arrowhead.
    
    Returns
    -------
    X : np.ndarray
        Selected inversion of the matrix.
    """
    
    X = np.zeros(L.shape, dtype=L.dtype)
    
    L_last_blk_inv = np.zeros((arrow_blocksize, arrow_blocksize), dtype=L.dtype)
    U_last_blk_inv = np.zeros((arrow_blocksize, arrow_blocksize), dtype=L.dtype)

    L_last_blk_inv = la.solve_triangular(L[-arrow_blocksize:, -arrow_blocksize:], np.eye(arrow_blocksize), lower=True)
    U_last_blk_inv = la.solve_triangular(U[-arrow_blocksize:, -arrow_blocksize:], np.eye(arrow_blocksize), lower=False)
    
    X[-arrow_blocksize:, -arrow_blocksize:] = U_last_blk_inv @ L_last_blk_inv

    L_blk_inv = np.zeros((diag_blocksize, diag_blocksize), dtype=L.dtype)
    U_blk_inv = np.zeros((diag_blocksize, diag_blocksize), dtype=L.dtype)

    L_blk_inv = la.solve_triangular(L[-arrow_blocksize-diag_blocksize:-arrow_blocksize, -arrow_blocksize-diag_blocksize:-arrow_blocksize], np.eye(diag_blocksize), lower=True)
    U_blk_inv = la.solve_triangular(U[-arrow_blocksize-diag_blocksize:-arrow_blocksize, -arrow_blocksize-diag_blocksize:-arrow_blocksize], np.eye(diag_blocksize), lower=False)

    # X_{ndb+1, ndb} = -X_{ndb+1, ndb+1} L_{ndb+1, ndb} L_{ndb, ndb}^{-1}
    X[-arrow_blocksize:, -arrow_blocksize-diag_blocksize:-arrow_blocksize] = -X[-arrow_blocksize:, -arrow_blocksize:] @ L[-arrow_blocksize:, -arrow_blocksize-diag_blocksize:-arrow_blocksize] @ L_blk_inv

    # X_{ndb, ndb+1} = -U_{ndb, ndb}^{-1} U_{ndb, ndb+1} X_{ndb+1, ndb+1}
    X[-arrow_blocksize-diag_blocksize:-arrow_blocksize, -arrow_blocksize:] = -U_blk_inv @ U[-arrow_blocksize-diag_blocksize:-arrow_blocksize, -arrow_blocksize:] @ X[-arrow_blocksize:, -arrow_blocksize:]

    # X_{ndb, ndb} = (U_{ndb, ndb}^{-1} - X_{ndb, ndb+1} L_{ndb+1, ndb}) L_{ndb, ndb}^{-1}
    X[-arrow_blocksize-diag_blocksize:-arrow_blocksize, -arrow_blocksize-diag_blocksize:-arrow_blocksize] = (U_blk_inv - X[-arrow_blocksize-diag_blocksize:-arrow_blocksize, -arrow_blocksize:] @ L[-arrow_blocksize:, -arrow_blocksize-diag_blocksize:-arrow_blocksize]) @ L_blk_inv

    n_diag_blocks = (L.shape[0]-arrow_blocksize) // diag_blocksize 
    for i in range(n_diag_blocks-2, -1, -1):
        L_blk_inv = la.solve_triangular(L[i*diag_blocksize:(i+1)*diag_blocksize, i*diag_blocksize:(i+1)*diag_blocksize], np.eye(diag_blocksize), lower=True)
        U_blk_inv = la.solve_triangular(U[i*diag_blocksize:(i+1)*diag_blocksize, i*diag_blocksize:(i+1)*diag_blocksize], np.eye(diag_blocksize), lower=False)

        # --- Off-diagonal block part --- 
        # X_{i+1, i} = (-X_{i+1, i+1} L_{i+1, i} - X_{i+1, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1}
        X[(i+1)*diag_blocksize:(i+2)*diag_blocksize, i*diag_blocksize:(i+1)*diag_blocksize] = (-X[(i+1)*diag_blocksize:(i+2)*diag_blocksize, (i+1)*diag_blocksize:(i+2)*diag_blocksize] @ L[(i+1)*diag_blocksize:(i+2)*diag_blocksize, i*diag_blocksize:(i+1)*diag_blocksize] - X[(i+1)*diag_blocksize:(i+2)*diag_blocksize, -arrow_blocksize:] @ L[-arrow_blocksize:, i*diag_blocksize:(i+1)*diag_blocksize]) @ L_blk_inv

        # X_{i, i+1} = U_{i, i}^{-1} (- U_{i, i+1} X_{i+1, i+1} - U_{i, i+1} X_{i+1, i+1})
        X[i*diag_blocksize:(i+1)*diag_blocksize, (i+1)*diag_blocksize:(i+2)*diag_blocksize] = U_blk_inv @ (- U[i*diag_blocksize:(i+1)*diag_blocksize, (i+1)*diag_blocksize:(i+2)*diag_blocksize] @ X[(i+1)*diag_blocksize:(i+2)*diag_blocksize, (i+1)*diag_blocksize:(i+2)*diag_blocksize] - U[i*diag_blocksize:(i+1)*diag_blocksize, -arrow_blocksize:] @ X[-arrow_blocksize:, (i+1)*diag_blocksize:(i+2)*diag_blocksize])

        # --- Arrowhead part --- 
        # X_{ndb+1, i} = (- X_{ndb+1, i+1} L_{i+1, i} - X_{ndb+1, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1}
        X[-arrow_blocksize:, i*diag_blocksize:(i+1)*diag_blocksize] = (- X[-arrow_blocksize:, (i+1)*diag_blocksize:(i+2)*diag_blocksize] @ L[(i+1)*diag_blocksize:(i+2)*diag_blocksize, i*diag_blocksize:(i+1)*diag_blocksize] - X[-arrow_blocksize:, -arrow_blocksize:] @ L[-arrow_blocksize:, i*diag_blocksize:(i+1)*diag_blocksize]) @ L_blk_inv

        # X_{i, ndb+1} = U_{i, i}^{-1} (- U_{i, i+1} X_{i+1, ndb+1} - U_{i, ndb+1} X_{ndb+1, ndb+1})
        X[i*diag_blocksize:(i+1)*diag_blocksize, -arrow_blocksize:] = U_blk_inv @ (- U[i*diag_blocksize:(i+1)*diag_blocksize, (i+1)*diag_blocksize:(i+2)*diag_blocksize] @ X[(i+1)*diag_blocksize:(i+2)*diag_blocksize, -arrow_blocksize:] - U[i*diag_blocksize:(i+1)*diag_blocksize, -arrow_blocksize:] @ X[-arrow_blocksize:, -arrow_blocksize:]) 

        # --- Diagonal block part --- 
        # X_{i, i} = (U_{i, i}^{-1} - X_{i, i+1} L_{i+1, i} - X_{i, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1}
        X[i*diag_blocksize:(i+1)*diag_blocksize, i*diag_blocksize:(i+1)*diag_blocksize] = (U_blk_inv - X[i*diag_blocksize:(i+1)*diag_blocksize, (i+1)*diag_blocksize:(i+2)*diag_blocksize] @ L[(i+1)*diag_blocksize:(i+2)*diag_blocksize, i*diag_blocksize:(i+1)*diag_blocksize] - X[i*diag_blocksize:(i+1)*diag_blocksize, -arrow_blocksize:] @ L[-arrow_blocksize:, i*diag_blocksize:(i+1)*diag_blocksize]) @ L_blk_inv

    return X



def lu_sinv_ndiags(
    L: np.ndarray,
    U: np.ndarray,
    ndiags: int,
    blocksize: int,
) -> np.ndarray:
    """ Perform a selected inversion from a lu decomposed matrix with a
    block n-diagonals structure.
    
    Parameters
    ----------
    L : np.ndarray
        Lower factor of the lu factorization fo the matrix.
    U : np.ndarray
        Upper factor of the lu factorization fo the matrix.
    ndiags : int
        Number of diagonals.
    blocksize : int
        Size of the blocks.
    
    Returns
    -------
    X : np.ndarray
        Selected inversion of the matrix.
    """

    X = np.zeros(L.shape, dtype=L.dtype)

    L_blk_inv = np.zeros((blocksize, blocksize), dtype=L.dtype)
    U_blk_inv = np.zeros((blocksize, blocksize), dtype=L.dtype)

    nblocks = L.shape[0] // blocksize
    n_offdiags_blk = ndiags // 2
    for i in range(nblocks-1, -1, -1):
        # L_blk_inv = L_{i, i}^{-1}
        L_blk_inv = la.solve_triangular(L[i*blocksize:(i+1)*blocksize, i*blocksize:(i+1)*blocksize], np.eye(blocksize), lower=True)
        # U_blk_inv = U_{i, i}^{-1}
        U_blk_inv = la.solve_triangular(U[i*blocksize:(i+1)*blocksize, i*blocksize:(i+1)*blocksize], np.eye(blocksize), lower=False)


        # Off-diagonal block part
        for j in range(min(i+n_offdiags_blk, nblocks-1), i, -1):
            for k in range(i+1, min(i+n_offdiags_blk+1, nblocks), 1):
                # X_{j, i} = X_{j, i} - X_{j, k} L_{k, i}
                X[j*blocksize:(j+1)*blocksize, i*blocksize:(i+1)*blocksize] -= \
                    X[j*blocksize:(j+1)*blocksize, k*blocksize:(k+1)*blocksize] @ \
                    L[k*blocksize:(k+1)*blocksize, i*blocksize:(i+1)*blocksize]

                # X_{i, j} = X_{i, j} - U_{i, k} X_{k, j}
                X[i*blocksize:(i+1)*blocksize, j*blocksize:(j+1)*blocksize] -= \
                    U[i*blocksize:(i+1)*blocksize, k*blocksize:(k+1)*blocksize] @ \
                    X[k*blocksize:(k+1)*blocksize, j*blocksize:(j+1)*blocksize]
                
                print(f"X{[j*blocksize,(j+1)*blocksize, i*blocksize,(i+1)*blocksize]} -=\
                    X{[j*blocksize,(j+1)*blocksize, k*blocksize,(k+1)*blocksize]} @ \
                    L{[k*blocksize,(k+1)*blocksize, i*blocksize,(i+1)*blocksize]}")

                # X_{i, j} = X_{i, j} - U_{i, k} X_{k, j}
                print(f"X{[i*blocksize,(i+1)*blocksize, j*blocksize,(j+1)*blocksize]} -= \
                    U{[i*blocksize,(i+1)*blocksize, k*blocksize,(k+1)*blocksize]},\
                    X{[k*blocksize,(k+1)*blocksize, j*blocksize,(j+1)*blocksize]}")

            # X_{j, i} = X_{j, i} L_{i, i}^{-1}
            X[j*blocksize:(j+1)*blocksize, i*blocksize:(i+1)*blocksize] = \
                X[j*blocksize:(j+1)*blocksize, i*blocksize:(i+1)*blocksize] @ \
                    L_blk_inv
        
            # X_{i, j} = U_{i, i}^{-1} X_{i, j}
            X[i*blocksize:(i+1)*blocksize, j*blocksize:(j+1)*blocksize] = \
                U_blk_inv @ \
                    X[i*blocksize:(i+1)*blocksize, j*blocksize:(j+1)*blocksize]


        # Diagonal block part
        # X_{i, i} = (U_{i, i}^{-1} - sum_{k=i+1}^{min(i+ndiags/2, nblocks)} X_{i, k} L_{k, i}) L_{i, i}^{-1}
        
        # X_{i, i} = U_{i, i}^{-1}
        X[i*blocksize:(i+1)*blocksize, i*blocksize:(i+1)*blocksize] = U_blk_inv

        for k in range(i+1, min(i+n_offdiags_blk+1, nblocks), 1):
            # X_{i, i} = X_{i, i} - X_{i, k} L_{k, i}
            X[i*blocksize:(i+1)*blocksize, i*blocksize:(i+1)*blocksize] -= \
                X[i*blocksize:(i+1)*blocksize, k*blocksize:(k+1)*blocksize] @ \
                    L[k*blocksize:(k+1)*blocksize, i*blocksize:(i+1)*blocksize]

        # X_{i, i} = X_{i, i} L_{i, i}^{-1}
        X[i*blocksize:(i+1)*blocksize, i*blocksize:(i+1)*blocksize] = X[i*blocksize:(i+1)*blocksize, i*blocksize:(i+1)*blocksize] @ L_blk_inv

    return X



def lu_sinv_ndiags_explicit(
    L: np.ndarray,
    U: np.ndarray,
    ndiags: int,
    blocksize: int,
) -> np.ndarray:
    """ Perform a selected inversion from a lu decomposed matrix with a
    block n-diagonals structure.
    
    Parameters
    ----------
    L : np.ndarray
        Lower factor of the lu factorization fo the matrix.
    U : np.ndarray
        Upper factor of the lu factorization fo the matrix.
    ndiags : int
        Number of diagonals.
    blocksize : int
        Size of the blocks.
    
    Returns
    -------
    X : np.ndarray
        Selected inversion of the matrix.
    """

    X = np.zeros(L.shape, dtype=L.dtype)

    L_blk_inv = np.zeros((blocksize, blocksize), dtype=L.dtype)
    U_blk_inv = np.zeros((blocksize, blocksize), dtype=L.dtype)

    nblocks = L.shape[0] // blocksize
    n_offdiags_blk = ndiags // 2
    for i in range(nblocks-1, -1, -1):
    # for i in range(nblocks-1):
        # L_blk_inv = L_{i, i}^{-1}
        # L_blk_inv = la.solve_triangular(L[i*blocksize:(i+1)*blocksize, i*blocksize:(i+1)*blocksize], np.eye(blocksize), lower=True)
        # # U_blk_inv = U_{i, i}^{-1}
        # U_blk_inv = la.solve_triangular(U[i*blocksize:(i+1)*blocksize, i*blocksize:(i+1)*blocksize], np.eye(blocksize), lower=False)


        LU = L[i*blocksize:(i+1)*blocksize, i*blocksize:(i+1)*blocksize] + U[i*blocksize:(i+1)*blocksize, i*blocksize:(i+1)*blocksize] - np.eye(blocksize, dtype=L.dtype)
        for ii in range(blocksize):
            LU[ii, ii] = 1 / LU[ii, ii]
            for jj in range(ii):
                LU[jj,ii] = LU[jj, ii] * LU[jj, jj]
                for kk in range(jj+1, ii):                    
                    LU[jj,ii] += LU[kk, ii] * LU[jj, kk]                
                LU[jj,ii] = - LU[jj,ii] * LU[ii, ii]
        
        for ii in range(blocksize):
            for jj in range(ii):
                for kk in range(jj+1, ii):     
                    LU[ii, jj] += LU[ii, kk] * LU[kk, jj]
                LU[ii,jj] = - LU[ii, jj]


        L_blk_inv = np.tril(LU, k=-1) + np.eye(blocksize, dtype=LU.dtype)
        U_blk_inv = np.triu(LU, k=0)   



        # Off-diagonal block part
        for j in range(min(i+n_offdiags_blk, nblocks-1), i, -1):
            for k in range(i+1, min(i+n_offdiags_blk+1, nblocks), 1):
                # X_{j, i} = X_{j, i} - X_{j, k} L_{k, i}
                X[j*blocksize:(j+1)*blocksize, i*blocksize:(i+1)*blocksize] -= X[j*blocksize:(j+1)*blocksize, k*blocksize:(k+1)*blocksize] @ L[k*blocksize:(k+1)*blocksize, i*blocksize:(i+1)*blocksize]

                # X_{i, j} = X_{i, j} - U_{i, k} X_{k, j}
                X[i*blocksize:(i+1)*blocksize, j*blocksize:(j+1)*blocksize] -= U[i*blocksize:(i+1)*blocksize, k*blocksize:(k+1)*blocksize] @ X[k*blocksize:(k+1)*blocksize, j*blocksize:(j+1)*blocksize]

            # X_{j, i} = X_{j, i} L_{i, i}^{-1}
            X[j*blocksize:(j+1)*blocksize, i*blocksize:(i+1)*blocksize] = X[j*blocksize:(j+1)*blocksize, i*blocksize:(i+1)*blocksize] @ L_blk_inv
        
            # X_{i, j} = U_{i, i}^{-1} X_{i, j}
            X[i*blocksize:(i+1)*blocksize, j*blocksize:(j+1)*blocksize] = U_blk_inv @ X[i*blocksize:(i+1)*blocksize, j*blocksize:(j+1)*blocksize]


        # Diagonal block part
        # X_{i, i} = (U_{i, i}^{-1} - sum_{k=i+1}^{min(i+ndiags/2, nblocks)} X_{i, k} L_{k, i}) L_{i, i}^{-1}
        
        # X_{i, i} = U_{i, i}^{-1}
        X[i*blocksize:(i+1)*blocksize, i*blocksize:(i+1)*blocksize] = U_blk_inv

        for k in range(i+1, min(i+n_offdiags_blk+1, nblocks), 1):
            # X_{i, i} = X_{i, i} - X_{i, k} L_{k, i}
            X[i*blocksize:(i+1)*blocksize, i*blocksize:(i+1)*blocksize] -= X[i*blocksize:(i+1)*blocksize, k*blocksize:(k+1)*blocksize] @ L[k*blocksize:(k+1)*blocksize, i*blocksize:(i+1)*blocksize]

        # X_{i, i} = X_{i, i} L_{i, i}^{-1}
        X[i*blocksize:(i+1)*blocksize, i*blocksize:(i+1)*blocksize] = X[i*blocksize:(i+1)*blocksize, i*blocksize:(i+1)*blocksize] @ L_blk_inv

    return X



def lu_sinv_ndiags_explicit_blocksize1(
    L: np.ndarray,
    U: np.ndarray,
    ndiags: int,
) -> np.ndarray:
    """ Perform a selected inversion from a lu decomposed matrix with a
    block n-diagonals structure.
    
    Parameters
    ----------
    L : np.ndarray
        Lower factor of the lu factorization fo the matrix.
    U : np.ndarray
        Upper factor of the lu factorization fo the matrix.
    ndiags : int
        Number of diagonals.
    blocksize : int
        Size of the blocks.
    
    Returns
    -------
    X : np.ndarray
        Selected inversion of the matrix.
    """

    X = np.zeros(L.shape, dtype=L.dtype)

    n = L.shape[0]
    for i in range(n-1, -1, -1):

        # Off-diagonal block part
        for j in range(min(i+ndiags, n-1), i, -1):
            for k in range(i+1, min(i+ndiags+1, n), 1):
                # X_{j, i} = X_{j, i} - X_{j, k} L_{k, i}
                X[j, i] -= X[j, k] * L[k, i]

                # X_{i, j} = X_{i, j} - U_{i, k} X_{k, j}
                X[i, j] -= U[i, k] * X[k, j]

            # X_{j, i} = X_{j, i} L_{i, i}^{-1}
            # X[j, i] = X[j, i] 
        
            # X_{i, j} = U_{i, i}^{-1} X_{i, j}
            X[i, j] = X[i, j] / U[i,i]


        # Diagonal block part
        # X_{i, i} = (U_{i, i}^{-1} - sum_{k=i+1}^{min(i+ndiags/2, nblocks)} X_{i, k} L_{k, i}) L_{i, i}^{-1}
        
        # X_{i, i} = U_{i, i}^{-1}
        X[i, i] = 1 / U[i,i]

        for k in range(i+1, min(i+ndiags+1, n), 1):
            # X_{i, i} = X_{i, i} - X_{i, k} L_{k, i}
            X[i, i] -= X[i, k] * L[k, i]


    return X






def lu_sinv_ndiags_arrowhead(
    L: np.ndarray,
    U: np.ndarray,
    ndiags: int,
    diag_blocksize: int,
    arrow_blocksize: int,
) -> np.ndarray:
    """ Perform a selected inversion from a lu decomposed matrix with a
    block tridiagonal arrowhead structure.
    
    Parameters
    ----------
    L : np.ndarray
        Lower factor of the lu factorization fo the matrix.
    U : np.ndarray
        Upper factor of the lu factorization fo the matrix.
    ndiags : int
        Number of diagonals.
    diag_blocksize : int
        Blocksize of the diagonals blocks of the matrix.
    arrow_blocksize : int
        Blocksize of the blocks composing the arrowhead.
    
    Returns
    -------
    X : np.ndarray
        Selected inversion of the matrix.
    """

    X = np.zeros(L.shape, dtype=L.dtype)

    L_last_blk_inv = np.zeros((arrow_blocksize, arrow_blocksize), dtype=L.dtype)
    U_last_blk_inv = np.zeros((arrow_blocksize, arrow_blocksize), dtype=L.dtype)

    L_last_blk_inv = la.solve_triangular(L[-arrow_blocksize:, -arrow_blocksize:], np.eye(arrow_blocksize), lower=True)
    U_last_blk_inv = la.solve_triangular(U[-arrow_blocksize:, -arrow_blocksize:], np.eye(arrow_blocksize), lower=False)
    
    X[-arrow_blocksize:, -arrow_blocksize:] = U_last_blk_inv @ L_last_blk_inv

    L_blk_inv = np.zeros((diag_blocksize, diag_blocksize), dtype=L.dtype)
    U_blk_inv = np.zeros((diag_blocksize, diag_blocksize), dtype=L.dtype)

    n_diag_blocks = (L.shape[0]-arrow_blocksize) // diag_blocksize 
    n_offdiags_blk = ndiags // 2
    for i in range(n_diag_blocks-1, -1, -1):
        # L_blk_inv = L_{i, i}^{-1}
        L_blk_inv = la.solve_triangular(L[i*diag_blocksize:(i+1)*diag_blocksize, i*diag_blocksize:(i+1)*diag_blocksize], np.eye(diag_blocksize), lower=True)

        # U_blk_inv = U_{i, i}^{-1}
        U_blk_inv = la.solve_triangular(U[i*diag_blocksize:(i+1)*diag_blocksize, i*diag_blocksize:(i+1)*diag_blocksize], np.eye(diag_blocksize), lower=False)


        # Arrowhead part
        # X_{ndb+1, i} = - X_{ndb+1, ndb+1} L_{ndb+1, i}
        X[-arrow_blocksize:, i*diag_blocksize:(i+1)*diag_blocksize] = - X[-arrow_blocksize:, -arrow_blocksize:] @ L[-arrow_blocksize:, i*diag_blocksize:(i+1)*diag_blocksize]

        # X_{i, ndb+1} = - U_{i, ndb+1} X_{ndb+1, ndb+1}
        X[i*diag_blocksize:(i+1)*diag_blocksize, -arrow_blocksize:] = - U[i*diag_blocksize:(i+1)*diag_blocksize, -arrow_blocksize:] @ X[-arrow_blocksize:, -arrow_blocksize:]

        for k in range(i+1, min(i+n_offdiags_blk+1, n_diag_blocks), 1):
            # X_{ndb+1, i} = X_{ndb+1, i} - X_{ndb+1, k} L_{k, i}
            X[-arrow_blocksize:, i*diag_blocksize:(i+1)*diag_blocksize] -= X[-arrow_blocksize:, k*diag_blocksize:(k+1)*diag_blocksize] @ L[k*diag_blocksize:(k+1)*diag_blocksize, i*diag_blocksize:(i+1)*diag_blocksize]

            # X_{i, ndb+1} = X_{i, ndb+1} - U_{i, k} X_{k, ndb+1}
            X[i*diag_blocksize:(i+1)*diag_blocksize, -arrow_blocksize:] -= U[i*diag_blocksize:(i+1)*diag_blocksize, k*diag_blocksize:(k+1)*diag_blocksize] @ X[k*diag_blocksize:(k+1)*diag_blocksize, -arrow_blocksize:]

        # X_{ndb+1, i} = X_{ndb+1, i} L_{i, i}^{-1}
        X[-arrow_blocksize:, i*diag_blocksize:(i+1)*diag_blocksize] = X[-arrow_blocksize:, i*diag_blocksize:(i+1)*diag_blocksize] @ L_blk_inv

        # X_{i, ndb+1} = U_{i, i}^{-1} X_{i, ndb+1}
        X[i*diag_blocksize:(i+1)*diag_blocksize, -arrow_blocksize:] = U_blk_inv @ X[i*diag_blocksize:(i+1)*diag_blocksize, -arrow_blocksize:]


        # Off-diagonal block part
        for j in range(min(i+n_offdiags_blk, n_diag_blocks-1), i, -1):
            # Take the effect of the arrowhead part into account
            # X_{j, i} = - X_{j, ndb+1} L_{ndb+1, i}
            X[j*diag_blocksize:(j+1)*diag_blocksize, i*diag_blocksize:(i+1)*diag_blocksize] = - X[j*diag_blocksize:(j+1)*diag_blocksize, -arrow_blocksize:] @ L[-arrow_blocksize:, i*diag_blocksize:(i+1)*diag_blocksize]

            # X_{i, j} = - U_{i, ndb+1} X_{ndb+1, j}
            X[i*diag_blocksize:(i+1)*diag_blocksize, j*diag_blocksize:(j+1)*diag_blocksize] = - U[i*diag_blocksize:(i+1)*diag_blocksize, -arrow_blocksize:] @ X[-arrow_blocksize:, j*diag_blocksize:(j+1)*diag_blocksize]

            for k in range(i+1, min(i+n_offdiags_blk+1, n_diag_blocks), 1):
                # X_{j, i} = X_{j, i} - X_{j, k} L_{k, i}
                X[j*diag_blocksize:(j+1)*diag_blocksize, i*diag_blocksize:(i+1)*diag_blocksize] -= X[j*diag_blocksize:(j+1)*diag_blocksize, k*diag_blocksize:(k+1)*diag_blocksize] @ L[k*diag_blocksize:(k+1)*diag_blocksize, i*diag_blocksize:(i+1)*diag_blocksize]

                # X_{i, j} = X_{i, j} - U_{i, k} X_{k, j}
                X[i*diag_blocksize:(i+1)*diag_blocksize, j*diag_blocksize:(j+1)*diag_blocksize] -= U[i*diag_blocksize:(i+1)*diag_blocksize, k*diag_blocksize:(k+1)*diag_blocksize] @ X[k*diag_blocksize:(k+1)*diag_blocksize, j*diag_blocksize:(j+1)*diag_blocksize]

            # X_{j, i} = X_{j, i} L_{i, i}^{-1}
            X[j*diag_blocksize:(j+1)*diag_blocksize, i*diag_blocksize:(i+1)*diag_blocksize] = X[j*diag_blocksize:(j+1)*diag_blocksize, i*diag_blocksize:(i+1)*diag_blocksize] @ L_blk_inv
        
            # X_{i, j} = U_{i, i}^{-1} X_{i, j}
            X[i*diag_blocksize:(i+1)*diag_blocksize, j*diag_blocksize:(j+1)*diag_blocksize] = U_blk_inv @ X[i*diag_blocksize:(i+1)*diag_blocksize, j*diag_blocksize:(j+1)*diag_blocksize]


        # Diagonal block part
        # X_{i, i} = (U_{i, i}^{-1} - X_{i, ndb+1} L_{ndb+1, i} - sum_{k=i+1}^{min(i+ndiags/2, n_diag_blocks)} X_{i, k} L_{k, i}) L_{i, i}^{-1}
        
        # X_{i, i} = U_{i, i}^{-1} - X_{i, ndb+1} L_{ndb+1, i}
        X[i*diag_blocksize:(i+1)*diag_blocksize, i*diag_blocksize:(i+1)*diag_blocksize] = U_blk_inv - X[i*diag_blocksize:(i+1)*diag_blocksize, -arrow_blocksize:] @ L[-arrow_blocksize:, i*diag_blocksize:(i+1)*diag_blocksize]

        for k in range(i+1, min(i+n_offdiags_blk+1, n_diag_blocks), 1):
            # X_{i, i} = X_{i, i} - X_{i, k} L_{k, i}
            X[i*diag_blocksize:(i+1)*diag_blocksize, i*diag_blocksize:(i+1)*diag_blocksize] -= X[i*diag_blocksize:(i+1)*diag_blocksize, k*diag_blocksize:(k+1)*diag_blocksize] @ L[k*diag_blocksize:(k+1)*diag_blocksize, i*diag_blocksize:(i+1)*diag_blocksize]

        # X_{i, i} = X_{i, i} L_{i, i}^{-1}
        X[i*diag_blocksize:(i+1)*diag_blocksize, i*diag_blocksize:(i+1)*diag_blocksize] = X[i*diag_blocksize:(i+1)*diag_blocksize, i*diag_blocksize:(i+1)*diag_blocksize] @ L_blk_inv
   
    return X











def GEMM_UL(invLU: np.ndarray) -> np.ndarray:
    """
    Perfom A_inv = U_inv @ L_inv
    Input invLU = L^(-1) + U^(-1) - eye(n)
    """
    C = np.zeros(invLU.shape, dtype=invLU.dtype)
    n = invLU.shape[0]

    # this is a triangular GEMM, observe the loop ranges.
    # also! The diagonal is a tricky one becasuse the diagonal
    # of matrix L^(1) is 1, so it is ommited. The elements on the 
    # diagonal of matrix invLU belong to U^(^1). When counting the 
    # diagonal A_inv, we don't want to square the diagonal!
    
    # We can explictly materialize L^(-1) with "proper" diagonal 
    # with ones on it, and U^(-1) separately. What would make more sense?
    for k in range(n):
        for i in range(k+1):            
            for j in range(k):
                C[i,j] += invLU[i,k]*invLU[k,j]
            C[i,k] += invLU[i,k]

    return C    
    # this is 


def invert_LU(in_LU: np.ndarray) -> np.ndarray:
    """
    NOTE: input matrix LU already has the inverted diagonal!
    """
    LU = copy.deepcopy(in_LU)
    n = LU.shape[0]
    A_inv = np.zeros(LU.shape, dtype=LU.dtype)
    for k in range(n):
        kk = n - k - 1
        # A_inv[kk, kk] = 1/LU[kk,kk]
        A_inv[kk, kk] = LU[kk,kk]
        for i in range(k):
            ii = n - i - 1
            # l = LU[k,i]
            # u = LU[kk,ii] * A_inv[ii, ii]
            LU[kk,ii]  = LU[kk,ii] * A_inv[ii, ii]

            for j in range(i+1, k):
                jj = n - j - 1
                # l += LU[k,j] * A_inv[j, i]
                LU[k,i] += LU[k,j] * A_inv[j, i]
                # u += LU[kk,jj] * A_inv[jj, ii]
                LU[kk,ii] += LU[kk,jj] * A_inv[jj, ii]
            # A_inv[k, i] = -l
            A_inv[k, i] = -LU[k,i]
            # A_inv[kk, ii] = -u * A_inv[kk,kk]
            A_inv[kk, ii] = -LU[kk,ii] * A_inv[kk,kk]
    return A_inv


def invert_L(L: np.ndarray) -> np.ndarray:
    n = L.shape[0]
    L_inv = np.zeros(L.shape, dtype=L.dtype)
    for k in range(n):
        L_inv[k, k] = 1 # 1/L[k,k]
        for i in range(k):
            s = 0
            for j in range(i, k):
                s = s+ L[k,j] * L_inv[j, i]
                # L_inv[i, k] -= L_inv[i, j] * L[j, k]
            # L_inv[k, i] = 0
            L_inv[k, i] = -s * L[k,k]
    return L_inv


def invert_U(U: np.ndarray) -> np.ndarray:
    n = U.shape[0]
    U_inv = np.zeros(U.shape, dtype=U.dtype)
    for kk in range(n):
        k = n - kk - 1
        U_inv[k, k] = 1/U[k,k]
        for ii in range(kk):
            i = n - ii - 1
            s = 0
            for jj in range(ii, kk):
                j = n - jj - 1
                s = s+ U[k,j] * U_inv[j, i]
                # L_inv[i, k] -= L_inv[i, j] * L[j, k]
            # L_inv[k, i] = 0
            U_inv[k, i] = -s * U_inv[k,k]
    return U_inv


def inplace_LU(in_A: np.ndarray,
            ndiags: int
) -> np.ndarray:
    """
    In-place LU decomposition of a matrix with a block n-diagonals structure."""
    A = copy.deepcopy(in_A)
    n = A.shape[0]
    # this is purely sequential
    for k in range(n):
        A[k, k] = 1 / A[k, k]  # <- this is NOT the part of LU! This is already a first step of INVERSION

        # 1D parallelism (order of size of the band)
        for i in range(k+1,min(k+1 + ndiags, n)):
            A[i, k] = A[i,k] * A[k,k]
        
        # 2D parallelism (order of size of the band^2).
        # this is a purest form of a GEMM call
        for i in range(k+1,min(k+1 + ndiags, n)):
            for j in range(k+1,min(k+1 + ndiags, n)):
                A[i,j]  -= A[i,k]*A[k,j]     
    return A




def inplace_double_trsm(in_A: np.ndarray,
                 in_B: np.ndarray,
                 in_C: np.ndarray):
    """
    In-place double triangular solve with multiple right-hand sides:
    B = tril(A)^-1 * B,
    C = triu(A)^-1 * B,
    where tril and triu are lower and upper triangular matrices, respectively.
    A is a result of in-place LU decomposition of A', so tril(A) = L, triu(A) = U, such that A' = L*U.
    
    !!! Note !!!
    The diagonal of A is already inverted! Normally, in-place LU decomposition does not invert the diagonal of A,
    and you need to DIVIDE by A[k,k]. Here, we MULTIPLY by A[k,k] instead, because later on, we need an inverse of
    this diagonal in the backward substitution step.
    Parameters:
    -----------
    in_A : np.ndarray
        Input in-place LU-factorized matrix,
    in_B : np.ndarray
        Tall-and-skinny matrix (vertical toblerone),
    in_C : np.ndarray
        short-and-fat matrix (horizontal toblerone),
    """
    B = copy.deepcopy(in_B)
    C = copy.deepcopy(in_C)
    n, b = in_B.shape
    for k in range(b):
        for i in range(n):
            # trailing column of B has to be divided by the diagonal element of L
            B[i, k] = B[i,k ] * in_A[k,k ]
        for i in range(n):
            for j in range(k+1,b):
                B[i,j]  -= B[i,k]*in_A[k,j]

                C[j,i]  -= C[k,i]*in_A[j,k]

    return B,C



def inplace_GEMM(in_A: np.ndarray,
                 in_B: np.ndarray,
                 in_C: np.ndarray):
    """
    Nothing to add. Just good ol' GEMM  C = A*B + C
    """
    C = copy.deepcopy(in_C)
    M,K = in_A.shape
    N = in_B.shape[1]
    for k in range(K):
        for i in range(N):
            for j in range(M):
                C[i,j] += in_A[i,k]*in_B[k,j]
    return C