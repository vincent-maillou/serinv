import numpy as np
import pytest
import copy
seed = 10


def sinv_ndiags(in_A: np.ndarray,
            ndiags: int
) -> np.ndarray:
    """
    Selected inversion of a band matrix with ndiags diagonals.
    There is no blocking (block size is 1).
    This results in poor parallelism (only 1D, and occasionally, 2D parallelism).
    """
    A = copy.deepcopy(in_A)
    n = A.shape[0]
    A_inv = np.zeros(in_A.shape, dtype=in_A.dtype)
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
                  
        # END in-place LU decomposition without pivoting

    # this is purely sequential
    for k in range(n-1, -1, -1):
        A_inv[k, k] = A[k,k]
        # 2D parallelism (order of size of the band^2), but more tricky
        for i in range(min(k+ndiags, n-1), k, -1):
            for j in range(min(k+ndiags, n-1), k, -1):
                A_inv[i, k] -= A_inv[i, j] * A[j, k]
                A_inv[k, i] -= A[k, j] * A_inv[j, i]
        
        # s = k
        # e = min(k+ndiags, n-1)
        # A_inv[s:e, k] -= A_inv[s:e, s:e] @ A[s:e, k]
        # A_inv[k, s:e] -= A_inv[s:e, s:e] @ A[k, s:e]

            A_inv[k, i] = A_inv[k, i]  * A[k,k]
            A_inv[k, k] -= A_inv[k, i] * A[i, k]
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
    # END debug only

    # in the following, we denote an additional iteration variable kk that would loop over b (for k in range(b))
    # to expose additinal degree of paralellism.

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


# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# ---- HELPER FUNCTIONS for in-place routines -------------------------
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
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






np.random.seed(seed)

def cut_to_banded(
    A: np.ndarray,
    ndiags : int
):
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if abs(i-j) > ndiags:
                A[i, j] = 0
    return A


def create_banded_matrix(
        matrix_size: int,
        ndiags: int
):
    A = np.zeros((matrix_size, matrix_size), dtype=np.float32)
    assert matrix_size >= 2*ndiags+1
    tmp = np.random.randint(1,10, size=(matrix_size,  2*ndiags+1)) #+ 1j * np.random.rand(matrix_size, 2*ndiags+1)
    for i in range(matrix_size):
        for j in range(max(0,i-ndiags), min(matrix_size, i+ndiags+1)):
            A[i, j] = tmp[i, j-i+ndiags]
    np.fill_diagonal(A, np.sum(np.abs(A), axis=1)+10)
    return A


np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))

@pytest.mark.parametrize(
        "matrix_size, ndiags",
        [(1, 0),
         (2, 0),
         (3, 0),
         (4, 0),
         (5, 0),
         (3, 1),
         (4, 1),
         (5, 1),
         (5, 2),
         (6, 2),
         (7, 2),
         (8, 2),
         (9, 3),
         (10, 2),
         (10, 4),
         (11, 4),
         (12, 4),
         (20, 5),
         (21, 5),
         (23, 5),
         (25, 5),
         (20, 6),
         (21, 6),
         (23, 6),
         (25, 6),
        ]
)


def test_sinv_ndiags(
        matrix_size: int,
        ndiags: int
):
    matrix_size = 12
    ndiags = 3
    if matrix_size % ndiags != 0:
        return
    A = create_banded_matrix(matrix_size, ndiags)
    # assert(np.allclose(reference_inverse @ A, np.eye(matrix_size), atol=1e-07))
    reference_inverse = np.linalg.inv(A)
    assert np.linalg.norm(reference_inverse @ A- np.eye(matrix_size))/np.linalg.norm(A) < 1e-7
    
    test_inverse = sinv_ndiags_with_enough_parallelism(A, ndiags, b=4)

    cut_to_banded(reference_inverse, ndiags)
    cut_to_banded(test_inverse, ndiags)
    assert np.allclose(test_inverse, reference_inverse)


if __name__ == "__main__":
    
    # [(1, 0),
    #      (2, 0),
    #      (3, 0),
    #      (4, 0),
    #      (5, 0),
    params =        [(3, 1),
         (4, 1),
         (5, 1),
         (5, 2),
         (6, 2),
         (7, 2),
         (8, 2),
         (9, 3),
         (10, 2),
         (10, 4),
         (11, 4),
         (12, 4),
         (20, 5),
         (21, 5),
         (23, 5),
         (25, 5),
         (18, 6),
         (24, 6),
         (30, 6),
         (36, 6),
         (24, 8),
         (32, 8),
         (40, 8),
         (56, 8)]
    # params = \
    #     [(128,2),
    #      (128,8),
    #     (128,16),
    #     (256,2),
    #     (256,8),
    #     (256,16),
    #     ]
    for param in params:
        test_sinv_ndiags(*param)