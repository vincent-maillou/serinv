from serinv import _get_module_from_array

import numpy as np
from numpy.linalg import matmul

from scipy.linalg.blas import get_blas_funcs
from scipy.linalg._misc import _datacopied
from scipy.linalg._decomp import _asarray_validated

try:
    import cupy as cp
    from cupy_backends.cuda.libs import cublas
    from cupy import _core
    from cupy.cuda import device
except (ImportError, ImportWarning, ModuleNotFoundError):
    pass

def gemm (a, b, c=None, alpha=1.0, beta=0.0, trans_a ='N', trans_b ='N'):
    """Wrapper to call GeMM for host or device"""
    xp, la = _get_module_from_array(a)

    if xp == np:
        return matmul_gemm_host(a, b, alpha, beta, c, trans_a, trans_b)
    elif xp == cp:
        return matmul_gemm_device(trans_a, trans_b, a, b, c, alpha, beta)
    else:
        ModuleNotFoundError("Unknown Module")


def matmul_gemm_host(a, b, alpha=1.0, beta=0.0, c=None, trans_a=0, trans_b=0, overwrite_c=0, check_finite=False):
    """
    Solve the equation ``a x = b`` for `x`, assuming a is a triangular matrix.

    Parameters
    ----------
    a : (M, M) array_like
        A triangular matrix
    b : (M,) or (M, N) array_like
        Right-hand side matrix in ``a x = b``
    lower : bool, optional
        Use only data contained in the lower triangle of `a`.
        Default is to use upper triangle.
    trans : {0, 1, 2, 'N', 'T', 'C'}, optional
        Type of system to solve:

        ========  =========
        trans     system
        ========  =========
        0 or 'N'  a x  = b
        1 or 'T'  a^T x = b
        2 or 'C'  a^H x = b
        ========  =========
    unit_diagonal : bool, optional
        If True, diagonal elements of `a` are assumed to be 1 and
        will not be referenced.
    overwrite_b : bool, optional
        Allow overwriting data in `b` (may enhance performance)
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    x : (M,) or (M, N) ndarray
        Solution to the system ``a x = b``.  Shape of return matches `b`.

    Raises
    ------
    LinAlgError
        If `a` is singular

    Notes
    -----
    .. versionadded:: 0.9.0

    Examples
    --------
    Solve the lower triangular system a x = b, where::

             [3  0  0  0]       [4]
        a =  [2  1  0  0]   b = [2]
             [1  0  1  0]       [4]
             [1  1  1  1]       [2]

    >>> import numpy as np
    >>> from scipy.linalg import solve_triangular
    >>> a = np.array([[3, 0, 0, 0], [2, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]])
    >>> b = np.array([4, 2, 4, 2])
    >>> x = solve_triangular(a, b, lower=True)
    >>> x
    array([ 1.33333333, -0.66666667,  2.66666667, -1.33333333])
    >>> a.dot(x)  # Check the result
    array([ 4.,  2.,  4.,  2.])

    """

    a1 = _asarray_validated(a, check_finite=check_finite)
    b1 = _asarray_validated(b, check_finite=check_finite)
    if c != None:
        c1 = _asarray_validated(c, check_finite=check_finite)
    else:
        c1 = None

    if len(a1.shape) != 2 or a1.shape[0] != a1.shape[1]:
        raise ValueError('expected square matrix')

    if a1.shape[0] != b1.shape[0]:
        raise ValueError(f'shapes of a {a1.shape} and b {b1.shape} are incompatible')
    
    if beta != 0 and c1 == None:
        raise ValueError('expected C matrix')

    # accommodate empty arrays
    if b1.size == 0:
        dt_nonempty = matmul_gemm_host(
            np.eye(2, dtype=a1.dtype), np.ones(2, dtype=b1.dtype)
        ).dtype
        return np.empty_like(b1, dtype=dt_nonempty)
    
    x = _matmul_gemm(a1, b1, alpha, beta, c1, trans_a, trans_b, overwrite_c)
    return x


# solve_triangular without the input validation
def _matmul_gemm(a1, b1, alpha=1.0, beta=0.0, c1=None, trans_a=0, trans_b=0, overwrite_c=0):

    trans_a = {'N': 0, 'T': 1, 'C': 2}.get(trans_a, trans_a)
    trans_b = {'N': 0, 'T': 1, 'C': 2}.get(trans_b, trans_b)
    gemm, = get_blas_funcs(('gemm',), (a1, b1))

    if a1.dtype.char in 'fd':
        dtype = a1.dtype
    else:
        dtype = np.promote_types(a1.dtype.char, 'f')

    if beta == 0:
        x = gemm(alpha, a1, b1, beta=beta, trans_a=trans_a, trans_b=trans_b, overwrite_c=overwrite_c)
    else:
        x = gemm(alpha, a1, b1, beta, c1, trans_a, trans_b, overwrite_c)
    

    return x


# Util functions for cupy gemm
def _trans_to_cublas_op(trans):
    if trans == 'N' or trans == cublas.CUBLAS_OP_N:
        trans = cublas.CUBLAS_OP_N
    elif trans == 'T' or trans == cublas.CUBLAS_OP_T:
        trans = cublas.CUBLAS_OP_T
    elif trans == 'C' or trans == cublas.CUBLAS_OP_C:
        trans = cublas.CUBLAS_OP_C
    else:
        raise TypeError('invalid trans (actual: {})'.format(trans))
    return trans

def _decide_ld_and_trans(a, trans):
    ld = None
    if trans in (cublas.CUBLAS_OP_N, cublas.CUBLAS_OP_T):
        if a._f_contiguous:
            ld = a.shape[0]
        elif a._c_contiguous:
            ld = a.shape[1]
            trans = 1 - trans
    return ld, trans


def _change_order_if_necessary(a, lda):
    if lda is None:
        lda = a.shape[0]
        if not a._f_contiguous:
            a = a.copy(order='F')
    return a, lda

def _get_scalar_ptr(a, dtype):
    if isinstance(a, cp.ndarray):
        if a.dtype != dtype:
            a = cp.array(a, dtype=dtype)
        a_ptr = a.data.ptr
    else:
        if not (isinstance(a, np.ndarray) and a.dtype == dtype):
            a = np.array(a, dtype=dtype)
        a_ptr = a.ctypes.data
    return a, a_ptr
# Util functions for cupy gemm end


def matmul_gemm_device(transa, transb, a, b, out=None, alpha=1.0, beta=0.0):
    """Computes out = alpha * op(a) @ op(b) + beta * out

    op(a) = a if transa is 'N', op(a) = a.T if transa is 'T',
    op(a) = a.T.conj() if transa is 'C'.
    op(b) = b if transb is 'N', op(b) = b.T if transb is 'T',
    op(b) = b.T.conj() if transb is 'C'.
    """
    assert a.ndim == b.ndim == 2
    assert a.dtype == b.dtype
    dtype = a.dtype.char
    if dtype == 'f':
        func = cublas.sgemm
    elif dtype == 'd':
        func = cublas.dgemm
    elif dtype == 'F':
        func = cublas.cgemm
    elif dtype == 'D':
        func = cublas.zgemm
    else:
        raise TypeError('invalid dtype')
    

    transa = _trans_to_cublas_op(transa)
    transb = _trans_to_cublas_op(transb)
    if transa == cublas.CUBLAS_OP_N:
        m, k = a.shape
    else:
        k, m = a.shape
    if transb == cublas.CUBLAS_OP_N:
        n = b.shape[1]
        assert b.shape[0] == k
    else:
        n = b.shape[0]
        assert b.shape[1] == k
    if out is None:
        out = cp.empty((m, n), dtype=dtype, order='F')
        beta = 0.0
    else:
        assert out.ndim == 2
        assert out.shape == (m, n)
        assert out.dtype == dtype

    alpha, alpha_ptr = _get_scalar_ptr(alpha, a.dtype)
    beta, beta_ptr = _get_scalar_ptr(beta, a.dtype)
    handle = device.get_cublas_handle()
    orig_mode = cublas.getPointerMode(handle)
    if isinstance(alpha, cp.ndarray) or isinstance(beta, cp.ndarray):
        if not isinstance(alpha, cp.ndarray):
            alpha = cp.array(alpha)
            alpha_ptr = alpha.data.ptr
        if not isinstance(beta, cp.ndarray):
            beta = cp.array(beta)
            beta_ptr = beta.data.ptr
        cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_DEVICE)
    else:
        cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)

    lda, transa = _decide_ld_and_trans(a, transa)
    ldb, transb = _decide_ld_and_trans(b, transb)
    if not (lda is None or ldb is None):
        if out._f_contiguous:
            try:
                func(handle, transa, transb, m, n, k, alpha_ptr,
                     a.data.ptr, lda, b.data.ptr, ldb, beta_ptr, out.data.ptr,
                     m)
            finally:
                cublas.setPointerMode(handle, orig_mode)
            return out
        elif out._c_contiguous:
            # Computes out.T = alpha * b.T @ a.T + beta * out.T
            try:
                func(handle, 1 - transb, 1 - transa, n, m, k, alpha_ptr,
                     b.data.ptr, ldb, a.data.ptr, lda, beta_ptr, out.data.ptr,
                     n)
            finally:
                cublas.setPointerMode(handle, orig_mode)
            return out

    a, lda = _change_order_if_necessary(a, lda)
    b, ldb = _change_order_if_necessary(b, ldb)
    c = out
    if not out._f_contiguous:
        c = out.copy(order='F')
    try:
        func(handle, transa, transb, m, n, k, alpha_ptr, a.data.ptr, lda,
             b.data.ptr, ldb, beta_ptr, c.data.ptr, m)
    finally:
        cublas.setPointerMode(handle, orig_mode)
    if not out._f_contiguous:
        _core.elementwise_copy(c, out)
    return out