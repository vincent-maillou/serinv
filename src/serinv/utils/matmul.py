from serinv import _get_module_from_array

import numpy as np
from numpy.linalg import matmul

from scipy.linalg.blas import get_blas_funcs
from scipy.linalg._misc import _datacopied
from scipy.linalg._decomp import _asarray_validated

try:
    import cupy as cp
    from cupy.cublas import gemm
except (ImportError, ImportWarning, ModuleNotFoundError):
    pass

def serinv_matmul (a, b, trans_a = 'N', trans_b = 'N'):
    """Wrapper to call GeMM for host or device"""
    xp, la = _get_module_from_array(a)

    if xp == np:
        return matmul_gemm_host(a, b, trans_a=trans_a, trans_b=trans_b)
    elif xp == cp:
        return gemm(trans_a, trans_b, a, b)
    else:
        ModuleNotFoundError("Unknown Module")


def matmul_gemm_host(a, b, trans_a=0, trans_b=0, overwrite_c=0, check_finite=False):
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

    if len(a1.shape) != 2 or a1.shape[0] != a1.shape[1]:
        raise ValueError('expected square matrix')

    if a1.shape[0] != b1.shape[0]:
        raise ValueError(f'shapes of a {a1.shape} and b {b1.shape} are incompatible')

    # accommodate empty arrays
    if b1.size == 0:
        dt_nonempty = matmul_gemm_host(
            np.eye(2, dtype=a1.dtype), np.ones(2, dtype=b1.dtype)
        ).dtype
        return np.empty_like(b1, dtype=dt_nonempty)

    x = _matmul_gemm(a1, b1, trans_a, trans_b, overwrite_c)
    return x


# solve_triangular without the input validation
def _matmul_gemm(a1, b1, trans_a=0, trans_b=0, overwrite_c=0):

    trans_a = {'N': 0, 'T': 1, 'C': 2}.get(trans_a, trans_a)
    trans_b = {'N': 0, 'T': 1, 'C': 2}.get(trans_b, trans_b)
    gemm, = get_blas_funcs(('gemm',), (a1, b1))

    if a1.dtype.char in 'fd':
        dtype = a1.dtype
    else:
        dtype = np.promote_types(a1.dtype.char, 'f')

    alpha = 1
    beta = 0
    
    x = gemm(alpha, a1, b1, beta=beta, trans_a=trans_a, trans_b=trans_b, overwrite_c=overwrite_c)
    

    return x
