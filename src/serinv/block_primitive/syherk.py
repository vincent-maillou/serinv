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

def syherk(a, c=None, trans=0, lower = False, unit_diagonal=False,
                            overwrite_c=False, check_finite=False, side=0):
    """Wrapper for the trsm function to call depending on wheter the solve happens on the host or the device
    
        For Compatibility this function accepts exactly the same parameters as what the scipy and cupy implementations accept
        plus the side parameter which can either be 0 or 1 for left or right hand side
    """
    xp, la = _get_module_from_array(a)
    if  xp == np:
        return matmul_syherk_host(a, c, trans, lower, unit_diagonal, overwrite_c, check_finite, side)
    elif xp == cp:
        return matmul_syherk_device(a, c, trans, lower, unit_diagonal, overwrite_c, check_finite, side)
    else:
        ModuleNotFoundError("Unknown Module")

def matmul_syherk_host(a, c=None, alpha=1.0, beta=1.0, trans=0, lower=False, unit_diagonal=False,
                     overwrite_c=False, check_finite=True, side=0):
    """Computes out = alpha * op(a) @ op(a)^T + beta * b

    op(a) = a if transa is 'N', op(a) = a.T if transa is 'T',
    op(a) = a.T.conj() if transa is 'C'.
    """

    a1 = _asarray_validated(a, check_finite=check_finite)
    if c is None:
        c1 = None
    else:
        c1 = _asarray_validated(c, check_finite=check_finite)


    if a1.shape[0] != c1.shape[0]:
        raise ValueError(f'shapes of a {a1.shape} and c {c1.shape} are incompatible')

    
    overwrite_c = overwrite_c or _datacopied(c1, c)

    x = _syherk(a1, c1, alpha, beta, trans, lower, unit_diagonal, overwrite_c, side)
    return x


# syherk without the input validation
def _syherk(a1, c1=None, alpha=1.0, beta=0.0, trans=0, lower=False, unit_diagonal=False,
                      overwrite_c=False, side=0):

    trans = {'N': 0, 'T': 1, 'C': 2}.get(trans, trans)

    if np.iscomplexobj(a1):
        syherk, = get_blas_funcs(('herk'), (a1, c1))
    else:
        syherk, = get_blas_funcs(('syrk'), (a1, c1))

    out = syherk(alpha, a1, beta, c1, trans, lower, overwrite_c)

    return out