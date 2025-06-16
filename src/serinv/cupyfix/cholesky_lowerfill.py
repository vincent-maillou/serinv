# Copyright 2023-2025 ETH Zurich. All rights reserved.
# Forked and modified from cupy.linalg.cholesky: https://github.com/cupy/cupy/blob/main/cupy/linalg/_decomposition.py#L157

import cupy
import numpy
from cupy.cuda import device
from cupy.linalg import _util

from cupy.cuda.nvtx import RangePush, RangePop


def cholesky(a: cupy.ndarray, lower=True) -> cupy.ndarray:
    """Cholesky decomposition.

    Decompose a given two-dimensional square matrix into ``L * L.H``,
    where ``L`` is a lower-triangular matrix and ``.H`` is a conjugate
    transpose operator.

    Note:
        This function call the cublas fill mode lower. Triggering the Cholesky
        "fast" kernel in cuSOLVER.

    Args:
        a (cupy.ndarray): Hermitian (symmetric if all elements are real),
            positive-definite input matrix with dimension ``(..., M, M)``.

    Returns:
        cupy.ndarray: The lower-triangular matrix of shape ``(..., M, M)``.

    .. warning::
        This function calls one or more cuSOLVER routine(s) which may yield
        invalid results if input conditions are not met.
        To detect these invalid results, you can set the `linalg`
        configuration to a value that is not `ignore` in
        :func:`cupyx.errstate` or :func:`cupyx.seterr`.

    .. seealso:: :func:`numpy.linalg.cholesky`
    """
    RangePush("cholesky")
    from cupy_backends.cuda.libs import cublas, cusolver

    _util._assert_cupy_array(a)
    _util._assert_stacked_2d(a)
    _util._assert_stacked_square(a)

    dtype, out_dtype = _util.linalg_common_type(a)
    if a.size == 0:
        return cupy.empty(a.shape, out_dtype)

    x = a.astype(dtype, order="C", copy=False)
    n = len(a)
    handle = device.get_cusolver_handle()
    dev_info = cupy.empty(1, dtype=numpy.int32)

    if lower:
        lower = cublas.CUBLAS_FILL_MODE_LOWER
    else:
        lower = cublas.CUBLAS_FILL_MODE_UPPER

    if dtype == "f":
        potrf = cusolver.spotrf
        potrf_bufferSize = cusolver.spotrf_bufferSize
    elif dtype == "d":
        potrf = cusolver.dpotrf
        potrf_bufferSize = cusolver.dpotrf_bufferSize
    elif dtype == "F":
        potrf = cusolver.cpotrf
        potrf_bufferSize = cusolver.cpotrf_bufferSize
    else:  # dtype == 'D':
        potrf = cusolver.zpotrf
        potrf_bufferSize = cusolver.zpotrf_bufferSize

    buffersize = potrf_bufferSize(
        handle, cublas.CUBLAS_FILL_MODE_LOWER, n, x.data.ptr, n
    )
    workspace = cupy.empty(buffersize, dtype=dtype)
    potrf(
        handle,
        lower,
        n,
        x.data.ptr,
        n,
        workspace.data.ptr,
        buffersize,
        dev_info.data.ptr,
    )
    cupy.linalg._util._check_cusolver_dev_info_if_synchronization_allowed(
        potrf, dev_info
    )

    _util._triu(x, k=0)
    cupy.conjugate(x, out=x)
    RangePop()
    return x.astype(out_dtype, copy=False).T
