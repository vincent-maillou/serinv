# distutils: language = c++

"""Thin wrapper of CUBLAS."""

cimport cython  # NOQA

from libc.stdint cimport intptr_t

from cupy_backends.cuda.api import runtime
from cupy_backends.cuda import stream as stream_module

###############################################################################
# Extern
###############################################################################

cdef extern from '../../cupy_complex.h':
    ctypedef struct cuComplex 'cuComplex':
        float x, y

    ctypedef struct cuDoubleComplex 'cuDoubleComplex':
        double x, y

cdef extern from '../../cupy_blas.h' nogil:
    ctypedef void* Stream 'cudaStream_t'
    ctypedef int DataType 'cudaDataType'

    # Stream
    int cublasSetStream(Handle handle, Stream streamId)
    int cublasGetStream(Handle handle, Stream* streamId)

    # BLAS Level 3
    int cublasCherk(
        Handle handle, FillMode uplo, Operation trans, int n, int k,
        cuComplex* alpha, cuComplex* A, int lda,
        cuComplex* beta, cuComplex* C, int ldc)
    int cublasZherk(
        Handle handle, FillMode uplo, Operation trans, int n, int k,
        cuDoubleComplex* alpha, cuDoubleComplex* A, int lda,
        cuDoubleComplex* beta, cuDoubleComplex* C, int ldc)

###############################################################################
# Error handling
###############################################################################

cdef dict STATUS = {
    0: 'CUBLAS_STATUS_SUCCESS',
    1: 'CUBLAS_STATUS_NOT_INITIALIZED',
    3: 'CUBLAS_STATUS_ALLOC_FAILED',
    7: 'CUBLAS_STATUS_INVALID_VALUE',
    8: 'CUBLAS_STATUS_ARCH_MISMATCH',
    11: 'CUBLAS_STATUS_MAPPING_ERROR',
    13: 'CUBLAS_STATUS_EXECUTION_FAILED',
    14: 'CUBLAS_STATUS_INTERNAL_ERROR',
    15: 'CUBLAS_STATUS_NOT_SUPPORTED',
    16: 'CUBLAS_STATUS_LICENSE_ERROR',
}


cdef dict HIP_STATUS = {
    0: 'HIPBLAS_STATUS_SUCCESS',
    1: 'HIPBLAS_STATUS_NOT_INITIALIZED',
    2: 'HIPBLAS_STATUS_ALLOC_FAILED',
    3: 'HIPBLAS_STATUS_INVALID_VALUE',
    4: 'HIPBLAS_STATUS_MAPPING_ERROR',
    5: 'HIPBLAS_STATUS_EXECUTION_FAILED',
    6: 'HIPBLAS_STATUS_INTERNAL_ERROR',
    7: 'HIPBLAS_STATUS_NOT_SUPPORTED',
    8: 'HIPBLAS_STATUS_ARCH_MISMATCH',
    9: 'HIPBLAS_STATUS_HANDLE_IS_NULLPTR',
}


class CUBLASError(RuntimeError):

    def __init__(self, status):
        self.status = status
        cdef str err
        if runtime._is_hip_environment:
            err = HIP_STATUS[status]
        else:
            err = STATUS[status]
        super(CUBLASError, self).__init__(err)

    def __reduce__(self):
        return (type(self), (self.status,))


@cython.profile(False)
cpdef inline check_status(int status):
    if status != 0:
        raise CUBLASError(status)


cpdef setStream(intptr_t handle, size_t stream):
    # TODO(leofang): It seems most of cuBLAS APIs support stream capture (as of
    # CUDA 11.5) under certain conditions, see
    # https://docs.nvidia.com/cuda/cublas/index.html#CUDA-graphs
    # Before we come up with a robust strategy to test the support conditions,
    # we disable this functionality.
    if not runtime._is_hip_environment and runtime.streamIsCapturing(stream):
        raise NotImplementedError(
            'calling cuBLAS API during stream capture is currently '
            'unsupported')

    with nogil:
        status = cublasSetStream(<Handle>handle, <Stream>stream)
    check_status(status)

cdef _setStream(intptr_t handle):
    """Set current stream"""
    setStream(handle, stream_module.get_current_stream_ptr())

###############################################################################
# BLAS Level 3
###############################################################################

cpdef cherk(intptr_t handle, int uplo, int trans, int n, int k,
            size_t alpha, size_t A, int lda, size_t beta, size_t C, int ldc):
    _setStream(handle)
    with nogil:
        status = cublasCherk(
            <Handle>handle, <FillMode>uplo, <Operation>trans, n, k,
            <const cuComplex*>alpha, <const cuComplex*>A, lda,
            <const cuComplex*>beta, <cuComplex*>C, ldc)
    check_status(status)


cpdef zherk(intptr_t handle, int uplo, int trans, int n, int k,
            size_t alpha, size_t A, int lda, size_t beta, size_t C, int ldc):
    _setStream(handle)
    with nogil:
        status = cublasZherk(
            <Handle>handle, <FillMode>uplo, <Operation>trans, n, k,
            <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>A, lda,
            <const cuDoubleComplex*>beta, <cuDoubleComplex*>C, ldc)
    check_status(status)