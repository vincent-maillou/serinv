"""Thin wrapper of CUBLAS."""
from libc.stdint cimport intptr_t


###############################################################################
# Types
###############################################################################

cdef extern from *:
    ctypedef void* cuComplexPtr 'cuComplex*'
    ctypedef void* cuDoubleComplexPtr 'cuDoubleComplex*'


cdef extern from *:
    ctypedef void* Handle 'cublasHandle_t'

    ctypedef int DiagType 'cublasDiagType_t'
    ctypedef int FillMode 'cublasFillMode_t'
    ctypedef int Operation 'cublasOperation_t'
    ctypedef int PointerMode 'cublasPointerMode_t'
    ctypedef int SideMode 'cublasSideMode_t'
    ctypedef int GemmAlgo 'cublasGemmAlgo_t'
    ctypedef int Math 'cublasMath_t'
    ctypedef int ComputeType 'cublasComputeType_t'

###############################################################################
# BLAS Level 3
###############################################################################


cpdef cherk(intptr_t handle, int uplo, int trans, int n, int k,
            size_t alpha, size_t A, int lda, size_t beta, size_t C, int ldc)
cpdef zherk(intptr_t handle, int uplo, int trans, int n, int k,
            size_t alpha, size_t A, int lda, size_t beta, size_t C, int ldc)
