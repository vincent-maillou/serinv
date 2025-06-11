from setuptools import setup, Extension
from Cython.Build import cythonize

ext = Extension(
    name="cupyfix_backends.cuda.libs.cublas",
    sources=["cupyfix_backends/cuda/libs/cublas.pyx",
             "cupyfix_backends/cuda/cupy_cublas.h",
             "cupyfix_backends/cuda/hip/cupy_cuComplex.h",
             "cupyfix_backends/cuda/hip/cupy_hip_common.h",
             "cupyfix_backends/cuda/hip/cupy_hipblas.h",
             "cupyfix_backends/cuda/stub/cupy_cublas.h",
             "cupyfix_backends/cuda/stub/cupy_cuComplex.h",
             "cupyfix_backends/cuda/cupy_blas.h"
             "cupyfix_backends/cuda/cupy_complex.h"],
             include_dirs=["cupyfix_backends"],
)

setup(
    name="cupyfix_backends",
    ext_modules=cythonize([ext]),
    packages=["cupyfix_backend"],
)