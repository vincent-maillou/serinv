from setuptools import setup, Extension
from Cython.Build import cythonize

ext = Extension(
    name="cupyfix_backends.cuda.libs.cublas",
    sources=["serinv/cupyfix_backends/cuda/libs/cublas.pyx",
             "serinv/cupyfix_backends/cuda/cupy_cublas.h",
             "serinv/cupyfix_backends/cuda/hip/cupy_cuComplex.h",
             "serinv/cupyfix_backends/cuda/hip/cupy_hip_common.h",
             "serinv/cupyfix_backends/cuda/hip/cupy_hipblas.h",
             "serinv/cupyfix_backends/cuda/stub/cupy_cublas.h",
             "serinv/cupyfix_backends/cuda/stub/cupy_cuComplex.h",
             "serinv/cupyfix_backends/cuda/cupy_blas.h"
             "serinv/cupyfix_backends/cuda/cupy_complex.h"],
             include_dirs=["cupyfix_backends"],
)

setup(
    name="cupyfix_backends",
    ext_modules=cythonize([ext]),
    packages=["cupyfix_backend"],
)