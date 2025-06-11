from setuptools import setup, Extension
from Cython.Build import cythonize

ext = Extension(
    name="cupyfix_backends.cuda.libs.cublas",
    sources=["src/serinv/cupyfix_backends/cuda/libs/cublas.pxd",
             "src/serinv/cupyfix_backends/cuda/libs/cublas.pyx",
             "src/serinv/cupyfix_backends/cuda/cupy_cublas.h",
             "src/serinv/cupyfix_backends/cuda/hip/cupy_cuComplex.h",
             "src/serinv/cupyfix_backends/cuda/hip/cupy_hip_common.h",
             "src/serinv/cupyfix_backends/cuda/hip/cupy_hipblas.h",
             "src/serinv/cupyfix_backends/cuda/stub/cupy_cublas.h",
             "src/serinv/cupyfix_backends/cuda/stub/cupy_cuComplex.h",
             "src/serinv/cupyfix_backends/cuda/cupy_blas.h"
             "src/serinv/cupyfix_backends/cuda/cupy_complex.h"],
             include_dirs=["cupyfix_backends"],
)

setup(
    name="cupyfix_backends",
    ext_modules=cythonize([ext]),
    packages=["src/serinv/cupyfix_backends.cuda.libs"],
)