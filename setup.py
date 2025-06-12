from setuptools import setup, Extension
from Cython.Build import cythonize
import os

CONDA_PREFIX = os.environ.get("CONDA_PREFIX", "")
CUDA_INCLUDE = os.path.join(CONDA_PREFIX, "include")



ext = Extension(
    name="cupyfix_backends.cuda.libs.cublas",
    sources=[
             "src/serinv/cupyfix_backends/cuda/libs/cublas.pyx"],
             include_dirs=["cupyfix_backends/cuda/libs",
                           "cupyfix_backends/hip",
                           "cupyfix_backends/stub",
                           "cupyfix_backends/cuda",
                           "cupyfix_backends",
                           CUDA_INCLUDE],
)

setup(
    name="cupyfix_backends",
    ext_modules=cythonize([ext]),
    packages=["src/serinv/cupyfix_backends.cuda.libs"],
)