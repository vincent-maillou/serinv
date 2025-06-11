from setuptools import setup, Extension
from Cython.Build import cythonize

ext = Extension(
    name="cupyfix_backends.cuda.libs.cublas",
    sources=[
             "src/serinv/cupyfix_backends/cuda/libs/cublas.pyx"],
             include_dirs=["cupyfix_backends/cuda/libs",
                           "cupyfix_backends/hip",
                           "cupyfix_backends/stub",
                           "cupyfix_backends/cuda",
                           "cupyfix_backends"],
)

setup(
    name="cupyfix_backends",
    ext_modules=cythonize([ext]),
    packages=["src/serinv/cupyfix_backends.cuda.libs"],
)