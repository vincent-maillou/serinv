.. _sdr_docs_mainpage:

.. toctree::
    :maxdepth: 1
    :hidden:

    API Reference <api/index>

SDR: Selected Decomposition Routines
====================================

Welcome to the documentation for sdr!

What is SDR?
------------

SDR is a python package implementing a collection of algorithms performing
selected factorization, solving and inversion on block matrices. 

These algorithms uses the structured sparsity pattern of the input matrix to save 
computation and memory in the factorization and solving process. In the case of 
the selected inversion the algorithms will also exploit the sparsity pattern of
the output matrix by only computing and producing user-defined selected elements.

How it's implemented?
---------------------

SDR algorithms are implemented in Python, performing blocks-wise operations, the
computationaly expensive section of the algorithms (block-wise inversion and 
block-wise gemm) calles ''numpy' that relies on your BLAS/LAPACK implementation.

What is implemented?
--------------------

Are currently supported:  

* Cholesky (selected) factorization/solve/inversion  
* LU (selected) factorization/solve/inversion  

Check out the full :doc:`api/index` for details on usage.
