.. _serinv_docs_mainpage:

.. toctree::
    :maxdepth: 1
    :hidden:

    API Reference <api/index>

SerinV: Selected Decomposition Routines
====================================

Welcome to the documentation of SerinV!

What is SerinV?
------------

SerinV is a python package implementing a collection of algorithms performing
selected-factorization, selected-solving and selected-inversion on block matrices. 

These algorithms uses the structured sparsity pattern of the input matrix to save 
computation and memory in the factorization and solving process. In the case of 
the selected inversion the algorithms will also exploit the sparsity pattern of
the output matrix by only computing elements of the output matching the sparsity 
pattern of the input.

How is it implemented?
---------------------

SerinV algorithms are implemented in Python, performing blocks-wise operations, the
computationaly expensive section of the algorithms (block-wise inversion and 
block-wise gemm) calles ''numpy' that relies on your BLAS/LAPACK implementation.

GPU support is also available through the use of the ''cupy'' library, this allow
the user to use GPU acceleration on both NVIDIA and AMD GPUs.

What is implemented?
--------------------

Are currently supported:  

* Cholesky (selected) factorization/solve/inversion  
* Cholesky distributed (selected) factorization/inversion [only arrowhead matrices]
* LU (selected) factorization/solve/inversion  
* LU distributed (selected) factorization/inversion [only arrowhead matrices]

Check out the full :doc:`api/index` for details on usage.
