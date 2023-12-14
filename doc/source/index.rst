.. _sdr_docs_mainpage:

.. toctree::
    :maxdepth: 1
    :hidden:

    API Reference <api/index>

SDR
=======

Welcome to the documentation for bsparse!

This is a Python package implementing block sparse matrices for
scientific computing.

Check out the full :doc:`api/index` for details on usage.


What bsparse is:
----------------
- bsparse implements sparse data containers (that always have an
  equivalent dense 2D array representation). As such, bsparse data
  containers can store ...

  - ... dense matrices (of arbitrary shape and type).
  - ... scipy.sparse matrices (of arbitrary shape and type).
  - ... bsparse matrices (of arbitrary shape and type).
  - ... a mixture of all three with the caveat that matrix rows and
    columns ***must not be ragged***. (Matrix sub-blocks have to be
    aligned along both axes.)

- bsparse Matrices implement...

  - ... all basic arithmetic operations (``+``, ``-``, ``*``, ``/`` )
  - ... dedicated matrix multiplication (``@``)
  - ... (conjugate) transposition (``.H``, ``.T``)
  - ... symmetry flags for square (skew) symmetric/Hermitian matrices
    to save memory
  - ... (sliced) element access, modification, and changes to the
    sparsity structure
  - ... conversion routines (between extending classes and to dense
    arrays)
  - ... instantiation routines (from `numpy.ndarray`, from
    `scipy.sparse.sparray`, ``eye``, ``diag`` (with a clever overlap
    functionality!), ``random`` (with a density parameter))
  - ... loading and saving to disk in ``.npz`` format

- A kind of bare bones (arguably worse) version of `scipy.sparse`
- Useful if you work with very large block-sparse matrices and want to
  implement algorithms making use of this structure.
- Pretty Pythonic and straightforward to use if you know a little bit of
  `numpy` and `scipy.sparse`.


What bsparse isn't:
-------------------
* A comprehensive library for sparse matrices. It is rather focused on
  providing a specific set of functionalities that were useful to the
  developers.
* A package that supports high performance linear algebra (for now).
  Check out `scipy.sparse` if optimized sparse algorithms are what
  you're after.


Why bsparse?
------------
In our work we often encounter large, sparse, often diagonally dominant,
block matrices. What we couldn't find was a suitable data structure in
Python, hence this little package.