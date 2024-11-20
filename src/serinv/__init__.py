# Copyright 2023-2024 ETH Zurich. All rights reserved.

import os
from warnings import warn

import numpy as np
import numpy.linalg.cholesky as np_cholesky
import scipy.linalg as np_la

from numpy.typing import ArrayLike

from serinv.__about__ import __version__

CupyAvail = False
try:
    import cupy as cp
    import cupyx.scipy.linalg as cu_la

    # In the case of CuPy, we want to use the lowerfill version
    # tweaked in serinv. (More performances)
    from serinv.cupyfix.cholesky_lowerfill import cholesky_lowerfill as cu_cholesky

    # Check if cupy is actually working. This could still raise
    # a cudaErrorInsufficientDriver error or something.
    cp.abs(1)

    CupyAvail = True
except ImportError as e:
    warn(f"'CuPy' is unavailable. ({e})")

# Allows user to specify the device streaming behavior via an
# environment variable.
DEVICE_STREAMING = os.environ.get("DEVICE_STREAMING")
if DEVICE_STREAMING is None:
    # Default behavior is to stream on the device.
    DEVICE_STREAMING = True

def _get_array_module(arr: ArrayLike):
    """Return the array module of the input array.

    Parameters
    ----------
    arr : ArrayLike
        Input array.

    Returns
    -------
    module : module
        The array module of the input array. (numpy or cupy)
    la : module
        The linear algebra module of the array module. (scipy.linalg or cupyx.scipy.linalg)
    """
    if CupyAvail:
        xp = cp.get_array_module(arr)

        if xp == cp:
            return cp, cu_la
    
    return np, np_la

def _get_cholesky(module):
    """Return the Cholesky factorization function of the input module.

    Parameters
    ----------
    module : module
        The array module. (numpy or cupy)

    Returns
    -------
    cholesky : function
        The Cholesky factorization function of the input module. (numpy.linalg.cholesky or serinv.cupyfix.cholesky_lowerfill)
    """
    if module == np:
        return np_cholesky
    elif CupyAvail and module == cp:
        return cu_cholesky
    else:
        raise ValueError(f"Unknown module '{module}'.")

__all__ = [
    "__version__",
    "ArrayLike",
    "CupyAvail",
    "_get_array_module",
    "_get_cholesky",
]
