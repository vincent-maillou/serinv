# Copyright 2023-2025 ETH Zurich. All rights reserved.

from warnings import warn

import numpy as np
from numpy.linalg import cholesky as np_cholesky
import scipy.linalg as np_la

from numpy.typing import ArrayLike

from serinv.__about__ import __version__

CUPY_AVAIL = False
try:
    import cupy as cp
    import cupyx.scipy.linalg as cu_la

    # In the case of CuPy, we want to use the lowerfill version
    # tweaked in serinv. (More performances)
    from serinv.cupyfix.cholesky_lowerfill import cholesky_lowerfill as cu_cholesky

    # Check if cupy is actually working. This could still raise
    # a cudaErrorInsufficientDriver error or something.
    cp.abs(1)

    CUPY_AVAIL = True
except ImportWarning as w:
    warn(f"'CuPy' is unavailable. ({w})")


def _get_module_from_array(arr: ArrayLike):
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
    if CUPY_AVAIL:
        xp = cp.get_array_module(arr)

        if xp == cp:
            return cp, cu_la

    return np, np_la


def _get_module_from_str(module_str: str):
    """Return the array module of the input string.

    Parameters
    ----------
    module_str : str
        The array module string. ("numpy" or "cupy")

    Returns
    -------
    module : module
        The array module of the input string. (numpy or cupy)
    la : module
        The linear algebra module of the array module. (scipy.linalg or cupyx.scipy.linalg)
    """
    if module_str == "numpy":
        return np, np_la
    elif module_str == "cupy":
        if CUPY_AVAIL:
            return cp, cu_la
        else:
            raise ImportError(
                "CuPy module have been requested but CuPy is not available."
            )
    else:
        raise ValueError(f"Unknown module '{module_str}'.")


def _get_cholesky(module_str: str):
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
    if module_str == "numpy":
        return np_cholesky
    elif CUPY_AVAIL and module_str == "cupy":
        return cu_cholesky
    else:
        raise ValueError(f"Unknown module '{module_str}'.")


__all__ = [
    "__version__",
    "ArrayLike",
    "CUPY_AVAIL",
    "_get_module_from_array",
    "_get_module_from_str",
    "_get_cholesky",
]
