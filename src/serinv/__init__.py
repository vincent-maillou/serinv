# Copyright 2023-2025 ETH Zurich. All rights reserved.

import os
from warnings import warn

import numpy as np
import scipy.linalg as np_la
from scipy.linalg import cholesky as sp_cholesky

from numpy.typing import ArrayLike
from functools import partial

from serinv.__about__ import __version__

backend_flags = {
    "cupy_avail": False,
    "nccl_avail": False,
    "mpi_avail": False,
    "mpi_cuda_aware": False,
}

try:
    import cupy as cp
    import cupyx.scipy.linalg as cu_la

    # In the case of CuPy, we want to use the lowerfill version
    # tweaked in serinv. (More performances)
    from serinv.cupyfix.cholesky_lowerfill import cholesky as cu_cholesky

    # Check if cupy is actually working. This could still raise
    # a cudaErrorInsufficientDriver error or something.
    cp.abs(1)

    backend_flags["cupy_avail"] = True
    try:
        # Check if NCCL is available
        from cupy.cuda import nccl

        nccl.get_version()
        backend_flags["nccl_avail"] = True

        nccl_datatype = {
            np.float32: nccl.NCCL_FLOAT,
            cp.float32: nccl.NCCL_FLOAT,
            np.complex64: nccl.NCCL_FLOAT,
            cp.complex64: nccl.NCCL_FLOAT,
            np.float64: nccl.NCCL_DOUBLE,
            cp.float64: nccl.NCCL_DOUBLE,
            np.complex128: nccl.NCCL_DOUBLE,
            cp.complex128: nccl.NCCL_DOUBLE,
        }

    except (AttributeError, ImportError, ModuleNotFoundError) as w:
        warn(f"'NCCL' is unavailable. ({w})")
except (ImportError, ImportWarning, ModuleNotFoundError) as w:
    warn(f"'CuPy' is unavailable. ({w})")


try:
    # Check if mpi4py is available
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    # Create a small GPU array
    array = np.array([comm_rank], dtype=np.float32)

    # Perform an MPI operation to check working
    if comm_size > 1:
        if comm_rank == 0:
            comm.Send([array, MPI.FLOAT], dest=1)
        elif comm_rank == 1:
            comm.Recv([array, MPI.FLOAT], source=0)

    backend_flags["mpi_avail"] = True
    if backend_flags["cupy_avail"] and os.environ.get("MPI_CUDA_AWARE", "0") == "1":
        backend_flags["mpi_cuda_aware"] = True

except (ImportError, ImportWarning, ModuleNotFoundError) as w:
    warn(f"'mpi4py' is unavailable. ({w})")


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
    if backend_flags["cupy_avail"]:
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
        if backend_flags["cupy_avail"]:
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
        return partial(sp_cholesky, lower=True, overwrite_a=False, check_finite=False)
    elif backend_flags["cupy_avail"] and module_str == "cupy":
        return cu_cholesky
    else:
        raise ValueError(f"Unknown module '{module_str}'.")


def _use_nccl(comm):
    """Handle NCCL GPU-to-GPU communication."""
    if backend_flags["nccl_avail"]:
        if isinstance(comm, nccl.NcclCommunicator):
            return True
    return False


def _get_nccl_parameters(arr, comm, rank, op: str):
    """Get the NCCL parameters for the given operation."""
    if np.iscomplexobj(arr):
        factor = 2
    else:
        factor = 1

    if backend_flags["nccl_avail"]:
        if op == "allgather":
            count = (arr.size // comm.size()) * factor
            displacement = count * rank * (arr.dtype.itemsize // factor)
        elif op == "allreduce":
            count = arr.size * factor
            displacement = 0
        else:
            raise ValueError(f"Not supported NCCL operation '{op}'.")

        datatype = nccl_datatype[arr.dtype.type]
    else:
        count = arr.size * factor
        displacement = 0
        datatype = None

    return count, displacement, datatype


__all__ = [
    "__version__",
    "ArrayLike",
    "backend_flags",
    "_get_module_from_array",
    "_get_module_from_str",
    "_get_cholesky",
    "_use_nccl",
    "_get_nccl_parameters",
]
