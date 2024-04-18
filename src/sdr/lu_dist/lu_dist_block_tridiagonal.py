"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Lisa Gaedke-Merzhaeuser  (lisa.gaedke.merzhaeuser@usi.ch)
@date: 2024-03

Distributed implementation of lu factorization and selected inversion for 
block tridiagonal arrowhead matrices.

Copyright 2023-2024 ETH Zurich and USI. All rights reserved.
"""

import numpy as np
import scipy.linalg as la
from mpi4py import MPI

from sdr.lu.lu_factorize import lu_factorize_tridiag
from sdr.lu.lu_selected_inversion import lu_sinv_tridiag


def lu_dist_block_tridiagonal(
    A_diagonal_blocks_local: np.ndarray,
    A_lower_diagonal_blocks_local: np.ndarray,
    A_upper_diagonal_blocks_local: np.ndarray,
    A_bridges_lower: np.ndarray,
    A_bridges_upper: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    diag_blocksize = A_diagonal_blocks_local.shape[0]
    n_diag_blocks_partition = A_diagonal_blocks_local.shape[1] // diag_blocksize

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()

    if comm_rank == 0:
        () = top_factorize()
        () = create_reduced_system()
    elif comm_rank == comm.Get_size() - 1:
        () = bottom_factorize()
        () = create_reduced_system()
    else:
        () = middle_factorize()
        () = create_reduced_system()

    () = inverse_reduced_system()

    () = update_sinv_reduced_system()

    if comm_rank == 0:
        () = top_sinv()
    elif comm_rank == comm.Get_size() - 1:
        () = bottom_sinv()
    else:
        () = middle_sinv()

    return ()


def top_factorize():
    return ()


def bottom_factorize():
    return ()


def middle_factorize():
    return ()


def create_reduced_system():
    return ()


def inverse_reduced_system():
    return ()


def update_sinv_reduced_system():
    return ()


def top_sinv():
    return ()


def bottom_sinv():
    return ()


def middle_sinv():
    return ()

