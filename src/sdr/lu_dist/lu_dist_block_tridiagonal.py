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
        (
            L_diagonal_blocks_inv,
            L_lower_diagonal_blocks,
            U_diagonal_blocks,
            U_upper_diagonal_blocks,
        ) = top_factorize(
            A_diagonal_blocks_local,
            A_lower_diagonal_blocks_local,
            A_upper_diagonal_blocks_local,
        )

        (
            A_rs_diagonal_blocks,
            A_rs_lower_diagonal_blocks,
            A_rs_upper_diagonal_blocks,
        ) = create_reduced_system(
            A_diagonal_blocks_local,
            A_bridges_lower,
            A_bridges_upper,
        )
    elif comm_rank == comm.Get_size() - 1:
        (
            L_diagonal_blocks_inv,
            L_lower_diagonal_blocks,
            U_diagonal_blocks,
            U_upper_diagonal_blocks,
        ) = bottom_factorize(
            A_diagonal_blocks_local,
            A_lower_diagonal_blocks_local,
            A_upper_diagonal_blocks_local,
        )

        (
            A_rs_diagonal_blocks,
            A_rs_lower_diagonal_blocks,
            A_rs_upper_diagonal_blocks,
        ) = create_reduced_system(
            A_diagonal_blocks_local,
            A_bridges_lower,
            A_bridges_upper,
        )
    else:
        # Arrays that store the update of the 2sided pattern for the middle processes
        A_top_2sided_arrow_blocks_local = np.empty(
            (diag_blocksize, n_diag_blocks_partition * diag_blocksize),
            dtype=A_diagonal_blocks_local.dtype,
        )
        A_left_2sided_arrow_blocks_local = np.empty(
            (n_diag_blocks_partition * diag_blocksize, diag_blocksize),
            dtype=A_diagonal_blocks_local.dtype,
        )

        (
            L_diagonal_blocks_inv,
            L_lower_diagonal_blocks,
            L_upper_2sided_arrow_blocks,
            U_diagonal_blocks,
            U_upper_diagonal_blocks,
            U_left_2sided_arrow_blocks,
        ) = middle_factorize(
            A_diagonal_blocks_local,
            A_lower_diagonal_blocks_local,
            A_upper_diagonal_blocks_local,
            A_top_2sided_arrow_blocks_local,
            A_left_2sided_arrow_blocks_local,
        )

        (
            A_rs_diagonal_blocks,
            A_rs_lower_diagonal_blocks,
            A_rs_upper_diagonal_blocks,
        ) = create_reduced_system(
            A_diagonal_blocks_local,
            A_bridges_lower,
            A_bridges_upper,
            A_top_2sided_arrow_blocks_local,
            A_left_2sided_arrow_blocks_local,
        )

    (
        X_rs_diagonal_blocks,
        X_rs_lower_diagonal_blocks,
        X_rs_upper_diagonal_blocks,
    ) = inverse_reduced_system(
        A_rs_diagonal_blocks,
        A_rs_lower_diagonal_blocks,
        A_rs_upper_diagonal_blocks,
    )

    (
        X_diagonal_blocks_local,
        X_lower_diagonal_blocks_local,
        X_upper_diagonal_blocks_local,
        X_top_2sided_arrow_blocks_local,
        X_left_2sided_arrow_blocks_local,
        X_bridges_lower,
        X_bridges_upper,
    ) = update_sinv_reduced_system(
        X_rs_diagonal_blocks,
        X_rs_lower_diagonal_blocks,
        X_rs_upper_diagonal_blocks,
        n_diag_blocks_partition,
        diag_blocksize,
    )

    if comm_rank == 0:
        (
            X_diagonal_blocks_local,
            X_lower_diagonal_blocks_local,
            X_upper_diagonal_blocks_local,
        ) = top_sinv(
            X_diagonal_blocks_local,
            X_lower_diagonal_blocks_local,
            X_upper_diagonal_blocks_local,
            L_diagonal_blocks_inv,
            L_lower_diagonal_blocks,
            U_diagonal_blocks,
            U_upper_diagonal_blocks,
        )
    elif comm_rank == comm.Get_size() - 1:
        (
            X_diagonal_blocks_local,
            X_lower_diagonal_blocks_local,
            X_upper_diagonal_blocks_local,
        ) = bottom_sinv(
            X_diagonal_blocks_local,
            X_lower_diagonal_blocks_local,
            X_upper_diagonal_blocks_local,
            L_diagonal_blocks_inv,
            L_lower_diagonal_blocks,
            U_diagonal_blocks,
            U_upper_diagonal_blocks,
        )
    else:
        (
            X_diagonal_blocks_local,
            X_lower_diagonal_blocks_local,
            X_upper_diagonal_blocks_local,
        ) = middle_sinv(
            X_diagonal_blocks_local,
            X_lower_diagonal_blocks_local,
            X_upper_diagonal_blocks_local,
            X_top_2sided_arrow_blocks_local,
            X_left_2sided_arrow_blocks_local,
            L_diagonal_blocks_inv,
            L_lower_diagonal_blocks,
            L_upper_2sided_arrow_blocks,
            U_diagonal_blocks,
            U_upper_diagonal_blocks,
            U_left_2sided_arrow_blocks,
        )

    return (
        X_diagonal_blocks_local,
        X_lower_diagonal_blocks_local,
        X_upper_diagonal_blocks_local,
        X_bridges_lower,
        X_bridges_upper,
    )


def top_factorize(
    A_diagonal_blocks_local,
    A_lower_diagonal_blocks_local,
    A_upper_diagonal_blocks_local,
):
    return ()


def bottom_factorize(
    A_diagonal_blocks_local,
    A_lower_diagonal_blocks_local,
    A_upper_diagonal_blocks_local,
):
    return ()


def middle_factorize(
    A_diagonal_blocks_local,
    A_lower_diagonal_blocks_local,
    A_upper_diagonal_blocks_local,
    A_top_2sided_arrow_blocks_local,
    A_left_2sided_arrow_blocks_local,
):
    return ()


def create_reduced_system(
    A_diagonal_blocks_local,
    A_bridges_lower,
    A_bridges_upper,
    A_top_2sided_arrow_blocks_local=None,
    A_left_2sided_arrow_blocks_local=None,
):
    return ()


def inverse_reduced_system(
    A_rs_diagonal_blocks,
    A_rs_lower_diagonal_blocks,
    A_rs_upper_diagonal_blocks,
):
    return ()


def update_sinv_reduced_system(
    X_rs_diagonal_blocks,
    X_rs_lower_diagonal_blocks,
    X_rs_upper_diagonal_blocks,
    n_diag_blocks_partition,
    diag_blocksize,
):
    return ()


def top_sinv(
    X_diagonal_blocks_local,
    X_lower_diagonal_blocks_local,
    X_upper_diagonal_blocks_local,
    L_diagonal_blocks_inv,
    L_lower_diagonal_blocks,
    U_diagonal_blocks,
    U_upper_diagonal_blocks,
):
    return ()


def bottom_sinv(
    X_diagonal_blocks_local,
    X_lower_diagonal_blocks_local,
    X_upper_diagonal_blocks_local,
    L_diagonal_blocks_inv,
    L_lower_diagonal_blocks,
    U_diagonal_blocks,
    U_upper_diagonal_blocks,
):
    return ()


def middle_sinv(
    X_diagonal_blocks_local,
    X_lower_diagonal_blocks_local,
    X_upper_diagonal_blocks_local,
    X_top_2sided_arrow_blocks_local,
    X_left_2sided_arrow_blocks_local,
    L_diagonal_blocks_inv,
    L_lower_diagonal_blocks,
    L_upper_2sided_arrow_blocks,
    U_diagonal_blocks,
    U_upper_diagonal_blocks,
    U_left_2sided_arrow_blocks,
):
    return ()
