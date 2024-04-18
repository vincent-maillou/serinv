"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Lisa Gaedke-Merzhaeuser  (lisa.gaedke.merzhaeuser@usi.ch)
@date: 2024-03

Distributed implementation of lu factorization and selected inversion for 
block tridiagonal arrowhead matrices.

Copyright 2023-2024 ETH Zurich and USI. All rights reserved.
"""

import numpy as np
import numpy.linalg as npla
import scipy.linalg as scla
from mpi4py import MPI

from sdr.cholesky.cholesky_factorize import (
    cholesky_factorize_block_tridiagonal_arrowhead,
)
from sdr.cholesky.cholesky_selected_inversion import (
    cholesky_sinv_block_tridiagonal_arrowhead,
)


def cholesky_dist_block_tridiagonal_arrowhead(
    A_diagonal_blocks_local: np.ndarray,
    A_lower_diagonal_blocks_local: np.ndarray,
    A_arrow_bottom_blocks_local: np.ndarray,
    A_arrow_tip_block: np.ndarray,
    A_bridges_lower: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    diag_blocksize = A_diagonal_blocks_local.shape[0]
    arrow_blocksize = A_arrow_bottom_blocks_local.shape[0]
    n_diag_blocks_partition = A_diagonal_blocks_local.shape[1] // diag_blocksize

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()

    if comm_rank == 0:
        (
            L_diagonal_blocks_inv,
            L_lower_diagonal_blocks,
            L_arrow_bottom_blocks,
            Update_arrow_tip,
        ) = top_factorize(
            A_diagonal_blocks_local,
            A_lower_diagonal_blocks_local,
            A_upper_diagonal_blocks_local,
            A_arrow_bottom_blocks_local,
            A_arrow_right_blocks_local,
            A_arrow_tip_block,
        )

        (
            A_rs_diagonal_blocks,
            A_rs_lower_diagonal_blocks,
            A_rs_upper_diagonal_blocks,
            A_rs_arrow_bottom_blocks,
            A_rs_arrow_right_blocks,
            A_rs_arrow_tip_block,
        ) = create_reduced_system(
            A_diagonal_blocks_local,
            A_arrow_bottom_blocks_local,
            A_arrow_right_blocks_local,
            A_arrow_tip_block,
            Update_arrow_tip,
            A_bridges_lower,
            A_bridges_upper,
        )
    else:
        # Arrays that store the update of the 2sided pattern for the middle processes
        A_left_2sided_arrow_blocks_local = np.empty(
            (n_diag_blocks_partition * diag_blocksize, diag_blocksize),
            dtype=A_diagonal_blocks_local.dtype,
        )

        A_left_2sided_arrow_blocks_local[:diag_blocksize, :] = A_diagonal_blocks_local[
            :, :diag_blocksize
        ]
        A_left_2sided_arrow_blocks_local[diag_blocksize : 2 * diag_blocksize, :] = (
            A_lower_diagonal_blocks_local[:, :diag_blocksize]
        )

        (
            L_diagonal_blocks_inv,
            L_lower_diagonal_blocks,
            L_arrow_bottom_blocks,
            L_upper_2sided_arrow_blocks,
            Update_arrow_tip,
        ) = middle_factorize(
            A_diagonal_blocks_local,
            A_lower_diagonal_blocks_local,
            A_arrow_bottom_blocks_local,
            A_left_2sided_arrow_blocks_local,
            A_arrow_tip_block,
        )

        (
            A_rs_diagonal_blocks,
            A_rs_lower_diagonal_blocks,
            A_rs_upper_diagonal_blocks,
            A_rs_arrow_bottom_blocks,
            A_rs_arrow_right_blocks,
            A_rs_arrow_tip_block,
        ) = create_reduced_system(
            A_diagonal_blocks_local,
            A_arrow_bottom_blocks_local,
            A_arrow_tip_block,
            Update_arrow_tip,
            A_bridges_lower,
            A_left_2sided_arrow_blocks_local,
        )

    return ()


def top_factorize():
    return ()


def middle_factorize():
    return ()


def create_reduced_system():
    return ()
