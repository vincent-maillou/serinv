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

from sdr.lu.lu_factorize import lu_factorize_tridiag_arrowhead
from sdr.lu.lu_selected_inversion import lu_sinv_tridiag_arrowhead


def lu_dist_tridiagonal_arrowhead(
    A_diagonal_blocks_local: np.ndarray,
    A_lower_diagonal_blocks_local: np.ndarray,
    A_upper_diagonal_blocks_local: np.ndarray,
    A_arrow_bottom_blocks_local: np.ndarray,
    A_arrow_right_blocks_local: np.ndarray,
    A_arrow_tip_block: np.ndarray,
    A_bridges_lower: np.ndarray,
    A_bridges_upper: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Perform the distributed LU factorization and selected inversion of a
    block tridiagonal arrowhead matrix.

    Parameters
    ----------
    A_diagonal_blocks_local : np.ndarray
        Local par of the diagonal array.
    A_lower_diagonal_blocks_local : np.ndarray
        Local part of the lower diagonal array.
    A_upper_diagonal_blocks_local : np.ndarray
        Local part of the upper diagonal array.
    A_arrow_bottom_blocks_local : np.ndarray
        Local part of the arrow bottom array.
    A_arrow_right_blocks_local : np.ndarray
        Local part of the arrow right array.
    A_arrow_tip_block : np.ndarray
        Tip block of the arrow.
    A_bridges_lower : np.ndarray
        Lower bridges.
    A_bridges_upper : np.ndarray
        Upper bridges.

    Returns
    -------
    X_diagonal_blocks_local : np.ndarray
        Local part of the diagonal array of the inverse.
    X_lower_diagonal_blocks_local : np.ndarray
        Local part of the lower diagonal array of the inverse.
    X_upper_diagonal_blocks_local : np.ndarray
        Local part of the upper diagonal array of the inverse.
    X_arrow_bottom_blocks_local : np.ndarray
        Local part of the arrow bottom array of the inverse.
    X_arrow_right_blocks_local : np.ndarray
        Local part of the arrow right array of the inverse.
    X_global_arrow_tip : np.ndarray
        Global part of the arrow tip array of the inverse.
    X_bridges_lower : np.ndarray
        Lower bridges of the inverse. Only the one that belong to the local
        process are correct
    X_bridges_upper : np.ndarray
        Upper bridges of the inverse. Only the one that belong to the local
        process are correct

    Notes
    -----
    The algorithm use a non-pivoting LU factorization, hence the input matrix
    is considered diagonally dominant or block diagonally dominant.
    """
    diag_blocksize = A_diagonal_blocks_local.shape[0]
    arrow_blocksize = A_arrow_bottom_blocks_local.shape[0]
    n_diag_blocks_partition = A_diagonal_blocks_local.shape[1] // diag_blocksize

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()

    if comm_rank == 0:
        (
            L_diagonal_blocks,
            L_lower_diagonal_blocks,
            L_arrow_bottom_blocks,
            U_diagonal_blocks,
            U_upper_diagonal_blocks,
            U_arrow_right_blocks,
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
        A_top_2sided_arrow_blocks_local = np.empty(
            (diag_blocksize, n_diag_blocks_partition * diag_blocksize),
            dtype=A_diagonal_blocks_local.dtype,
        )
        A_left_2sided_arrow_blocks_local = np.empty(
            (n_diag_blocks_partition * diag_blocksize, diag_blocksize),
            dtype=A_diagonal_blocks_local.dtype,
        )

        A_top_2sided_arrow_blocks_local[:, :diag_blocksize] = A_diagonal_blocks_local[
            :, :diag_blocksize
        ]
        A_top_2sided_arrow_blocks_local[:, diag_blocksize : 2 * diag_blocksize] = (
            A_upper_diagonal_blocks_local[:, :diag_blocksize]
        )

        A_left_2sided_arrow_blocks_local[:diag_blocksize, :] = A_diagonal_blocks_local[
            :, :diag_blocksize
        ]
        A_left_2sided_arrow_blocks_local[diag_blocksize : 2 * diag_blocksize, :] = (
            A_lower_diagonal_blocks_local[:, :diag_blocksize]
        )

        (
            L_diagonal_blocks,
            L_lower_diagonal_blocks,
            L_arrow_bottom_blocks,
            L_upper_2sided_arrow_blocks,
            U_diagonal_blocks,
            U_upper_diagonal_blocks,
            U_arrow_right_blocks,
            U_left_2sided_arrow_blocks,
            Update_arrow_tip,
        ) = middle_factorize(
            A_diagonal_blocks_local,
            A_lower_diagonal_blocks_local,
            A_upper_diagonal_blocks_local,
            A_arrow_bottom_blocks_local,
            A_arrow_right_blocks_local,
            A_top_2sided_arrow_blocks_local,
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
            A_arrow_right_blocks_local,
            A_arrow_tip_block,
            Update_arrow_tip,
            A_bridges_lower,
            A_bridges_upper,
            A_top_2sided_arrow_blocks_local,
            A_left_2sided_arrow_blocks_local,
        )

    (
        X_rs_diagonal_blocks,
        X_rs_lower_diagonal_blocks,
        X_rs_upper_diagonal_blocks,
        X_rs_arrow_bottom_blocks,
        X_rs_arrow_right_blocks,
        X_rs_arrow_tip_block,
    ) = inverse_reduced_system(
        A_rs_diagonal_blocks,
        A_rs_lower_diagonal_blocks,
        A_rs_upper_diagonal_blocks,
        A_rs_arrow_bottom_blocks,
        A_rs_arrow_right_blocks,
        A_rs_arrow_tip_block,
    )

    (
        X_diagonal_blocks_local,
        X_lower_diagonal_blocks_local,
        X_upper_diagonal_blocks_local,
        X_arrow_bottom_blocks_local,
        X_arrow_right_blocks_local,
        X_top_2sided_arrow_blocks_local,
        X_left_2sided_arrow_blocks_local,
        X_global_arrow_tip,
        X_bridges_lower,
        X_bridges_upper,
    ) = update_sinv_reduced_system(
        X_rs_diagonal_blocks,
        X_rs_lower_diagonal_blocks,
        X_rs_upper_diagonal_blocks,
        X_rs_arrow_bottom_blocks,
        X_rs_arrow_right_blocks,
        X_rs_arrow_tip_block,
        n_diag_blocks_partition,
        diag_blocksize,
        arrow_blocksize,
    )

    if comm_rank == 0:
        (
            X_diagonal_blocks_local,
            X_lower_diagonal_blocks_local,
            X_upper_diagonal_blocks_local,
            X_arrow_bottom_blocks_local,
            X_arrow_right_blocks_local,
            _,
        ) = top_sinv(
            X_diagonal_blocks_local,
            X_lower_diagonal_blocks_local,
            X_upper_diagonal_blocks_local,
            X_arrow_bottom_blocks_local,
            X_arrow_right_blocks_local,
            X_global_arrow_tip,
            L_diagonal_blocks,
            L_lower_diagonal_blocks,
            L_arrow_bottom_blocks,
            U_diagonal_blocks,
            U_upper_diagonal_blocks,
            U_arrow_right_blocks,
        )
    else:
        (
            X_diagonal_blocks_local,
            X_lower_diagonal_blocks_local,
            X_upper_diagonal_blocks_local,
            X_arrow_bottom_blocks_local,
            X_arrow_right_blocks_local,
            _,
        ) = middle_sinv(
            X_diagonal_blocks_local,
            X_lower_diagonal_blocks_local,
            X_upper_diagonal_blocks_local,
            X_arrow_bottom_blocks_local,
            X_arrow_right_blocks_local,
            X_top_2sided_arrow_blocks_local,
            X_left_2sided_arrow_blocks_local,
            X_global_arrow_tip,
            L_diagonal_blocks,
            L_lower_diagonal_blocks,
            L_arrow_bottom_blocks,
            L_upper_2sided_arrow_blocks,
            U_diagonal_blocks,
            U_upper_diagonal_blocks,
            U_arrow_right_blocks,
            U_left_2sided_arrow_blocks,
        )

    return (
        X_diagonal_blocks_local,
        X_lower_diagonal_blocks_local,
        X_upper_diagonal_blocks_local,
        X_arrow_bottom_blocks_local,
        X_arrow_right_blocks_local,
        X_global_arrow_tip,
        X_bridges_lower,
        X_bridges_upper,
    )


def top_factorize(
    A_diagonal_blocks_local: np.ndarray,
    A_lower_diagonal_blocks_local: np.ndarray,
    A_upper_diagonal_blocks_local: np.ndarray,
    A_arrow_bottom_blocks_local: np.ndarray,
    A_arrow_right_blocks_local: np.ndarray,
    A_arrow_tip_block: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Proceed to the top-down LU factorization, called by the first process.

    Parameters
    ----------
    A_diagonal_blocks_local : np.ndarray
        Local part of the diagonal array.
    A_lower_diagonal_blocks_local : np.ndarray
        Local part of the lower diagonal array.
    A_upper_diagonal_blocks_local : np.ndarray
        Local part of the upper diagonal array.
    A_arrow_bottom_blocks_local : np.ndarray
        Local part of the arrow bottom array.
    A_arrow_right_blocks_local : np.ndarray
        Local part of the arrow right array.
    A_arrow_tip_block : np.ndarray
        Tip block of the arrow.

    Returns
    -------
    L_diagonal_blocks_local : np.ndarray
        Diagonal blocks of the local L factor.
    L_lower_diagonal_blocks_local : np.ndarray
        Lower diagonal blocks of the local L factor.
    L_arrow_bottom_blocks_local : np.ndarray
        Arrow bottom blocks of the local L factor.
    U_diagonal_blocks_local : np.ndarray
        Diagonal blocks of the local U factor.
    U_upper_diagonal_blocks_local : np.ndarray
        Upper diagonal blocks of the local U factor.
    U_arrow_right_blocks_local : np.ndarray
        Arrow right blocks of the local U factor.
    Update_arrow_tip_local : np.ndarray
        Local update of the arrow tip block.
    """
    diag_blocksize = A_diagonal_blocks_local.shape[0]
    nblocks = A_diagonal_blocks_local.shape[1] // diag_blocksize

    L_diagonal_blocks_local = np.empty_like(A_diagonal_blocks_local)
    L_lower_diagonal_blocks_local = np.empty_like(A_lower_diagonal_blocks_local)
    L_arrow_bottom_blocks_local = np.empty_like(A_arrow_bottom_blocks_local)
    U_diagonal_blocks_local = np.empty_like(A_diagonal_blocks_local)
    U_upper_diagonal_blocks_local = np.empty_like(A_upper_diagonal_blocks_local)
    U_arrow_right_blocks_local = np.empty_like(A_arrow_right_blocks_local)
    Update_arrow_tip_local = np.zeros_like(
        A_arrow_tip_block
    )  # Have to be zero-initialized

    for i in range(nblocks - 1):
        # L_{i, i}, U_{i, i} = lu_dcmp(A_{i, i})
        (
            L_diagonal_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize],
            U_diagonal_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize],
        ) = la.lu(
            A_diagonal_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize],
            permute_l=True,
        )

        # Compute lower factors
        U_inv_temp = la.solve_triangular(
            U_diagonal_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize],
            np.eye(diag_blocksize),
            lower=False,
        )

        # L_{i+1, i} = A_{i+1, i} @ U_local{i, i}^{-1}
        L_lower_diagonal_blocks_local[
            :, i * diag_blocksize : (i + 1) * diag_blocksize
        ] = (
            A_lower_diagonal_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            @ U_inv_temp
        )

        # L_{ndb+1, i} = A_{ndb+1, i} @ U{i, i}^{-1}
        L_arrow_bottom_blocks_local[
            :, i * diag_blocksize : (i + 1) * diag_blocksize
        ] = (
            A_arrow_bottom_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            @ U_inv_temp
        )

        # Compute upper factors
        L_inv_temp = la.solve_triangular(
            L_diagonal_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize],
            np.eye(diag_blocksize),
            lower=True,
        )

        # U_{i, i+1} = L_local{i, i}^{-1} @ A_{i, i+1}
        U_upper_diagonal_blocks_local[
            :, i * diag_blocksize : (i + 1) * diag_blocksize
        ] = (
            L_inv_temp
            @ A_upper_diagonal_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
        )

        # U_{i, ndb+1} = L{i, i}^{-1} @ A_{i, ndb+1}
        U_arrow_right_blocks_local[i * diag_blocksize : (i + 1) * diag_blocksize, :] = (
            L_inv_temp
            @ A_arrow_right_blocks_local[
                i * diag_blocksize : (i + 1) * diag_blocksize, :
            ]
        )

        # Update next diagonal block
        # A_{i+1, i+1} = A_{i+1, i+1} - L_{i+1, i} @ U_{i, i+1}
        A_diagonal_blocks_local[
            :, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize
        ] = (
            A_diagonal_blocks_local[
                :, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize
            ]
            - L_lower_diagonal_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            @ U_upper_diagonal_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
        )

        # Update next upper/lower blocks of the arrowhead
        # A_{ndb+1, i+1} = A_{ndb+1, i+1} - L_{ndb+1, i} @ U_{i, i+1}
        A_arrow_bottom_blocks_local[
            :, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize
        ] = (
            A_arrow_bottom_blocks_local[
                :, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize
            ]
            - L_arrow_bottom_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            @ U_upper_diagonal_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
        )

        # A_{i+1, ndb+1} = A_{i+1, ndb+1} - L_{i+1, i} @ U_{i, ndb+1}
        A_arrow_right_blocks_local[
            (i + 1) * diag_blocksize : (i + 2) * diag_blocksize, :
        ] = (
            A_arrow_right_blocks_local[
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize, :
            ]
            - L_lower_diagonal_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            @ U_arrow_right_blocks_local[
                i * diag_blocksize : (i + 1) * diag_blocksize, :
            ]
        )

        # Update the block at the tip of the arrowhead
        # A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, i} @ U_{i, ndb+1}
        Update_arrow_tip_local[:, :] = (
            Update_arrow_tip_local[:, :]
            - L_arrow_bottom_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            @ U_arrow_right_blocks_local[
                i * diag_blocksize : (i + 1) * diag_blocksize, :
            ]
        )

    # L_{nblocks, nblocks}, U_{nblocks, nblocks} = lu_dcmp(A_{nblocks, nblocks})
    (
        L_diagonal_blocks_local[:, -diag_blocksize:],
        U_diagonal_blocks_local[:, -diag_blocksize:],
    ) = la.lu(A_diagonal_blocks_local[:, -diag_blocksize:], permute_l=True)

    return (
        L_diagonal_blocks_local,
        L_lower_diagonal_blocks_local,
        L_arrow_bottom_blocks_local,
        U_diagonal_blocks_local,
        U_upper_diagonal_blocks_local,
        U_arrow_right_blocks_local,
        Update_arrow_tip_local,
    )


def middle_factorize(
    A_diagonal_blocks_local: np.ndarray,
    A_lower_diagonal_blocks_local: np.ndarray,
    A_upper_diagonal_blocks_local: np.ndarray,
    A_arrow_bottom_blocks_local: np.ndarray,
    A_arrow_right_blocks_local: np.ndarray,
    A_top_2sided_arrow_blocks_local: np.ndarray,
    A_left_2sided_arrow_blocks_local: np.ndarray,
    A_arrow_tip_block: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Proceed to the 2-sided LU factorization, called by the middle processes.

    Parameters
    ----------
    A_diagonal_blocks_local : np.ndarray
        Local part of the diagonal array.
    A_lower_diagonal_blocks_local : np.ndarray
        Local part of the lower diagonal array.
    A_upper_diagonal_blocks_local : np.ndarray
        Local part of the upper diagonal array.
    A_arrow_bottom_blocks_local : np.ndarray
        Local part of the arrow bottom array.
    A_arrow_right_blocks_local : np.ndarray
        Local part of the arrow right array.
    A_top_2sided_arrow_blocks_local : np.ndarray
        Array that stores the top update of the 2sided pattern for the middle processes.
    A_left_2sided_arrow_blocks_local : np.ndarray
        Array that stores the left update of the 2sided pattern for the middle processes.
    A_arrow_tip_block : np.ndarray
        Tip block of the arrow.

    Returns
    -------
    L_diagonal_blocks_local : np.ndarray
        Diagonal blocks of the local L factor.
    L_lower_diagonal_blocks_local : np.ndarray
        Lower diagonal blocks of the local L factor.
    L_arrow_bottom_blocks_local : np.ndarray
        Arrow bottom blocks of the local L factor.
    L_upper_2sided_arrow_blocks_local : np.ndarray
        Upper 2sided arrow blocks of the local L factor.
    U_diagonal_blocks_local : np.ndarray
        Diagonal blocks of the local U factor.
    U_upper_diagonal_blocks_local : np.ndarray
        Upper diagonal blocks of the local U factor.
    U_arrow_right_blocks_local : np.ndarray
        Arrow right blocks of the local U factor.
    U_left_2sided_arrow_blocks_local : np.ndarray
        Left 2sided arrow blocks of the local U factor.
    Update_arrow_tip_local : np.ndarray
        Local update of the arrow tip block.
    """
    diag_blocksize = A_diagonal_blocks_local.shape[0]
    n_blocks = A_diagonal_blocks_local.shape[1] // diag_blocksize

    L_diagonal_blocks_local = np.empty_like(A_diagonal_blocks_local)
    L_lower_diagonal_blocks_local = np.empty_like(A_lower_diagonal_blocks_local)
    L_arrow_bottom_blocks_local = np.empty_like(A_arrow_bottom_blocks_local)
    L_upper_2sided_arrow_blocks_local = np.empty_like(A_top_2sided_arrow_blocks_local)
    U_diagonal_blocks_local = np.empty_like(A_diagonal_blocks_local)
    U_upper_diagonal_blocks_local = np.empty_like(A_upper_diagonal_blocks_local)
    U_arrow_right_blocks_local = np.empty_like(A_arrow_right_blocks_local)
    U_left_2sided_arrow_blocks_local = np.empty_like(A_left_2sided_arrow_blocks_local)
    Update_arrow_tip_local = np.zeros_like(
        A_arrow_tip_block
    )  # Have to be zero-initialized

    for i in range(1, n_blocks - 1):
        # L_{i, i}, U_{i, i} = lu_dcmp(A_{i, i})
        (
            L_diagonal_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize],
            U_diagonal_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize],
        ) = la.lu(
            A_diagonal_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize],
            permute_l=True,
        )

        # Compute lower factors
        U_inv_temp = la.solve_triangular(
            U_diagonal_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize],
            np.eye(diag_blocksize),
            lower=False,
        )

        # L_{i+1, i} = A_{i+1, i} @ U{i, i}^{-1}
        L_lower_diagonal_blocks_local[
            :, i * diag_blocksize : (i + 1) * diag_blocksize
        ] = (
            A_lower_diagonal_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            @ U_inv_temp
        )

        # L_{top, i} = A_{top, i} @ U{i, i}^{-1}
        L_upper_2sided_arrow_blocks_local[
            :, i * diag_blocksize : (i + 1) * diag_blocksize
        ] = (
            A_top_2sided_arrow_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            @ U_inv_temp
        )

        # L_{ndb+1, i} = A_{ndb+1, i} @ U{i, i}^{-1}
        L_arrow_bottom_blocks_local[
            :, i * diag_blocksize : (i + 1) * diag_blocksize
        ] = (
            A_arrow_bottom_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            @ U_inv_temp
        )

        # Compute upper factors
        L_inv_temp = la.solve_triangular(
            L_diagonal_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize],
            np.eye(diag_blocksize),
            lower=True,
        )

        # U_{i, i+1} = L{i, i}^{-1} @ A_{i, i+1}
        U_upper_diagonal_blocks_local[
            :, i * diag_blocksize : (i + 1) * diag_blocksize
        ] = (
            L_inv_temp
            @ A_upper_diagonal_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
        )

        # U_{i, top} = L{i, i}^{-1} @ A_{i, top}
        U_left_2sided_arrow_blocks_local[
            i * diag_blocksize : (i + 1) * diag_blocksize, :
        ] = (
            L_inv_temp
            @ A_left_2sided_arrow_blocks_local[
                i * diag_blocksize : (i + 1) * diag_blocksize, :
            ]
        )

        # U_{i, ndb+1} = L{i, i}^{-1} @ A_{i, ndb+1}
        U_arrow_right_blocks_local[i * diag_blocksize : (i + 1) * diag_blocksize, :] = (
            L_inv_temp
            @ A_arrow_right_blocks_local[
                i * diag_blocksize : (i + 1) * diag_blocksize, :
            ]
        )

        # Update next diagonal block
        # A_{i+1, i+1} = A_{i+1, i+1} - L_{i+1, i} @ U_{i, i+1}
        A_diagonal_blocks_local[
            :, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize
        ] = (
            A_diagonal_blocks_local[
                :, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize
            ]
            - L_lower_diagonal_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            @ U_upper_diagonal_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
        )

        # Update next upper/lower blocks of the arrowhead
        # A_{ndb+1, i+1} = A_{ndb+1, i+1} - L_{ndb+1, i} @ U_{i, i+1}
        A_arrow_bottom_blocks_local[
            :, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize
        ] = (
            A_arrow_bottom_blocks_local[
                :, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize
            ]
            - L_arrow_bottom_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            @ U_upper_diagonal_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
        )

        # A_{i+1, ndb+1} = A_{i+1, ndb+1} - L_{i+1, i} @ U_{i, ndb+1}
        A_arrow_right_blocks_local[
            (i + 1) * diag_blocksize : (i + 2) * diag_blocksize, :
        ] = (
            A_arrow_right_blocks_local[
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize, :
            ]
            - L_lower_diagonal_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            @ U_arrow_right_blocks_local[
                i * diag_blocksize : (i + 1) * diag_blocksize, :
            ]
        )

        # Update the block at the tip of the arrowhead
        # A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, i} @ U_{i, ndb+1}
        Update_arrow_tip_local[:, :] = (
            Update_arrow_tip_local[:, :]
            - L_arrow_bottom_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            @ U_arrow_right_blocks_local[
                i * diag_blocksize : (i + 1) * diag_blocksize, :
            ]
        )

        # Update top and next upper/lower blocks of 2-sided factorization pattern
        # A_{top, top} = A_{top, top} - L_{top, i} @ U_{i, top}
        A_diagonal_blocks_local[:, :diag_blocksize] = (
            A_diagonal_blocks_local[:, :diag_blocksize]
            - L_upper_2sided_arrow_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            @ U_left_2sided_arrow_blocks_local[
                i * diag_blocksize : (i + 1) * diag_blocksize, :
            ]
        )

        # A_{i+1, top} = - L_{i+1, i} @ U_{i, top}
        A_left_2sided_arrow_blocks_local[
            (i + 1) * diag_blocksize : (i + 2) * diag_blocksize, :
        ] = (
            -L_lower_diagonal_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            @ U_left_2sided_arrow_blocks_local[
                i * diag_blocksize : (i + 1) * diag_blocksize, :
            ]
        )

        # A_local[top, i+1] = - L[top, i] @ U_[i, i+1]
        A_top_2sided_arrow_blocks_local[
            :, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize
        ] = (
            -L_upper_2sided_arrow_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            @ U_upper_diagonal_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
        )

        # Update the top (first blocks) of the arrowhead
        # A_{ndb+1, top} = A_{ndb+1, top} - L_{ndb+1, i} @ U_{i, top}
        A_arrow_bottom_blocks_local[:, :diag_blocksize] = (
            A_arrow_bottom_blocks_local[:, :diag_blocksize]
            - L_arrow_bottom_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            @ U_left_2sided_arrow_blocks_local[
                i * diag_blocksize : (i + 1) * diag_blocksize, :
            ]
        )

        # A_{top, ndb+1} = A_{top, ndb+1} - L_{top, i} @ U_{i, ndb+1}
        A_arrow_right_blocks_local[:diag_blocksize, :] = (
            A_arrow_right_blocks_local[:diag_blocksize, :]
            - L_upper_2sided_arrow_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            @ U_arrow_right_blocks_local[
                i * diag_blocksize : (i + 1) * diag_blocksize, :
            ]
        )

    # Compute the last LU blocks of the 2-sided factorization pattern
    # L_{nblocks, nblocks}, U_{nblocks, nblocks} = lu_dcmp(A_{nblocks, nblocks})
    (
        L_diagonal_blocks_local[:, -diag_blocksize:],
        U_diagonal_blocks_local[:, -diag_blocksize:],
    ) = la.lu(A_diagonal_blocks_local[:, -diag_blocksize:], permute_l=True)

    # Compute last lower factors
    U_inv_temp = la.solve_triangular(
        U_diagonal_blocks_local[:, -diag_blocksize:],
        np.eye(diag_blocksize),
        lower=False,
    )

    # L_{top, nblocks} = A_{top, nblocks} @ U{nblocks, nblocks}^{-1}
    L_upper_2sided_arrow_blocks_local[:, -diag_blocksize:] = (
        A_top_2sided_arrow_blocks_local[:, -diag_blocksize:] @ U_inv_temp
    )

    # L_{ndb+1, nblocks} = A_{ndb+1, nblocks} @ U{nblocks, nblocks}^{-1}
    L_arrow_bottom_blocks_local[:, -diag_blocksize:] = (
        A_arrow_bottom_blocks_local[:, -diag_blocksize:] @ U_inv_temp
    )

    # Compute last upper factors
    L_inv_temp = la.solve_triangular(
        L_diagonal_blocks_local[:, -diag_blocksize:],
        np.eye(diag_blocksize),
        lower=True,
    )

    # U_{nblocks, top} = L{nblocks, nblocks}^{-1} @ A_{nblocks, top}
    U_left_2sided_arrow_blocks_local[-diag_blocksize:, :] = (
        L_inv_temp @ A_left_2sided_arrow_blocks_local[-diag_blocksize:, :]
    )

    # U_{nblocks, ndb+1} = L{nblocks, nblocks}^{-1} @ A_{nblocks, ndb+1}
    U_arrow_right_blocks_local[-diag_blocksize:, :] = (
        L_inv_temp @ A_arrow_right_blocks_local[-diag_blocksize:, :]
    )

    # NOTE: On purpose, we don't update the tip of the arrowhead since the
    # propagation will appear during the inversion of the reduced system

    # Compute the top (first) LU blocks of the 2-sided factorization pattern
    # and its respective parts of the arrowhead
    # L_{top, top}, U_{top, top} = lu_dcmp(A_{top, top})
    (
        L_diagonal_blocks_local[:, :diag_blocksize],
        U_diagonal_blocks_local[:, :diag_blocksize],
    ) = la.lu(A_diagonal_blocks_local[:, :diag_blocksize], permute_l=True)

    # Compute top lower factors
    U_inv_temp = la.solve_triangular(
        U_diagonal_blocks_local[:, :diag_blocksize],
        np.eye(diag_blocksize),
        lower=False,
    )

    # L_{top+1, top} = A_{top+1, top} @ U{top, top}^{-1}
    L_lower_diagonal_blocks_local[:, :diag_blocksize] = (
        A_lower_diagonal_blocks_local[:, :diag_blocksize] @ U_inv_temp
    )

    # L_{ndb+1, top} = A_{ndb+1, top} @ U{top, top}^{-1}
    L_arrow_bottom_blocks_local[:, :diag_blocksize] = (
        A_arrow_bottom_blocks_local[:, :diag_blocksize] @ U_inv_temp
    )

    # Compute top upper factors
    L_inv_temp = la.solve_triangular(
        L_diagonal_blocks_local[:, :diag_blocksize],
        np.eye(diag_blocksize),
        lower=True,
    )

    # U_{top, top+1} = L{top, top}^{-1} @ A_{top, top+1}
    U_upper_diagonal_blocks_local[:, :diag_blocksize] = (
        L_inv_temp @ A_upper_diagonal_blocks_local[:, :diag_blocksize]
    )

    # U_{top, ndb+1} = L{top, top}^{-1} @ A_{top, ndb+1}
    U_arrow_right_blocks_local[:diag_blocksize, :] = (
        L_inv_temp @ A_arrow_right_blocks_local[:diag_blocksize, :]
    )

    return (
        L_diagonal_blocks_local,
        L_lower_diagonal_blocks_local,
        L_arrow_bottom_blocks_local,
        L_upper_2sided_arrow_blocks_local,
        U_diagonal_blocks_local,
        U_upper_diagonal_blocks_local,
        U_arrow_right_blocks_local,
        U_left_2sided_arrow_blocks_local,
        Update_arrow_tip_local,
    )


def create_reduced_system(
    A_diagonal_blocks_local: np.ndarray,
    A_arrow_bottom_blocks_local: np.ndarray,
    A_arrow_right_blocks_local: np.ndarray,
    A_arrow_tip_block: np.ndarray,
    Update_arrow_tip: np.ndarray,
    A_bridges_lower: np.ndarray,
    A_bridges_upper: np.ndarray,
    A_top_2sided_arrow_blocks_local: np.ndarray = None,
    A_left_2sided_arrow_blocks_local: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create the reduced system and broadcast it to all the processes.

    Parameters
    ----------
    A_diagonal_blocks_local : np.ndarray
        Local part of the diagonal array.
    A_arrow_bottom_blocks_local : np.ndarray
        Local part of the arrow bottom array.
    A_arrow_right_blocks_local : np.ndarray
        Local part of the arrow right array.
    A_arrow_tip_block : np.ndarray
        Tip block of the arrow.
    Update_arrow_tip : np.ndarray
        Update of the arrow tip block.
    A_bridges_lower : np.ndarray
        Lower part of the bridges array.
    A_bridges_upper : np.ndarray
        Upper part of the bridges array.
    A_top_2sided_arrow_blocks_local : np.ndarray, optional
        Array that stores the top update of the 2sided pattern for the middle processes.
    A_left_2sided_arrow_blocks_local : np.ndarray, optional
        Array that stores the left update of the 2sided pattern for the middle processes.

    Returns
    -------
    A_rs_diagonal_blocks_sum : np.ndarray
        Diagonal blocks of the reduced system.
    A_rs_lower_diagonal_blocks_sum : np.ndarray
        Lower diagonal blocks of the reduced system.
    A_rs_upper_diagonal_blocks_sum : np.ndarray
        Upper diagonal blocks of the reduced system.
    A_rs_arrow_bottom_blocks_sum : np.ndarray
        Arrow bottom blocks of the reduced system.
    A_rs_arrow_right_blocks_sum : np.ndarray
        Arrow right blocks of the reduced system.
    A_rs_arrow_tip_block_sum : np.ndarray
        Tip block of the reduced system.

    Notes
    -----
    This function represent the parallel bottleneck of the algorithm. It uses
    the MPI_Allreduce operation to sum the reduced system of each process and
    broadcast it to all the processes. After this communication step no more
    communication is needed.
    """
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    # Create empty matrix for reduced system -> (2 * #process - 1) * diag_blocksize + arrowhead_size
    diag_blocksize = A_diagonal_blocks_local.shape[0]
    arrow_blocksize = A_arrow_bottom_blocks_local.shape[0]
    n_diag_blocks_reduced_system = 2 * comm_size - 1

    A_rs_diagonal_blocks = np.zeros(
        (diag_blocksize, n_diag_blocks_reduced_system * diag_blocksize),
        dtype=A_diagonal_blocks_local.dtype,
    )
    A_rs_lower_diagonal_blocks = np.zeros(
        (diag_blocksize, (n_diag_blocks_reduced_system - 1) * diag_blocksize),
        dtype=A_diagonal_blocks_local.dtype,
    )
    A_rs_upper_diagonal_blocks = np.zeros(
        (diag_blocksize, (n_diag_blocks_reduced_system - 1) * diag_blocksize),
        dtype=A_diagonal_blocks_local.dtype,
    )
    A_rs_arrow_bottom_blocks = np.zeros(
        (arrow_blocksize, n_diag_blocks_reduced_system * diag_blocksize),
        dtype=A_diagonal_blocks_local.dtype,
    )
    A_rs_arrow_right_blocks = np.zeros(
        (n_diag_blocks_reduced_system * diag_blocksize, arrow_blocksize),
        dtype=A_diagonal_blocks_local.dtype,
    )
    A_rs_arrow_tip_block = np.zeros(
        (arrow_blocksize, arrow_blocksize), dtype=A_diagonal_blocks_local.dtype
    )

    A_rs_arrow_tip_block = Update_arrow_tip

    if comm_rank == 0:
        A_rs_diagonal_blocks[:, :diag_blocksize] = A_diagonal_blocks_local[
            :, -diag_blocksize:
        ]
        A_rs_upper_diagonal_blocks[:, :diag_blocksize] = A_bridges_upper[
            :, comm_rank * diag_blocksize : (comm_rank + 1) * diag_blocksize
        ]

        A_rs_arrow_bottom_blocks[:, :diag_blocksize] = A_arrow_bottom_blocks_local[
            :, -diag_blocksize:
        ]
        A_rs_arrow_right_blocks[:diag_blocksize, :] = A_arrow_right_blocks_local[
            -diag_blocksize:, :
        ]
    else:
        start_index = diag_blocksize + (comm_rank - 1) * 2 * diag_blocksize

        A_rs_lower_diagonal_blocks[:, start_index - diag_blocksize : start_index] = (
            A_bridges_lower[
                :, (comm_rank - 1) * diag_blocksize : comm_rank * diag_blocksize
            ]
        )

        A_rs_diagonal_blocks[:, start_index : start_index + diag_blocksize] = (
            A_diagonal_blocks_local[:, :diag_blocksize]
        )

        A_rs_upper_diagonal_blocks[:, start_index : start_index + diag_blocksize] = (
            A_top_2sided_arrow_blocks_local[:, -diag_blocksize:]
        )

        A_rs_lower_diagonal_blocks[:, start_index : start_index + diag_blocksize] = (
            A_left_2sided_arrow_blocks_local[-diag_blocksize:, :]
        )

        A_rs_diagonal_blocks[
            :, start_index + diag_blocksize : start_index + 2 * diag_blocksize
        ] = A_diagonal_blocks_local[:, -diag_blocksize:]

        if comm_rank != comm_size - 1:
            A_rs_upper_diagonal_blocks[
                :, start_index + diag_blocksize : start_index + 2 * diag_blocksize
            ] = A_bridges_upper[
                :, comm_rank * diag_blocksize : (comm_rank + 1) * diag_blocksize
            ]

        A_rs_arrow_bottom_blocks[:, start_index : start_index + diag_blocksize] = (
            A_arrow_bottom_blocks_local[:, :diag_blocksize]
        )

        A_rs_arrow_bottom_blocks[
            :, start_index + diag_blocksize : start_index + 2 * diag_blocksize
        ] = A_arrow_bottom_blocks_local[:, -diag_blocksize:]

        A_rs_arrow_right_blocks[start_index : start_index + diag_blocksize, :] = (
            A_arrow_right_blocks_local[:diag_blocksize, :]
        )

        A_rs_arrow_right_blocks[
            start_index + diag_blocksize : start_index + 2 * diag_blocksize, :
        ] = A_arrow_right_blocks_local[-diag_blocksize:, :]

    # Send the reduced_system with MPIallReduce SUM operation
    A_rs_diagonal_blocks_sum = np.zeros_like(A_rs_diagonal_blocks)
    A_rs_lower_diagonal_blocks_sum = np.zeros_like(A_rs_lower_diagonal_blocks)
    A_rs_upper_diagonal_blocks_sum = np.zeros_like(A_rs_upper_diagonal_blocks)
    A_rs_arrow_bottom_blocks_sum = np.zeros_like(A_rs_arrow_bottom_blocks)
    A_rs_arrow_right_blocks_sum = np.zeros_like(A_rs_arrow_right_blocks)
    A_rs_arrow_tip_block_sum = np.zeros_like(A_rs_arrow_tip_block)

    comm.Allreduce(
        [A_rs_diagonal_blocks, MPI.DOUBLE],
        [A_rs_diagonal_blocks_sum, MPI.DOUBLE],
        op=MPI.SUM,
    )
    comm.Allreduce(
        [A_rs_lower_diagonal_blocks, MPI.DOUBLE],
        [A_rs_lower_diagonal_blocks_sum, MPI.DOUBLE],
        op=MPI.SUM,
    )
    comm.Allreduce(
        [A_rs_upper_diagonal_blocks, MPI.DOUBLE],
        [A_rs_upper_diagonal_blocks_sum, MPI.DOUBLE],
        op=MPI.SUM,
    )
    comm.Allreduce(
        [A_rs_arrow_bottom_blocks, MPI.DOUBLE],
        [A_rs_arrow_bottom_blocks_sum, MPI.DOUBLE],
        op=MPI.SUM,
    )
    comm.Allreduce(
        [A_rs_arrow_right_blocks, MPI.DOUBLE],
        [A_rs_arrow_right_blocks_sum, MPI.DOUBLE],
        op=MPI.SUM,
    )
    comm.Allreduce(
        [A_rs_arrow_tip_block, MPI.DOUBLE],
        [A_rs_arrow_tip_block_sum, MPI.DOUBLE],
        op=MPI.SUM,
    )

    # Add the global arrow tip to the reduced system arrow-tip summation
    A_rs_arrow_tip_block_sum += A_arrow_tip_block

    return (
        A_rs_diagonal_blocks_sum,
        A_rs_lower_diagonal_blocks_sum,
        A_rs_upper_diagonal_blocks_sum,
        A_rs_arrow_bottom_blocks_sum,
        A_rs_arrow_right_blocks_sum,
        A_rs_arrow_tip_block_sum,
    )


def inverse_reduced_system(
    A_rs_diagonal_blocks: np.ndarray,
    A_rs_lower_diagonal_blocks: np.ndarray,
    A_rs_upper_diagonal_blocks: np.ndarray,
    A_rs_arrow_bottom_blocks: np.ndarray,
    A_rs_arrow_right_blocks: np.ndarray,
    A_rs_arrow_tip_block: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Invert the reduced system using a sequential selected inversion algorithm.

    Parameters
    ----------
    A_rs_diagonal_blocks : np.ndarray
        Diagonal blocks of the reduced system.
    A_rs_lower_diagonal_blocks : np.ndarray
        Lower diagonal blocks of the reduced system.
    A_rs_upper_diagonal_blocks : np.ndarray
        Upper diagonal blocks of the reduced system.
    A_rs_arrow_bottom_blocks : np.ndarray
        Arrow bottom blocks of the reduced system.
    A_rs_arrow_right_blocks : np.ndarray
        Arrow right blocks of the reduced system.
    A_rs_arrow_tip_block : np.ndarray
        Tip block of the reduced system.

    Returns
    -------
    X_rs_diagonal_blocks : np.ndarray
        Diagonal blocks of the inverse of the reduced system.
    X_rs_lower_diagonal_blocks : np.ndarray
        Lower diagonal blocks of the inverse of the reduced system.
    X_rs_upper_diagonal_blocks : np.ndarray
        Upper diagonal blocks of the inverse of the reduced system.
    X_rs_arrow_bottom_blocks : np.ndarray
        Arrow bottom blocks of the inverse of the reduced system.
    X_rs_arrow_right_blocks : np.ndarray
        Arrow right blocks of the inverse of the reduced system.
    X_rs_arrow_tip_block : np.ndarray
        Tip block of the inverse of the reduced system.

    Notes
    -----
    The inversion of the reduced system is performed using a sequential
    selected inversion algorithm.
    """

    (
        L_diagonal_blocks,
        L_lower_diagonal_blocks,
        L_arrow_bottom_blocks,
        U_diagonal_blocks,
        U_upper_diagonal_blocks,
        U_arrow_right_blocks,
    ) = lu_factorize_tridiag_arrowhead(
        A_rs_diagonal_blocks,
        A_rs_lower_diagonal_blocks,
        A_rs_upper_diagonal_blocks,
        A_rs_arrow_bottom_blocks,
        A_rs_arrow_right_blocks,
        A_rs_arrow_tip_block,
    )

    (
        X_rs_diagonal_blocks,
        X_rs_lower_diagonal_blocks,
        X_rs_upper_diagonal_blocks,
        X_rs_arrow_bottom_blocks,
        X_rs_arrow_right_blocks,
        X_rs_arrow_tip_block,
    ) = lu_sinv_tridiag_arrowhead(
        L_diagonal_blocks,
        L_lower_diagonal_blocks,
        L_arrow_bottom_blocks,
        U_diagonal_blocks,
        U_upper_diagonal_blocks,
        U_arrow_right_blocks,
    )

    return (
        X_rs_diagonal_blocks,
        X_rs_lower_diagonal_blocks,
        X_rs_upper_diagonal_blocks,
        X_rs_arrow_bottom_blocks,
        X_rs_arrow_right_blocks,
        X_rs_arrow_tip_block,
    )


def update_sinv_reduced_system(
    X_rs_diagonal_blocks: np.ndarray,
    X_rs_lower_diagonal_blocks: np.ndarray,
    X_rs_upper_diagonal_blocks: np.ndarray,
    X_rs_arrow_bottom_blocks: np.ndarray,
    X_rs_arrow_right_blocks: np.ndarray,
    X_rs_arrow_tip_block: np.ndarray,
    n_diag_blocks_partition: int,
    diag_blocksize: int,
    arrow_blocksize: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Fill the local parts of the inverse with the suited blocks of the
    inverted reduced system.

    Parameters
    ----------
    X_rs_diagonal_blocks : np.ndarray
        Diagonal blocks of the inverse of the reduced system.
    X_rs_lower_diagonal_blocks : np.ndarray
        Lower diagonal blocks of the inverse of the reduced system.
    X_rs_upper_diagonal_blocks : np.ndarray
        Upper diagonal blocks of the inverse of the reduced system.
    X_rs_arrow_bottom_blocks : np.ndarray
        Arrow bottom blocks of the inverse of the reduced system.
    X_rs_arrow_right_blocks : np.ndarray
        Arrow right blocks of the inverse of the reduced system.
    X_rs_arrow_tip_block : np.ndarray
        Tip block of the inverse of the reduced system.
    n_diag_blocks_partition : int
        Number of diagonal blocks in the partition.
    diag_blocksize : int
        Size of the diagonal blocks.
    arrow_blocksize : int
        Size of the arrow blocks.

    Returns
    -------
    X_diagonal_blocks_local : np.ndarray
        Local part of the diagonal array of the inverse.
    X_lower_diagonal_blocks_local : np.ndarray
        Local part of the lower diagonal array of the inverse.
    X_upper_diagonal_blocks_local : np.ndarray
        Local part of the upper diagonal array of the inverse.
    X_arrow_bottom_blocks_local : np.ndarray
        Local part of the arrow bottom array of the inverse.
    X_arrow_right_blocks_local : np.ndarray
        Local part of the arrow right array of the inverse.
    X_top_2sided_arrow_blocks_local : np.ndarray
        2sided pattern array storing corner blocks of the inverse, for the middle processes.
    X_left_2sided_arrow_blocks_local : np.ndarray
        2sided pattern array storing corner blocks of the inverse, for the middle processes.
    X_global_arrow_tip : np.ndarray
        Global arrow tip block of the inverse.
    X_bridges_lower : np.ndarray
        Lower part of the bridges array of the inverse.
    X_bridges_upper : np.ndarray
        Upper part of the bridges array of the inverse.
    """
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    X_diagonal_blocks_local = np.empty(
        (diag_blocksize, n_diag_blocks_partition * diag_blocksize),
        dtype=X_rs_diagonal_blocks.dtype,
    )
    X_lower_diagonal_blocks_local = np.empty(
        (diag_blocksize, (n_diag_blocks_partition - 1) * diag_blocksize),
        dtype=X_rs_diagonal_blocks.dtype,
    )
    X_upper_diagonal_blocks_local = np.empty(
        (diag_blocksize, (n_diag_blocks_partition - 1) * diag_blocksize),
        dtype=X_rs_diagonal_blocks.dtype,
    )
    X_arrow_bottom_blocks_local = np.empty(
        (arrow_blocksize, n_diag_blocks_partition * diag_blocksize),
        dtype=X_rs_diagonal_blocks.dtype,
    )
    X_arrow_right_blocks_local = np.empty(
        (n_diag_blocks_partition * diag_blocksize, arrow_blocksize),
        dtype=X_rs_diagonal_blocks.dtype,
    )

    X_bridges_upper = np.empty(
        (diag_blocksize, (comm_size - 1) * diag_blocksize),
        dtype=X_rs_diagonal_blocks.dtype,
    )
    X_bridges_lower = np.empty(
        (diag_blocksize, (comm_size - 1) * diag_blocksize),
        dtype=X_rs_diagonal_blocks.dtype,
    )

    if comm_rank == 0:
        X_top_2sided_arrow_blocks_local = None
        X_left_2sided_arrow_blocks_local = None

        X_diagonal_blocks_local[:, -diag_blocksize:] = X_rs_diagonal_blocks[
            :, :diag_blocksize
        ]

        X_bridges_upper[
            :, comm_rank * diag_blocksize : (comm_rank + 1) * diag_blocksize
        ] = X_rs_upper_diagonal_blocks[:, :diag_blocksize]

        X_arrow_bottom_blocks_local[:, -diag_blocksize:] = X_rs_arrow_bottom_blocks[
            :, :diag_blocksize
        ]
        X_arrow_right_blocks_local[-diag_blocksize:, :] = X_rs_arrow_right_blocks[
            :diag_blocksize, :
        ]
    else:
        X_top_2sided_arrow_blocks_local = np.empty(
            (diag_blocksize, n_diag_blocks_partition * diag_blocksize),
            dtype=X_rs_diagonal_blocks.dtype,
        )
        X_left_2sided_arrow_blocks_local = np.empty(
            (n_diag_blocks_partition * diag_blocksize, diag_blocksize),
            dtype=X_rs_diagonal_blocks.dtype,
        )

        start_index = diag_blocksize + (comm_rank - 1) * 2 * diag_blocksize

        X_bridges_lower[
            :, (comm_rank - 1) * diag_blocksize : comm_rank * diag_blocksize
        ] = X_rs_lower_diagonal_blocks[:, start_index - diag_blocksize : start_index]

        X_diagonal_blocks_local[:, :diag_blocksize] = X_rs_diagonal_blocks[
            :, start_index : start_index + diag_blocksize
        ]

        X_top_2sided_arrow_blocks_local[:, -diag_blocksize:] = (
            X_rs_upper_diagonal_blocks[:, start_index : start_index + diag_blocksize]
        )

        X_left_2sided_arrow_blocks_local[-diag_blocksize:, :diag_blocksize] = (
            X_rs_lower_diagonal_blocks[:, start_index : start_index + diag_blocksize]
        )

        X_diagonal_blocks_local[:, -diag_blocksize:] = X_rs_diagonal_blocks[
            :, start_index + diag_blocksize : start_index + 2 * diag_blocksize
        ]

        if comm_rank != comm_size - 1:
            X_bridges_upper[
                :, comm_rank * diag_blocksize : (comm_rank + 1) * diag_blocksize
            ] = X_rs_upper_diagonal_blocks[
                :, start_index + diag_blocksize : start_index + 2 * diag_blocksize
            ]

        X_arrow_bottom_blocks_local[:, :diag_blocksize] = X_rs_arrow_bottom_blocks[
            :, start_index : start_index + diag_blocksize
        ]

        X_arrow_bottom_blocks_local[:, -diag_blocksize:] = X_rs_arrow_bottom_blocks[
            :, start_index + diag_blocksize : start_index + 2 * diag_blocksize
        ]

        X_arrow_right_blocks_local[:diag_blocksize, :] = X_rs_arrow_right_blocks[
            start_index : start_index + diag_blocksize, :
        ]

        X_arrow_right_blocks_local[-diag_blocksize:, :] = X_rs_arrow_right_blocks[
            start_index + diag_blocksize : start_index + 2 * diag_blocksize, :
        ]

    X_global_arrow_tip = X_rs_arrow_tip_block

    return (
        X_diagonal_blocks_local,
        X_lower_diagonal_blocks_local,
        X_upper_diagonal_blocks_local,
        X_arrow_bottom_blocks_local,
        X_arrow_right_blocks_local,
        X_top_2sided_arrow_blocks_local,
        X_left_2sided_arrow_blocks_local,
        X_global_arrow_tip,
        X_bridges_lower,
        X_bridges_upper,
    )


def top_sinv(
    X_diagonal_blocks_local: np.ndarray,
    X_lower_diagonal_blocks_local: np.ndarray,
    X_upper_diagonal_blocks_local: np.ndarray,
    X_arrow_bottom_blocks_local: np.ndarray,
    X_arrow_right_blocks_local: np.ndarray,
    X_global_arrow_tip: np.ndarray,
    L_diagonal_blocks_local: np.ndarray,
    L_lower_diagonal_blocks_local: np.ndarray,
    L_arrow_bottom_blocks_local: np.ndarray,
    U_diagonal_blocks_local: np.ndarray,
    U_upper_diagonal_blocks_local: np.ndarray,
    U_arrow_right_blocks_local: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """

    Parameters
    ----------
    X_diagonal_blocks_local : np.ndarray
        Local part of the diagonal array of the inverse.
    X_lower_diagonal_blocks_local : np.ndarray
        Local part of the lower diagonal array of the inverse.
    X_upper_diagonal_blocks_local : np.ndarray
        Local part of the upper diagonal array of the inverse.
    X_arrow_bottom_blocks_local : np.ndarray
        Local part of the arrow bottom array of the inverse.
    X_arrow_right_blocks_local : np.ndarray
        Local part of the arrow right array of the inverse.
    X_global_arrow_tip : np.ndarray
        Global arrow tip block of the inverse.
    L_diagonal_blocks_local : np.ndarray
        Diagonal blocks of the lower factor of the local partition.
    L_lower_diagonal_blocks_local : np.ndarray
        Lower diagonal blocks of the lower factor of the local partition.
    L_arrow_bottom_blocks_local : np.ndarray
        Arrow bottom blocks of the lower factor of the local partition.
    U_diagonal_blocks_local : np.ndarray
        Diagonal blocks of the upper factor of the local partition.
    U_upper_diagonal_blocks_local : np.ndarray
        Upper diagonal blocks of the upper factor of the local partition.
    U_arrow_right_blocks_local : np.ndarray
        Arrow right blocks of the upper factor of the local partition.

    Returns
    -------
    X_diagonal_blocks_local : np.ndarray
        Local part of the diagonal array of the inverse.
    X_lower_diagonal_blocks_local : np.ndarray
        Local part of the lower diagonal array of the inverse.
    X_upper_diagonal_blocks_local : np.ndarray
        Local part of the upper diagonal array of the inverse.
    X_arrow_bottom_blocks_local : np.ndarray
        Local part of the arrow bottom array of the inverse.
    X_arrow_right_blocks_local : np.ndarray
        Local part of the arrow right array of the inverse.
    X_global_arrow_tip : np.ndarray
        Global arrow tip block of the inverse.
    """
    diag_blocksize = X_diagonal_blocks_local.shape[0]
    n_blocks = X_diagonal_blocks_local.shape[1] // diag_blocksize

    L_blk_inv = np.empty(
        (diag_blocksize, diag_blocksize), dtype=L_diagonal_blocks_local.dtype
    )
    U_blk_inv = np.empty(
        (diag_blocksize, diag_blocksize), dtype=U_diagonal_blocks_local.dtype
    )

    for i in range(n_blocks - 2, -1, -1):
        # ----- Block-tridiagonal solver -----
        L_blk_inv = la.solve_triangular(
            L_diagonal_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize],
            np.eye(diag_blocksize),
            lower=True,
        )
        U_blk_inv = la.solve_triangular(
            U_diagonal_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize],
            np.eye(diag_blocksize),
            lower=False,
        )

        # --- Lower-diagonal blocks ---
        # X_{i+1, i} = (-X_{i+1, i+1} L_{i+1, i} - X_{i+1, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1}
        X_lower_diagonal_blocks_local[
            :, i * diag_blocksize : (i + 1) * diag_blocksize
        ] = (
            -X_diagonal_blocks_local[
                :, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize
            ]
            @ L_lower_diagonal_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            - X_arrow_right_blocks_local[
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize, :
            ]
            @ L_arrow_bottom_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
        ) @ L_blk_inv

        # X_{ndb+1, i} = (- X_{ndb+1, i+1} L_{i+1, i} - X_{ndb+1, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1}
        X_arrow_bottom_blocks_local[
            :, i * diag_blocksize : (i + 1) * diag_blocksize
        ] = (
            -X_arrow_bottom_blocks_local[
                :, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize
            ]
            @ L_lower_diagonal_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            - X_global_arrow_tip[:, :]
            @ L_arrow_bottom_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
        ) @ L_blk_inv

        # --- Upper-diagonal blocks ---
        # X_{i, i+1} = U_{i, i}^{-1} (- U_{i, i+1} X_{i+1, i+1} - U_{i, ndb+1} X_{ndb+1, i+1})
        X_upper_diagonal_blocks_local[
            :, i * diag_blocksize : (i + 1) * diag_blocksize
        ] = U_blk_inv @ (
            -U_upper_diagonal_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            @ X_diagonal_blocks_local[
                :, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize
            ]
            - U_arrow_right_blocks_local[
                i * diag_blocksize : (i + 1) * diag_blocksize, :
            ]
            @ X_arrow_bottom_blocks_local[
                :, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize
            ]
        )

        # X_{i, ndb+1} = U_{i, i}^{-1} (- U_{i, i+1} X_{i+1, ndb+1} - U_{i, ndb+1} X_{ndb+1, ndb+1})
        X_arrow_right_blocks_local[i * diag_blocksize : (i + 1) * diag_blocksize, :] = (
            U_blk_inv
            @ (
                -U_upper_diagonal_blocks_local[
                    :, i * diag_blocksize : (i + 1) * diag_blocksize
                ]
                @ X_arrow_right_blocks_local[
                    (i + 1) * diag_blocksize : (i + 2) * diag_blocksize, :
                ]
                - U_arrow_right_blocks_local[
                    i * diag_blocksize : (i + 1) * diag_blocksize, :
                ]
                @ X_global_arrow_tip[:, :]
            )
        )

        # # --- Diagonal blocks ---
        # X_{i, i} = (U_{i, i}^{-1} - X_{i, i+1} L_{i+1, i} - X_{i, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1}
        X_diagonal_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize] = (
            U_blk_inv
            - X_upper_diagonal_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            @ L_lower_diagonal_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            - X_arrow_right_blocks_local[
                i * diag_blocksize : (i + 1) * diag_blocksize, :
            ]
            @ L_arrow_bottom_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
        ) @ L_blk_inv

    return (
        X_diagonal_blocks_local,
        X_lower_diagonal_blocks_local,
        X_upper_diagonal_blocks_local,
        X_arrow_bottom_blocks_local,
        X_arrow_right_blocks_local,
        X_global_arrow_tip,
    )


def middle_sinv(
    X_diagonal_blocks_local: np.ndarray,
    X_lower_diagonal_blocks_local: np.ndarray,
    X_upper_diagonal_blocks_local: np.ndarray,
    X_arrow_bottom_blocks_local: np.ndarray,
    X_arrow_right_blocks_local: np.ndarray,
    X_top_2sided_arrow_blocks_local: np.ndarray,
    X_left_2sided_arrow_blocks_local: np.ndarray,
    X_global_arrow_tip_block_local: np.ndarray,
    L_diagonal_blocks_local: np.ndarray,
    L_lower_diagonal_blocks_local: np.ndarray,
    L_arrow_bottom_blocks_local: np.ndarray,
    L_upper_2sided_arrow_blocks_local: np.ndarray,
    U_diagonal_blocks_local: np.ndarray,
    U_upper_diagonal_blocks_local: np.ndarray,
    U_arrow_right_blocks_local: np.ndarray,
    U_left_2sided_arrow_blocks_local: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """

    Parameters
    ----------
    X_diagonal_blocks_local : np.ndarray
        Local part of the diagonal array of the inverse.
    X_lower_diagonal_blocks_local : np.ndarray
        Local part of the lower diagonal array of the inverse.
    X_upper_diagonal_blocks_local : np.ndarray
        Local part of the upper diagonal array of the inverse.
    X_arrow_bottom_blocks_local : np.ndarray
        Local part of the arrow bottom array of the inverse.
    X_arrow_right_blocks_local : np.ndarray
        Local part of the arrow right array of the inverse.
    X_top_2sided_arrow_blocks_local : np.ndarray
        2-sided pattern array storing top blocks of the inverse.
    X_left_2sided_arrow_blocks_local : np.ndarray
        2-sided pattern array storing left blocks of the inverse.
    X_global_arrow_tip_block_local : np.ndarray
        Global arrow tip block of the inverse.
    L_diagonal_blocks_local : np.ndarray
        Diagonal blocks of the lower factor of the local partition.
    L_lower_diagonal_blocks_local : np.ndarray
        Lower diagonal blocks of the lower factor of the local partition.
    L_arrow_bottom_blocks_local : np.ndarray
        Arrow bottom blocks of the lower factor of the local partition.
    L_upper_2sided_arrow_blocks_local : np.ndarray
        2-sided pattern array storing top blocks of the lower factor of the local partition.
    U_diagonal_blocks_local : np.ndarray
        Diagonal blocks of the upper factor of the local partition.
    U_upper_diagonal_blocks_local : np.ndarray
        Upper diagonal blocks of the upper factor of the local partition.
    U_arrow_right_blocks_local : np.ndarray
        Arrow right blocks of the upper factor of the local partition.
    U_left_2sided_arrow_blocks_local : np.ndarray
        2-sided pattern array storing left blocks of the upper factor of the local partition.

    Returns
    -------
    X_diagonal_blocks_local : np.ndarray
        Local part of the diagonal array of the inverse.
    X_lower_diagonal_blocks_local : np.ndarray
        Local part of the lower diagonal array of the inverse.
    X_upper_diagonal_blocks_local : np.ndarray
        Local part of the upper diagonal array of the inverse.
    X_arrow_bottom_blocks_local : np.ndarray
        Local part of the arrow bottom array of the inverse.
    X_arrow_right_blocks_local : np.ndarray
        Local part of the arrow right array of the inverse.
    X_global_arrow_tip_block_local : np.ndarray
        Global arrow tip block of the inverse.
    """
    diag_blocksize = X_diagonal_blocks_local.shape[0]
    n_blocks = X_diagonal_blocks_local.shape[1] // diag_blocksize

    L_blk_inv = np.empty(
        (diag_blocksize, diag_blocksize), dtype=L_diagonal_blocks_local.dtype
    )
    U_blk_inv = np.empty(
        (diag_blocksize, diag_blocksize), dtype=U_diagonal_blocks_local.dtype
    )

    for i in range(n_blocks - 2, 0, -1):
        # ----- Block-tridiagonal solver -----
        L_blk_inv = la.solve_triangular(
            L_diagonal_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize],
            np.eye(diag_blocksize),
            lower=True,
        )
        U_blk_inv = la.solve_triangular(
            U_diagonal_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize],
            np.eye(diag_blocksize),
            lower=False,
        )

        # X_{i+1, i} = (- X_{i+1, top} L_{top, i} - X_{i+1, i+1} L_{i+1, i} - X_{i+1, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1}
        X_lower_diagonal_blocks_local[
            :, i * diag_blocksize : (i + 1) * diag_blocksize
        ] = (
            -X_left_2sided_arrow_blocks_local[
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize, :
            ]
            @ L_upper_2sided_arrow_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            - X_diagonal_blocks_local[
                :, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize
            ]
            @ L_lower_diagonal_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            - X_arrow_right_blocks_local[
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize, :
            ]
            @ L_arrow_bottom_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
        ) @ L_blk_inv

        # X_{i, i+1} = U_{i, i}^{-1} (- U_{i, i+1} X_{i+1, i+1} - U_{i, top} X_{top, i+1} - U_{i, ndb+1} X_{ndb+1, i+1})
        X_upper_diagonal_blocks_local[
            :, i * diag_blocksize : (i + 1) * diag_blocksize
        ] = U_blk_inv @ (
            -U_upper_diagonal_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            @ X_diagonal_blocks_local[
                :, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize
            ]
            - U_left_2sided_arrow_blocks_local[
                i * diag_blocksize : (i + 1) * diag_blocksize, :
            ]
            @ X_top_2sided_arrow_blocks_local[
                :, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize
            ]
            - U_arrow_right_blocks_local[
                i * diag_blocksize : (i + 1) * diag_blocksize, :
            ]
            @ X_arrow_bottom_blocks_local[
                :, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize
            ]
        )

        # X_{top, i} = (- X_{top, i+1} L_{i+1, i} - X_{top, top} L_{top, i} - X_{top, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1}
        X_top_2sided_arrow_blocks_local[
            :, i * diag_blocksize : (i + 1) * diag_blocksize
        ] = (
            -X_top_2sided_arrow_blocks_local[
                :, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize
            ]
            @ L_lower_diagonal_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            - X_diagonal_blocks_local[:, :diag_blocksize]
            @ L_upper_2sided_arrow_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            - X_arrow_right_blocks_local[:diag_blocksize, :]
            @ L_arrow_bottom_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
        ) @ L_blk_inv

        # X_{i, top} = U_{i, i}^{-1} (- U_{i, i+1} X_{i+1, top} - U_{i, top} X_{top, top} - U_{i, ndb+1} X_{ndb+1, top})
        X_left_2sided_arrow_blocks_local[
            i * diag_blocksize : (i + 1) * diag_blocksize, :
        ] = U_blk_inv @ (
            -U_upper_diagonal_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            @ X_left_2sided_arrow_blocks_local[
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize, :
            ]
            - U_left_2sided_arrow_blocks_local[
                i * diag_blocksize : (i + 1) * diag_blocksize, :
            ]
            @ X_diagonal_blocks_local[:, :diag_blocksize]
            - U_arrow_right_blocks_local[
                i * diag_blocksize : (i + 1) * diag_blocksize, :
            ]
            @ X_arrow_bottom_blocks_local[:, :diag_blocksize]
        )

        # Arrowhead
        # X_{ndb+1, i} = (- X_{ndb+1, i+1} L_{i+1, i} - X_{ndb+1, top} L_{top, i} - X_{ndb+1, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1}
        X_arrow_bottom_blocks_local[
            :, i * diag_blocksize : (i + 1) * diag_blocksize
        ] = (
            -X_arrow_bottom_blocks_local[
                :, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize
            ]
            @ L_lower_diagonal_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            - X_arrow_bottom_blocks_local[:, :diag_blocksize]
            @ L_upper_2sided_arrow_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            - X_global_arrow_tip_block_local[:, :]
            @ L_arrow_bottom_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
        ) @ L_blk_inv

        # X_{i, ndb+1} = U_{i, i}^{-1} (- U_{i, i+1} X_{i+1, ndb+1} - U_{i, top} X_{top, ndb+1} - U_{i, ndb+1} X_{ndb+1, ndb+1})
        X_arrow_right_blocks_local[i * diag_blocksize : (i + 1) * diag_blocksize, :] = (
            U_blk_inv
            @ (
                -U_upper_diagonal_blocks_local[
                    :, i * diag_blocksize : (i + 1) * diag_blocksize
                ]
                @ X_arrow_right_blocks_local[
                    (i + 1) * diag_blocksize : (i + 2) * diag_blocksize, :
                ]
                - U_left_2sided_arrow_blocks_local[
                    i * diag_blocksize : (i + 1) * diag_blocksize, :
                ]
                @ X_arrow_right_blocks_local[:diag_blocksize, :]
                - U_arrow_right_blocks_local[
                    i * diag_blocksize : (i + 1) * diag_blocksize, :
                ]
                @ X_global_arrow_tip_block_local[:, :]
            )
        )

        # X_{i, i} = (U_{i, i}^{-1} - X_{i, i+1} L_{i+1, i} - X_{i, top} L_{top, i} - X_{i, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1}
        X_diagonal_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize] = (
            U_blk_inv
            - X_upper_diagonal_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            @ L_lower_diagonal_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            - X_left_2sided_arrow_blocks_local[
                i * diag_blocksize : (i + 1) * diag_blocksize, :
            ]
            @ L_upper_2sided_arrow_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            - X_arrow_right_blocks_local[
                i * diag_blocksize : (i + 1) * diag_blocksize, :
            ]
            @ L_arrow_bottom_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
        ) @ L_blk_inv

    # Copy back the 2 first blocks that have been produced in the 2-sided pattern
    # to the tridiagonal storage.
    X_upper_diagonal_blocks_local[:, :diag_blocksize] = X_top_2sided_arrow_blocks_local[
        :, diag_blocksize : 2 * diag_blocksize
    ]
    X_lower_diagonal_blocks_local[:, :diag_blocksize] = (
        X_left_2sided_arrow_blocks_local[diag_blocksize : 2 * diag_blocksize, :]
    )

    return (
        X_diagonal_blocks_local,
        X_lower_diagonal_blocks_local,
        X_upper_diagonal_blocks_local,
        X_arrow_bottom_blocks_local,
        X_arrow_right_blocks_local,
        X_global_arrow_tip_block_local,
    )
