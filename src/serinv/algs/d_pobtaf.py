# Copyright 2023-2024 ETH Zurich. All rights reserved.

try:
    import cupy as cp
    import cupyx.scipy.linalg as cu_la

    CUPY_AVAIL = True

except:
    CUPY_AVAIL = False

import numpy as np
import scipy.linalg as np_la
from numpy.typing import ArrayLike
from mpi4py import MPI

comm_rank = MPI.COMM_WORLD.Get_rank()
comm_size = MPI.COMM_WORLD.Get_size()


def d_pobtaf(
    A_diagonal_blocks_local: ArrayLike,
    A_lower_diagonal_blocks_local: ArrayLike,
    A_arrow_bottom_blocks_local: ArrayLike,
    A_arrow_tip_block_global: ArrayLike,
    device_streaming: bool = False,
) -> tuple[
    ArrayLike,
    ArrayLike,
    ArrayLike,
    ArrayLike,
    ArrayLike,
]:
    """Perform the distributed Cholesky factorization of a block tridiagonal
    with arrowhead matrix.

    Note:
    -----
    - The matrix, A, is assumed to be symmetric positive definite.
    - Will overwrite the inputs and store the results in them (in-place).
    - If a device array is given, the algorithm will run on the GPU.

    Parameters
    ----------
    A_diagonal_blocks_local : ArrayLike
        Local slice of the diagonal blocks of A.
    A_lower_diagonal_blocks_local : ArrayLike
        Local slice of the lower diagonal blocks of A.
    A_arrow_bottom_blocks_local : ArrayLike
        Local slice of the arrow bottom blocks of A.
    A_arrow_tip_block_global : ArrayLike
        Arrow tip block of A.
    device_streaming : bool
        Whether to use streamed GPU computation.

    Returns
    -------
    L_diagonal_blocks_local : ArrayLike
        Local slice of the diagonal blocks of L.
    L_lower_diagonal_blocks_local : ArrayLike
        Local slice of the lower diagonal blocks of L.
    L_arrow_bottom_blocks_local : ArrayLike
        Local slice of the arrow bottom blocks of L.
    L_arrow_tip_block_global : ArrayLike
        Arrow tip block of L.
    L_upper_nested_dissection_buffer_local : ArrayLike, optional
        Local upper buffer used in the nested dissection factorization.
    """

    if (
        CUPY_AVAIL
        and cp.get_array_module(A_diagonal_blocks_local) == np
        and device_streaming
    ):
        return _streaming_d_pobtaf(
            A_diagonal_blocks_local,
            A_lower_diagonal_blocks_local,
            A_arrow_bottom_blocks_local,
            A_arrow_tip_block_global,
        )

    return _d_pobtaf(
        A_diagonal_blocks_local,
        A_lower_diagonal_blocks_local,
        A_arrow_bottom_blocks_local,
        A_arrow_tip_block_global,
    )


def _d_pobtaf(
    A_diagonal_blocks_local: ArrayLike,
    A_lower_diagonal_blocks_local: ArrayLike,
    A_arrow_bottom_blocks_local: ArrayLike,
    A_arrow_tip_block_global: ArrayLike,
) -> tuple[
    ArrayLike,
    ArrayLike,
    ArrayLike,
    ArrayLike,
]:
    la = np_la
    if CUPY_AVAIL:
        xp = cp.get_array_module(A_diagonal_blocks_local)
        if xp == cp:
            la = cu_la
    else:
        xp = np

    diag_blocksize = A_diagonal_blocks_local.shape[1]
    n_diag_blocks = A_diagonal_blocks_local.shape[0]

    L_diagonal_blocks_local = A_diagonal_blocks_local
    L_lower_diagonal_blocks_local = A_lower_diagonal_blocks_local
    L_arrow_bottom_blocks_local = A_arrow_bottom_blocks_local

    L_upper_nested_dissection_buffer_local = None
    L_inv_temp = xp.zeros_like(A_diagonal_blocks_local[0])

    Update_arrow_tip_block_local = xp.zeros_like(A_arrow_tip_block_global)
    Update_arrow_tip_block_global = xp.zeros_like(Update_arrow_tip_block_local)

    if comm_rank == 0:
        # Forward block-Cholesky, performed by a "top" process
        for i in range(0, n_diag_blocks - 1):
            # L_{i, i} = chol(A_{i, i})
            L_diagonal_blocks_local[i, :, :] = xp.linalg.cholesky(
                A_diagonal_blocks_local[i, :, :],
            )

            # Compute lower factors
            L_inv_temp[:, :] = la.solve_triangular(
                L_diagonal_blocks_local[i, :, :],
                xp.eye(diag_blocksize),
                lower=True,
            )

            # L_{i+1, i} = A_{i+1, i} @ L_{i, i}^{-T}
            L_lower_diagonal_blocks_local[i, :, :] = (
                A_lower_diagonal_blocks_local[i, :, :] @ L_inv_temp[:, :].T
            )

            # L_{ndb+1, i} = A_{ndb+1, i} @ L_{i, i}^{-T}
            L_arrow_bottom_blocks_local[i, :, :] = (
                A_arrow_bottom_blocks_local[i, :, :] @ L_inv_temp[:, :].T
            )

            # Update next diagonal block
            # A_{i+1, i+1} = A_{i+1, i+1} - L_{i+1, i} @ L_{i+1, i}.T
            A_diagonal_blocks_local[i + 1, :, :] = (
                A_diagonal_blocks_local[i + 1, :, :]
                - L_lower_diagonal_blocks_local[i, :, :]
                @ L_lower_diagonal_blocks_local[i, :, :].T
            )

            # A_{ndb+1, i+1} = A_{ndb+1, i+1} - L_{ndb+1, i} @ L_{i+1, i}.T
            A_arrow_bottom_blocks_local[i + 1, :, :] = (
                A_arrow_bottom_blocks_local[i + 1, :, :]
                - L_arrow_bottom_blocks_local[i, :, :]
                @ L_lower_diagonal_blocks_local[i, :, :].T
            )

            # A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, i} @ L_{ndb+1, i}.T
            Update_arrow_tip_block_local[:, :] = (
                Update_arrow_tip_block_local[:, :]
                - L_arrow_bottom_blocks_local[i, :, :]
                @ L_arrow_bottom_blocks_local[i, :, :].T
            )

        """ Here to check weather we perform this cholesky now or later
        as we want the updated A matrix for the reduced system... 
        # L_{ndb, ndb} = chol(A_{ndb, ndb})
        L_diagonal_blocks_local[-1, :, :] = xp.linalg.cholesky(
            A_diagonal_blocks_local[-1, :, :]
        )

        # L_{ndb+1, ndb} = A_{ndb+1, ndb} @ L_{ndb, ndb}^{-T}
        L_arrow_bottom_blocks_local[-1, :, :] = A_arrow_bottom_blocks_local[
            -1, :, :
        ] @ la.solve_triangular(
            L_diagonal_blocks_local[-1, :, :],
            xp.eye(diag_blocksize),
            lower=True,
        ) """

    else:
        A_upper_nested_dissection_buffer_local = xp.zeros_like(A_diagonal_blocks_local)
        L_upper_nested_dissection_buffer_local = A_upper_nested_dissection_buffer_local

        # Forward block-Cholesky, performed by a "middle" process
        for i in range(1, n_diag_blocks - 1):
            # L_{i, i} = chol(A_{i, i})
            L_diagonal_blocks_local[i, :, :] = xp.linalg.cholesky(
                A_diagonal_blocks_local[i, :, :]
            )

            # Compute lower factors
            L_inv_temp[:, :] = la.solve_triangular(
                L_diagonal_blocks_local[i, :, :],
                xp.eye(diag_blocksize),
                lower=True,
            )

            # L_{i+1, i} = A_{i+1, i} @ L_{i, i}^{-T}
            L_lower_diagonal_blocks_local[i, :, :] = (
                A_lower_diagonal_blocks_local[i, :, :] @ L_inv_temp[:, :].T
            )

            # L_{top, i} = A_{top, i} @ U{i, i}^{-1}
            L_upper_nested_dissection_buffer_local[i, :, :] = (
                A_upper_nested_dissection_buffer_local[i, :, :] @ L_inv_temp[:, :].T
            )

            # L_{ndb+1, i} = A_{ndb+1, i} @ L_{i, i}^{-T}
            L_arrow_bottom_blocks_local[i, :, :] = (
                A_arrow_bottom_blocks_local[i, :, :] @ L_inv_temp[:, :].T
            )

            # Update next diagonal block
            # A_{i+1, i+1} = A_{i+1, i+1} - L_{i+1, i} @ L_{i+1, i}.T
            A_diagonal_blocks_local[i + 1, :, :] = (
                A_diagonal_blocks_local[i + 1, :, :]
                - L_lower_diagonal_blocks_local[i, :, :]
                @ L_lower_diagonal_blocks_local[i, :, :].T
            )

            # A_{ndb+1, i+1} = A_{ndb+1, i+1} - L_{ndb+1, i} @ L_{i+1, i}.T
            A_arrow_bottom_blocks_local[i + 1, :, :] = (
                A_arrow_bottom_blocks_local[i + 1, :, :]
                - L_arrow_bottom_blocks_local[i, :, :]
                @ L_lower_diagonal_blocks_local[i, :, :].T
            )

            # Update the block at the tip of the arrowhead
            # A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, i} @ L_{ndb+1, i}.T
            Update_arrow_tip_block_local[:, :] = (
                Update_arrow_tip_block_local[:, :]
                - L_arrow_bottom_blocks_local[i, :, :]
                @ L_arrow_bottom_blocks_local[i, :, :].T
            )

            # Update top and next upper/lower blocks of 2-sided factorization pattern
            # A_{top, top} = A_{top, top} - L_{top, i} @ L_{top, i}.T
            A_diagonal_blocks_local[0, :, :] = (
                A_diagonal_blocks_local[0, :, :]
                - L_upper_nested_dissection_buffer_local[i, :, :]
                @ L_upper_nested_dissection_buffer_local[i, :, :].T
            )

            # A_{top, i+1} = - L{top, i} @ L_{i+1, i}.T
            A_upper_nested_dissection_buffer_local[i + 1, :, :] = (
                -L_upper_nested_dissection_buffer_local[i, :, :]
                @ L_lower_diagonal_blocks_local[i, :, :].T
            )

            # Update the top (first blocks) of the arrowhead
            # A_{ndb+1, top} = A_{ndb+1, top} - L_{ndb+1, i} @ L_{top, i}.T
            A_arrow_bottom_blocks_local[0, :, :] = (
                A_arrow_bottom_blocks_local[0, :, :]
                - L_arrow_bottom_blocks_local[i, :, :]
                @ L_upper_nested_dissection_buffer_local[i, :, :].T
            )

    # Accumulate the distributed update of the arrow tip block
    MPI.COMM_WORLD.Allreduce(
        [Update_arrow_tip_block_local, MPI.DOUBLE],
        [Update_arrow_tip_block_global, MPI.DOUBLE],
        op=MPI.SUM,
    )

    A_arrow_tip_block_global[:, :] += Update_arrow_tip_block_global[:, :]
    L_arrow_tip_block_global = A_arrow_tip_block_global

    return (
        L_diagonal_blocks_local,
        L_lower_diagonal_blocks_local,
        L_arrow_bottom_blocks_local,
        L_arrow_tip_block_global,
        L_upper_nested_dissection_buffer_local,
    )


def _streaming_d_pobtaf(
    A_diagonal_blocks_local: ArrayLike,
    A_lower_diagonal_blocks_local: ArrayLike,
    A_arrow_bottom_blocks_local: ArrayLike,
    A_arrow_tip_block_global: ArrayLike,
) -> tuple[
    ArrayLike,
    ArrayLike,
    ArrayLike,
    ArrayLike,
]:
    raise NotImplementedError
