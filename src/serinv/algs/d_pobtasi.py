# Copyright 2023-2024 ETH Zurich. All rights reserved.

try:
    import cupy as cp
    import cupyx as cpx
    import cupyx.scipy.linalg as cu_la

    CUPY_AVAIL = True

except ImportError:
    CUPY_AVAIL = False

import numpy as np
import scipy.linalg as np_la
from numpy.typing import ArrayLike
from mpi4py import MPI

comm_rank = MPI.COMM_WORLD.Get_rank()
comm_size = MPI.COMM_WORLD.Get_size()

from serinv.algs import pobtaf, pobtasi


def d_pobtasi(
    L_diagonal_blocks_local: ArrayLike,
    L_lower_diagonal_blocks_local: ArrayLike,
    L_arrow_bottom_blocks_local: ArrayLike,
    L_arrow_tip_block_global: ArrayLike,
    L_upper_nested_dissection_buffer_local: ArrayLike = None,
    device_streaming: bool = False,
) -> tuple[
    ArrayLike,
    ArrayLike,
    ArrayLike,
    ArrayLike,
]:
    """Perform a distributed selected-inversion of a block tridiagonal with
    arrowhead matrix.

    Note:
    -----
    - Will overwrite the inputs and store the results in them (in-place). Returns
        are aliases on the inputs.
    - If a device array is given, the algorithm will run on the GPU.

    Parameters
    ----------
    L_diagonal_blocks_local : ArrayLike
        Local slice of the diagonal blocks of L.
    L_lower_diagonal_blocks_local : ArrayLike
        Local slice of the lower diagonal blocks of L.
    L_arrow_bottom_blocks_local : ArrayLike
        Local slice of the arrow bottom blocks of L.
    L_arrow_tip_block_global : ArrayLike
        Arrow tip block of L.
    L_upper_nested_dissection_buffer_local : ArrayLike, optional
        Local upper buffer used in the nested dissection factorization. None for
        uppermost process.
    device_streaming : bool
        Whether to use streamed GPU computation.

    Returns
    -------
    X_diagonal_blocks_local : ArrayLike
        Local slice of the diagonal blocks of X, alias on L_diagonal_blocks_local.
    X_lower_diagonal_blocks_local : ArrayLike
        Local slice of the lower diagonal blocks of X, alias on L_lower_diagonal_blocks_local.
    X_arrow_bottom_blocks_local : ArrayLike
        Local slice of the arrow bottom blocks of X, alias on L_arrow_bottom_blocks_local.
    X_arrow_tip_block_global : ArrayLike
        Arrow tip block of X, alias on L_arrow_tip_block_global.
    """

    if (
        CUPY_AVAIL
        and cp.get_array_module(L_diagonal_blocks_local) == np
        and device_streaming
    ):
        return _streaming_d_pobtasi(
            L_diagonal_blocks_local,
            L_lower_diagonal_blocks_local,
            L_arrow_bottom_blocks_local,
            L_arrow_tip_block_global,
            L_upper_nested_dissection_buffer_local,
        )

    return _d_pobtasi(
        L_diagonal_blocks_local,
        L_lower_diagonal_blocks_local,
        L_arrow_bottom_blocks_local,
        L_arrow_tip_block_global,
        L_upper_nested_dissection_buffer_local,
    )


def _d_pobtasi(
    L_diagonal_blocks_local: ArrayLike,
    L_lower_diagonal_blocks_local: ArrayLike,
    L_arrow_bottom_blocks_local: ArrayLike,
    L_arrow_tip_block_global: ArrayLike,
    L_upper_nested_dissection_buffer_local: ArrayLike,
) -> tuple[
    ArrayLike,
    ArrayLike,
    ArrayLike,
    ArrayLike,
]:
    la = np_la
    if CUPY_AVAIL:
        xp = cp.get_array_module(L_diagonal_blocks_local)
        if xp == cp:
            la = cu_la
    else:
        xp = np

    diag_blocksize = L_diagonal_blocks_local.shape[1]
    n_diag_blocks_local = L_diagonal_blocks_local.shape[0]

    X_diagonal_blocks_local = L_diagonal_blocks_local
    X_lower_diagonal_blocks_local = L_lower_diagonal_blocks_local
    X_arrow_bottom_blocks_local = L_arrow_bottom_blocks_local
    X_upper_nested_dissection_buffer_local = L_upper_nested_dissection_buffer_local

    A_reduced_system_diagonal_blocks = xp.zeros(
        (2 * comm_size - 1, *L_diagonal_blocks_local.shape[1:]),
        dtype=L_diagonal_blocks_local.dtype,
    )
    A_reduced_system_lower_diagonal_blocks = xp.zeros(
        (2 * comm_size - 2, *L_lower_diagonal_blocks_local.shape[1:]),
        dtype=L_lower_diagonal_blocks_local.dtype,
    )
    A_reduced_system_arrow_bottom_blocks = xp.zeros(
        (2 * comm_size - 1, *L_arrow_bottom_blocks_local.shape[1:]),
        dtype=L_arrow_bottom_blocks_local.dtype,
    )
    # Alias on the tip block for the reduced system
    A_reduced_system_arrow_tip_block = L_arrow_tip_block_global

    # Construct the reduced system from the factorized blocks distributed over the
    # processes.
    if comm_rank == 0:
        A_reduced_system_diagonal_blocks[0, :, :] = L_diagonal_blocks_local[-1, :, :]
        A_reduced_system_lower_diagonal_blocks[0, :, :] = L_lower_diagonal_blocks_local[
            -1, :, :
        ]
        A_reduced_system_arrow_bottom_blocks[0, :, :] = L_arrow_bottom_blocks_local[
            -1, :, :
        ]
    else:
        A_reduced_system_diagonal_blocks[2 * comm_rank - 1, :, :] = (
            L_diagonal_blocks_local[0, :, :]
        )
        A_reduced_system_diagonal_blocks[2 * comm_rank, :, :] = L_diagonal_blocks_local[
            -1, :, :
        ]

        A_reduced_system_lower_diagonal_blocks[2 * comm_rank - 1, :, :] = (
            L_upper_nested_dissection_buffer_local[-1, :, :].conj().T
        )
        if comm_rank < comm_size - 1:
            A_reduced_system_lower_diagonal_blocks[2 * comm_rank, :, :] = (
                L_lower_diagonal_blocks_local[-1, :, :]
            )

        A_reduced_system_arrow_bottom_blocks[2 * comm_rank - 1, :, :] = (
            L_arrow_bottom_blocks_local[0, :, :]
        )
        A_reduced_system_arrow_bottom_blocks[2 * comm_rank, :, :] = (
            L_arrow_bottom_blocks_local[-1, :, :]
        )

    if CUPY_AVAIL and xp == cp:
        A_reduced_system_diagonal_blocks_host = cpx.zeros_like_pinned(
            A_reduced_system_diagonal_blocks
        )
        A_reduced_system_lower_diagonal_blocks_host = cpx.zeros_like_pinned(
            A_reduced_system_lower_diagonal_blocks
        )
        A_reduced_system_arrow_bottom_blocks_host = cpx.zeros_like_pinned(
            A_reduced_system_arrow_bottom_blocks
        )

        A_reduced_system_diagonal_blocks.get(out=A_reduced_system_diagonal_blocks_host)
        A_reduced_system_lower_diagonal_blocks.get(
            out=A_reduced_system_lower_diagonal_blocks_host
        )
        A_reduced_system_arrow_bottom_blocks.get(
            out=A_reduced_system_arrow_bottom_blocks_host
        )
    else:
        A_reduced_system_diagonal_blocks_host = A_reduced_system_diagonal_blocks
        A_reduced_system_lower_diagonal_blocks_host = (
            A_reduced_system_lower_diagonal_blocks
        )
        A_reduced_system_arrow_bottom_blocks_host = A_reduced_system_arrow_bottom_blocks

    MPI.COMM_WORLD.Allreduce(
        MPI.IN_PLACE,
        A_reduced_system_diagonal_blocks_host,
        op=MPI.SUM,
    )
    MPI.COMM_WORLD.Allreduce(
        MPI.IN_PLACE,
        A_reduced_system_lower_diagonal_blocks_host,
        op=MPI.SUM,
    )
    MPI.COMM_WORLD.Allreduce(
        MPI.IN_PLACE,
        A_reduced_system_arrow_bottom_blocks_host,
        op=MPI.SUM,
    )

    if CUPY_AVAIL and xp == cp:
        A_reduced_system_diagonal_blocks.set(arr=A_reduced_system_diagonal_blocks_host)
        A_reduced_system_lower_diagonal_blocks.set(
            arr=A_reduced_system_lower_diagonal_blocks_host
        )
        A_reduced_system_arrow_bottom_blocks.set(
            arr=A_reduced_system_arrow_bottom_blocks_host
        )

    # Perform the inversion of the reduced system.
    (
        L_reduced_system_diagonal_blocks,
        L_reduced_system_lower_diagonal_blocks,
        L_reduced_system_arrow_bottom_blocks,
        L_reduced_system_arrow_tip_block,
    ) = pobtaf(
        A_reduced_system_diagonal_blocks,
        A_reduced_system_lower_diagonal_blocks,
        A_reduced_system_arrow_bottom_blocks,
        A_reduced_system_arrow_tip_block,
        device_streaming=True if CUPY_AVAIL and xp == cp else False,
    )

    (
        X_reduced_system_diagonal_blocks,
        X_reduced_system_lower_diagonal_blocks,
        X_reduced_system_arrow_bottom_blocks,
        X_reduced_system_arrow_tip_block,
    ) = pobtasi(
        L_reduced_system_diagonal_blocks,
        L_reduced_system_lower_diagonal_blocks,
        L_reduced_system_arrow_bottom_blocks,
        L_reduced_system_arrow_tip_block,
        device_streaming=True if CUPY_AVAIL and xp == cp else False,
    )

    X_arrow_tip_block_global = X_reduced_system_arrow_tip_block

    # Update of the local slices by there corresponding blocks in the inverted
    # reduced system.
    if comm_rank == 0:
        X_diagonal_blocks_local[-1, :, :] = X_reduced_system_diagonal_blocks[0, :, :]
        X_lower_diagonal_blocks_local[-1, :, :] = (
            X_reduced_system_lower_diagonal_blocks[0, :, :]
        )
        X_arrow_bottom_blocks_local[-1, :, :] = X_reduced_system_arrow_bottom_blocks[
            0, :, :
        ]
    else:
        X_diagonal_blocks_local[0, :, :] = X_reduced_system_diagonal_blocks[
            2 * comm_rank - 1, :, :
        ]
        X_diagonal_blocks_local[-1, :, :] = X_reduced_system_diagonal_blocks[
            2 * comm_rank, :, :
        ]

        X_upper_nested_dissection_buffer_local[-1, :, :] = (
            X_reduced_system_lower_diagonal_blocks[2 * comm_rank - 1, :, :].conj().T
        )
        if comm_rank < comm_size - 1:
            X_lower_diagonal_blocks_local[-1, :, :] = (
                X_reduced_system_lower_diagonal_blocks[2 * comm_rank, :, :]
            )

        X_arrow_bottom_blocks_local[0, :, :] = X_reduced_system_arrow_bottom_blocks[
            2 * comm_rank - 1, :, :
        ]
        X_arrow_bottom_blocks_local[-1, :, :] = X_reduced_system_arrow_bottom_blocks[
            2 * comm_rank, :, :
        ]

    # Backward selected-inversion
    L_inv_temp = xp.zeros_like(L_diagonal_blocks_local[0])
    L_lower_diagonal_blocks_temp = xp.zeros_like(L_lower_diagonal_blocks_local[0])
    L_arrow_bottom_blocks_temp = xp.zeros_like(L_arrow_bottom_blocks_local[0])

    if comm_rank == 0:
        for i in range(n_diag_blocks_local - 2, -1, -1):
            # Compute lower factors
            L_inv_temp[:, :] = la.solve_triangular(
                L_diagonal_blocks_local[i, :, :],
                xp.eye(diag_blocksize),
                lower=True,
            )

            L_lower_diagonal_blocks_temp[:, :] = L_lower_diagonal_blocks_local[i, :, :]
            L_arrow_bottom_blocks_temp[:, :] = L_arrow_bottom_blocks_local[i, :, :]

            # --- Lower-diagonal blocks ---
            # X_{i+1, i} = (-X_{i+1, i+1} L_{i+1, i} - X_{i+1, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1}
            X_lower_diagonal_blocks_local[i, :, :] = (
                -X_diagonal_blocks_local[i + 1, :, :]
                @ L_lower_diagonal_blocks_temp[:, :]
                - X_arrow_bottom_blocks_local[i + 1, :, :].conj().T
                @ L_arrow_bottom_blocks_temp[:, :]
            ) @ L_inv_temp[:, :]

            # X_{ndb+1, i} = (- X_{ndb+1, i+1} L_{i+1, i} - X_{ndb+1, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1}
            X_arrow_bottom_blocks_local[i, :, :] = (
                -X_arrow_bottom_blocks_local[i + 1, :, :]
                @ L_lower_diagonal_blocks_temp[:, :]
                - X_arrow_tip_block_global[:, :] @ L_arrow_bottom_blocks_temp[:, :]
            ) @ L_inv_temp[:, :]

            # --- Diagonal block part ---
            # X_{i, i} = (L_{i, i}^{-T} - X_{i+1, i}^{T} L_{i+1, i} - X_{ndb+1, i}.T L_{ndb+1, i}) L_{i, i}^{-1}
            X_diagonal_blocks_local[i, :, :] = (
                L_inv_temp[:, :].conj().T
                - X_lower_diagonal_blocks_local[i, :, :].conj().T
                @ L_lower_diagonal_blocks_temp[:, :]
                - X_arrow_bottom_blocks_local[i, :, :].conj().T
                @ L_arrow_bottom_blocks_temp[:, :]
            ) @ L_inv_temp[:, :]
    else:
        L_upper_nested_dissection_buffer_temp = xp.zeros_like(
            L_upper_nested_dissection_buffer_local[0, :, :]
        )

        for i in range(n_diag_blocks_local - 2, 0, -1):
            L_inv_temp[:, :] = la.solve_triangular(
                L_diagonal_blocks_local[i, :, :],
                xp.eye(diag_blocksize),
                lower=True,
            )

            L_lower_diagonal_blocks_temp[:, :] = L_lower_diagonal_blocks_local[i, :, :]
            L_arrow_bottom_blocks_temp[:, :] = L_arrow_bottom_blocks_local[i, :, :]
            L_upper_nested_dissection_buffer_temp[:, :] = (
                L_upper_nested_dissection_buffer_local[i, :, :]
            )

            # X_{i+1, i} = (- X_{top, i+1}.T L_{top, i} - X_{i+1, i+1} L_{i+1, i} - X_{ndb+1, i+1}.T L_{ndb+1, i}) L_{i, i}^{-1}
            X_lower_diagonal_blocks_local[i, :, :] = (
                -X_upper_nested_dissection_buffer_local[i + 1, :, :].conj().T
                @ L_upper_nested_dissection_buffer_temp[:, :]
                - X_diagonal_blocks_local[i + 1, :, :]
                @ L_lower_diagonal_blocks_temp[:, :]
                - X_arrow_bottom_blocks_local[i + 1, :, :].conj().T
                @ L_arrow_bottom_blocks_temp[:, :]
            ) @ L_inv_temp[:, :]

            # X_{top, i} = (- X_{top, i+1} L_{i+1, i} - X_{top, top} L_{top, i} - X_{ndb+1, top}.T L_{ndb+1, i}) L_{i, i}^{-1}
            X_upper_nested_dissection_buffer_local[i, :, :] = (
                -X_upper_nested_dissection_buffer_local[i + 1, :, :]
                @ L_lower_diagonal_blocks_temp[:, :]
                - X_diagonal_blocks_local[0, :, :]
                @ L_upper_nested_dissection_buffer_temp[:, :]
                - X_arrow_bottom_blocks_local[0, :, :].conj().T
                @ L_arrow_bottom_blocks_temp[:, :]
            ) @ L_inv_temp[:, :]

            # Arrowhead
            # X_{ndb+1, i} = (- X_{ndb+1, i+1} L_{i+1, i} - X_{ndb+1, top} L_{top, i} - X_{ndb+1, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1}
            X_arrow_bottom_blocks_local[i, :, :] = (
                -X_arrow_bottom_blocks_local[i + 1, :, :]
                @ L_lower_diagonal_blocks_temp[:, :]
                - X_arrow_bottom_blocks_local[0, :, :]
                @ L_upper_nested_dissection_buffer_temp[:, :]
                - X_arrow_tip_block_global[:, :] @ L_arrow_bottom_blocks_temp[:, :]
            ) @ L_inv_temp[:, :]

            # X_{i, i} = (U_{i, i}^{-1} - X_{i+1, i}.T L_{i+1, i} - X_{top, i}.T L_{top, i} - X_{ndb+1, i}.T L_{ndb+1, i}) L_{i, i}^{-1}
            X_diagonal_blocks_local[i, :, :] = (
                L_inv_temp[:, :].conj().T
                - X_lower_diagonal_blocks_local[i, :, :].conj().T
                @ L_lower_diagonal_blocks_temp[:, :]
                - X_upper_nested_dissection_buffer_local[i, :, :].conj().T
                @ L_upper_nested_dissection_buffer_temp[:, :]
                - X_arrow_bottom_blocks_local[i, :, :].conj().T
                @ L_arrow_bottom_blocks_temp[:, :]
            ) @ L_inv_temp[:, :]

        # Copy back the 2 first blocks that have been produced in the 2-sided pattern
        # to the tridiagonal storage.
        X_lower_diagonal_blocks_local[0, :, :] = (
            X_upper_nested_dissection_buffer_local[1, :, :].conj().T
        )

    return (
        X_diagonal_blocks_local,
        X_lower_diagonal_blocks_local,
        X_arrow_bottom_blocks_local,
        X_arrow_tip_block_global,
    )


def _streaming_d_pobtasi(
    L_diagonal_blocks_local: ArrayLike,
    L_lower_diagonal_blocks_local: ArrayLike,
    L_arrow_bottom_blocks_local: ArrayLike,
    L_arrow_tip_block_global: ArrayLike,
    L_upper_nested_dissection_buffer_local: ArrayLike,
) -> tuple[
    ArrayLike,
    ArrayLike,
    ArrayLike,
    ArrayLike,
]:
    raise NotImplementedError
