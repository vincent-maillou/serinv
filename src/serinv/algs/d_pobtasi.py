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
    nested_solving: bool = False
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

    Complexity analysis:
        Parameters:
            n : number of diagonal blocks
            b : diagonal block size
            a : arrow block size
            p : number of processes
            n_r = (2*p-1): number of diagonal blocks of the reduced system

        The FLOPS count of the top process is the same as the one of the sequential
        block selected inversion algorithm (pobtasi). Following the assumption
        of a load balancing between the first processes and the others ("middle"), the FLOPS
        count is derived assuming only "middle" processes.

        The selected inversion procedure require the inversion of a reduced system
        made of boundary blocks of the distributed factorization of the matrix. The
        reduced system is constructed by gathering the boundary blocks of the factorization
        on each processes and performing a selected inversion of the reduced system.
        This selected inversion is assumed to be performed using the sequential
        block-Cholesky and selected inversion algorithm (pobtaf + pobtasi).

        Inversion of the reduced system:

        FLOPS count:
            Selected inversion of the reduced system:
                T_{flops_{POBTAF}} = n_r * (10/3 * b^3 + (1/2 + 3*a) * b^2 + (1/6 + 2*a^2) * b) - 3 * b^3 - 2*a*b^2 + 1/3 * a^3 + 1/2 * a^2 + 1/6 * a
                T_{flops_{POBTASI}} = (n_r-1) * (9*b^3 + 10*b^2 + 2*a^2*b) + 2*b^3 + 5*a*b^2 + 2*a^2*b + 2*a^3

            Distributed selected inversion:
                GEMM_b^3 : (n/p-1) * 18 * b^3
                GEMM_b^2_a : (n/p-1) * 14 * b^2 * a
                TRSM_b^3 : (n/p-1) * b^3

                Total FLOPS: (n/p-1) * (19*b^3 + 14*a*b^2)

            T_{flops_{d\_pobtasi}} = (n/p-1) * (19*b^3 + 14*a*b^2) + T_{flops_{POBTAF}} + T_{flops_{POBTASI}}

        Complexity:
            By making the assumption that b >> a, the complexity of the d_pobtasi
            algorithm is O(n/p * b^3 + p * b^3) or O((n+p^2)/p * b^3).

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
    nested_solving : bool
        If true, recursively calls d_pobtasi to solve the reduced system.

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
            nested_solving=nested_solving
        )

    return _d_pobtasi(
        L_diagonal_blocks_local,
        L_lower_diagonal_blocks_local,
        L_arrow_bottom_blocks_local,
        L_arrow_tip_block_global,
        L_upper_nested_dissection_buffer_local,
        nested_solving=nested_solving
    )


def _d_pobtasi(
    L_diagonal_blocks_local: ArrayLike,
    L_lower_diagonal_blocks_local: ArrayLike,
    L_arrow_bottom_blocks_local: ArrayLike,
    L_arrow_tip_block_global: ArrayLike,
    L_upper_nested_dissection_buffer_local: ArrayLike,
    nested_solving: bool = False
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
        A_reduced_system_diagonal_blocks_host = cpx.empty_like_pinned(
            A_reduced_system_diagonal_blocks
        )
        A_reduced_system_lower_diagonal_blocks_host = cpx.empty_like_pinned(
            A_reduced_system_lower_diagonal_blocks
        )
        A_reduced_system_arrow_bottom_blocks_host = cpx.empty_like_pinned(
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
    if nested_solving:
        # Nested Solving
        # Reduced system has 2 * size - 1 diagonal blocks.
        # Assing to 1 block to rank 0 and 2 blocks to all other ranks.
        m0 = 1
        m = 2
        if comm_rank == 0:
            A_reduced_system_diagonal_blocks_local = A_reduced_system_diagonal_blocks[:m0, :, :]
            A_reduced_system_lower_diagonal_blocks_local = A_reduced_system_lower_diagonal_blocks[:m0-1, :, :] if m0 > 1 else None
            A_reduced_system_arrow_bottom_blocks_local = A_reduced_system_arrow_bottom_blocks[:m0, :, :]
        else:
            A_reduced_system_diagonal_blocks_local = A_reduced_system_diagonal_blocks[m0 - 1 + (comm_rank - 1) * 2 : m0 + comm_rank * m, :, :]
            A_reduced_system_lower_diagonal_blocks_local = A_reduced_system_lower_diagonal_blocks[min(0, m0 - 2 + (comm_rank - 1) * 2) : m0 - 2 + comm_rank * m, :, :]
            A_reduced_system_arrow_bottom_blocks_local = A_reduced_system_arrow_bottom_blocks[m0 - 1 + (comm_rank - 1) * 2 : m0 + comm_rank * m, :, :]
        (
            X_reduced_system_diagonal_blocks_local,
            X_reduced_system_lower_diagonal_blocks_local,
            X_reduced_system_arrow_bottom_blocks_local,
            X_reduced_system_arrow_tip_block
        ) = d_pobtasi(
            A_reduced_system_diagonal_blocks_local,
            A_reduced_system_lower_diagonal_blocks_local,
            A_reduced_system_arrow_bottom_blocks_local,
            A_reduced_system_arrow_tip_block,
            device_streaming=True if CUPY_AVAIL and xp == cp else False,
        )

        # TODO: Gather the results to the X_reduced_system buffers. Is this needed or can we just use the local buffers?
    else:
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
    L_inv_temp = xp.empty_like(L_diagonal_blocks_local[0])
    L_lower_diagonal_blocks_temp = xp.empty_like(L_lower_diagonal_blocks_local[0])
    L_arrow_bottom_blocks_temp = xp.empty_like(L_arrow_bottom_blocks_local[0])

    if comm_rank == 0:
        for i in range(n_diag_blocks_local - 2, -1, -1):
            L_lower_diagonal_blocks_temp[:, :] = L_lower_diagonal_blocks_local[i, :, :]
            L_arrow_bottom_blocks_temp[:, :] = L_arrow_bottom_blocks_local[i, :, :]

            # Compute lower factors
            L_inv_temp[:, :] = la.solve_triangular(
                L_diagonal_blocks_local[i, :, :],
                xp.eye(diag_blocksize),
                lower=True,
            )

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
        L_upper_nested_dissection_buffer_temp = xp.empty_like(
            L_upper_nested_dissection_buffer_local[0, :, :]
        )

        for i in range(n_diag_blocks_local - 2, 0, -1):
            L_lower_diagonal_blocks_temp[:, :] = L_lower_diagonal_blocks_local[i, :, :]
            L_arrow_bottom_blocks_temp[:, :] = L_arrow_bottom_blocks_local[i, :, :]
            L_upper_nested_dissection_buffer_temp[:, :] = (
                L_upper_nested_dissection_buffer_local[i, :, :]
            )

            L_inv_temp[:, :] = la.solve_triangular(
                L_diagonal_blocks_local[i, :, :],
                xp.eye(diag_blocksize),
                lower=True,
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
    nested_solving: bool = False
) -> tuple[
    ArrayLike,
    ArrayLike,
    ArrayLike,
    ArrayLike,
]:
    diag_blocksize = L_diagonal_blocks_local.shape[1]
    n_diag_blocks_local = L_diagonal_blocks_local.shape[0]

    X_diagonal_blocks_local = L_diagonal_blocks_local
    X_lower_diagonal_blocks_local = L_lower_diagonal_blocks_local
    X_arrow_bottom_blocks_local = L_arrow_bottom_blocks_local
    X_upper_nested_dissection_buffer_local = L_upper_nested_dissection_buffer_local

    A_reduced_system_diagonal_blocks = cpx.zeros_pinned(
        (2 * comm_size - 1, *L_diagonal_blocks_local.shape[1:]),
        dtype=L_diagonal_blocks_local.dtype,
    )
    A_reduced_system_lower_diagonal_blocks = cpx.zeros_pinned(
        (2 * comm_size - 2, *L_lower_diagonal_blocks_local.shape[1:]),
        dtype=L_lower_diagonal_blocks_local.dtype,
    )
    A_reduced_system_arrow_bottom_blocks = cpx.zeros_pinned(
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

    MPI.COMM_WORLD.Allreduce(
        MPI.IN_PLACE,
        A_reduced_system_diagonal_blocks,
        op=MPI.SUM,
    )
    MPI.COMM_WORLD.Allreduce(
        MPI.IN_PLACE,
        A_reduced_system_lower_diagonal_blocks,
        op=MPI.SUM,
    )
    MPI.COMM_WORLD.Allreduce(
        MPI.IN_PLACE,
        A_reduced_system_arrow_bottom_blocks,
        op=MPI.SUM,
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
        device_streaming=True,
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
        device_streaming=True,
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
    # Device buffers
    L_diagonal_blocks_d = cp.empty(
        (2, *L_diagonal_blocks_local.shape[1:]), dtype=L_diagonal_blocks_local.dtype
    )
    L_lower_diagonal_blocks_d = cp.empty_like(L_diagonal_blocks_local[0])
    L_arrow_bottom_blocks_d = cp.empty(
        (2, *L_arrow_bottom_blocks_local.shape[1:]),
        dtype=L_arrow_bottom_blocks_local.dtype,
    )
    X_arrow_tip_block_d = cp.empty_like(X_arrow_tip_block_global)

    # X Device buffers arrays pointers
    X_diagonal_blocks_d = L_diagonal_blocks_d
    X_lower_diagonal_blocks_d = L_lower_diagonal_blocks_d
    X_arrow_bottom_blocks_d = L_arrow_bottom_blocks_d

    # Buffers for the intermediate results of the backward block-selected inversion
    L_inv_temp_d = cp.empty_like(L_diagonal_blocks_local[0])
    L_lower_diagonal_blocks_d_i = cp.empty_like(L_lower_diagonal_blocks_local[0])
    L_arrow_bottom_blocks_d_i = cp.empty_like(L_arrow_bottom_blocks_local[0])

    if comm_rank == 0:
        # --- Host 2 Device transfers ---
        X_diagonal_blocks_d[(n_diag_blocks_local - 1) % 2, :, :].set(
            arr=L_diagonal_blocks_local[-1, :, :]
        )
        X_arrow_bottom_blocks_d[(n_diag_blocks_local - 1) % 2, :, :].set(
            arr=L_arrow_bottom_blocks_local[-1, :, :]
        )
        X_arrow_tip_block_d.set(arr=X_arrow_tip_block_global)

        for i in range(n_diag_blocks_local - 2, -1, -1):
            # --- Host 2 Device transfers ---
            L_diagonal_blocks_d[i % 2, :, :].set(arr=L_diagonal_blocks_local[i, :, :])
            L_lower_diagonal_blocks_d[:, :].set(
                arr=L_lower_diagonal_blocks_local[i, :, :]
            )
            L_arrow_bottom_blocks_d[i % 2, :, :].set(
                arr=L_arrow_bottom_blocks_local[i, :, :]
            )

            L_lower_diagonal_blocks_d_i[:, :] = L_lower_diagonal_blocks_d[:, :]
            L_arrow_bottom_blocks_d_i[:, :] = L_arrow_bottom_blocks_d[i % 2, :, :]

            # --- Computations ---
            L_inv_temp_d[:, :] = cu_la.solve_triangular(
                L_diagonal_blocks_d[i % 2, :, :],
                cp.eye(diag_blocksize),
                lower=True,
            )

            # --- Off-diagonal block part ---
            # X_{i+1, i} = (-X_{i+1, i+1} L_{i+1, i} - X_{i+1, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1}
            X_lower_diagonal_blocks_d[:, :] = (
                -X_diagonal_blocks_d[(i + 1) % 2, :, :]
                @ L_lower_diagonal_blocks_d_i[:, :]
                - X_arrow_bottom_blocks_d[(i + 1) % 2, :, :].conj().T
                @ L_arrow_bottom_blocks_d_i[:, :]
            ) @ L_inv_temp_d[:, :]

            # --- Arrowhead part ---
            # X_{ndb+1, i} = (- X_{ndb+1, i+1} L_{i+1, i} - X_{ndb+1, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1}
            X_arrow_bottom_blocks_d[i % 2, :, :] = (
                -X_arrow_bottom_blocks_d[(i + 1) % 2, :, :]
                @ L_lower_diagonal_blocks_d_i[:, :]
                - X_arrow_tip_block_d[:, :] @ L_arrow_bottom_blocks_d_i[:, :]
            ) @ L_inv_temp_d[:, :]

            # --- Diagonal block part ---
            # X_{i, i} = (L_{i, i}^{-T} - X_{i+1, i}^{T} L_{i+1, i} - X_{ndb+1, i}.T L_{ndb+1, i}) L_{i, i}^{-1}
            X_diagonal_blocks_d[i % 2, :, :] = (
                L_inv_temp_d[:, :].conj().T
                - X_lower_diagonal_blocks_d[:, :].conj().T
                @ L_lower_diagonal_blocks_d_i[:, :]
                - X_arrow_bottom_blocks_d[i % 2, :, :].conj().T
                @ L_arrow_bottom_blocks_d_i[:, :]
            ) @ L_inv_temp_d[:, :]

            # --- Device 2 Host transfers ---
            X_diagonal_blocks_d[i % 2, :, :].get(out=X_diagonal_blocks_local[i, :, :])
            X_lower_diagonal_blocks_d[:, :].get(
                out=X_lower_diagonal_blocks_local[i, :, :]
            )
            X_arrow_bottom_blocks_d[i % 2, :, :].get(
                out=X_arrow_bottom_blocks_local[i, :, :]
            )
    else:
        # Device aliases & buffers specific to the middle process
        X_diagonal_top_block_d = cp.empty_like(X_diagonal_blocks_local[0])
        X_arrow_bottom_top_block_d = cp.empty_like(X_arrow_bottom_blocks_local[0])
        L_upper_nested_dissection_buffer_d = cp.empty(
            (2, *L_upper_nested_dissection_buffer_local.shape[1:]),
            dtype=L_upper_nested_dissection_buffer_local.dtype,
        )

        X_upper_nested_dissection_buffer_d = L_upper_nested_dissection_buffer_d

        L_upper_nested_dissection_buffer_d_i = cp.empty_like(
            L_upper_nested_dissection_buffer_local[0, :, :]
        )

        # --- Host 2 Device transfers ---
        X_diagonal_blocks_d[(n_diag_blocks_local - 1) % 2, :, :].set(
            arr=L_diagonal_blocks_local[-1, :, :]
        )
        X_arrow_bottom_blocks_d[(n_diag_blocks_local - 1) % 2, :, :].set(
            arr=L_arrow_bottom_blocks_local[-1, :, :]
        )
        X_arrow_tip_block_d.set(arr=X_arrow_tip_block_global)

        X_diagonal_top_block_d.set(arr=X_diagonal_blocks_local[0, :, :])
        X_arrow_bottom_top_block_d.set(arr=X_arrow_bottom_blocks_local[0, :, :])
        L_upper_nested_dissection_buffer_d[(n_diag_blocks_local - 1) % 2, :, :].set(
            arr=L_upper_nested_dissection_buffer_local[-1, :, :]
        )

        for i in range(n_diag_blocks_local - 2, 0, -1):
            # --- Host 2 Device transfers ---
            L_diagonal_blocks_d[i % 2, :, :].set(arr=L_diagonal_blocks_local[i, :, :])
            L_lower_diagonal_blocks_d[:, :].set(
                arr=L_lower_diagonal_blocks_local[i, :, :]
            )
            L_arrow_bottom_blocks_d[i % 2, :, :].set(
                arr=L_arrow_bottom_blocks_local[i, :, :]
            )
            L_upper_nested_dissection_buffer_d[i % 2, :, :].set(
                arr=L_upper_nested_dissection_buffer_local[i, :, :]
            )

            L_lower_diagonal_blocks_d_i[:, :] = L_lower_diagonal_blocks_d[:, :]
            L_arrow_bottom_blocks_d_i[:, :] = L_arrow_bottom_blocks_d[i % 2, :, :]
            L_upper_nested_dissection_buffer_d_i[:, :] = (
                L_upper_nested_dissection_buffer_d[i % 2, :, :]
            )

            # --- Computations ---
            L_inv_temp_d[:, :] = cu_la.solve_triangular(
                L_diagonal_blocks_d[i % 2, :, :],
                cp.eye(diag_blocksize),
                lower=True,
            )

            # X_{i+1, i} = (- X_{top, i+1}.T L_{top, i} - X_{i+1, i+1} L_{i+1, i} - X_{ndb+1, i+1}.T L_{ndb+1, i}) L_{i, i}^{-1}
            X_lower_diagonal_blocks_d[:, :] = (
                -X_upper_nested_dissection_buffer_d[(i + 1) % 2, :, :].conj().T
                @ L_upper_nested_dissection_buffer_d_i[:, :]
                - X_diagonal_blocks_d[(i + 1) % 2, :, :]
                @ L_lower_diagonal_blocks_d_i[:, :]
                - X_arrow_bottom_blocks_d[(i + 1) % 2, :, :].conj().T
                @ L_arrow_bottom_blocks_d_i[:, :]
            ) @ L_inv_temp_d[:, :]

            # X_{top, i} = (- X_{top, i+1} L_{i+1, i} - X_{top, top} L_{top, i} - X_{ndb+1, top}.T L_{ndb+1, i}) L_{i, i}^{-1}
            X_upper_nested_dissection_buffer_d[i % 2, :, :] = (
                -X_upper_nested_dissection_buffer_d[(i + 1) % 2, :, :]
                @ L_lower_diagonal_blocks_d_i[:, :]
                - X_diagonal_top_block_d[:, :]
                @ L_upper_nested_dissection_buffer_d_i[:, :]
                - X_arrow_bottom_top_block_d[:, :].conj().T
                @ L_arrow_bottom_blocks_d_i[:, :]
            ) @ L_inv_temp_d[:, :]

            # Arrowhead
            # X_{ndb+1, i} = (- X_{ndb+1, i+1} L_{i+1, i} - X_{ndb+1, top} L_{top, i} - X_{ndb+1, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1}
            X_arrow_bottom_blocks_d[i % 2, :, :] = (
                -X_arrow_bottom_blocks_d[(i + 1) % 2, :, :]
                @ L_lower_diagonal_blocks_d_i[:, :]
                - X_arrow_bottom_top_block_d[:, :]
                @ L_upper_nested_dissection_buffer_d_i[:, :]
                - X_arrow_tip_block_d[:, :] @ L_arrow_bottom_blocks_d_i[:, :]
            ) @ L_inv_temp_d[:, :]

            # X_{i, i} = (U_{i, i}^{-1} - X_{i+1, i}.T L_{i+1, i} - X_{top, i}.T L_{top, i} - X_{ndb+1, i}.T L_{ndb+1, i}) L_{i, i}^{-1}
            X_diagonal_blocks_d[i % 2, :, :] = (
                L_inv_temp_d[:, :].conj().T
                - X_lower_diagonal_blocks_d[:, :].conj().T
                @ L_lower_diagonal_blocks_d_i[:, :]
                - X_upper_nested_dissection_buffer_d[i % 2, :, :].conj().T
                @ L_upper_nested_dissection_buffer_d_i[:, :]
                - X_arrow_bottom_blocks_d[i % 2, :, :].conj().T
                @ L_arrow_bottom_blocks_d_i[:, :]
            ) @ L_inv_temp_d[:, :]

            # --- Device 2 Host transfers ---
            X_diagonal_blocks_d[i % 2, :, :].get(out=X_diagonal_blocks_local[i, :, :])
            X_lower_diagonal_blocks_d[:, :].get(
                out=X_lower_diagonal_blocks_local[i, :, :]
            )
            X_arrow_bottom_blocks_d[i % 2, :, :].get(
                out=X_arrow_bottom_blocks_local[i, :, :]
            )
            X_upper_nested_dissection_buffer_d[i % 2, :, :].get(
                out=X_upper_nested_dissection_buffer_local[i, :, :]
            )

        # Copy back the 2 first blocks that have been produced in the 2-sided pattern
        # to the tridiagonal storage.
        X_lower_diagonal_blocks_local[0, :, :] = (
            X_upper_nested_dissection_buffer_local[1, :, :].conj().T
        )

    cp.cuda.Device().synchronize()
    cp.cuda.nvtx.RangePop()

    return (
        X_diagonal_blocks_local,
        X_lower_diagonal_blocks_local,
        X_arrow_bottom_blocks_local,
        X_arrow_tip_block_global,
    )
