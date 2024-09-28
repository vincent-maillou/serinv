# Copyright 2023-2024 ETH Zurich. All rights reserved.
from serinv import SolverConfig
from serinv.algs import ddbtaf, ddbtasi

try:
    import cupy as cp
    import cupyx as cpx

    CUPY_AVAIL = True

except ImportError:
    CUPY_AVAIL = False

import numpy as np
import scipy.linalg as np_la
from mpi4py import MPI
from numpy.typing import ArrayLike

comm_rank = MPI.COMM_WORLD.Get_rank()
comm_size = MPI.COMM_WORLD.Get_size()


def d_ddbtasi(
    LU_diagonal_blocks_local: ArrayLike,
    LU_lower_diagonal_blocks_local: ArrayLike,
    LU_upper_diagonal_blocks_local: ArrayLike,
    LU_arrow_bottom_blocks_local: ArrayLike,
    LU_arrow_right_blocks_local: ArrayLike,
    LU_arrow_tip_block_global: ArrayLike,
    L_permutation_upper: ArrayLike = None,
    U_permutation_lower: ArrayLike = None,
    solver_config: SolverConfig = SolverConfig(),
) -> tuple[
    ArrayLike,
    ArrayLike,
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
    LU_diagonal_blocks_local : ArrayLike
        Local slice of the diagonal blocks of LU.
    LU_lower_diagonal_blocks_local : ArrayLike
        Local slice of the lower diagonal blocks of LU.
    LU_upper_diagonal_blocks_local : ArrayLike
        Local slice of the upper diagonal blocks of LU.
    LU_arrow_bottom_blocks_local : ArrayLike
        Local slice of the arrow bottom blocks of LU.
    LU_arrow_right_blocks_local : ArrayLike
        Local slice of the arrow top blocks of LU.
    LU_arrow_tip_block_global : ArrayLike
        Arrow tip block of LU.
    L_permutation_upper : ArrayLike, optional
        Local upper buffer used in the nested dissection factorization. None for
        uppermost process.
    U_permutation_lower : ArrayLike, optional
        Local lower buffer used in the nested dissection factorization. None for
        lowermost process.
    solver_config : SolverConfig, optional
        Configuration of the solver.

    Returns
    -------
    X_diagonal_blocks_local : ArrayLike
        Local slice of the diagonal blocks of X, alias on LU_diagonal_blocks_local.
    X_lower_diagonal_blocks_local : ArrayLike
        Local slice of the lower diagonal blocks of X, alias on LU_lower_diagonal_blocks_local.
    X_upper_diagonal_blocks_local : ArrayLike
        Local slice of the upper diagonal blocks of X, alias on LU_upper_diagonal_blocks_local.
    X_arrow_bottom_blocks_local : ArrayLike
        Local slice of the arrow bottom blocks of X, alias on LU_arrow_bottom_blocks_local.
    X_arrow_top_blocks_local : ArrayLike
        Local slice of the arrow top blocks of X, alias on LU_arrow_right_blocks_local.
    X_arrow_tip_block_global : ArrayLike
        Arrow tip block of X, alias on LU_arrow_tip_block_global.
    """

    # array_module = cp.get_array_module(LU_diagonal_blocks_local)

    # if array_module == cp:
    #     # Device computation
    #     return _device_d_ddbtasi(
    #         LU_diagonal_blocks_local,
    #         LU_lower_diagonal_blocks_local,
    #         LU_upper_diagonal_blocks_local,
    #         LU_arrow_bottom_blocks_local,
    #         LU_arrow_right_blocks_local,
    #         LU_arrow_tip_block_global,
    #         L_permutation_upper,
    #         U_permutation_lower,
    #         solver_config,
    #     )
    # else:
    #     # Host computation
    #     return _host_d_ddbtasi(
    #         LU_diagonal_blocks_local,
    #         LU_lower_diagonal_blocks_local,
    #         LU_upper_diagonal_blocks_local,
    #         LU_arrow_bottom_blocks_local,
    #         LU_arrow_right_blocks_local,
    #         LU_arrow_tip_block_global,
    #         L_permutation_upper,
    #         U_permutation_lower,
    #     )


def d_ddbtasi_rss(
    LU_diagonal_blocks_local: ArrayLike,
    LU_lower_diagonal_blocks_local: ArrayLike,
    LU_upper_diagonal_blocks_local: ArrayLike,
    LU_arrow_bottom_blocks_local: ArrayLike,
    LU_arrow_right_blocks_local: ArrayLike,
    LU_arrow_tip_block_global: ArrayLike,
    L_permutation_upper: ArrayLike = None,
    U_permutation_lower: ArrayLike = None,
    solver_config: SolverConfig = SolverConfig(),
) -> tuple[
    ArrayLike,
    ArrayLike,
    ArrayLike,
    ArrayLike,
    ArrayLike,
    ArrayLike,
    ArrayLike,
    ArrayLike,
]:
    """Perform the selected-inversion of the reduced system constructed from the
    distributed factorization of the block tridiagonal with arrowhead matrix.

    Parameters
    ----------
    LU_diagonal_blocks_local : ArrayLike
        Local slice of the diagonal blocks of LU.
    LU_lower_diagonal_blocks_local : ArrayLike
        Local slice of the lower diagonal blocks of LU.
    LU_upper_diagonal_blocks_local : ArrayLike
        Local slice of the upper diagonal blocks of LU.
    LU_arrow_bottom_blocks_local : ArrayLike
        Local slice of the arrow bottom blocks of LU.
    LU_arrow_right_blocks_local : ArrayLike
        Local slice of the arrow top blocks of LU.
    LU_arrow_tip_block_global : ArrayLike
        Arrow tip block of LU.
    L_permutation_upper : ArrayLike, optional
        Local upper buffer used in the permuted factorization. None for
        uppermost process.
    U_permutation_lower : ArrayLike, optional
        Local lower buffer used in the permuted factorization. None for
        lowermost process.
    solver_config : SolverConfig, optional
        Configuration of the solver.

    Returns
    -------
    LU_diagonal_blocks_local : ArrayLike
        Local slice of the diagonal blocks of LU, initialized with the solution
        of the reduced system.
    LU_lower_diagonal_blocks_local : ArrayLike
        Local slice of the lower diagonal blocks of LU, initialized with the solution
        of the reduced system.
    LU_upper_diagonal_blocks_local : ArrayLike
        Local slice of the upper diagonal blocks of LU, initialized with the solution
        of the reduced system.
    LU_arrow_bottom_blocks_local : ArrayLike
        Local slice of the arrow bottom blocks of LU, initialized with the solution
        of the reduced system.
    LU_arrow_right_blocks_local : ArrayLike
        Local slice of the arrow top blocks of LU, initialized with the solution
        of the reduced system.
    LU_arrow_tip_block_global : ArrayLike
        Arrow tip block of LU, initialized with the solution of the reduced system.
    L_permutation_upper : ArrayLike, optional
        Local upper buffer used in the permuted factorization, initialized
        with the solution of the reduced system. None for uppermost process.
    U_permutation_lower : ArrayLike, optional
        Local lower buffer used in the permuted factorization, initialized
        with the solution of the reduced system. None for lowermost process.
    """

    array_module = cp.get_array_module(LU_diagonal_blocks_local)

    if array_module == cp:
        # Device streaming
        return _device_d_ddbtasi_rss(
            LU_diagonal_blocks_local,
            LU_lower_diagonal_blocks_local,
            LU_upper_diagonal_blocks_local,
            LU_arrow_bottom_blocks_local,
            LU_arrow_right_blocks_local,
            LU_arrow_tip_block_global,
            L_permutation_upper,
            U_permutation_lower,
            solver_config,
        )
    else:
        # Host computation
        return _host_d_ddbtasi_rss(
            LU_diagonal_blocks_local,
            LU_lower_diagonal_blocks_local,
            LU_upper_diagonal_blocks_local,
            LU_arrow_bottom_blocks_local,
            LU_arrow_right_blocks_local,
            LU_arrow_tip_block_global,
            L_permutation_upper,
            U_permutation_lower,
            solver_config,
        )


def _host_d_ddbtasi_rss(
    LU_diagonal_blocks_local: ArrayLike,
    LU_lower_diagonal_blocks_local: ArrayLike,
    LU_upper_diagonal_blocks_local: ArrayLike,
    LU_arrow_bottom_blocks_local: ArrayLike,
    LU_arrow_right_blocks_local: ArrayLike,
    LU_arrow_tip_block_global: ArrayLike,
    L_permutation_upper: ArrayLike,
    U_permutation_lower: ArrayLike,
    solver_config: SolverConfig = SolverConfig(),
) -> tuple[
    ArrayLike,
    ArrayLike,
    ArrayLike,
    ArrayLike,
    ArrayLike,
    ArrayLike,
    ArrayLike,
    ArrayLike,
]:
    A_reduced_system_diagonal_blocks = np.zeros(
        (2 * comm_size, *LU_diagonal_blocks_local.shape[1:]),
        dtype=LU_diagonal_blocks_local.dtype,
    )

    A_reduced_system_lower_diagonal_blocks = np.zeros(
        (2 * comm_size, *LU_lower_diagonal_blocks_local.shape[1:]),
        dtype=LU_lower_diagonal_blocks_local.dtype,
    )
    A_reduced_system_upper_diagonal_blocks = np.zeros(
        (2 * comm_size, *LU_upper_diagonal_blocks_local.shape[1:]),
        dtype=LU_upper_diagonal_blocks_local.dtype,
    )

    A_reduced_system_arrow_bottom_blocks = np.zeros(
        (2 * comm_size, *LU_arrow_bottom_blocks_local.shape[1:]),
        dtype=LU_arrow_bottom_blocks_local.dtype,
    )
    A_reduced_system_arrow_right_blocks = np.zeros(
        (2 * comm_size, *LU_arrow_right_blocks_local.shape[1:]),
        dtype=LU_arrow_right_blocks_local.dtype,
    )

    # Alias on the tip block for the reduced system
    A_reduced_system_arrow_tip_block = LU_arrow_tip_block_global

    # Construct the reduced system from the factorized blocks distributed over the
    # processes.
    if comm_rank == 0:
        # R_{0,0} = A^{p_0}_{0,0}
        A_reduced_system_diagonal_blocks[1, :, :] = LU_diagonal_blocks_local[-1, :, :]

        # R_{1, 0} = A^{p_0}_{1, 0}
        A_reduced_system_lower_diagonal_blocks[1, :, :] = (
            LU_lower_diagonal_blocks_local[-1, :, :]
        )
        A_reduced_system_upper_diagonal_blocks[1, :, :] = (
            LU_upper_diagonal_blocks_local[-1, :, :]
        )

        # R_{n, 0} = A^{p_0}_{n, 0}
        A_reduced_system_arrow_bottom_blocks[1, :, :] = LU_arrow_bottom_blocks_local[
            -1, :, :
        ]
        A_reduced_system_arrow_right_blocks[1, :, :] = LU_arrow_right_blocks_local[
            -1, :, :
        ]
    else:
        # R_{2p_i-1, 2p_i-1} = A^{p_i}_{0, 0}
        A_reduced_system_diagonal_blocks[2 * comm_rank, :, :] = (
            LU_diagonal_blocks_local[0, :, :]
        )
        # R_{2p_i, 2p_i-1} = A^{p_i}_{-1, -1}
        A_reduced_system_diagonal_blocks[2 * comm_rank + 1, :, :] = (
            LU_diagonal_blocks_local[-1, :, :]
        )

        # R_{2p_i-1, 2p_i-2} = B^{p_i}_{-1}^\dagger
        A_reduced_system_lower_diagonal_blocks[2 * comm_rank, :, :] = (
            U_permutation_lower[-1, :, :]
        )
        A_reduced_system_upper_diagonal_blocks[2 * comm_rank, :, :] = (
            L_permutation_upper[-1, :, :]
        )

        if comm_rank < comm_size - 1:
            # R_{2p_i, 2p_i-1} = A^{p_i}_{1, 0}
            A_reduced_system_lower_diagonal_blocks[2 * comm_rank + 1, :, :] = (
                LU_lower_diagonal_blocks_local[-1, :, :]
            )
            A_reduced_system_upper_diagonal_blocks[2 * comm_rank + 1, :, :] = (
                LU_upper_diagonal_blocks_local[-1, :, :]
            )

        # R_{n, 2p_i-1} = A^{p_i}_{n, 0}
        A_reduced_system_arrow_bottom_blocks[2 * comm_rank, :, :] = (
            LU_arrow_bottom_blocks_local[0, :, :]
        )
        # R_{n, 2p_i} = A^{p_i}_{n, -1}
        A_reduced_system_arrow_bottom_blocks[2 * comm_rank + 1, :, :] = (
            LU_arrow_bottom_blocks_local[-1, :, :]
        )

        A_reduced_system_arrow_right_blocks[2 * comm_rank, :, :] = (
            LU_arrow_right_blocks_local[0, :, :]
        )
        A_reduced_system_arrow_right_blocks[2 * comm_rank + 1, :, :] = (
            LU_arrow_right_blocks_local[-1, :, :]
        )

    # Communicate the reduced system
    MPI.COMM_WORLD.Allgather(
        MPI.IN_PLACE,
        A_reduced_system_diagonal_blocks,
    )

    MPI.COMM_WORLD.Allgather(
        MPI.IN_PLACE,
        A_reduced_system_lower_diagonal_blocks,
    )
    MPI.COMM_WORLD.Allgather(
        MPI.IN_PLACE,
        A_reduced_system_upper_diagonal_blocks,
    )

    MPI.COMM_WORLD.Allgather(
        MPI.IN_PLACE,
        A_reduced_system_arrow_bottom_blocks,
    )
    MPI.COMM_WORLD.Allgather(
        MPI.IN_PLACE,
        A_reduced_system_arrow_right_blocks,
    )

    # Perform the inversion of the reduced system.
    (
        LU_reduced_system_diagonal_blocks,
        LU_reduced_system_lower_diagonal_blocks,
        LU_reduced_system_upper_diagonal_blocks,
        LU_reduced_system_arrow_bottom_blocks,
        LU_reduced_system_arrow_right_blocks,
        LU_reduced_system_arrow_tip_block,
    ) = ddbtaf(
        A_reduced_system_diagonal_blocks[1:, :, :],
        A_reduced_system_lower_diagonal_blocks[1:-1, :, :],
        A_reduced_system_upper_diagonal_blocks[1:-1, :, :],
        A_reduced_system_arrow_bottom_blocks[1:, :, :],
        A_reduced_system_arrow_right_blocks[1:, :, :],
        A_reduced_system_arrow_tip_block,
    )

    (
        X_reduced_system_diagonal_blocks,
        X_reduced_system_lower_diagonal_blocks,
        X_reduced_system_upper_diagonal_blocks,
        X_reduced_system_arrow_bottom_blocks,
        X_reduced_system_arrow_right_blocks,
        X_reduced_system_arrow_tip_block,
    ) = ddbtasi(
        LU_reduced_system_diagonal_blocks,
        LU_reduced_system_lower_diagonal_blocks,
        LU_reduced_system_upper_diagonal_blocks,
        LU_reduced_system_arrow_bottom_blocks,
        LU_reduced_system_arrow_right_blocks,
        LU_reduced_system_arrow_tip_block,
    )

    # Update of the local slices by there corresponding blocks in the inverted
    # reduced system.
    if comm_rank == 0:
        LU_diagonal_blocks_local[-1, :, :] = X_reduced_system_diagonal_blocks[0, :, :]

        LU_lower_diagonal_blocks_local[-1, :, :] = (
            X_reduced_system_lower_diagonal_blocks[0, :, :]
        )
        LU_upper_diagonal_blocks_local[-1, :, :] = (
            X_reduced_system_upper_diagonal_blocks[0, :, :]
        )

        LU_arrow_bottom_blocks_local[-1, :, :] = X_reduced_system_arrow_bottom_blocks[
            0, :, :
        ]
        LU_arrow_right_blocks_local[-1, :, :] = X_reduced_system_arrow_right_blocks[
            0, :, :
        ]
    else:
        LU_diagonal_blocks_local[0, :, :] = X_reduced_system_diagonal_blocks[
            2 * comm_rank - 1, :, :
        ]
        LU_diagonal_blocks_local[-1, :, :] = X_reduced_system_diagonal_blocks[
            2 * comm_rank, :, :
        ]

        U_permutation_lower[-1, :, :] = X_reduced_system_lower_diagonal_blocks[
            2 * comm_rank - 1, :, :
        ]
        L_permutation_upper[-1, :, :] = X_reduced_system_upper_diagonal_blocks[
            2 * comm_rank - 1, :, :
        ]

        if comm_rank < comm_size - 1:
            LU_lower_diagonal_blocks_local[-1, :, :] = (
                X_reduced_system_lower_diagonal_blocks[2 * comm_rank, :, :]
            )
            LU_upper_diagonal_blocks_local[-1, :, :] = (
                X_reduced_system_upper_diagonal_blocks[2 * comm_rank, :, :]
            )

        LU_arrow_bottom_blocks_local[0, :, :] = X_reduced_system_arrow_bottom_blocks[
            2 * comm_rank - 1, :, :
        ]
        LU_arrow_bottom_blocks_local[-1, :, :] = X_reduced_system_arrow_bottom_blocks[
            2 * comm_rank, :, :
        ]

        LU_arrow_right_blocks_local[0, :, :] = X_reduced_system_arrow_right_blocks[
            2 * comm_rank - 1, :, :
        ]
        LU_arrow_right_blocks_local[-1, :, :] = X_reduced_system_arrow_right_blocks[
            2 * comm_rank, :, :
        ]

    return (
        LU_diagonal_blocks_local,
        LU_lower_diagonal_blocks_local,
        LU_upper_diagonal_blocks_local,
        LU_arrow_bottom_blocks_local,
        LU_arrow_right_blocks_local,
        LU_arrow_tip_block_global,
        L_permutation_upper,
        U_permutation_lower,
    )


def _device_d_ddbtasi_rss(
    LU_diagonal_blocks_local: ArrayLike,
    LU_lower_diagonal_blocks_local: ArrayLike,
    LU_upper_diagonal_blocks_local: ArrayLike,
    LU_arrow_bottom_blocks_local: ArrayLike,
    LU_arrow_right_blocks_local: ArrayLike,
    LU_arrow_tip_block_global: ArrayLike,
    L_permutation_upper: ArrayLike,
    U_permutation_lower: ArrayLike,
    solver_config: SolverConfig = SolverConfig(),
) -> tuple[
    ArrayLike,
    ArrayLike,
    ArrayLike,
    ArrayLike,
    ArrayLike,
    ArrayLike,
    ArrayLike,
    ArrayLike,
]:
    A_reduced_system_diagonal_blocks = cp.zeros(
        (2 * comm_size, *LU_diagonal_blocks_local.shape[1:]),
        dtype=LU_diagonal_blocks_local.dtype,
    )

    A_reduced_system_lower_diagonal_blocks = cp.zeros(
        (2 * comm_size, *LU_lower_diagonal_blocks_local.shape[1:]),
        dtype=LU_lower_diagonal_blocks_local.dtype,
    )
    A_reduced_system_upper_diagonal_blocks = cp.zeros(
        (2 * comm_size, *LU_upper_diagonal_blocks_local.shape[1:]),
        dtype=LU_upper_diagonal_blocks_local.dtype,
    )

    A_reduced_system_arrow_bottom_blocks = cp.zeros(
        (2 * comm_size, *LU_arrow_bottom_blocks_local.shape[1:]),
        dtype=LU_arrow_bottom_blocks_local.dtype,
    )
    A_reduced_system_arrow_right_blocks = cp.zeros(
        (2 * comm_size, *LU_arrow_right_blocks_local.shape[1:]),
        dtype=LU_arrow_right_blocks_local.dtype,
    )

    # Alias on the tip block for the reduced system
    A_reduced_system_arrow_tip_block = LU_arrow_tip_block_global

    # Construct the reduced system from the factorized blocks distributed over the
    # processes.
    if comm_rank == 0:
        # R_{0,0} = A^{p_0}_{0,0}
        A_reduced_system_diagonal_blocks[1, :, :] = LU_diagonal_blocks_local[-1, :, :]

        # R_{1, 0} = A^{p_0}_{1, 0}
        A_reduced_system_lower_diagonal_blocks[1, :, :] = (
            LU_lower_diagonal_blocks_local[-1, :, :]
        )
        A_reduced_system_upper_diagonal_blocks[1, :, :] = (
            LU_upper_diagonal_blocks_local[-1, :, :]
        )

        # R_{n, 0} = A^{p_0}_{n, 0}
        A_reduced_system_arrow_bottom_blocks[1, :, :] = LU_arrow_bottom_blocks_local[
            -1, :, :
        ]
        A_reduced_system_arrow_right_blocks[1, :, :] = LU_arrow_right_blocks_local[
            -1, :, :
        ]
    else:
        # R_{2p_i-1, 2p_i-1} = A^{p_i}_{0, 0}
        A_reduced_system_diagonal_blocks[2 * comm_rank, :, :] = (
            LU_diagonal_blocks_local[0, :, :]
        )
        # R_{2p_i, 2p_i-1} = A^{p_i}_{-1, -1}
        A_reduced_system_diagonal_blocks[2 * comm_rank + 1, :, :] = (
            LU_diagonal_blocks_local[-1, :, :]
        )

        # R_{2p_i-1, 2p_i-2} = B^{p_i}_{-1}^\dagger
        A_reduced_system_lower_diagonal_blocks[2 * comm_rank, :, :] = (
            U_permutation_lower[-1, :, :]
        )
        A_reduced_system_upper_diagonal_blocks[2 * comm_rank, :, :] = (
            L_permutation_upper[-1, :, :]
        )

        if comm_rank < comm_size - 1:
            # R_{2p_i, 2p_i-1} = A^{p_i}_{1, 0}
            A_reduced_system_lower_diagonal_blocks[2 * comm_rank + 1, :, :] = (
                LU_lower_diagonal_blocks_local[-1, :, :]
            )
            A_reduced_system_upper_diagonal_blocks[2 * comm_rank + 1, :, :] = (
                LU_upper_diagonal_blocks_local[-1, :, :]
            )

        # R_{n, 2p_i-1} = A^{p_i}_{n, 0}
        A_reduced_system_arrow_bottom_blocks[2 * comm_rank, :, :] = (
            LU_arrow_bottom_blocks_local[0, :, :]
        )
        # R_{n, 2p_i} = A^{p_i}_{n, -1}
        A_reduced_system_arrow_bottom_blocks[2 * comm_rank + 1, :, :] = (
            LU_arrow_bottom_blocks_local[-1, :, :]
        )

        A_reduced_system_arrow_right_blocks[2 * comm_rank, :, :] = (
            LU_arrow_right_blocks_local[0, :, :]
        )
        A_reduced_system_arrow_right_blocks[2 * comm_rank + 1, :, :] = (
            LU_arrow_right_blocks_local[-1, :, :]
        )

    # Allocate reduced system on host using pinned memory
    A_reduced_system_diagonal_blocks_host = cpx.empty_like_pinned(
        A_reduced_system_diagonal_blocks
    )

    A_reduced_system_lower_diagonal_blocks_host = cpx.empty_like_pinned(
        A_reduced_system_lower_diagonal_blocks
    )
    A_reduced_system_upper_diagonal_blocks_host = cpx.empty_like_pinned(
        A_reduced_system_upper_diagonal_blocks
    )

    A_reduced_system_arrow_bottom_blocks_host = cpx.empty_like_pinned(
        A_reduced_system_arrow_bottom_blocks
    )
    A_reduced_system_arrow_right_blocks_host = cpx.empty_like_pinned(
        A_reduced_system_arrow_right_blocks
    )

    # Copy the reduced system to the host
    A_reduced_system_diagonal_blocks.get(out=A_reduced_system_diagonal_blocks_host)

    A_reduced_system_lower_diagonal_blocks.get(
        out=A_reduced_system_lower_diagonal_blocks_host
    )
    A_reduced_system_upper_diagonal_blocks.get(
        out=A_reduced_system_upper_diagonal_blocks_host
    )

    A_reduced_system_arrow_bottom_blocks.get(
        out=A_reduced_system_arrow_bottom_blocks_host
    )
    A_reduced_system_arrow_right_blocks.get(
        out=A_reduced_system_arrow_right_blocks_host
    )

    # Communicate the reduced system
    MPI.COMM_WORLD.Allgather(
        MPI.IN_PLACE,
        A_reduced_system_diagonal_blocks_host,
    )

    MPI.COMM_WORLD.Allgather(
        MPI.IN_PLACE,
        A_reduced_system_lower_diagonal_blocks_host,
    )
    MPI.COMM_WORLD.Allgather(
        MPI.IN_PLACE,
        A_reduced_system_upper_diagonal_blocks_host,
    )

    MPI.COMM_WORLD.Allgather(
        MPI.IN_PLACE,
        A_reduced_system_arrow_bottom_blocks_host,
    )
    MPI.COMM_WORLD.Allgather(
        MPI.IN_PLACE,
        A_reduced_system_arrow_right_blocks_host,
    )

    # Copy the reduced system back to the device
    A_reduced_system_diagonal_blocks.set(arr=A_reduced_system_diagonal_blocks_host)

    A_reduced_system_lower_diagonal_blocks.set(
        arr=A_reduced_system_lower_diagonal_blocks_host
    )
    A_reduced_system_upper_diagonal_blocks.set(
        arr=A_reduced_system_upper_diagonal_blocks_host
    )

    A_reduced_system_arrow_bottom_blocks.set(
        arr=A_reduced_system_arrow_bottom_blocks_host
    )
    A_reduced_system_arrow_right_blocks.set(
        arr=A_reduced_system_arrow_right_blocks_host
    )

    # Perform the inversion of the reduced system.
    (
        LU_reduced_system_diagonal_blocks,
        LU_reduced_system_lower_diagonal_blocks,
        LU_reduced_system_upper_diagonal_blocks,
        LU_reduced_system_arrow_bottom_blocks,
        LU_reduced_system_arrow_right_blocks,
        LU_reduced_system_arrow_tip_block,
    ) = ddbtaf(
        A_reduced_system_diagonal_blocks[1:, :, :],
        A_reduced_system_lower_diagonal_blocks[1:-1, :, :],
        A_reduced_system_upper_diagonal_blocks[1:-1, :, :],
        A_reduced_system_arrow_bottom_blocks[1:, :, :],
        A_reduced_system_arrow_right_blocks[1:, :, :],
        A_reduced_system_arrow_tip_block,
    )

    (
        X_reduced_system_diagonal_blocks,
        X_reduced_system_lower_diagonal_blocks,
        X_reduced_system_upper_diagonal_blocks,
        X_reduced_system_arrow_bottom_blocks,
        X_reduced_system_arrow_right_blocks,
        X_reduced_system_arrow_tip_block,
    ) = ddbtasi(
        LU_reduced_system_diagonal_blocks,
        LU_reduced_system_lower_diagonal_blocks,
        LU_reduced_system_upper_diagonal_blocks,
        LU_reduced_system_arrow_bottom_blocks,
        LU_reduced_system_arrow_right_blocks,
        LU_reduced_system_arrow_tip_block,
    )

    # Update of the local slices by there corresponding blocks in the inverted
    # reduced system.
    if comm_rank == 0:
        LU_diagonal_blocks_local[-1, :, :] = X_reduced_system_diagonal_blocks[0, :, :]

        LU_lower_diagonal_blocks_local[-1, :, :] = (
            X_reduced_system_lower_diagonal_blocks[0, :, :]
        )
        LU_upper_diagonal_blocks_local[-1, :, :] = (
            X_reduced_system_upper_diagonal_blocks[0, :, :]
        )

        LU_arrow_bottom_blocks_local[-1, :, :] = X_reduced_system_arrow_bottom_blocks[
            0, :, :
        ]
        LU_arrow_right_blocks_local[-1, :, :] = X_reduced_system_arrow_right_blocks[
            0, :, :
        ]
    else:
        LU_diagonal_blocks_local[0, :, :] = X_reduced_system_diagonal_blocks[
            2 * comm_rank - 1, :, :
        ]
        LU_diagonal_blocks_local[-1, :, :] = X_reduced_system_diagonal_blocks[
            2 * comm_rank, :, :
        ]

        U_permutation_lower[-1, :, :] = X_reduced_system_lower_diagonal_blocks[
            2 * comm_rank - 1, :, :
        ]
        L_permutation_upper[-1, :, :] = X_reduced_system_upper_diagonal_blocks[
            2 * comm_rank - 1, :, :
        ]

        if comm_rank < comm_size - 1:
            LU_lower_diagonal_blocks_local[-1, :, :] = (
                X_reduced_system_lower_diagonal_blocks[2 * comm_rank, :, :]
            )
            LU_upper_diagonal_blocks_local[-1, :, :] = (
                X_reduced_system_upper_diagonal_blocks[2 * comm_rank, :, :]
            )

        LU_arrow_bottom_blocks_local[0, :, :] = X_reduced_system_arrow_bottom_blocks[
            2 * comm_rank - 1, :, :
        ]
        LU_arrow_bottom_blocks_local[-1, :, :] = X_reduced_system_arrow_bottom_blocks[
            2 * comm_rank, :, :
        ]

        LU_arrow_right_blocks_local[0, :, :] = X_reduced_system_arrow_right_blocks[
            2 * comm_rank - 1, :, :
        ]
        LU_arrow_right_blocks_local[-1, :, :] = X_reduced_system_arrow_right_blocks[
            2 * comm_rank, :, :
        ]

    # print(f"comm_rank:{comm_rank}, B_upper:{L_permutation_upper}")
    # print(f"comm_rank:{comm_rank}, B_lower:{U_permutation_lower}")

    return (
        LU_diagonal_blocks_local,
        LU_lower_diagonal_blocks_local,
        LU_upper_diagonal_blocks_local,
        LU_arrow_bottom_blocks_local,
        LU_arrow_right_blocks_local,
        LU_arrow_tip_block_global,
        L_permutation_upper,
        U_permutation_lower,
    )


def _host_d_pobtasi(
    LU_diagonal_blocks_local: ArrayLike,
    LU_lower_diagonal_blocks_local: ArrayLike,
    LU_upper_diagonal_blocks_local: ArrayLike,
    LU_arrow_bottom_blocks_local: ArrayLike,
    LU_arrow_right_blocks_local: ArrayLike,
    LU_arrow_tip_block_global: ArrayLike,
    L_permutation_upper: ArrayLike,
    U_permutation_lower: ArrayLike,
) -> tuple[
    ArrayLike,
    ArrayLike,
    ArrayLike,
    ArrayLike,
    ArrayLike,
    ArrayLike,
]:
    diag_blocksize = LU_diagonal_blocks_local.shape[1]
    n_diag_blocks_local = LU_diagonal_blocks_local.shape[0]

    X_diagonal_blocks_local = LU_diagonal_blocks_local
    X_lower_diagonal_blocks_local = LU_lower_diagonal_blocks_local
    X_upper_diagonal_blocks_local = LU_upper_diagonal_blocks_local
    X_arrow_bottom_blocks_local = LU_arrow_bottom_blocks_local
    X_arrow_right_blocks_local = LU_arrow_right_blocks_local
    X_arrow_tip_block_global = LU_arrow_tip_block_global

    # Backward selected-inversion
    L_inv_temp = np.empty_like(LU_diagonal_blocks_local[0])
    U_inv_temp = np.empty_like(LU_diagonal_blocks_local[0])
    LU_lower_diagonal_blocks_temp = np.empty_like(LU_lower_diagonal_blocks_local[0])
    LU_upper_diagonal_blocks_temp = np.empty_like(LU_upper_diagonal_blocks_local[0])
    LU_arrow_bottom_blocks_temp = np.empty_like(LU_arrow_bottom_blocks_local[0])
    LU_arrow_right_blocks_temp = np.empty_like(LU_arrow_right_blocks_local[0])

    if comm_rank == 0:
        for i in range(n_diag_blocks_local - 2, -1, -1):
            LU_lower_diagonal_blocks_temp[:, :] = LU_lower_diagonal_blocks_local[
                i, :, :
            ]
            LU_upper_diagonal_blocks_temp[:, :] = LU_upper_diagonal_blocks_local[
                i, :, :
            ]

            LU_arrow_bottom_blocks_temp[:, :] = LU_arrow_bottom_blocks_local[i, :, :]
            LU_arrow_right_blocks_temp[:, :] = LU_arrow_right_blocks_local[i, :, :]

            L_inv_temp = np_la.solve_triangular(
                LU_diagonal_blocks_local[i, :, :],
                np.eye(diag_blocksize),
                lower=True,
                unit_diagonal=True,
            )

            U_inv_temp = np_la.solve_triangular(
                LU_diagonal_blocks_local[i, :, :],
                np.eye(diag_blocksize),
                lower=False,
            )

            # --- Off-diagonal block part ---
            # X_{i+1, i} = (-X_{i+1, i+1} L_{i+1, i} - X_{i+1, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1}
            X_lower_diagonal_blocks_local[i, :, :] = (
                -X_diagonal_blocks_local[i + 1, :, :]
                @ LU_lower_diagonal_blocks_local[i, :, :]
                - X_arrow_right_blocks_local[i + 1, :, :]
                @ LU_arrow_bottom_blocks_local[i, :, :]
            ) @ L_inv_temp

            # X_{i, i+1} = U_{i, i}^{-1} (- U_{i, i+1} X_{i+1, i+1} - U_{i, ndb+1} X_{ndb+1, i+1})
            X_upper_diagonal_blocks_local[i, :, :] = U_inv_temp @ (
                -LU_upper_diagonal_blocks_local[i, :, :]
                @ X_diagonal_blocks_local[i + 1, :, :]
                - LU_arrow_right_blocks_local[i, :, :]
                @ X_arrow_bottom_blocks_local[i + 1, :, :]
            )

            # --- Arrowhead part ---
            # X_{ndb+1, i} = (- X_{ndb+1, i+1} L_{i+1, i} - X_{ndb+1, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1}
            X_arrow_bottom_blocks_local[i, :, :] = (
                -X_arrow_bottom_blocks_local[i + 1, :, :]
                @ LU_lower_diagonal_blocks_local[i, :, :]
                - X_arrow_tip_block_global[:, :] @ LU_arrow_bottom_blocks_local[i, :, :]
            ) @ L_inv_temp

            # X_{i, ndb+1} = U_{i, i}^{-1} (- U_{i, i+1} X_{i+1, ndb+1} - U_{i, ndb+1} X_{ndb+1, ndb+1})
            X_arrow_right_blocks_local[i, :, :] = U_inv_temp @ (
                -LU_upper_diagonal_blocks_local[i, :, :]
                @ X_arrow_right_blocks_local[i + 1, :, :]
                - LU_arrow_right_blocks_local[i, :, :] @ X_arrow_tip_block_global[:, :]
            )

            # --- Diagonal block part ---
            # X_{i, i} = (U_{i, i}^{-1} - X_{i, i+1} L_{i+1, i} - X_{i, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1}
            X_diagonal_blocks_local[i, :, :] = (
                U_inv_temp
                - X_upper_diagonal_blocks_local[i, :, :]
                @ LU_lower_diagonal_blocks_local[i, :, :]
                - X_arrow_right_blocks_local[i, :, :]
                @ LU_arrow_bottom_blocks_local[i, :, :]
            ) @ L_inv_temp
    """ else:
        LU_upper_nested_dissection_buffer_temp = np.empty_like(
            L_permutation_upper[0, :, :]
        )

        for i in range(n_diag_blocks_local - 2, 0, -1):
            LU_lower_diagonal_blocks_temp[:, :] = LU_lower_diagonal_blocks_local[
                i, :, :
            ]
            LU_arrow_bottom_blocks_temp[:, :] = LU_arrow_bottom_blocks_local[i, :, :]
            LU_upper_nested_dissection_buffer_temp[:, :] = L_permutation_upper[i, :, :]

            L_inv_temp[:, :] = np_la.solve_triangular(
                LU_diagonal_blocks_local[i, :, :],
                np.eye(diag_blocksize),
                lower=True,
            )

            # X_{i+1, i} = (- X_{top, i+1}.T L_{top, i} - X_{i+1, i+1} L_{i+1, i} - X_{ndb+1, i+1}.T L_{ndb+1, i}) L_{i, i}^{-1}
            X_lower_diagonal_blocks_local[i, :, :] = (
                -L_permutation_upper[i + 1, :, :].conj().T
                @ LU_upper_nested_dissection_buffer_temp[:, :]
                - X_diagonal_blocks_local[i + 1, :, :]
                @ LU_lower_diagonal_blocks_temp[:, :]
                - X_arrow_bottom_blocks_local[i + 1, :, :].conj().T
                @ LU_arrow_bottom_blocks_temp[:, :]
            ) @ L_inv_temp[:, :]

            # X_{top, i} = (- X_{top, i+1} L_{i+1, i} - X_{top, top} L_{top, i} - X_{ndb+1, top}.T L_{ndb+1, i}) L_{i, i}^{-1}
            L_permutation_upper[i, :, :] = (
                -L_permutation_upper[i + 1, :, :] @ LU_lower_diagonal_blocks_temp[:, :]
                - X_diagonal_blocks_local[0, :, :]
                @ LU_upper_nested_dissection_buffer_temp[:, :]
                - X_arrow_bottom_blocks_local[0, :, :].conj().T
                @ LU_arrow_bottom_blocks_temp[:, :]
            ) @ L_inv_temp[:, :]

            # Arrowhead
            # X_{ndb+1, i} = (- X_{ndb+1, i+1} L_{i+1, i} - X_{ndb+1, top} L_{top, i} - X_{ndb+1, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1}
            X_arrow_bottom_blocks_local[i, :, :] = (
                -X_arrow_bottom_blocks_local[i + 1, :, :]
                @ LU_lower_diagonal_blocks_temp[:, :]
                - X_arrow_bottom_blocks_local[0, :, :]
                @ LU_upper_nested_dissection_buffer_temp[:, :]
                - X_arrow_tip_block_global[:, :] @ LU_arrow_bottom_blocks_temp[:, :]
            ) @ L_inv_temp[:, :]

            # X_{i, i} = (L_{i, i}^{-T} - X_{i+1, i}.T L_{i+1, i} - X_{top, i}.T L_{top, i} - X_{ndb+1, i}.T L_{ndb+1, i}) L_{i, i}^{-1}
            X_diagonal_blocks_local[i, :, :] = (
                L_inv_temp[:, :].conj().T
                - X_lower_diagonal_blocks_local[i, :, :].conj().T
                @ LU_lower_diagonal_blocks_temp[:, :]
                - L_permutation_upper[i, :, :].conj().T
                @ LU_upper_nested_dissection_buffer_temp[:, :]
                - X_arrow_bottom_blocks_local[i, :, :].conj().T
                @ LU_arrow_bottom_blocks_temp[:, :]
            ) @ L_inv_temp[:, :]

        # Copy back the 2 first blocks that have been produced in the 2-sided pattern
        # to the tridiagonal storage.
        X_lower_diagonal_blocks_local[0, :, :] = L_permutation_upper[1, :, :].conj().T
    """
    return (
        X_diagonal_blocks_local,
        X_lower_diagonal_blocks_local,
        X_upper_diagonal_blocks_local,
        X_arrow_bottom_blocks_local,
        X_arrow_right_blocks_local,
        X_arrow_tip_block_global,
    )
