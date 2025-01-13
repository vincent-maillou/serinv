# Copyright 2023-2025 ETH Zurich. All rights reserved.

from mpi4py import MPI

from serinv import (
    ArrayLike,
    CUPY_AVAIL,
    _get_module_from_array,
)

if CUPY_AVAIL:
    import cupyx as cpx

from serinv.algs import pobtaf

comm_rank = MPI.COMM_WORLD.Get_rank()
comm_size = MPI.COMM_WORLD.Get_size()


def ppobtaf(
    A_diagonal_blocks: ArrayLike,
    A_lower_diagonal_blocks: ArrayLike,
    A_arrow_bottom_blocks: ArrayLike,
    A_arrow_tip_block: ArrayLike,
    **kwargs,
) -> ArrayLike:
    """Perform the parallel factorization of a block tridiagonal with arrowhead matrix
    (pointing downward by convention).

    Parameters
    ----------
    A_diagonal_blocks : ArrayLike
        The diagonal blocks of the block tridiagonal with arrowhead matrix.
    A_lower_diagonal_blocks : ArrayLike
        The lower diagonal blocks of the block tridiagonal with arrowhead matrix.
    A_arrow_bottom_blocks : ArrayLike
        The arrow bottom blocks of the block tridiagonal with arrowhead matrix.
    A_arrow_tip_block : ArrayLike
        The arrow tip block of the block tridiagonal with arrowhead matrix.

    Keyword Arguments
    -----------------
    device_streaming : bool, optional
        If True, the algorithm will perform host-device streaming. (default: False)

    Note:
    -----
    - The matrix, A, is assumed to be symmetric positive definite.
    - Will overwrite the inputs and store the results in them (in-place).
    - If a device array is given, the algorithm will run on the GPU.
    """
    if comm_size == 1:
        raise ValueError("The number of MPI processes must be greater than 1.")

    xp, _ = _get_module_from_array(arr=A_diagonal_blocks)

    device_streaming: bool = kwargs.get("device_streaming", False)

    A_tip_update = xp.zeros_like(A_arrow_tip_block)

    buffer = None

    if comm_rank == 0:
        pobtaf(
            A_diagonal_blocks,
            A_lower_diagonal_blocks,
            A_arrow_bottom_blocks,
            A_tip_update,
            device_streaming=device_streaming,
            factorize_last_block=False,
        )
    else:
        if device_streaming:
            buffer = cpx.empty_like_pinned(A_diagonal_blocks)
        else:
            buffer = xp.empty_like(A_diagonal_blocks)

        pobtaf(
            A_diagonal_blocks,
            A_lower_diagonal_blocks,
            A_arrow_bottom_blocks,
            A_tip_update,
            device_streaming=device_streaming,
            buffer=buffer,
        )

    # Assemble the reduced system
    _n = 2 * comm_size - 1

    if device_streaming:
        zeros = cpx.zeros_pinned
    else:
        zeros = xp.zeros

    _L_diagonal_blocks = zeros(
        (_n, A_diagonal_blocks[0].shape[0], A_diagonal_blocks[0].shape[1]),
        dtype=A_diagonal_blocks.dtype,
    )
    _L_lower_diagonal_blocks = zeros(
        (
            _n - 1,
            A_lower_diagonal_blocks[0].shape[0],
            A_lower_diagonal_blocks[0].shape[1],
        ),
        dtype=A_lower_diagonal_blocks.dtype,
    )
    _L_lower_arrow_blocks = zeros(
        (_n, A_arrow_bottom_blocks[0].shape[0], A_arrow_bottom_blocks[0].shape[1]),
        dtype=A_arrow_bottom_blocks.dtype,
    )

    if comm_rank == 0:
        _L_diagonal_blocks[0] = A_diagonal_blocks[-1]
        _L_lower_diagonal_blocks[0] = A_lower_diagonal_blocks[-1]
        _L_lower_arrow_blocks[0] = A_arrow_bottom_blocks[-1]
    else:
        _L_diagonal_blocks[2 * comm_rank - 1] = A_diagonal_blocks[0]
        _L_diagonal_blocks[2 * comm_rank] = A_diagonal_blocks[-1]

        _L_lower_diagonal_blocks[2 * comm_rank - 1] = buffer[-1].conj().T

        _L_lower_arrow_blocks[2 * comm_rank - 1] = A_arrow_bottom_blocks[0]
        _L_lower_arrow_blocks[2 * comm_rank] = A_arrow_bottom_blocks[-1]

    # Can be done with AllGather (need resize of buffer, assuming P0 get 2 blocks)
    MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, _L_diagonal_blocks, op=MPI.SUM)
    MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, _L_lower_diagonal_blocks, op=MPI.SUM)
    MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, _L_lower_arrow_blocks, op=MPI.SUM)
    MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, A_tip_update, op=MPI.SUM)

    MPI.COMM_WORLD.Barrier()

    A_arrow_tip_block[:, :] = A_arrow_tip_block[:, :] + A_tip_update[:, :]

    # --- Factorize the reduced system ---
    pobtaf(
        _L_diagonal_blocks,
        _L_lower_diagonal_blocks,
        _L_lower_arrow_blocks,
        A_arrow_tip_block,
        device_streaming=device_streaming,
    )

    return (
        _L_diagonal_blocks,
        _L_lower_diagonal_blocks,
        _L_lower_arrow_blocks,
        buffer,
    )
