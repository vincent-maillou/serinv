# Copyright 2023-2025 ETH Zurich. All rights reserved.

from mpi4py import MPI

from serinv import (
    ArrayLike,
    _get_module_from_array,
)

from serinv.algs import pobtasi
from serinv.wrappers.pobtars import (
    map_pobtars_to_ppobtax,
    scatter_pobtars,
    allocate_pinned_pobtars,
)

comm_rank = MPI.COMM_WORLD.Get_rank()
comm_size = MPI.COMM_WORLD.Get_size()


def ppobtasi(
    L_diagonal_blocks: ArrayLike,
    L_lower_diagonal_blocks: ArrayLike,
    L_arrow_bottom_blocks: ArrayLike,
    L_arrow_tip_block: ArrayLike,
    _L_diagonal_blocks: ArrayLike,
    _L_lower_diagonal_blocks: ArrayLike,
    _L_lower_arrow_blocks: ArrayLike,
    L_permutation_buffer: ArrayLike,
    **kwargs,
):
    """Perform a selected inversion of a block tridiagonal with arrowhead matrix (pointing downward by convention).

    Parameters
    ----------
    L_diagonal_blocks : ArrayLike
        Diagonal blocks of the Cholesky factor of the matrix.
    L_lower_diagonal_blocks : ArrayLike
        Lower diagonal blocks of the Cholesky factor of the matrix.
    L_arrow_bottom_blocks : ArrayLike
        Arrow bottom blocks of the Cholesky factor of the matrix.
    L_arrow_tip_block : ArrayLike
        Arrow tip block of the Cholesky factor of the matrix.
    _L_diagonal_blocks : ArrayLike
        Diagonal blocks of the reduced system.
    _L_lower_diagonal_blocks : ArrayLike
        Lower diagonal blocks of the reduced system.
    _L_lower_arrow_blocks : ArrayLike
        Arrow bottom blocks of the reduced system.
    L_permutation_buffer : ArrayLike
        Buffer array for the permuted arrowhead.

    Keyword Arguments
    -----------------
    device_streaming : bool, optional
        If True, the algorithm will run on the GPU. (default: False)
    strategy : str, optional
        The communication strategy to use. (default: "allgather")
    root : int, optional
        The root rank for the communication strategy. (default: 0)

    Note:
    -----
    - Will overwrite the inputs and store the results in them (in-place).
    - If a device array is given, the algorithm will run on the GPU.

    Currently implemented:
    ----------------------
    |              | Natural | Permuted |
    | ------------ | ------- | -------- |
    | Direct-array | x       | x        |
    | Streaming    | x       | x        |
    """
    if comm_size == 1:
        raise ValueError("The number of MPI processes must be greater than 1.")

    xp, _ = _get_module_from_array(arr=L_diagonal_blocks)

    # Check for optional parameters
    device_streaming: bool = kwargs.get("device_streaming", False)
    strategy: str = kwargs.get("strategy", "allgather")
    root: int = kwargs.get("root", 0)

    # Selected-inversion of the reduced system
    if strategy == "gather-scatter":
        if comm_rank == root:
            pobtasi(
                _L_diagonal_blocks[1:],
                _L_lower_diagonal_blocks[1:-1],
                _L_lower_arrow_blocks[1:],
                L_arrow_tip_block,
                device_streaming=device_streaming,
            )

        MPI.COMM_WORLD.Barrier()

        if xp.__name__ == "cupy":
            # Check for given pinned memory buffers
            _L_diagonal_blocks_h: ArrayLike = kwargs.get("_L_diagonal_blocks_h", None)
            _L_lower_diagonal_blocks_h: ArrayLike = kwargs.get(
                "_L_lower_diagonal_blocks_h", None
            )
            _L_lower_arrow_blocks_h: ArrayLike = kwargs.get(
                "_L_lower_arrow_blocks_h", None
            )
            _L_tip_update_h: ArrayLike = kwargs.get("_L_tip_update_h", None)

            if any(
                buffers is None
                for buffers in [
                    _L_diagonal_blocks_h,
                    _L_lower_diagonal_blocks_h,
                    _L_lower_arrow_blocks_h,
                    _L_tip_update_h,
                ]
            ):
                (
                    _L_diagonal_blocks_h,
                    _L_lower_diagonal_blocks_h,
                    _L_lower_arrow_blocks_h,
                    _L_tip_update_h,
                ) = allocate_pinned_pobtars(
                    _L_diagonal_blocks,
                    _L_lower_diagonal_blocks,
                    _L_lower_arrow_blocks,
                    L_arrow_tip_block,
                )

            if comm_rank == root:
                _L_diagonal_blocks.get(out=_L_diagonal_blocks_h)
                _L_lower_diagonal_blocks.get(out=_L_lower_diagonal_blocks_h)
                _L_lower_arrow_blocks.get(out=_L_lower_arrow_blocks_h)
                L_arrow_tip_block.get(out=_L_tip_update_h)

            scatter_pobtars(
                _L_diagonal_blocks=_L_diagonal_blocks_h,
                _L_lower_diagonal_blocks=_L_lower_diagonal_blocks_h,
                _L_lower_arrow_blocks=_L_lower_arrow_blocks_h,
                L_arrow_tip_block=_L_tip_update_h,
                strategy=strategy,
                root=root,
            )

            _L_diagonal_blocks.set(arr=_L_diagonal_blocks_h)
            _L_lower_diagonal_blocks.set(arr=_L_lower_diagonal_blocks_h)
            _L_lower_arrow_blocks.set(arr=_L_lower_arrow_blocks_h)
            L_arrow_tip_block.set(arr=_L_tip_update_h)
        else:
            scatter_pobtars(
                _L_diagonal_blocks,
                _L_lower_diagonal_blocks,
                _L_lower_arrow_blocks,
                L_arrow_tip_block,
                strategy=strategy,
                root=root,
            )
    else:
        pobtasi(
            _L_diagonal_blocks,
            _L_lower_diagonal_blocks,
            _L_lower_arrow_blocks,
            L_arrow_tip_block,
            device_streaming=device_streaming,
        )

    # Map result of the reduced system back to the original system
    map_pobtars_to_ppobtax(
        L_diagonal_blocks=L_diagonal_blocks,
        L_lower_diagonal_blocks=L_lower_diagonal_blocks,
        L_arrow_bottom_blocks=L_arrow_bottom_blocks,
        L_arrow_tip_block=L_arrow_tip_block,
        L_permutation_buffer=L_permutation_buffer,
        _L_diagonal_blocks=_L_diagonal_blocks,
        _L_lower_diagonal_blocks=_L_lower_diagonal_blocks,
        _L_lower_arrow_blocks=_L_lower_arrow_blocks,
        _L_tip_update=L_arrow_tip_block,
        strategy=strategy,
    )

    # Parallel selected inversion of the original system
    if comm_rank == 0:
        pobtasi(
            L_diagonal_blocks,
            L_lower_diagonal_blocks,
            L_arrow_bottom_blocks,
            L_arrow_tip_block,
            device_streaming=device_streaming,
            inverse_last_block=False,
        )
    else:
        pobtasi(
            L_diagonal_blocks,
            L_lower_diagonal_blocks,
            L_arrow_bottom_blocks,
            L_arrow_tip_block,
            device_streaming=device_streaming,
            buffer=L_permutation_buffer,
        )
