# Copyright 2023-2024 ETH Zurich. All rights reserved.

import numpy as np
import pytest
import time

from mpi4py import MPI
from numpy.typing import ArrayLike
from serinv import SolverConfig
from serinv.algs import d_pobtaf, d_pobtasi, d_pobtasi_rss


SEED = 63

try:
    import cupy as cp
    import cupyx as cpx

    CUPY_AVAIL = True
    cp.random.seed(cp.uint64(SEED))

except ImportError:
    CUPY_AVAIL = False


np.random.seed(SEED)

NCCL_AVAIL = False
if CUPY_AVAIL:
    try:
        from cupy.cuda import nccl
        nccl.get_version()  # Check if NCCL is available

        NCCL_AVAIL = True
    except (AttributeError, ImportError, ModuleNotFoundError):
        pass

from os import environ

environ["OMP_NUM_THREADS"] = "1"


comm_rank = MPI.COMM_WORLD.Get_rank()
comm_size = MPI.COMM_WORLD.Get_size()

# cuda_aware = CUPY_AVAIL and device_array and not device_streaming
cuda_aware = False
use_nccl = True
nested_solving = True
if NCCL_AVAIL:
    from cupy.cuda import nccl
    comm_id = nccl.get_unique_id()
    comm_id = MPI.COMM_WORLD.bcast(comm_id, root=0)
    start = time.perf_counter_ns()
    nccl_comm = nccl.NcclCommunicator(comm_size, comm_id, comm_rank)
    cp.cuda.runtime.deviceSynchronize()
    finish = time.perf_counter_ns()
    print(f"Rank {comm_rank} took {finish - start} ns to create nccl_comm")
    if nested_solving:
        reduced_size = comm_size // 2
        reduced_rank = comm_rank
        reduced_color = int(comm_rank < reduced_size)
        if reduced_rank >= reduced_size:
            reduced_rank -= reduced_size
        reduced_key = reduced_rank
        reduced_comm = MPI.COMM_WORLD.Split(reduced_color, reduced_key)
        reduced_comm_id = nccl.get_unique_id()
        # reduced_comm_id = MPI.COMM_WORLD.bcast(reduced_comm_id, root=0)
        reduced_comm_id = reduced_comm.bcast(reduced_comm_id, root=0)
        start = time.perf_counter_ns()
        # if reduced_color == 1:
        #     nccl_reduced_comm = nccl.NcclCommunicator(reduced_size, reduced_comm_id, reduced_rank)
        # else:
        #     nccl_reduced_comm = None
        nccl_reduced_comm = nccl.NcclCommunicator(reduced_size, reduced_comm_id, reduced_rank)
        cp.cuda.runtime.deviceSynchronize()
        finish = time.perf_counter_ns()
        print(f"Rank {comm_rank} took {finish - start} ns to create nccl_reduced_comm")
    else:
        nccl_reduced_comm = None
else:
    use_nccl = False


def dd_bta(
    diagonal_blocksize: int,
    arrowhead_blocksize: int,
    n_diag_blocks: int,
    device_array: bool,
    dtype: np.dtype,
):
    """Returns a random, diagonaly dominant general, block tridiagonal arrowhead matrix."""
    xp = cp if device_array and CUPY_AVAIL else np

    DD_BTA = xp.zeros(
        (
            diagonal_blocksize * n_diag_blocks + arrowhead_blocksize,
            diagonal_blocksize * n_diag_blocks + arrowhead_blocksize,
        ),
        dtype=dtype,
    )

    rc = (1.0 + 1.0j) if dtype == np.complex128 else 1.0

    # Fill the lower arrowhead blocks
    DD_BTA[-arrowhead_blocksize:, :-arrowhead_blocksize] = rc * xp.random.rand(
        arrowhead_blocksize, n_diag_blocks * diagonal_blocksize
    )
    # Fill the right arrowhead blocks
    DD_BTA[:-arrowhead_blocksize, -arrowhead_blocksize:] = rc * xp.random.rand(
        n_diag_blocks * diagonal_blocksize, arrowhead_blocksize
    )

    # Fill the tip of the arrowhead
    DD_BTA[-arrowhead_blocksize:, -arrowhead_blocksize:] = rc * xp.random.rand(
        arrowhead_blocksize, arrowhead_blocksize
    )

    # Fill the diagonal blocks
    for i in range(n_diag_blocks):
        DD_BTA[
            i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
            i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
        ] = rc * xp.random.rand(diagonal_blocksize, diagonal_blocksize) + rc * xp.eye(
            diagonal_blocksize
        )

        # Fill the off-diagonal blocks
        if i > 0:
            DD_BTA[
                i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
                (i - 1) * diagonal_blocksize : i * diagonal_blocksize,
            ] = rc * xp.random.rand(diagonal_blocksize, diagonal_blocksize)

        if i < n_diag_blocks - 1:
            DD_BTA[
                i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
                (i + 1) * diagonal_blocksize : (i + 2) * diagonal_blocksize,
            ] = rc * xp.random.rand(diagonal_blocksize, diagonal_blocksize)

    # Make the matrix diagonally dominant
    for i in range(DD_BTA.shape[0]):
        DD_BTA[i, i] = 1 + xp.sum(DD_BTA[i, :])

    return DD_BTA

def bta_symmetrize(bta: ArrayLike):
    """Symmetrizes a block tridiagonal arrowhead matrix."""

    return (bta + bta.conj().T) / 2


def bta_dense_to_arrays(
    bta: ArrayLike,
    diagonal_blocksize: int,
    arrowhead_blocksize: int,
    n_diag_blocks: int,
):
    """Converts a block tridiagonal arrowhead matrix from a dense representation to arrays of blocks."""
    if CUPY_AVAIL:
        xp = cp.get_array_module(bta)
    else:
        xp = np

    A_diagonal_blocks = xp.zeros(
        (n_diag_blocks, diagonal_blocksize, diagonal_blocksize),
        dtype=bta.dtype,
    )

    A_lower_diagonal_blocks = xp.zeros(
        (n_diag_blocks - 1, diagonal_blocksize, diagonal_blocksize),
        dtype=bta.dtype,
    )
    A_upper_diagonal_blocks = xp.zeros(
        (n_diag_blocks - 1, diagonal_blocksize, diagonal_blocksize),
        dtype=bta.dtype,
    )

    A_arrow_bottom_blocks = xp.zeros(
        (n_diag_blocks, arrowhead_blocksize, diagonal_blocksize),
        dtype=bta.dtype,
    )

    A_arrow_right_blocks = xp.zeros(
        (n_diag_blocks, diagonal_blocksize, arrowhead_blocksize),
        dtype=bta.dtype,
    )

    for i in range(n_diag_blocks):
        A_diagonal_blocks[i, :, :] = bta[
            i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
            i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
        ]
        if i > 0:
            A_lower_diagonal_blocks[i - 1, :, :] = bta[
                i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
                (i - 1) * diagonal_blocksize : i * diagonal_blocksize,
            ]
        if i < n_diag_blocks - 1:
            A_upper_diagonal_blocks[i, :, :] = bta[
                i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
                (i + 1) * diagonal_blocksize : (i + 2) * diagonal_blocksize,
            ]

        A_arrow_bottom_blocks[i, :, :] = bta[
            -arrowhead_blocksize:,
            i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
        ]

        A_arrow_right_blocks[i, :, :] = bta[
            i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
            -arrowhead_blocksize:,
        ]

    A_arrow_tip_block = bta[-arrowhead_blocksize:, -arrowhead_blocksize:]

    return (
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_upper_diagonal_blocks,
        A_arrow_bottom_blocks,
        A_arrow_right_blocks,
        A_arrow_tip_block,
    )


# @pytest.mark.mpi(min_size=2)
# @pytest.mark.parametrize("diagonal_blocksize", [2, 3])
# @pytest.mark.parametrize("arrowhead_blocksize", [2, 3])
# @pytest.mark.parametrize("n_diag_blocks", [comm_size * 3, comm_size * 4, comm_size * 5])
# @pytest.mark.parametrize("device_array", [False, True], ids=["host", "device"])
# @pytest.mark.parametrize("dtype", [np.float64, np.complex128])
# @pytest.mark.parametrize("device_streaming", [False, True])
def test_d_pobtasi(
    # dd_bta,
    # bta_dense_to_arrays,
    # bta_symmetrize,
    diagonal_blocksize,
    arrowhead_blocksize,
    n_diag_blocks,
    device_array,
    dtype,
    device_streaming,
):
    data = dd_bta(diagonal_blocksize, arrowhead_blocksize, n_diag_blocks, device_array, dtype)
    if CUPY_AVAIL:
        xp = cp.get_array_module(data)
    else:
        xp = np

    # Input matrix
    A = bta_symmetrize(data)

    (
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        _,
        A_arrow_bottom_blocks,
        _,
        A_arrow_tip_block_global,
    ) = bta_dense_to_arrays(A, diagonal_blocksize, arrowhead_blocksize, n_diag_blocks)

    # Save the local slice of the array for each MPI process
    n_diag_blocks_per_processes = n_diag_blocks // comm_size
    A_diagonal_blocks_local = A_diagonal_blocks[
        comm_rank
        * n_diag_blocks_per_processes : (comm_rank + 1)
        * n_diag_blocks_per_processes,
        :,
        :,
    ]

    if comm_rank == comm_size - 1:
        A_lower_diagonal_blocks_local = A_lower_diagonal_blocks[
            comm_rank * n_diag_blocks_per_processes : n_diag_blocks - 1,
            :,
            :,
        ]
    else:
        A_lower_diagonal_blocks_local = A_lower_diagonal_blocks[
            comm_rank
            * n_diag_blocks_per_processes : (comm_rank + 1)
            * n_diag_blocks_per_processes,
            :,
            :,
        ]

    A_arrow_bottom_blocks_local = A_arrow_bottom_blocks[
        comm_rank
        * n_diag_blocks_per_processes : (comm_rank + 1)
        * n_diag_blocks_per_processes,
        :,
        :,
    ]

    if CUPY_AVAIL and device_streaming and not device_array:
        A_diagonal_blocks_local_pinned = cpx.zeros_like_pinned(A_diagonal_blocks_local)
        A_diagonal_blocks_local_pinned[:, :, :] = A_diagonal_blocks_local[:, :, :]
        A_lower_diagonal_blocks_local_pinned = cpx.zeros_like_pinned(
            A_lower_diagonal_blocks_local
        )
        A_lower_diagonal_blocks_local_pinned[:, :, :] = A_lower_diagonal_blocks_local[
            :, :, :
        ]
        A_arrow_bottom_blocks_local_pinned = cpx.zeros_like_pinned(
            A_arrow_bottom_blocks_local
        )
        A_arrow_bottom_blocks_local_pinned[:, :, :] = A_arrow_bottom_blocks_local[
            :, :, :
        ]
        A_arrow_tip_block_global_pinned = cpx.zeros_like_pinned(
            A_arrow_tip_block_global
        )
        A_arrow_tip_block_global_pinned[:, :] = A_arrow_tip_block_global[:, :]

        A_diagonal_blocks_local = A_diagonal_blocks_local_pinned
        A_lower_diagonal_blocks_local = A_lower_diagonal_blocks_local_pinned
        A_arrow_bottom_blocks_local = A_arrow_bottom_blocks_local_pinned
        A_arrow_tip_block_global = A_arrow_tip_block_global_pinned

    # Reference solution
    X_ref = xp.linalg.inv(A)

    (
        X_ref_diagonal_blocks,
        X_ref_lower_diagonal_blocks,
        _,
        X_ref_arrow_bottom_blocks,
        _,
        X_ref_arrow_tip_block_global,
    ) = bta_dense_to_arrays(
        X_ref, diagonal_blocksize, arrowhead_blocksize, n_diag_blocks
    )

    X_ref_diagonal_blocks_local = X_ref_diagonal_blocks[
        comm_rank
        * n_diag_blocks_per_processes : (comm_rank + 1)
        * n_diag_blocks_per_processes,
        :,
        :,
    ]

    if comm_rank == comm_size - 1:
        X_ref_lower_diagonal_blocks_local = X_ref_lower_diagonal_blocks[
            comm_rank * n_diag_blocks_per_processes : n_diag_blocks - 1,
            :,
            :,
        ]
    else:
        X_ref_lower_diagonal_blocks_local = X_ref_lower_diagonal_blocks[
            comm_rank
            * n_diag_blocks_per_processes : (comm_rank + 1)
            * n_diag_blocks_per_processes,
            :,
            :,
        ]

    X_ref_arrow_bottom_blocks_local = X_ref_arrow_bottom_blocks[
        comm_rank
        * n_diag_blocks_per_processes : (comm_rank + 1)
        * n_diag_blocks_per_processes,
        :,
        :,
    ]

    # SerinV solver configuration
    if device_array and use_nccl:
        comm = nccl_comm
        reduced_comm = nccl_reduced_comm
    else:
        comm = MPI.COMM_WORLD
        reduced_comm = None
    solver_config = SolverConfig(device_streaming=device_streaming, cuda_aware_mpi=cuda_aware, nccl=use_nccl, nested_solving=nested_solving)

    # Distributed factorization
    (
        L_diagonal_blocks_local,
        L_lower_diagonal_blocks_local,
        L_arrow_bottom_blocks_local,
        L_arrow_tip_block_global,
        L_upper_nested_dissection_buffer_local,
    ) = d_pobtaf(
        A_diagonal_blocks_local,
        A_lower_diagonal_blocks_local,
        A_arrow_bottom_blocks_local,
        A_arrow_tip_block_global,
        solver_config,
        comm,
    )

    # Distributed selected-inversion
    # Inversion of the reduced system
    (
        L_diagonal_blocks_local,
        L_lower_diagonal_blocks_local,
        L_arrow_bottom_blocks_local,
        L_arrow_tip_block_global,
        L_upper_nested_dissection_buffer_local,
    ) = d_pobtasi_rss(
        L_diagonal_blocks_local,
        L_lower_diagonal_blocks_local,
        L_arrow_bottom_blocks_local,
        L_arrow_tip_block_global,
        L_upper_nested_dissection_buffer_local,
        solver_config,
        comm,
        reduced_comm,
    )

    # Inversion of the full system
    (
        X_diagonal_blocks_local,
        X_lower_diagonal_blocks_local,
        X_arrow_bottom_blocks_local,
        X_arrow_tip_block_global,
    ) = d_pobtasi(
        L_diagonal_blocks_local,
        L_lower_diagonal_blocks_local,
        L_arrow_bottom_blocks_local,
        L_arrow_tip_block_global,
        L_upper_nested_dissection_buffer_local,
        solver_config,
        comm,
    )

    assert xp.allclose(X_diagonal_blocks_local, X_ref_diagonal_blocks_local)
    assert xp.allclose(
        X_lower_diagonal_blocks_local,
        X_ref_lower_diagonal_blocks_local,
    )
    assert xp.allclose(
        X_arrow_bottom_blocks_local,
        X_ref_arrow_bottom_blocks_local,
    )
    assert xp.allclose(X_arrow_tip_block_global, X_ref_arrow_tip_block_global)

    if device_array:
        assert X_diagonal_blocks_local.data == A_diagonal_blocks_local.data
        assert X_lower_diagonal_blocks_local.data == A_lower_diagonal_blocks_local.data
        assert X_arrow_bottom_blocks_local.data == A_arrow_bottom_blocks_local.data
        assert X_arrow_tip_block_global.data == A_arrow_tip_block_global.data
    else:
        assert (
            X_diagonal_blocks_local.ctypes.data == A_diagonal_blocks_local.ctypes.data
        )
        assert (
            X_lower_diagonal_blocks_local.ctypes.data
            == A_lower_diagonal_blocks_local.ctypes.data
        )
        assert (
            X_arrow_bottom_blocks_local.ctypes.data
            == A_arrow_bottom_blocks_local.ctypes.data
        )
        assert (
            X_arrow_tip_block_global.ctypes.data == A_arrow_tip_block_global.ctypes.data
        )


if __name__ == "__main__":

    # @pytest.mark.mpi(min_size=2)
    # @pytest.mark.parametrize("diagonal_blocksize", [2, 3])
    # @pytest.mark.parametrize("arrowhead_blocksize", [2, 3])
    # @pytest.mark.parametrize("n_diag_blocks", [comm_size * 3, comm_size * 4, comm_size * 5])
    # @pytest.mark.parametrize("device_array", [False, True], ids=["host", "device"])
    # @pytest.mark.parametrize("dtype", [np.float64, np.complex128])
    # @pytest.mark.parametrize("device_streaming", [False, True])

    for diagonal_blocksize in (2, 3):
        for arrowhead_blocksize in (2, 3):
            for n_diag_blocks in (comm_size * 3, comm_size * 4, comm_size * 5):
                for device_array in (False, True):
                    for dtype in (np.float64, np.complex128):
                        for device_streaming in (False, True):
                            print(f"diagonal_blocksize: {diagonal_blocksize}, arrowhead_blocksize: {arrowhead_blocksize}, n_diag_blocks: {n_diag_blocks}, device_array: {device_array}, dtype: {dtype}, device_streaming: {device_streaming}")
                            test_d_pobtasi(diagonal_blocksize, arrowhead_blocksize, n_diag_blocks, device_array, dtype, device_streaming)
