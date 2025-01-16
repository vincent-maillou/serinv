""" 
Run the sequential Serinv codes on CPU using the inlamat matrices.
"""

import time

tic = time.perf_counter()

import cupy as cp
import numpy as np
import scipy.stats
import argparse

from mpi4py import MPI

comm_rank = MPI.COMM_WORLD.Get_rank()
comm_size = MPI.COMM_WORLD.Get_size()

try:
    from cupy.cuda import nccl
    comm_id = nccl.get_unique_id()
    print(f"Process {comm_rank}: comm_id {comm_id}", flush=True)
    comm_id = MPI.COMM_WORLD.bcast(comm_id, root=0)
    print(f"Process {comm_rank}: comm_id {comm_id}", flush=True)
    comm = nccl.NcclCommunicator(comm_size, comm_id, comm_rank)
    cp.cuda.runtime.deviceSynchronize()
    comm_rank = comm.rank_id()
    comm_size = comm.size()
    # if solver_config.nested_solving:
    #     reduced_size = comm_size // 2
    #     reduced_rank = comm_rank
    #     reduced_color = int(comm_rank < reduced_size)
    #     reduced_key = comm_rank
    #     reduced_comm_id = nccl.get_unique_id()
    #     reduced_comm_id = MPI.COMM_WORLD.bcast(reduced_comm_id, root=0)
    #     if reduced_color == 1:
    #         reduced_comm = nccl.NcclCommunicator(reduced_size, reduced_comm_id, reduced_rank)
    #     else:
    #         reduced_comm = None
    #     cp.cuda.runtime.deviceSynchronize()
except Exception as e:
    print(f"Error: {e}", flush=True)
    comm = MPI.COMM_WORLD

from serinv.wrappers import ppobtaf, ppobtasi
from serinv.wrappers import (
    ppobtaf,
    ppobtasi,
    allocate_permutation_buffer,
    allocate_ppobtars,
)

from matutils import (
    csc_to_dense_bta,
    read_sym_CSC,
)


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return m, m - h, m + h


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--diagonal_blocksize",
        type=int,
        default=92,
        help="an integer for the diagonal block size",
    )
    parser.add_argument(
        "--arrowhead_blocksize",
        type=int,
        default=4,
        help="an integer for the arrowhead block size",
    )
    parser.add_argument(
        "--n_diag_blocks",
        type=int,
        default=50,
        help="an integer for the number of diagonal blocks",
    )
    parser.add_argument(
        "--density",
        type=float,
        default=0,
        help="an integer for the number of diagonal blocks",
    )
    parser.add_argument(
        "--file_path",
        type=str,
        default="/home/x_gaedkelb/serinv/dev/matrices/",
        help="a string for the file path",
    )
    parser.add_argument(
        "--n_iterations",
        type=int,
        default=1,
        help="number of iterations for the benchmarking",
    )
    parser.add_argument(
        "--n_warmups",
        type=int,
        default=1,
        help="number of warm-ups iterations",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["allgather", "gather-scatter"],
        default="allgather",
        help="communication strategy",
    )

    args = parser.parse_args()
    toc = time.perf_counter()
    print(f"Import and parsing took: {toc - tic:.5f} sec", flush=True)

    diagonal_blocksize = args.diagonal_blocksize
    arrowhead_blocksize = args.arrowhead_blocksize
    n_diag_blocks = args.n_diag_blocks
    density = args.density
    file_path = args.file_path
    n_iterations = args.n_iterations
    n_warmups = args.n_warmups
    comm_strategy = args.strategy

    n = diagonal_blocksize * n_diag_blocks + arrowhead_blocksize

    # read in file
    if density == 0:
        file = (
            "Qxy_ns"
            + str(diagonal_blocksize)
            + "_nt"
            + str(n_diag_blocks)
            + "_nss0_nb"
            + str(arrowhead_blocksize)
            + "_n"
            + str(n)
            + ".dat"
        )
    else:
        file = (
            "Qxy_ns"
            + str(diagonal_blocksize)
            + "_nt"
            + str(n_diag_blocks)
            + "_nss0_nb"
            + str(arrowhead_blocksize)
            + "_n"
            + str(n)
            + "_density"
            + str(density)
            + ".mtx"
        )

    filename = file_path + file

    tic = time.perf_counter()
    A = read_sym_CSC(filename)
    (
        A_diagonal_blocks_init,
        A_lower_diagonal_blocks_init,
        A_arrow_bottom_blocks_init,
        A_arrow_tip_block_init,
    ) = csc_to_dense_bta(A, diagonal_blocksize, arrowhead_blocksize, n_diag_blocks)
    toc = time.perf_counter()

    print(f"Reading matrix took: {toc - tic:.5f} sec", flush=True)
    print(
        f"    A_diagonal_blocks_init.shape: {A_diagonal_blocks_init.shape}", flush=True
    )
    print(
        f"    A_lower_diagonal_blocks_init.shape: {A_lower_diagonal_blocks_init.shape}",
        flush=True,
    )
    print(
        f"    A_arrow_bottom_blocks_init.shape: {A_arrow_bottom_blocks_init.shape}",
        flush=True,
    )
    print(
        f"    A_arrow_tip_block_init.shape: {A_arrow_tip_block_init.shape}", flush=True
    )

    # Compute local n_blocks size
    factor = 1.8
    alpha = n_diag_blocks / (factor + comm_size - 1)
    top_partition_blocks = int(alpha * factor)
    n_diag_blocks_per_processes = (n_diag_blocks - top_partition_blocks) // (comm_size - 1)
    n_locals = [top_partition_blocks] + [n_diag_blocks_per_processes for i in range(comm_size-1)]
    remainder = (n_diag_blocks - top_partition_blocks) % (comm_size - 1)
    for i in range(remainder):
        n_locals[i+1] += 1
    # n_diag_blocks_per_processes = n_diag_blocks // comm_size
    # n_locals = [n_diag_blocks_per_processes for i in range(comm_size)]
    # remainder = n_diag_blocks % comm_size
    # for i in range(remainder):
    #     n_locals[i] += 1
    starting_idx = [sum(n_locals[0:i]) for i in range(comm_size+1)]

    print(f"factor: {factor}", flush=True)
    print(f"n_locals: {n_locals}", flush=True)
    print(f"Starting idx: {starting_idx}", flush=True)

    # tic = time.perf_counter()
    # A_diagonal_blocks_local = cp.zeros((n_locals[comm_rank], diagonal_blocksize, diagonal_blocksize), dtype=A_diagonal_blocks_init.dtype)
    # if comm_rank != comm_size - 1:
    #     A_lower_diagonal_blocks_local = cp.zeros((n_locals[comm_rank], diagonal_blocksize, diagonal_blocksize), dtype=A_diagonal_blocks_init.dtype)
    # else:
    #     A_lower_diagonal_blocks_local = cp.zeros((n_locals[comm_rank]-1, diagonal_blocksize, diagonal_blocksize), dtype=A_diagonal_blocks_init.dtype)
    # A_arrow_bottom_blocks_local = cp.zeros((n_locals[comm_rank], arrowhead_blocksize, diagonal_blocksize), dtype=A_arrow_bottom_blocks_init.dtype)
    # A_arrow_tip_block_local = cp.zeros_like(A_arrow_tip_block_init)
    # toc = time.perf_counter()

    

    # print(f"Allocating testing buffers: {toc - tic:.5f} sec", flush=True)
    print("Initialization done..", flush=True)

    t_pobtaf = np.zeros(n_iterations)
    t_pobtasi = np.zeros(n_iterations)

    for i in range(n_warmups + n_iterations):
        print(f"Iteration: {i+1}/{n_warmups+n_iterations}", flush=True)

        print("comm_rank: ", comm_rank, flush=True)

        tic = time.perf_counter()
        A_diagonal_blocks_local = None
        A_lower_diagonal_blocks_local = None
        A_arrow_bottom_blocks_local = None
        A_arrow_tip_block_local = None
        permutation_buffer = None
        _L_diagonal_blocks = None
        _L_lower_diagonal_blocks = None
        _L_lower_arrow_blocks = None
        _L_tip_update = None

        A_diagonal_blocks_local = cp.asarray(A_diagonal_blocks_init[starting_idx[comm_rank]:starting_idx[comm_rank]+n_locals[comm_rank], :, :])
    
        # A_diagonal_blocks_local[:, :, :] = cp.asarray(A_diagonal_blocks_init[starting_idx[comm_rank]:starting_idx[comm_rank]+n_locals[comm_rank], :, :])
        if comm_rank != comm_size - 1:
            A_lower_diagonal_blocks_local = cp.array(A_lower_diagonal_blocks_init[starting_idx[comm_rank]:starting_idx[comm_rank]+n_locals[comm_rank], :, :])

            # A_lower_diagonal_blocks_local[:, :, :] = cp.array(A_lower_diagonal_blocks_init[starting_idx[comm_rank]:starting_idx[comm_rank]+n_locals[comm_rank], :, :])
        else:
            A_lower_diagonal_blocks_local = cp.asarray(A_lower_diagonal_blocks_init[starting_idx[comm_rank]:starting_idx[comm_rank]+n_locals[comm_rank]-1, :, :])
            # A_lower_diagonal_blocks_local[:, :, :] = cp.asarray(A_lower_diagonal_blocks_init[starting_idx[comm_rank]:starting_idx[comm_rank]+n_locals[comm_rank]-1, :, :])
        A_arrow_bottom_blocks_local = cp.asarray(A_arrow_bottom_blocks_init[starting_idx[comm_rank]:starting_idx[comm_rank]+n_locals[comm_rank], :, :])
        A_arrow_tip_block_local = cp.asarray(A_arrow_tip_block_init[:, :])
        
        # A_arrow_bottom_blocks_local[:, :, :] = cp.asarray(A_arrow_bottom_blocks_init[starting_idx[comm_rank]:starting_idx[comm_rank]+n_locals[comm_rank], :, :])
        # A_arrow_tip_block_local[:, :] = cp.asarray(A_arrow_tip_block_init[:, :])
        
        """ # Reset permutation buffer
        permutation_buffer.fill(0)

        # Reset reduced system
        _L_diagonal_blocks.fill(0)
        _L_lower_diagonal_blocks.fill(0)
        _L_lower_arrow_blocks.fill(0)
        _L_tip_update.fill(0) """

        permutation_buffer = allocate_permutation_buffer(
            A_diagonal_blocks_local,
            device_streaming=False,
        )

        (
            _L_diagonal_blocks,
            _L_lower_diagonal_blocks,
            _L_lower_arrow_blocks,
            _L_tip_update,
        ) = allocate_ppobtars(
            A_diagonal_blocks=A_diagonal_blocks_local,
            A_lower_diagonal_blocks=A_lower_diagonal_blocks_local,
            A_arrow_bottom_blocks=A_arrow_bottom_blocks_local,
            A_arrow_tip_block=A_arrow_tip_block_local,
            comm_size=comm_size,
            array_module="cupy",
            strategy=comm_strategy,
        )
        toc = time.perf_counter()

        print(
            f"  Copying initial data to testing array took: {toc - tic:.5f} sec",
            flush=True,
        )

        cp.cuda.runtime.deviceSynchronize()
        MPI.COMM_WORLD.Barrier()
        start_time = time.perf_counter()
        (
            _L_diagonal_blocks,
            _L_lower_diagonal_blocks,
            _L_lower_arrow_blocks,
            permutation_buffer,
        ) = ppobtaf(
            A_diagonal_blocks_local,
            A_lower_diagonal_blocks_local,
            A_arrow_bottom_blocks_local,
            A_arrow_tip_block_local,
            A_permutation_buffer=permutation_buffer,
            _L_diagonal_blocks=_L_diagonal_blocks,
            _L_lower_diagonal_blocks=_L_lower_diagonal_blocks,
            _L_lower_arrow_blocks=_L_lower_arrow_blocks,
            _L_tip_update=_L_tip_update,
            strategy=comm_strategy,
            comm=comm, comm_rank=comm_rank, comm_size=comm_size,
        )
        cp.cuda.runtime.deviceSynchronize()
        MPI.COMM_WORLD.Barrier()
        end_time = time.perf_counter()
        elapsed_time_pobtaf = end_time - start_time

        if i >= n_warmups:
            t_pobtaf[i - n_warmups] = elapsed_time_pobtaf

        cp.cuda.runtime.deviceSynchronize()
        MPI.COMM_WORLD.Barrier()
        start_time = time.perf_counter()
        ppobtasi(
            A_diagonal_blocks_local,
            A_lower_diagonal_blocks_local,
            A_arrow_bottom_blocks_local,
            A_arrow_tip_block_local,
            _L_diagonal_blocks,
            _L_lower_diagonal_blocks,
            _L_lower_arrow_blocks,
            permutation_buffer,
            strategy=comm_strategy,
        )
        cp.cuda.runtime.deviceSynchronize()
        MPI.COMM_WORLD.Barrier()
        end_time = time.perf_counter()
        elapsed_time_pobtasi = end_time - start_time

        if i >= n_warmups:
            t_pobtasi[i - n_warmups] = elapsed_time_pobtasi

        if i < n_warmups:
            print(
                f"  Warmup iteration: {i+1}/{n_warmups}, Time pobtaf: {elapsed_time_pobtaf:.5f} sec, Time pobtasi: {elapsed_time_pobtasi:.5f} sec"
            )
        else:
            print(
                f"  Bench iteration: {i+1-n_warmups}/{n_iterations} Time Chol: {elapsed_time_pobtaf:.5f} sec. Time selInv: {elapsed_time_pobtasi:.5f} sec"
            )

    mean_pobtaf, lb_mean_pobtaf, ub_mean_pobtaf = mean_confidence_interval(t_pobtaf)
    mean_pobtasi, lb_mean_pobtasi, ub_mean_pobtasi = mean_confidence_interval(t_pobtasi)

    print(
        f"Mean time pobtaf: {mean_pobtaf:.5f} sec, 95% CI: [{lb_mean_pobtaf:.5f}, {ub_mean_pobtaf:.5f}]"
    )

    print(
        f"Mean time pobtasi: {mean_pobtasi:.5f} sec, 95% CI: [{lb_mean_pobtasi:.5f}, {ub_mean_pobtasi:.5f}]"
    )

    np.save(
        f"timings_inlamat_pobtaf_b{diagonal_blocksize}_a{arrowhead_blocksize}_n{n_diag_blocks}.npy",
        t_pobtaf,
    )

    np.save(
        f"timings_inlamat_pobtasi_b{diagonal_blocksize}_a{arrowhead_blocksize}_n{n_diag_blocks}.npy",
        t_pobtasi,
    )
