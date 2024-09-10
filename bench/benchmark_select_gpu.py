import cupy as cp
import cupyx as cpx
import numpy as np
import scipy.stats
import time
import argparse
from ctypes import *


import mpi4py
mpi4py.rc.initialize = False  # do not initialize MPI automatically
mpi4py.rc.finalize = False  # do not finalize MPI automatically
from mpi4py import MPI

# comm_rank = MPI.COMM_WORLD.Get_rank()
# comm_size = MPI.COMM_WORLD.Get_size()

from serinv.algs import d_pobtaf, d_pobtasi

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return m, m - h, m + h


def set_device(
    rank_id: int,
    n_gpus_per_node,
    gpus: list,
    debug: bool = False,
):
    """
    Set the device for the current process.

    Parameters
    ----------
    rank_id : int
        The rank of the current process.
    n_gpus_per_node : int
        The number of GPUs per node.
    gpus: list[int]
        GPU IDs to use.
    debug : bool, optional
        Whether to print debug information.
    """
    local_id = rank_id % n_gpus_per_node
    if not isinstance(gpus, list):
        gpus = list(gpus)
    device_id = gpus[local_id]
    cp.cuda.Device(device_id).use()

    if debug:
        libc = CDLL("libc.so.6")
        print(f"Process {rank_id} of {comm_size} is using the GPU {device_id} from the CPU {libc.sched_getcpu()}", flush=True)

    return device_id


if __name__ == "__main__":

    MPI.Init_thread(MPI.THREAD_SINGLE)
    comm_rank = MPI.COMM_WORLD.Get_rank()
    comm_size = MPI.COMM_WORLD.Get_size()

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
        "--file_path",
        type=str,
        default="/home/vault/j101df/j101df10/inla_matrices/synthetic_dataset/distributed/",
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
        "--n_tasks_per_node",
        type=int,
        help="number of MPI ranks per node"
    )
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        help="IDs of the GPUs to use ordered by MPI rank"
    )

    args = parser.parse_args()

    diagonal_blocksize = args.diagonal_blocksize
    arrowhead_blocksize = args.arrowhead_blocksize
    n_diag_blocks = args.n_diag_blocks
    file_path = args.file_path
    n_iterations = args.n_iterations
    n_warmups = args.n_warmups
    n_tasks = args.n_tasks_per_node
    gpus = args.gpus

    color = comm_rank < n_tasks
    key = comm_rank
    comm = MPI.COMM_WORLD.Split(color=color, key=key)

    if color == 1:

        comm_rank = comm.Get_rank()
        comm_size = comm.Get_size()

        device_streaming = False
        n_gpus_per_node = 8

        device_id = set_device(comm_rank, n_gpus_per_node, gpus, debug=True)

        # Load the diagonal block belonging to the current process
        file_name = f"pobta_nb{n_diag_blocks}_ds{diagonal_blocksize}_as{arrowhead_blocksize}_blocks_p{comm_rank}_np{comm_size-1}.npz"

        blocks_data = np.load(file_path+file_name)

        A_diagonal_blocks_local = blocks_data["A_diagonal_blocks"]
        A_lower_diagonal_blocks_local = blocks_data["A_lower_diagonal_blocks"]
        A_arrow_bottom_blocks_local = blocks_data["A_arrow_bottom_blocks"]

        # Load the tip of the arrow
        file_name_arrowtip = f"pobta_nb{n_diag_blocks}_ds{diagonal_blocksize}_as{arrowhead_blocksize}_arrowtip_np{comm_size-1}.npz"

        tip_data = np.load(file_path+file_name_arrowtip)
        A_arrow_tip_block = tip_data["A_arrow_tip_block"]

        print(f"Process {comm_rank}: A_diagonal_blocks_local.shape", A_diagonal_blocks_local.shape, flush=True)
        print(f"Process {comm_rank}: A_lower_diagonal_blocks_local.shape", A_lower_diagonal_blocks_local.shape, flush=True)
        print(f"Process {comm_rank}: A_arrow_bottom_blocks_local.shape", A_arrow_bottom_blocks_local.shape, flush=True)
        print(f"Process {comm_rank}: A_arrow_tip_block.shape", A_arrow_tip_block.shape, flush=True)

        # # Allocate pinned memory buffers
        # A_diagonal_blocks_local_pinned = cpx.zeros_like_pinned(A_diagonal_blocks_local)
        # A_lower_diagonal_blocks_local_pinned = cpx.zeros_like_pinned(A_lower_diagonal_blocks_local)
        # A_arrow_bottom_blocks_local_pinned = cpx.zeros_like_pinned(A_arrow_bottom_blocks_local)
        # A_arrow_tip_block_pinned = cpx.zeros_like_pinned(A_arrow_tip_block)

        print(f"Initialization done.. (process {comm_rank})", flush=True)

        t_d_pobtaf = np.zeros(n_iterations)
        t_d_pobtasi = np.zeros(n_iterations)

        for i in range(n_warmups + n_iterations):
            # A_diagonal_blocks_local_pinned[:, :, :] = A_diagonal_blocks_local[:, :, :].copy()
            # A_lower_diagonal_blocks_local_pinned[:, :, :] = A_lower_diagonal_blocks_local[:, :, :].copy()
            # A_arrow_bottom_blocks_local_pinned[:, :, :] = A_arrow_bottom_blocks_local[:, :, :].copy()
            # A_arrow_tip_block_pinned[:, :] = A_arrow_tip_block[:, :].copy()

            A_diagonal_blocks_local_dev = cp.asarray(A_diagonal_blocks_local)
            A_lower_diagonal_blocks_local_dev = cp.asarray(A_lower_diagonal_blocks_local)
            A_arrow_bottom_blocks_local_dev = cp.asarray(A_arrow_bottom_blocks_local)
            A_arrow_tip_block_dev = cp.asarray(A_arrow_tip_block)

            comm.Barrier()

            start_time = time.perf_counter()

            # (
            #     L_diagonal_blocks_local,
            #     L_lower_diagonal_blocks_local,
            #     L_arrow_bottom_blocks_local,
            #     L_arrow_tip_block_global,
            #     L_upper_nested_dissection_buffer_local,
            # ) = d_pobtaf(
            #     A_diagonal_blocks_local_pinned,
            #     A_lower_diagonal_blocks_local_pinned,
            #     A_arrow_bottom_blocks_local_pinned,
            #     A_arrow_tip_block_pinned,
            #     device_streaming,
            # )

            (
                L_diagonal_blocks_local,
                L_lower_diagonal_blocks_local,
                L_arrow_bottom_blocks_local,
                L_arrow_tip_block_global,
                L_upper_nested_dissection_buffer_local,
            ) = d_pobtaf(
                A_diagonal_blocks_local_dev,
                A_lower_diagonal_blocks_local_dev,
                A_arrow_bottom_blocks_local_dev,
                A_arrow_tip_block_dev,
                device_streaming,
                comm
            )

            comm.Barrier()

            end_time = time.perf_counter()
            elapsed_time_pobtaf = end_time - start_time

            if i >= n_warmups:
                t_d_pobtaf[i - n_warmups] = elapsed_time_pobtaf

            start_time = time.perf_counter()

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
                device_streaming,
                comm
            )

            comm.Barrier()

            end_time = time.perf_counter()
            elapsed_time_selinv = end_time - start_time

            if i >= n_warmups:
                t_d_pobtasi[i - n_warmups] = elapsed_time_selinv

            if comm_rank == 0:
                if i < n_warmups:
                    print(
                        f"Warmup iteration: {i+1}/{n_warmups}, Time d_pobtaf: {elapsed_time_pobtaf:.5f} sec, Time d_pobtasi: {elapsed_time_selinv:.5f} sec", flush=True
                    )
                else:
                    print(
                        f"Bench iteration: {i+1-n_warmups}/{n_iterations} Time d_pobtaf: {elapsed_time_pobtaf:.5f} sec. Time d_pobtasi: {elapsed_time_selinv:.5f} sec", flush=True
                    )

        if comm_rank == 0:
            np.save(
                f"timings_d_pobtaf_bs{diagonal_blocksize}_as{arrowhead_blocksize}_nb{n_diag_blocks}_np{comm_size}.npy",
                t_d_pobtaf,
            )

            np.save(
                f"timings_d_pobtasi_bs{diagonal_blocksize}_as{arrowhead_blocksize}_nb{n_diag_blocks}_np{comm_size}.npy",
                t_d_pobtasi,
            )

            mean_d_pobtaf, lb_mean_d_pobtaf, ub_mean_d_pobtaf = mean_confidence_interval(t_d_pobtaf)
            mean_d_pobtasi, lb_mean_d_pobtasi, ub_mean_d_pobtasi = mean_confidence_interval(t_d_pobtasi)

            print(
                f"Mean time d_pobtaf: {mean_d_pobtaf:.5f} sec, 95% CI: [{lb_mean_d_pobtaf:.5f}, {ub_mean_d_pobtaf:.5f}]", flush=True
            )

            print(
                f"Mean time d_pobtasi: {mean_d_pobtasi:.5f} sec, 95% CI: [{lb_mean_d_pobtasi:.5f}, {ub_mean_d_pobtasi:.5f}]", flush=True
            )

    MPI.COMM_WORLD.Barrier()
    
    MPI.Finalize()
