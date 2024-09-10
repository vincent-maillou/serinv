import time

tic = time.perf_counter()
try:
    import cupyx as cpx
except:
    pass

from serinv.algs import pobtaf, pobtasi
from load_datmat import csc_to_dense_bta, read_sym_CSC
from utility_functions import bta_arrays_to_dense, bta_dense_to_arrays
import numpy as np

import scipy.stats
import argparse


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

    args = parser.parse_args()
    toc = time.perf_counter()
    print(f"Import and parsing took: {toc - tic:.5f} sec", flush=True)

    diagonal_blocksize = args.diagonal_blocksize
    arrowhead_blocksize = args.arrowhead_blocksize
    n_diag_blocks = args.n_diag_blocks
    file_path = args.file_path
    n_iterations = args.n_iterations
    n_warmups = args.n_warmups
    timing_sections = True

    print("timing_sections", timing_sections)

    n = diagonal_blocksize * n_diag_blocks + arrowhead_blocksize

    device_streaming = True

    # if True: compare to reference solution
    DEBUG = False

    # read in file
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
    
    filename = file_path + file

    tic = time.perf_counter()
    A = read_sym_CSC(filename)

    (
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_arrow_bottom_blocks,
        A_arrow_tip_block,
    ) = csc_to_dense_bta(A, diagonal_blocksize, arrowhead_blocksize, n_diag_blocks)
    toc = time.perf_counter()

    print(f"Reading matrix took: {toc - tic:.5f} sec", flush=True)
    print(f"    A_diagonal_blocks.shape", A_diagonal_blocks.shape, flush=True)
    print(f"    A_lower_diagonal_blocks.shape", A_lower_diagonal_blocks.shape, flush=True)
    print(f"    A_arrow_bottom_blocks.shape", A_arrow_bottom_blocks.shape, flush=True)
    print(f"    A_arrow_tip_block.shape", A_arrow_tip_block.shape, flush=True)

    tic = time.perf_counter()
    A_diagonal_blocks_pinned = cpx.zeros_like_pinned(A_diagonal_blocks)
    A_lower_diagonal_blocks_pinned = cpx.zeros_like_pinned(A_lower_diagonal_blocks)
    A_arrow_bottom_blocks_pinned = cpx.zeros_like_pinned(A_arrow_bottom_blocks)
    A_arrow_tip_block_pinned = cpx.zeros_like_pinned(A_arrow_tip_block)
    toc = time.perf_counter()

    print(f"Allocating pinned memory took: {toc - tic:.5f} sec", flush=True)
    print(f"Initialization done..", flush=True)

    t_pobtaf = np.zeros(n_iterations)
    t_pobtasi = np.zeros(n_iterations)

    dict_timings_pobtaf = {}
    dict_timings_pobtaf["potrf"] = np.zeros(n_iterations)
    dict_timings_pobtaf["trsm"] = np.zeros(n_iterations)
    dict_timings_pobtaf["gemm"] = np.zeros(n_iterations)

    dict_timings_pobtasi = {}
    dict_timings_pobtasi["trsm"] = np.zeros(n_iterations)
    dict_timings_pobtasi["gemm"] = np.zeros(n_iterations)

    dict_timings = {}
    dict_timings["potrf"] = 0
    dict_timings["trsm"] = 0
    dict_timings["gemm"] = 0

    for i in range(n_warmups + n_iterations):
        print(f"Iteration: {i+1}/{n_warmups+n_iterations}", flush=True)

        tic = time.perf_counter()
        A_diagonal_blocks_pinned[:, :, :] = A_diagonal_blocks[:, :, :].copy()
        A_lower_diagonal_blocks_pinned[:, :, :] = A_lower_diagonal_blocks[:, :, :].copy()
        A_arrow_bottom_blocks_pinned[:, :, :] = A_arrow_bottom_blocks[:, :, :].copy()
        A_arrow_tip_block_pinned[:, :] = A_arrow_tip_block[:, :].copy()
        toc = time.perf_counter()

        print(f"  Copying data to pinned memory took: {toc - tic:.5f} sec", flush=True)


        start_time = time.perf_counter()
        (
            L_diagonal_blocks,
            L_lower_diagonal_blocks,
            L_arrow_bottom_blocks,
            L_arrow_tip_block,
            dict_timings,
        ) = pobtaf(
            A_diagonal_blocks_pinned,
            A_lower_diagonal_blocks_pinned,
            A_arrow_bottom_blocks_pinned,
            A_arrow_tip_block_pinned,
            device_streaming,
            timing_sections,
        )
        end_time = time.perf_counter()
        elapsed_time_pobtaf = end_time - start_time

        if i >= n_warmups:
            dict_timings_pobtaf["potrf"][i - n_warmups] = dict_timings["potrf"]
            dict_timings_pobtaf["trsm"][i - n_warmups] = dict_timings["trsm"]
            dict_timings_pobtaf["gemm"][i - n_warmups] = dict_timings["gemm"]
            t_pobtaf[i - n_warmups] = elapsed_time_pobtaf


        if DEBUG:
            L = bta_arrays_to_dense(
                L_diagonal_blocks,
                L_lower_diagonal_blocks,
                np.zeros(
                    (n_diag_blocks - 1, diagonal_blocksize, diagonal_blocksize),
                    dtype=A.dtype,
                ),
                L_arrow_bottom_blocks,
                np.zeros(
                    (n_diag_blocks, diagonal_blocksize, arrowhead_blocksize),
                    dtype=A.dtype,
                ),
                L_arrow_tip_block,
            )

            A_dense = A.todense()
            A_symmetric = A_dense + np.tril(A_dense, -1).T
            L_ref = np.linalg.cholesky(A_symmetric)
            print("norm(L - L_ref):", np.linalg.norm(L - L_ref))


        start_time = time.perf_counter()
        (
            X_diagonal_blocks,
            X_lower_diagonal_blocks,
            X_arrow_bottom_blocks,
            X_arrow_tip_block,
            dict_timings,
        ) = pobtasi(
            L_diagonal_blocks,
            L_lower_diagonal_blocks,
            L_arrow_bottom_blocks,
            L_arrow_tip_block,
            device_streaming,
            timing_sections,
        )
        end_time = time.perf_counter()
        elapsed_time_pobtasi = end_time - start_time

        if i >= n_warmups:
            dict_timings_pobtasi["trsm"][i - n_warmups] = dict_timings["trsm"]
            dict_timings_pobtasi["gemm"][i - n_warmups] = dict_timings["gemm"]
            t_pobtasi[i - n_warmups] = elapsed_time_pobtasi

        if i < n_warmups:
            print(
                f"  Warmup iteration: {i+1}/{n_warmups}, Time pobtaf: {elapsed_time_pobtaf:.5f} sec, Time pobtasi: {elapsed_time_pobtasi:.5f} sec"
            )
        else:
            print(
                f"  Bench iteration: {i+1-n_warmups}/{n_iterations} Time Chol: {elapsed_time_pobtaf:.5f} sec. Time selInv: {elapsed_time_pobtasi:.5f} sec"
            )
            print(f"  pobtaf: potrf = {dict_timings_pobtaf['potrf'][i-n_warmups]:.5f} sec, trsm = {dict_timings_pobtaf['trsm'][i-n_warmups]:.5f} sec, gemm = {dict_timings_pobtaf['gemm'][i-n_warmups]:.5f} sec")
            print(f"  pobtasi: trsm = {dict_timings_pobtasi['trsm'][i-n_warmups]:.5f} sec, gemm = {dict_timings_pobtasi['gemm'][i-n_warmups]:.5f} sec")

        if DEBUG:
            X_ref = np.linalg.inv(A_symmetric)
            (
                X_diagonal_blocks_ref,
                X_lower_diagonal_blocks_ref,
                _,
                X_arrow_bottom_blocks_ref,
                _,
                X_arrow_tip_block_ref,
            ) = bta_dense_to_arrays(
                X_ref, diagonal_blocksize, arrowhead_blocksize, n_diag_blocks
            )

            print(
                "norm(X_diagonal_blocks - X_diagonal_blocks_ref):             ",
                np.linalg.norm(X_diagonal_blocks - X_diagonal_blocks_ref),
            )
            print(
                "norm(X_lower_diagonal_blocks - X_lower_diagonal_blocks_ref): ",
                np.linalg.norm(X_lower_diagonal_blocks - X_lower_diagonal_blocks_ref),
            )
            print(
                "norm(X_arrow_bottom_blocks - X_arrow_bottom_blocks_ref):     ",
                np.linalg.norm(X_arrow_bottom_blocks - X_arrow_bottom_blocks_ref),
            )
            print(
                "norm(X_arrow_tip_block - X_arrow_tip_block_ref):             ",
                np.linalg.norm(X_arrow_tip_block - X_arrow_tip_block_ref),
            )

    mean_pobtaf, lb_mean_pobtaf, ub_mean_pobtaf = mean_confidence_interval(t_pobtaf)
    mean_pobtasi, lb_mean_pobtasi, ub_mean_pobtasi = mean_confidence_interval(t_pobtasi)

    print(
        f"Mean time pobtaf: {mean_pobtaf:.5f} sec, 95% CI: [{lb_mean_pobtaf:.5f}, {ub_mean_pobtaf:.5f}]"
    )

    print(
        f"Mean time pobtasi: {mean_pobtasi:.5f} sec, 95% CI: [{lb_mean_pobtasi:.5f}, {ub_mean_pobtasi:.5f}]"
    )

    if timing_sections:
        # Save the raw data
        np.save(
            f"dict_timings_inlamat_pobtaf_b{diagonal_blocksize}_a{arrowhead_blocksize}_n{n_diag_blocks}.npy",
            dict_timings_pobtaf,
        )

        np.save(
            f"dict_timings_inlamat_pobtasi_b{diagonal_blocksize}_a{arrowhead_blocksize}_n{n_diag_blocks}.npy",
            dict_timings_pobtasi,
        )
    else:
        np.save(
            f"timings_inlamat_pobtaf_b{diagonal_blocksize}_a{arrowhead_blocksize}_n{n_diag_blocks}.npy",
            t_pobtaf,
        )

        np.save(
            f"timings_inlamat_pobtasi_b{diagonal_blocksize}_a{arrowhead_blocksize}_n{n_diag_blocks}.npy",
            t_pobtasi,
        )
    
