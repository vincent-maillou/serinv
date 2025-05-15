import time

tic = time.perf_counter()
import argparse

import numpy as np
import cupy as cp
from cupy.cuda.nvtx import RangePush, RangePop

from serinv.algs import pobtaf, pobtas, pobtasi


def sequential_dataset(
    n_blocks: int,
    diagonal_blocksize: int,
    arrowhead_blocksize: int,
):
    A_diagonal_blocks = np.random.rand(n_blocks, diagonal_blocksize, diagonal_blocksize)
    A_lower_diagonal_blocks = np.random.rand(
        n_blocks - 1, diagonal_blocksize, diagonal_blocksize
    )
    A_arrow_bottom_blocks = np.random.rand(
        n_blocks, arrowhead_blocksize, diagonal_blocksize
    )
    A_arrow_tip_block = np.random.rand(arrowhead_blocksize, arrowhead_blocksize)

    # CODE TO MODIFY
    arrow_colsum = np.zeros((arrowhead_blocksize), dtype=A_diagonal_blocks.dtype)
    for i in range(A_diagonal_blocks.shape[0]):
        colsum = np.sum(A_diagonal_blocks[i, :, :], axis=1) - np.diag(
            A_diagonal_blocks[i, :, :]
        )
        if i > 0:
            colsum += np.sum(A_lower_diagonal_blocks[i - 1, :, :], axis=1)

        A_diagonal_blocks[i, :, :] += np.diag(colsum)

        arrow_colsum[:] += np.sum(A_arrow_bottom_blocks[i, :, :], axis=1)

    A_arrow_tip_block[:, :] += np.diag(
        arrow_colsum + np.sum(A_arrow_tip_block[:, :], axis=1)
    )

    return (
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_arrow_bottom_blocks,
        A_arrow_tip_block,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--b",
        type=int,
        default=128,
        help="an integer for the diagonal block size",
    )
    parser.add_argument(
        "--a",
        type=int,
        default=0,
        help="an integer for the diagonal block size",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=8,
        help="an integer for the number of diagonal blocks",
    )
    args = parser.parse_args()
    toc = time.perf_counter()
    print(f"Import and parsing took: {toc - tic:.5f} sec", flush=True)

    diagonal_blocksize = args.b
    arrowhead_blocksize = args.a
    n_blocks = args.n
    n_iterations = 10
    n_warmups = 2

    tic = time.perf_counter()
    (
        A_diagonal_blocks_cpu,
        A_lower_diagonal_blocks_cpu,
        A_arrow_bottom_blocks_cpu,
        A_arrow_tip_block_cpu,
    ) = sequential_dataset(
        n_blocks,
        diagonal_blocksize,
        arrowhead_blocksize,
    )
    B_cpu = np.random.rand(diagonal_blocksize * n_blocks + arrowhead_blocksize, 1)
    toc = time.perf_counter()
    print(f"Generate dataset took: {toc - tic:.5f} sec", flush=True)
    print(f"    b = {diagonal_blocksize}", flush=True)
    print(f"    a = {arrowhead_blocksize}", flush=True)
    print(f"    n = {n_blocks}", flush=True)
    print(f"    n_iterations = {n_iterations}", flush=True)
    print(f"    n_warmups = {n_warmups}", flush=True)

    total_memory = (
        A_diagonal_blocks_cpu.nbytes
        + A_lower_diagonal_blocks_cpu.nbytes
        + A_arrow_bottom_blocks_cpu.nbytes
        + A_arrow_tip_block_cpu.nbytes
        + B_cpu.nbytes
    )
    print(f"    Total memory: {total_memory / 1e9:.5f} GB", flush=True)

    tic = time.perf_counter()
    # Init device arrays
    A_diagonal_blocks_gpu = cp.empty_like(A_diagonal_blocks_cpu)
    A_lower_diagonal_blocks_gpu = cp.empty_like(A_lower_diagonal_blocks_cpu)
    A_arrow_bottom_blocks_gpu = cp.empty_like(A_arrow_bottom_blocks_cpu)
    A_arrow_tip_block_gpu = cp.empty_like(A_arrow_tip_block_cpu)
    B_gpu = cp.empty_like(B_cpu)
    toc = time.perf_counter()
    print(f"Init device arrays took: {toc - tic:.5f} sec", flush=True)

    t_pobtaf = []
    t_pobtas = []
    t_pobtasi = []

    for i in range(n_warmups + n_iterations):
        print(f"Iteration: {i+1}/{n_warmups+n_iterations}", flush=True)

        tic = time.perf_counter()
        A_diagonal_blocks_gpu.set(arr=A_diagonal_blocks_cpu)
        A_lower_diagonal_blocks_gpu.set(arr=A_lower_diagonal_blocks_cpu)
        A_arrow_bottom_blocks_gpu.set(arr=A_arrow_bottom_blocks_cpu)
        A_arrow_tip_block_gpu.set(arr=A_arrow_tip_block_cpu)
        B_gpu.set(arr=B_cpu)
        toc = time.perf_counter()
        print(f"Copying data to GPU took: {toc - tic:.5f} sec", flush=True)

        cp.cuda.runtime.deviceSynchronize()
        RangePush(f"pobtaf: i:{i}")
        tic = time.perf_counter()
        pobtaf(
            A_diagonal_blocks_gpu,
            A_lower_diagonal_blocks_gpu,
            A_arrow_bottom_blocks_gpu,
            A_arrow_tip_block_gpu,
            device_streaming=True
        )
        cp.cuda.runtime.deviceSynchronize()
        toc = time.perf_counter()
        RangePop()
        elapsed = toc - tic
        print(f"pobtaf took: {elapsed:.5f} sec", flush=True)
        if i >= n_warmups:
            t_pobtaf.append(elapsed)

        cp.cuda.runtime.deviceSynchronize()
        RangePush(f"pobtas: i:{i}")
        tic = time.perf_counter()
        pobtas(
            A_diagonal_blocks_gpu,
            A_lower_diagonal_blocks_gpu,
            A_arrow_bottom_blocks_gpu,
            A_arrow_tip_block_gpu,
            B_gpu,
            device_streaming=True
        )
        cp.cuda.runtime.deviceSynchronize()
        toc = time.perf_counter()
        RangePop()
        elapsed = toc - tic
        print(f"pobtas took: {elapsed:.5f} sec", flush=True)
        if i >= n_warmups:
            t_pobtas.append(elapsed)

        cp.cuda.runtime.deviceSynchronize()
        RangePush(f"pobtasi: i:{i}")
        tic = time.perf_counter()
        pobtasi(
            A_diagonal_blocks_gpu,
            A_lower_diagonal_blocks_gpu,
            A_arrow_bottom_blocks_gpu,
            A_arrow_tip_block_gpu,
        )
        cp.cuda.runtime.deviceSynchronize()
        toc = time.perf_counter()
        RangePop()
        elapsed = toc - tic
        print(f"pobtasi took: {elapsed:.5f} sec", flush=True)
        if i >= n_warmups:
            t_pobtasi.append(elapsed)

    print(f"t_pobtaf: {t_pobtaf}", flush=True)
    print(f"t_pobtas: {t_pobtas}", flush=True)
    print(f"t_pobtasi: {t_pobtasi}", flush=True)

    print(f"avg t_pobtaf: {np.mean(np.array(t_pobtaf)):.5f} sec", flush=True)
    print(f"avg t_pobtas: {np.mean(np.array(t_pobtas)):.5f} sec", flush=True)
    print(f"avg t_pobtasi: {np.mean(np.array(t_pobtasi)):.5f} sec", flush=True)