# Copyright 2023-2024 ETH Zurich. All rights reserved.

SEED = 63

import numpy as np

np.random.seed(SEED)

from serinv.utils.check_dd import check_ddbta
import argparse

def generate_synthetic_dataset_for_pobta(
        path: str,
        n_blocks: int,
        diagonal_blocksize: int,
        arrowhead_blocksize: int,
        dtype = np.float64,
):
    rc = (1.0 + 1.0j) if dtype == np.complex128 else 1.0
    
    A_diagonal_blocks = rc * np.random.rand(n_blocks, diagonal_blocksize, diagonal_blocksize)
    A_lower_diagonal_blocks = rc * np.random.rand(n_blocks - 1, diagonal_blocksize, diagonal_blocksize)
    A_arrow_bottom_blocks = rc * np.random.rand(n_blocks, arrowhead_blocksize, diagonal_blocksize)
    A_arrow_tip_block = rc * np.random.rand(arrowhead_blocksize, arrowhead_blocksize)


    # CODE TO MODIFY
    arrow_colsum = np.zeros((arrowhead_blocksize), dtype=A_diagonal_blocks.dtype)
    for i in range(A_diagonal_blocks.shape[0]):
        colsum = (
            np.sum(A_diagonal_blocks[i, :, :], axis=1)
            - np.diag(A_diagonal_blocks[i, :, :])
        )
        if i > 0:
            colsum += np.sum(A_lower_diagonal_blocks[i - 1, :, :], axis=1)

        A_diagonal_blocks[i, :, :] += np.diag(colsum)

        arrow_colsum[:] += np.sum(A_arrow_bottom_blocks[i, :, :], axis=1)

    A_arrow_tip_block[:, :] += np.diag(arrow_colsum + np.sum(A_arrow_tip_block[:, :], axis=1)) 

    print("A_diagonal_blocks.shape", A_diagonal_blocks.shape, flush=True)
    print("A_lower_diagonal_blocks.shape", A_lower_diagonal_blocks.shape, flush=True)
    print("A_arrow_bottom_blocks.shape", A_arrow_bottom_blocks.shape, flush=True)
    print("A_arrow_tip_block.shape", A_arrow_tip_block.shape, flush=True)

    ddbta = check_ddbta(
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        np.zeros((n_blocks, diagonal_blocksize, diagonal_blocksize), dtype=dtype),
        A_arrow_bottom_blocks,
        np.zeros((n_blocks, diagonal_blocksize, arrowhead_blocksize), dtype=dtype),
        A_arrow_tip_block,
    )

    if np.all(ddbta):
        print("All rows are diagonally dominant", flush=True)

    file_name = f"pobta_nb{n_blocks}_ds{diagonal_blocksize}_as{arrowhead_blocksize}.npz"

    np.savez(
        path+file_name,
        A_diagonal_blocks=A_diagonal_blocks,
        A_lower_diagonal_blocks=A_lower_diagonal_blocks,
        A_arrow_bottom_blocks=A_arrow_bottom_blocks,
        A_arrow_tip_block=A_arrow_tip_block,
    )


def generate_distributed_synthetic_dataset_for_d_pobta(
        path: str,
        n_blocks: int,
        diagonal_blocksize: int,
        arrowhead_blocksize: int,
        n_processes: int,
        dtype = np.float64,
):
    rc = (1.0 + 1.0j) if dtype == np.complex128 else 1.0
    
    arrow_colsum = np.zeros((arrowhead_blocksize), dtype=dtype)
    last_process_colsum = np.zeros((diagonal_blocksize), dtype=dtype)

    for p in range(n_processes):

        n_blocks_pi = n_blocks // n_processes

        A_diagonal_blocks_pi = rc * np.random.rand(n_blocks_pi, diagonal_blocksize, diagonal_blocksize)
        A_arrow_bottom_blocks_pi = rc * np.random.rand(n_blocks_pi, arrowhead_blocksize, diagonal_blocksize)

        A_diagonal_blocks_pi[0, :, :] += np.diag(last_process_colsum)

        if p == n_processes - 1:
            A_lower_diagonal_blocks_pi = rc * np.random.rand(n_blocks_pi - 1, diagonal_blocksize, diagonal_blocksize)
        else:
            A_lower_diagonal_blocks_pi = rc * np.random.rand(n_blocks_pi, diagonal_blocksize, diagonal_blocksize)
            last_process_colsum[:] = np.sum(A_lower_diagonal_blocks_pi[-1, :, :], axis=1)

        for i in range(A_diagonal_blocks_pi.shape[0]):
            colsum = (
                np.sum(A_diagonal_blocks_pi[i, :, :], axis=1)
                - np.diag(A_diagonal_blocks_pi[i, :, :])
            )
            if i > 0:
                colsum += np.sum(A_lower_diagonal_blocks_pi[i - 1, :, :], axis=1)

            A_diagonal_blocks_pi[i, :, :] += np.diag(colsum)
            arrow_colsum[:] += np.sum(A_arrow_bottom_blocks_pi[i, :, :], axis=1)

        print(f"A_diagonal_blocks_p{p}x{n_processes-1}.shape", A_diagonal_blocks_pi.shape, flush=True)
        print(f"A_lower_diagonal_blocks_p{p}x{n_processes-1}.shape", A_lower_diagonal_blocks_pi.shape, flush=True)
        print(f"A_arrow_bottom_blocks_p{p}x{n_processes-1}.shape", A_arrow_bottom_blocks_pi.shape, flush=True)

        file_name = f"pobta_nb{n_blocks}_ds{diagonal_blocksize}_as{arrowhead_blocksize}_blocks_p{p}_np{n_processes-1}.npz"

        np.savez(
            path+file_name,
            A_diagonal_blocks=A_diagonal_blocks_pi,
            A_lower_diagonal_blocks=A_lower_diagonal_blocks_pi,
            A_arrow_bottom_blocks=A_arrow_bottom_blocks_pi,
        )

    A_arrow_tip_block = rc * np.random.rand(arrowhead_blocksize, arrowhead_blocksize)
    A_arrow_tip_block[:, :] += np.diag(arrow_colsum + np.sum(A_arrow_tip_block[:, :], axis=1)) 
    print("A_arrow_tip_block.shape", A_arrow_tip_block.shape, flush=True)

    file_name_arrowtip = f"pobta_nb{n_blocks}_ds{diagonal_blocksize}_as{arrowhead_blocksize}_arrowtip_np{n_processes-1}.npz"

    np.savez(
        path+file_name_arrowtip,
        A_arrow_tip_block=A_arrow_tip_block,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--diagonal_blocksize",
        type=int,
        default=1024,
        help="an integer for the diagonal block size",
    )
    parser.add_argument(
        "--n_diag_blocks",
        type=int,
        default=8,
        help="an integer for the number of diagonal blocks",
    )
    parser.add_argument(
        "--n_processes",
        type=int,
        default=1,
        help="number of processes for the distributed case",
    )

    args = parser.parse_args()

    n_blocks = args.n_diag_blocks
    diagonal_blocksize = args.diagonal_blocksize
    arrowhead_blocksize = diagonal_blocksize//4
    n_processes = args.n_processes

    if n_processes == 1:
        PATH = "/home/vault/j101df/j101df10/inla_matrices/synthetic_dataset/sequential/"
        generate_synthetic_dataset_for_pobta(
            PATH, n_blocks, diagonal_blocksize, arrowhead_blocksize
        )
    else:
        PATH = "/home/vault/j101df/j101df10/inla_matrices/synthetic_dataset/distributed/"
        generate_distributed_synthetic_dataset_for_d_pobta(
            PATH, n_blocks, diagonal_blocksize, arrowhead_blocksize, n_processes
        )
        