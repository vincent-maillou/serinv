# Copyright 2023-2024 ETH Zurich. All rights reserved.

SEED = 63

import numpy as np

np.random.seed(SEED)

def generate_synthetic_dataset_for_pobta(
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
        diag = np.abs(np.diag(A_diagonal_blocks[i, :, :]))
        colsum = (
            np.sum(A_diagonal_blocks[i, :, :], axis=1)
            - np.diag(A_diagonal_blocks[i, :, :])
        )
        if i > 0:
            colsum += np.sum(A_lower_diagonal_blocks[i - 1, :, :], axis=1)

        A_diagonal_blocks[i, :, :] += np.diag(colsum)

        arrow_colsum[:] += np.sum(A_arrow_bottom_blocks[i, :, :], axis=1)

    A_arrow_tip_block[:, :] += np.diag(arrow_colsum + np.sum(A_arrow_tip_block[:, :], axis=1)) 

    return (
        A_diagonal_blocks, 
        A_lower_diagonal_blocks, 
        A_arrow_bottom_blocks, 
        A_arrow_tip_block
    )




if __name__ == "__main__":
    from serinv.utils.check_dd import check_ddbta

    n_blocks = 32
    diagonal_blocksize = 4096
    arrowhead_blocksize = diagonal_blocksize//4
    PATH = "/home/vault/j101df/j101df10/inla_matrices/synthetic_dataset/"
    FILE_NAME = f"pobta_nb{n_blocks}_ds{diagonal_blocksize}_as{arrowhead_blocksize}.npz"

    (
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_arrow_bottom_blocks,
        A_arrow_tip_block,
    ) = generate_synthetic_dataset_for_pobta(
        n_blocks, diagonal_blocksize, arrowhead_blocksize
    )

    print("A_diagonal_blocks.shape", A_diagonal_blocks.shape, flush=True)
    print("A_lower_diagonal_blocks.shape", A_lower_diagonal_blocks.shape, flush=True)
    print("A_arrow_bottom_blocks.shape", A_arrow_bottom_blocks.shape, flush=True)
    print("A_arrow_tip_block.shape", A_arrow_tip_block.shape, flush=True)

    ddbta = check_ddbta(
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        np.zeros((n_blocks, diagonal_blocksize, diagonal_blocksize), dtype=np.float64),
        A_arrow_bottom_blocks,
        np.zeros((n_blocks, diagonal_blocksize, arrowhead_blocksize), dtype=np.float64),
        A_arrow_tip_block,
    )

    if np.all(ddbta):
        print("All rows are diagonally dominant", flush=True)

    np.savez(
        PATH+FILE_NAME,
        A_diagonal_blocks=A_diagonal_blocks,
        A_lower_diagonal_blocks=A_lower_diagonal_blocks,
        A_arrow_bottom_blocks=A_arrow_bottom_blocks,
        A_arrow_tip_block=A_arrow_tip_block,
    )

    