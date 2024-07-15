try:
    import cupy as cp
    import cupyx as cpx

    CUPY_AVAIL = True

except:
    CUPY_AVAIL = False
    
print("CUPY_AVAIL: ", CUPY_AVAIL)

from serinv.algs import pobtaf, pobtasi, pobtasinv
from load_datmat import csc_to_dense_bta, read_sym_CSC
from utility_functions import bta_arrays_to_dense, bta_dense_to_arrays
import numpy as np
import time
import argparse
import matplotlib.pyplot as plt


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--diagonal_blocksize', type=int, default=92,
                        help='an integer for the diagonal block size')
    parser.add_argument('--arrowhead_blocksize', type=int, default=4,
                        help='an integer for the arrowhead block size')
    parser.add_argument('--n_diag_blocks', type=int, default=50,
                        help='an integer for the number of diagonal blocks')
    parser.add_argument('--file_path', type=str, default="/home/x_gaedkelb/serinv/dev/matrices/",
                        help='a string for the file path')
    parser.add_argument('--device_streaming', type=bool, default=True,
                        help='a boolean indicating if device streaming is enabled')
    parser.add_argument('--n_iterations', type=int, default=1,
                        help='number of iterations for the benchmarking')
    
    args = parser.parse_args()

    diagonal_blocksize = args.diagonal_blocksize
    arrowhead_blocksize = args.arrowhead_blocksize
    n_diag_blocks = args.n_diag_blocks
    file_path = args.file_path
    device_streaming = args.device_streaming
    n_iterations = args.n_iterations
    
    n = diagonal_blocksize*n_diag_blocks + arrowhead_blocksize

    # if True: compare to reference solution
    DEBUG = False
    
    # read in file
    file = "Qxy_ns" + str(diagonal_blocksize) + "_nt" + str(n_diag_blocks) + "_nss0_nb" + str(arrowhead_blocksize) + "_n" + str(n) + ".dat"
    filename = file_path + file

    ## CAREFUL output matrix is only lower triangular part
    A = read_sym_CSC(filename)
    
    device_streaming = True
    device_array = False
    
    # timings
    t_list_chol = np.zeros(n_iterations)
    t_list_selInv = np.zeros(n_iterations)
    
    for i in range(n_iterations):
        
        (
            A_diagonal_blocks,
            A_lower_diagonal_blocks,
            A_arrow_bottom_blocks,
            A_arrow_tip_block,
        ) = csc_to_dense_bta(A, diagonal_blocksize, arrowhead_blocksize, n_diag_blocks)
        
        if CUPY_AVAIL and device_streaming and not device_array:
            if i == 0:
                print("Using pinned memory", flush=True)
            A_diagonal_blocks_pinned = cpx.zeros_like_pinned(A_diagonal_blocks)
            A_diagonal_blocks_pinned[:, :, :] = A_diagonal_blocks[:, :, :]
            A_lower_diagonal_blocks_pinned = cpx.zeros_like_pinned(A_lower_diagonal_blocks)
            A_lower_diagonal_blocks_pinned[:, :, :] = A_lower_diagonal_blocks[:, :, :]
            A_arrow_bottom_blocks_pinned = cpx.zeros_like_pinned(A_arrow_bottom_blocks)
            A_arrow_bottom_blocks_pinned[:, :, :] = A_arrow_bottom_blocks[:, :, :]
            A_arrow_tip_block_pinned = cpx.zeros_like_pinned(A_arrow_tip_block)
            A_arrow_tip_block_pinned[:, :] = A_arrow_tip_block[:, :]

            A_diagonal_blocks = A_diagonal_blocks_pinned
            A_lower_diagonal_blocks = A_lower_diagonal_blocks_pinned
            A_arrow_bottom_blocks = A_arrow_bottom_blocks_pinned
            A_arrow_tip_block = A_arrow_tip_block_pinned
            
        start_time = time.time()
        
        (
            X_diagonal_blocks,
            X_lower_diagonal_blocks,
            X_arrow_bottom_blocks,
            X_arrow_tip_block,
        ) = pobtasinv (
            A_diagonal_blocks,
            A_lower_diagonal_blocks,
            A_arrow_bottom_blocks,
            A_arrow_tip_block,
            device_streaming,
        )
        
        end_time = time.time()
        elapsed_time_selinv = end_time - start_time
        t_list_selInv[i] = elapsed_time_selinv
        print(f"Iter: {i} Time direct selInv: {elapsed_time_selinv:.5f} sec")
        #print(f"Iter: {i} Time Chol: {elapsed_time_chol:.5f} sec. Time selInv: {elapsed_time_selinv:.5f} sec")

        if DEBUG:   
            A_dense = A.todense()
            A_symmetric = A_dense +  np.tril(A_dense, -1).T 
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

            print("norm(X_diagonal_blocks - X_diagonal_blocks_ref):             ", np.linalg.norm(X_diagonal_blocks - X_diagonal_blocks_ref))
            print("norm(X_lower_diagonal_blocks - X_lower_diagonal_blocks_ref): ", np.linalg.norm(X_lower_diagonal_blocks - X_lower_diagonal_blocks_ref))
            print("norm(X_arrow_bottom_blocks - X_arrow_bottom_blocks_ref):     ", np.linalg.norm(X_arrow_bottom_blocks - X_arrow_bottom_blocks_ref))
            print("norm(X_arrow_tip_block - X_arrow_tip_block_ref):             ", np.linalg.norm(X_arrow_tip_block - X_arrow_tip_block_ref))
        
    if n_iterations > 3:
        t_mean_chol = np.mean(t_list_chol[3:])
        print(f"\nAverage time chol:   {t_mean_chol:.5f}")
        
        t_mean_selInv = np.mean(t_list_selInv[3:])
        print(f"Average time selInv: {t_mean_selInv:.5f}")

 
      
     