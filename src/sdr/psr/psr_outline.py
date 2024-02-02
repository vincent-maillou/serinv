
import bsparse as bsp

import numpy as np



def extract_bridges()
    # 


def top_factorization()
    
    last_block_update = np.zeros((blocksize, blocksize))

    for i in range(0, nblocks):
        A_ii_inv = np.inv(A[i, i])
        L[i+1, i] =
        U[i, i+1] =
        A[i+1, i+1] = 

        L_bot_arrow_vector[i] =
        A_bot_arrow_vector[i+1] =

        U_right_arrow_vector[i] =
        A_right_arrow_vector[i+1] =

        last_block_update += f(bot_arrow_vector[i])
        last_block_update += f(right_arrow_vector[i])

    send_block_to_last_process(last_block_update)



def mid_factorization()
    pass

def bot_factorization(ndb, ndb-partition_size, -1):
    
    for i in range():
        L[i+1, i] =
        U[i, i+1] =

        bot_arrow_vector[i] =
        right_arrow_vector[i] =

    for p in range(0, comm_world-2)
        update = receive_tip_arrowhead_update(process=p)

        A[ndb+1, ndb+1] += update


def create_reduced_system(
    A_local, 
    A_arrow_bottom, 
    A_arrow_right, 
    blocksize,
    arrow_blocksize, 
    Bridges_upper, 
    Bridges_lower, 
    local_arrow_tip_update, 
    process,
    total_num_processes
):
    
    # create empty matrix for reduced system -> (2*#process - 1)*blocksize + arrowhead_size
    size_reduced_system = (2*total_num_processes - 1) * blocksize + arrow_blocksize
    reduced_system = np.zeros(size_reduced_system, size_reduced_system)

    if process == 0:
        pass
        reduced_system[:blocksize, :blocksize] = A_local[-blocksize:, -blocksize:]
        reduced_system[:blocksize, blocksize:2*blocksize]  = Bridges_upper[process, :, :]
        
        reduced_system[-arrow_blocksize:, :blocksize] = A_arrow_bottom[arrow_blocksize, -blocksize:]
        reduced_system[:blocksize, -arrow_blocksize:] = A_arrow_right[arrow_blocksize, -blocksize:]
        
        # send with MPI_Allgather
        
    else:
        
        # process
        start_index = blocksize + (process - 1) * 2 * blocksize
        reduced_system[start_index : start_index + blocksize, start_index - blocksize : start_index] = Bridges_lower[process]
        reduced_system[start_index : start_index + blocksize, start_index : start_index + blocksize] = A_local[:blocksize, :blocksize]
        reduced_system[start_index : start_index + blocksize, start_index + blocksize : start_index + 2 * blocksize] = A_local[:blocksize, -blocksize:]
        
        reduced_system[start_index + blocksize : start_index + 2 * blocksize, start_index : start_index + blocksize] = A_local[-blocksize : , : blocksize]
        reduced_system[start_index + blocksize : start_index + 2 * blocksize, start_index + blocksize : start_index + 2 * blocksize] = A_local[-blocksize : , -blocksize : ]
        reduced_system[start_index + blocksize : start_index + 2 * blocksize, start_index + 2 * blocksize : start_index + 3 * blocksize] = Bridges_upper[process, :, :]
        
        reduced_system[-arrow_blocksize:, start_index : start_index + blocksize] = A_arrow_bottom[:, :blocksize]
        reduced_system[-arrow_blocksize:, start_index + blocksize : start_index + 2*blocksize] = A_arrow_bottom[:, -blocksize:]
        
        reduced_system[start_index : start_index + blocksize, -arrow_blocksize:] = A_arrow_right[:blocksize, :]
        reduced_system[start_index + blocksize : start_index + 2*blocksize, -arrow_blocksize:] = A_arrow_right[-blocksize:, :]
        
        # send with MPI_Allgather

    # all processes 

def reduce_system_solve()
    pass


def top_solve()
    pass

def mid_solve()
    pass

def bot_solve()
    pass


def psr_arrowhead()
    
    if comm_rank == 0:
        top_factorization()
    if comm_rank == comm_size-1:
        bot_factorization()
    else:
        mid_factorization()

    send_reduce_system()

    reduce_system_solve()

    scatter_back_reduced_system()

    if comm_rank == 0:
        top_solve()
    if comm_rank == comm_size-1:
        bot_solve()
    else:
        mid_solve()