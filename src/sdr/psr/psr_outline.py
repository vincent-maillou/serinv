
import bsparse as bsp

import numpy as np






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