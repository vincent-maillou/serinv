from sdr.utils import matrix_generation
from sdr.lu import lu_dcmp_tridiag_arrowhead
from sdr.lu import lu_sinv_tridiag_arrowhead


import numpy as np
import matplotlib.pyplot as plt

def get_partitions_indices(
    n_partitions: int,
    total_size: int,
    partitions_distribution: list = None,
) -> [[], [], []]:
    """Create the partitions start/end indices and sizes for the entire problem.
    If the problem size doesn't match a perfect partitioning w.r.t the distribution,
    partitions will be resized starting from the first one.

    Parameters
    ----------
    n_partitions : int
        Total number of partitions.
    total_size : int
        Total number of blocks in the global matrix. Equal to the sum of the sizes of
        all partitions.
    partitions_distribution : list, optional
        Distribution of the partitions sizes, in percentage. The default is None
        and a uniform distribution is assumed.

    Returns
    -------
    start_blockrows : []
        List of the indices of the first blockrow of each partition in the
        global matrix.
    partition_sizes : []
        List of the sizes of each partition.
    end_blockrows : []
        List of the indices of the last blockrow of each partition in the
        global matrix.

    """

    if n_partitions > total_size:
        raise ValueError(
            "Number of partitions cannot be greater than the total size of the matrix."
        )

    if partitions_distribution is not None:
        if n_partitions != len(partitions_distribution):
            raise ValueError(
                "Number of partitions and number of entries in the distribution list do not match."
            )
        if sum(partitions_distribution) != 100:
            raise ValueError(
                "Sum of the entries in the distribution list is not equal to 100."
            )
    else:
        partitions_distribution = [100 / n_partitions] * n_partitions

    partitions_distribution = np.array(partitions_distribution) / 100

    start_blockrows = []
    partition_sizes = []
    end_blockrows = []

    for i in range(n_partitions):
        partition_sizes.append(round(partitions_distribution[i] * total_size))

    if sum(partition_sizes) != total_size:
        diff = total_size - sum(partition_sizes)
        for i in range(diff):
            partition_sizes[i] += 1

    for i in range(n_partitions):
        start_blockrows.append(sum(partition_sizes[:i]))
        end_blockrows.append(start_blockrows[i] + partition_sizes[i])

    return start_blockrows, partition_sizes, end_blockrows


def extract_partition(
    A_global: np.ndarray,
    start_blockrow: int, 
    partition_size: int,
    blocksize: int,
    arrow_blocksize: int,
):
    
    A_local = np.zeros((partition_size*blocksize, partition_size*blocksize), dtype=A_global.dtype)
    A_arrow_bottom = np.zeros((arrow_blocksize, partition_size*arrow_blocksize), dtype=A_global.dtype)
    A_arrow_right = np.zeros((partition_size*arrow_blocksize, arrow_blocksize), dtype=A_global.dtype)

    stop_blockrow = start_blockrow + partition_size

    A_local = A_global[start_blockrow*blocksize:stop_blockrow*blocksize, start_blockrow*blocksize:stop_blockrow*blocksize]
    A_arrow_bottom = A_global[-arrow_blocksize:-1, start_blockrow*blocksize:stop_blockrow*blocksize]
    A_arrow_right  = A_global[start_blockrow*blocksize:stop_blockrow*blocksize, -arrow_blocksize:-1]

    return A_local, A_arrow_bottom, A_arrow_right


def top_factorize(
    A_local: np.ndarray,
    A_arrow_bottom: np.ndarray, 
    A_arrow_right: np.ndarray,
    blocksize: int,
    arrowhead_blocksize: int,
) -> [np.ndarray, np.ndarray, np.ndarray]:

    LU_local = np.zeros_like(A_local)
    L_arrow_bottom = np.zeros_like(A_arrow_bottom)
    U_arrow_right = np.zeros_like(A_arrow_right)

    nblocks = A_local.shape[0] // blocksize

    for i in range(1, nblocks):
        # L[i, i-1] = A[i, i-1] @ A[i-1, i-1]^(-1)
        LU_local[i * blocksize : (i + 1) * blocksize, (i-1) * blocksize : i * blocksize] = A_local[i * blocksize : (i + 1) * blocksize, (i-1) * blocksize : i * blocksize] @ np.linalg.inv(A_local[(i-1) * blocksize : i * blocksize, (i-1) * blocksize : i * blocksize])
        #LU_local[i * blocksize : (i + 1) * blocksize, (i-1) * blocksize : i * blocksize] = np.linalg.solve(A_local[i * blocksize : (i + 1) * blocksize, (i-1) * blocksize : i * blocksize].T, A_local[(i-1) * blocksize : i * blocksize, (i-1) * blocksize : i * blocksize].T).T

        
        # U[i-1, i] = A[i-1, i-1]^(-1) @ A[i-1, i]
        LU_local[(i-1) * blocksize : i * blocksize, i * blocksize : (i+1) * blocksize]   = np.linalg.inv(A_local[(i-1) * blocksize : i * blocksize, (i-1) * blocksize : i * blocksize]) @ A_local[(i-1) * blocksize : i * blocksize, i * blocksize : (i+1) * blocksize]
        #LU_local[(i-1) * blocksize : i * blocksize, i * blocksize : (i+1) * blocksize]   = np.linalg.solve(A_local[(i-1) * blocksize : i * blocksize, (i-1) * blocksize : i * blocksize], A_local[(i-1) * blocksize : i * blocksize, i * blocksize : (i+1) * blocksize])

        # A_{i+1, ndb+1} = A_{i+1, ndb+1} - L_{i+1, i} @ U_{i, ndb+1}

        # A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, i} @ U_{i, ndb+1}

        # A[i, i] = A[i, i] - L[i, i-1] @ A[i-1, i]
        A_local[i * blocksize : (i+1) * blocksize, i * blocksize : (i+1) * blocksize]    = A_local[i * blocksize : (i+1) * blocksize, i * blocksize : (i+1) * blocksize] - LU_local[i * blocksize : (i + 1) * blocksize, (i-1) * blocksize : i * blocksize] @ A_local[(i-1) * blocksize : i * blocksize, i * blocksize : (i+1) * blocksize]


    return A_local, LU_local, L_arrow_bottom, U_arrow_right



def middle_factorize(
    A_local: np.ndarray,
    A_arrow_bottom: np.ndarray,
    A_arrow_right: np.ndarray,
    blocksize: int,
    arrowhead_blocksize: int,
) -> [np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    LU_local = np.zeros_like(A_local)
    L_arrow_bottom = np.zeros_like(A_arrow_bottom)
    U_arrow_right = np.zeros_like(A_arrow_right)

    n_blocks = A_local.shape[0] // blocksize

    for i in range(2, n_blocks):

        top = slice(0, blocksize)
        im1 = slice((i - 1) * blocksize, i * blocksize)
        i = slice(i * blocksize, (i + 1) * blocksize)

        A_im1im1_inv = np.linalg.inv(A_local[im1, im1])

        # L[i, i-1] = A[i, i-1] @ A[i-1, i-1]^(-1)
        LU_local[i, im1] = A_local[i, im1] @ A_im1im1_inv

        # L[top, i-1] = A[top, i-1] @ A[i-1, i-1]^(-1)
        LU_local[top, im1] = A_local[top, im1] @ A_im1im1_inv

        # U[i-1, i] = A[i-1, i-1]^(-1) @ A[i-1, i]
        LU_local[im1, i] = A_im1im1_inv @ A_local[im1, i]

        # U[i-1, top] = A[i-1, i-1]^(-1) @ A[i-1, top]
        LU_local[im1, top] = A_im1im1_inv @ A_local[im1, top]

        # A_local[i, i] = A[i, i] - L[i, i-1] @ A_local[i-1, i]
        A_local[i, i] = A_local[i, i] - LU_local[i, im1] @ A_local[im1, i]

        # A_local[top, top] = A[top, top] - L[top, i-1] @ A_local[i-1, top]
        A_local[top, top] = A_local[top, top] - LU_local[top, im1] @ A_local[im1, top]

        # A_local[i, top] = - L[i, i-1] @ A_local[i-1, top]
        A_local[i, top] = - LU_local[i, im1] @ A_local[im1, top]

        # A_local[top, i] = - L[top, i-1] @ A_local[i-1, i]
        A_local[top, i] = - LU_local[top, im1] @ A_local[im1, i]

    return A_local, LU_local, L_arrow_bottom, U_arrow_right


def create_reduced_system(
    A_local, 
    A_arrow_bottom, 
    A_arrow_right, 
    blocksize,
    arrow_blocksize, 
    Bridges_upper, 
    Bridges_lower, 
    A_global_arrow_tip,
    local_arrow_tip_update, 
    process,
    total_num_processes
):
    
    # create empty matrix for reduced system -> (2*#process - 1)*blocksize + arrowhead_size
    size_reduced_system = (2*total_num_processes - 1) * blocksize + arrow_blocksize
    reduced_system = np.zeros(size_reduced_system, size_reduced_system)
    reduced_system[-arrow_blocksize:, -arrow_blocksize:] = local_arrow_tip_update

    if process == 0:
        pass
        reduced_system[:blocksize, :blocksize] = A_local[-blocksize:, -blocksize:]
        reduced_system[:blocksize, blocksize:2*blocksize]  = Bridges_upper[process, :, :]
        
        reduced_system[-arrow_blocksize:, :blocksize] = A_arrow_bottom[arrow_blocksize, -blocksize:]
        reduced_system[:blocksize, -arrow_blocksize:] = A_arrow_right[arrow_blocksize, -blocksize:]
        
        # send with MPI_Allgather
        
    else:
        
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
        
        # send with MPI_AllReduce

    # all processes 
    
    reduced_system[-arrow_blocksize:, -arrow_blocksize:] = A_global_arrow_tip
    
    return reduced_system
    

def inverse_reduced_system(reduced_system, diag_blocksize, arrowhead_blocksize):
    
    #n_blocks = (reduced_system.shape[0] - arrowhead_blocksize) // diag_blocksize
    
    L_reduced, U_reduced = lu_dcmp_tridiag_arrowhead(reduced_system, diag_blocksize, arrowhead_blocksize)
    S_reduced  = lu_sinv_tridiag_arrowhead(L_reduced,  U_reduced, diag_blocksize, arrowhead_blocksize)
    
    return S_reduced

def update_sinv_reduced_system():
    
    # revert create_reduced_system() -> write into S_local but use indices from A_local ...
    pass

def top_sinv( 
    A_local: np.ndarray,
    A_arrow_bottom: np.ndarray, 
    A_arrow_right: np.ndarray,
    LU_local: np.ndarray, 
    L_arrow_bottom: np.ndarray, 
    U_arrow_right: np.ndarray, 
    S_local: np.ndarray, 
    S_arrow_bottom: np.ndarray, 
    S_arrow_right: np.ndarray,
    blocksize: int,
    arrowhead_blocksize: int
) -> [np.ndarray]:
        
    n_blocks = A_local.shape[0] // blocksize
    
    # for now initialize S_local[nblocks, nblocks] = inv(A_local[nblocks, nblocks])
    #S_local[(n_blocks - 1)*blocksize:n_blocks*blocksize, (n_blocks - 1)*blocksize:n_blocks*blocksize] = np.linalg.inv(A_local[(n_blocks - 1)*blocksize:n_blocks*blocksize, (n_blocks - 1)*blocksize:n_blocks*blocksize])

    for i in range(n_blocks-1, 0, -1):
        # S[i,i-1] = - S[i,i] @ L[i,i-1]
        S_local[i*blocksize: (i+1)*blocksize, (i-1)*blocksize: i*blocksize] = - S_local[i*blocksize: (i+1)*blocksize, i*blocksize: (i+1)*blocksize] @ LU_local[i*blocksize: (i+1)*blocksize, (i-1)*blocksize: i*blocksize]
       
        # S[i-1,i] = - U[i-1,i] @ S[i,i]
        S_local[(i-1)*blocksize: i*blocksize, i*blocksize: (i+1)*blocksize] = - LU_local[(i-1)*blocksize: i*blocksize, i*blocksize: (i+1)*blocksize] @ S_local[i*blocksize: (i+1)*blocksize, i*blocksize: (i+1)*blocksize]
        
        # S[i-1,i-1] = inv(A[i-1,i-1]) - U[i-1, i] @ S[i,i-1]
        S_local[(i-1)*blocksize: i*blocksize, (i-1)*blocksize: i*blocksize] = np.linalg.inv(A_local[(i-1)*blocksize: i*blocksize, (i-1)*blocksize: i*blocksize]) - LU_local[(i-1)*blocksize: i*blocksize, i*blocksize: (i+1)*blocksize] @ S_local[i*blocksize: (i+1)*blocksize, (i-1)*blocksize: i*blocksize]
    
    return S_local, S_arrow_bottom, S_arrow_right

def middle_sinv(
    A_local: int, 
    LU_local: int, 
    L_arrow_bottom: int, 
    U_arrow_right: int,
    blocksize: int,
    arrowhead_blocksize: int,
) -> [np.ndarray, np.ndarray, np.ndarray]:
    
    S_local = np.zeros_like(A_local)
    S_arrow_bottom = np.zeros_like(L_arrow_bottom)
    S_arrow_right = np.zeros_like(U_arrow_right)

    n_blocks = A_local.shape[0] // blocksize

    # S_local[bot, bot-1] = - S_local[bot, top] @ L[top, bot-1] - S_local[bot, bot] @ L[bot, bot-1]

    # S_local[bot-1, bot] = - U[bot-1, bot] @ S_local[bot, bot] - U[bot-1, top] @ S_local[top, bot]

    for i in range(n_blocks-2, 1, -1):
        # S_local[top, i] = - S_local[top, top] @ L[top, i] - S_local[top, i+1] @ L[i+1, i]

        # S_local[i, top] = - U[i, i+1] @ S_local[i+1, top] - U[i, top] @ S_local[top, top]
        
        pass

    for i in range(n_blocks-2, 2, -1):
        # S_local[i, i] = np.linalg.inv(A_local[i, i]) - U[i, top] @ S_local[top, i] - U[i, i+1] @ S_local[i+1, i]

        # S_local[i-1, i] = - U[i-1, top] @ S_local[top, i] - U[i-1, i] @ S_local[i, i]

        # S_local[i, i-1] = - S_local[i, top] @ L[top, i-1] - S_local[i, i] @ L[i, i-1]
        
        pass

    # S_local[top+1, top+1] = np.linalg.inv(A_local[top+1, top+1]) - U[top+1, top] @ S_local[top, top+1] - U[top+1, top+2] @ S_local[top+2, top+1]

    return S_local, S_arrow_bottom, S_arrow_right

import copy as cp

from sdr.lu.lu_decompose import lu_dcmp_tridiag
from sdr.lu.lu_selected_inversion import lu_sinv_tridiag


def psr_arrowhead(
    A_local: np.ndarray,
    A_arrow_bottom: np.ndarray, 
    A_arrow_right: np.ndarray,
    blocksize: int,
    arrowhead_blocksize: int,
    process: int,
):

    if process == 0:
        A_local, LU_local, L_arrow_bottom, U_arrow_right = top_factorize(A_local, A_arrow_bottom, A_arrow_right, blocksize, arrowhead_blocksize)
    else:
        A_local, LU_local, L_arrow_bottom, U_arrow_right = middle_factorize(A_local, A_arrow_bottom, A_arrow_right, blocksize, arrowhead_blocksize)

        
        S_local, S_arrow_bottom, S_arrow_right = middle_sinv(A_local, LU_local, L_arrow_bottom, U_arrow_right, blocksize, arrowhead_blocksize)

       
    return S_local, S_arrow_bottom, S_arrow_right



if __name__ == "__main__":
    nblocks = 30
    diag_blocksize = 3
    arrow_blocksize = 2
    symmetric = False
    diagonal_dominant = True
    seed = 63

    A = matrix_generation.generate_blocktridiag_arrowhead(
        nblocks, diag_blocksize, arrow_blocksize, symmetric, diagonal_dominant, 
        seed
    ) 

    # plt.matshow(A)
    # plt.title("A inital matrix")
 
    n_partitions = 3

    start_blockrows, partition_sizes, end_blockrows = get_partitions_indices(n_partitions=n_partitions, total_size=nblocks-1)


    for process in range(0, n_partitions):
        if process == 0:
            A_local, A_arrow_bottom, A_arrow_right = extract_partition(A, start_blockrows[process], partition_sizes[process], diag_blocksize, arrow_blocksize)

            psr_arrowhead(A_local, A_arrow_bottom, A_arrow_right, diag_blocksize, arrow_blocksize, process)

        if process == 1:
            A_local, A_arrow_bottom, A_arrow_right = extract_partition(A, start_blockrows[process], partition_sizes[process], diag_blocksize, arrow_blocksize)

            psr_arrowhead(A_local, A_arrow_bottom, A_arrow_right, diag_blocksize, arrow_blocksize, process)

        """ plt.matshow(A_local)
        plt.title("A_local process: " + str(process))

        plt.matshow(A_arrow_bottom)
        plt.title("A_arrow_bottom process: " + str(process))

        plt.matshow(A_arrow_right)
        plt.title("A_arrow_right process: " + str(process))
        plt.show()  """