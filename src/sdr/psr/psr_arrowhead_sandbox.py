from sdr.utils import matrix_generation


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


if __name__ == "__main__":
    nblocks = 11
    diag_blocksize = 3
    arrow_blocksize = 2
    symmetric = False
    diagonal_dominant = True
    seed = 63

    A = matrix_generation.generate_blocktridiag_arrowhead(
        nblocks, diag_blocksize, arrow_blocksize, symmetric, diagonal_dominant, 
        seed
    ) 

    plt.matshow(A)
 
    n_partitions = 3

    start_blockrows, partition_sizes, end_blockrows = get_partitions_indices(n_partitions=n_partitions, total_size=nblocks-1)


    for process in range(0, n_partitions):
        A_local, A_arrow_bottom, A_arrow_right = extract_partition(A, start_blockrows[process], partition_sizes[process], diag_blocksize, arrow_blocksize)

        plt.matshow(A_local)
        plt.title("A_local process: " + str(process))

        plt.matshow(A_arrow_bottom)
        plt.title("A_arrow_bottom process: " + str(process))

        plt.matshow(A_arrow_right)
        plt.title("A_arrow_right process: " + str(process))
        plt.show() 

