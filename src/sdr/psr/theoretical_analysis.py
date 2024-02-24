import sympy as sp
import numpy as np
import matplotlib.pyplot as plt


def getcost_bta_seq_factorization(
    n_blocks, 
    diag_blocksize,
    arrowhead_blocksize
):
    lu_cost = n_blocks * diag_blocksize**3 + arrowhead_blocksize**3
    triangular_solve_cost = 2 * n_blocks * diag_blocksize**2
    matmult_cost = 3 * (n_blocks - 1) * diag_blocksize**3 + 4 * (n_blocks - 1) * diag_blocksize**2 * arrowhead_blocksize + (n_blocks - 1) * diag_blocksize * arrowhead_blocksize**2
    addsub_cost = (n_blocks - 1) * diag_blocksize**2 + (n_blocks - 1) * diag_blocksize * arrowhead_blocksize + (n_blocks - 1) * arrowhead_blocksize**2

    bta_seq_factorization = lu_cost + triangular_solve_cost + matmult_cost + addsub_cost

    return bta_seq_factorization

def getcost_bta_seq_sinv(
    n_blocks, 
    diag_blocksize,
    arrowhead_blocksize
):
    triangular_solve_cost = 2 * n_blocks * diag_blocksize**2
    matmult_cost = (6 * (n_blocks - 1) + 3) * diag_blocksize**3 + 7 * (n_blocks - 1) * diag_blocksize**2 * arrowhead_blocksize + 2 * n_blocks * diag_blocksize * arrowhead_blocksize**2
    addsub_cost = (4 * (n_blocks - 1) + 1) * diag_blocksize**2 + 2 * (n_blocks - 1) * diag_blocksize * arrowhead_blocksize

    bta_seq_sinv = triangular_solve_cost + matmult_cost + addsub_cost

    return bta_seq_sinv

def get_partition_size(
    n_blocks,
    n_partitions,
):
    n_blocks_partition = n_blocks // n_partitions
    return n_blocks_partition

def get_reduced_system_size(
    n_partitions,
):
    reduced_system_size = 2*(n_partitions - 1)
    return reduced_system_size

def getcost_bta_middle_dist_factorization(
    n_blocks_partition,
    diag_blocksize,
    arrowhead_blocksize,
):
    lu_cost = n_blocks_partition * diag_blocksize**3
    triangular_solve = n_blocks_partition * diag_blocksize**2
    matmult_cost = (8 * (n_blocks_partition - 2) + 4) * diag_blocksize**3 + (6 * (n_blocks_partition - 2) + 4) * diag_blocksize**2 + arrowhead_blocksize + (n_blocks_partition - 2) * diag_blocksize + arrowhead_blocksize**2
    addsub_cost = 2 * (n_blocks_partition - 2) * diag_blocksize**2 + 4 * (n_blocks_partition - 2) * diag_blocksize * arrowhead_blocksize + (n_blocks_partition - 2) * arrowhead_blocksize**2

    bta_middle_dist_factorization = lu_cost + triangular_solve + matmult_cost + addsub_cost
    return bta_middle_dist_factorization

def getcost_bta_middle_dist_sinv(
    n_blocks_partition,
    diag_blocksize,
    arrowhead_blocksize,
):  
    triangular_solve_cost = 2 * (n_blocks_partition - 1) * diag_blocksize**2
    matmult_cost = 15 * (n_blocks_partition - 1) * diag_blocksize**3 + 11 * (n_blocks_partition - 1) * diag_blocksize**2 * arrowhead_blocksize + 2 * (n_blocks_partition - 1) * diag_blocksize * arrowhead_blocksize**2
    addsub_cost = 11 *(n_blocks_partition - 1) * diag_blocksize**2 + 4 *(n_blocks_partition - 1) * diag_blocksize * arrowhead_blocksize

    bta_middle_dist_sinv = triangular_solve_cost + matmult_cost + addsub_cost
    return bta_middle_dist_sinv

""" if __name__ == "__main__":
    # Define the symbols
    n_blocks = sp.symbols('n_blocks')
    diag_blocksize, arrowhead_blocksize = sp.symbols('diag_blocksize arrowhead_blocksize')
    n_partitions = sp.symbols('n_partitions')


    # Compute the cost of the BTA sequential algorithm
    cost_bta_seq_factorization = getcost_bta_seq_factorization(n_blocks, diag_blocksize, arrowhead_blocksize)
    cost_bta_seq_sinv = getcost_bta_seq_sinv(n_blocks, diag_blocksize, arrowhead_blocksize)

    cost_bta_seq = sp.simplify(cost_bta_seq_factorization)

    print("cost_bta_dist: \n", cost_bta_seq)


    # Compute the cost of the BTA distributed algorithm
    n_blocks_partition = get_partition_size(n_blocks, n_partitions)
    reduced_system_size = get_reduced_system_size(n_partitions)

    cost_bta_dist_middle_process_factorization = getcost_bta_middle_dist_factorization(n_blocks_partition, diag_blocksize, arrowhead_blocksize)
    cost_bta_dist_reduced_system_solve = getcost_bta_seq_factorization(reduced_system_size, diag_blocksize, arrowhead_blocksize) + getcost_bta_seq_sinv(reduced_system_size, diag_blocksize, arrowhead_blocksize)
    cost_bta_dist_middle_process_sinv = getcost_bta_middle_dist_sinv(n_blocks_partition, diag_blocksize, arrowhead_blocksize)

    cost_bta_dist = sp.simplify(cost_bta_dist_middle_process_factorization + cost_bta_dist_reduced_system_solve + cost_bta_dist_middle_process_sinv)

    print("cost_bta_dist: \n", cost_bta_dist) """



""" if __name__ == "__main__":
    n_blocks = 250
    diag_blocksize = 4000
    arrowhead_blocksize = 4

    max_partitions = 128

    costs_bta_seq = []
    costs_bta_dist = []
    costs_bta_seq_over_dist = []

    for n_partitions in range(1, max_partitions):
        # Compute the cost of the BTA sequential algorithm
        cost_bta_seq_factorization = getcost_bta_seq_factorization(n_blocks, diag_blocksize, arrowhead_blocksize)
        cost_bta_seq_sinv = getcost_bta_seq_sinv(n_blocks, diag_blocksize, arrowhead_blocksize)

        cost_bta_seq = sp.simplify(cost_bta_seq_factorization)

        # print("cost_bta_dist: \n", cost_bta_seq)
        costs_bta_seq.append(cost_bta_seq)


        # Compute the cost of the BTA distributed algorithm
        n_blocks_partition = get_partition_size(n_blocks, n_partitions)
        reduced_system_size = get_reduced_system_size(n_partitions)

        cost_bta_dist_middle_process_factorization = getcost_bta_middle_dist_factorization(n_blocks_partition, diag_blocksize, arrowhead_blocksize)
        cost_bta_dist_reduced_system_solve = getcost_bta_seq_factorization(reduced_system_size, diag_blocksize, arrowhead_blocksize) + getcost_bta_seq_sinv(reduced_system_size, diag_blocksize, arrowhead_blocksize)
        cost_bta_dist_middle_process_sinv = getcost_bta_middle_dist_sinv(n_blocks_partition, diag_blocksize, arrowhead_blocksize)

        cost_bta_dist = sp.simplify(cost_bta_dist_middle_process_factorization + cost_bta_dist_reduced_system_solve + cost_bta_dist_middle_process_sinv)

        # print("cost_bta_dist: \n", cost_bta_dist)
        costs_bta_dist.append(cost_bta_dist)

        costs_bta_seq_over_dist.append(cost_bta_seq / cost_bta_dist)

    plt.plot(range(1, max_partitions), costs_bta_seq, label="BTA Sequential")
    plt.plot(range(1, max_partitions), costs_bta_dist, label="BTA Distributed")
    plt.xlabel("Number of Partitions")
    plt.ylabel("Cost")
    plt.title("BTA Sequential vs BTA Distributed")
    plt.legend()
    plt.show()

    plt.plot(range(1, max_partitions), costs_bta_seq_over_dist, label="BTA Sequential / BTA Distributed")
    plt.xlabel("Number of Partitions")
    plt.ylabel("Cost ratio")
    plt.title("BTA Sequential vs BTA Distributed")
    plt.legend()
    plt.show() """

    
if __name__ == "__main__":
    diag_blocksize = 4000
    arrowhead_blocksize = 4

    max_n_blocks = 3000
    max_partitions = 128

    cost_matrix = np.zeros((max_partitions, max_n_blocks))

    for n_blocks in range(1, max_n_blocks):
        for n_partitions in range(1, max_partitions):
            # Compute the cost of the BTA sequential algorithm
            cost_bta_seq_factorization = getcost_bta_seq_factorization(n_blocks, diag_blocksize, arrowhead_blocksize)
            cost_bta_seq_sinv = getcost_bta_seq_sinv(n_blocks, diag_blocksize, arrowhead_blocksize)

            cost_bta_seq = sp.simplify(cost_bta_seq_factorization)


            # Compute the cost of the BTA distributed algorithm
            n_blocks_partition = get_partition_size(n_blocks, n_partitions)
            reduced_system_size = get_reduced_system_size(n_partitions)

            # Standard algorithm analysis
            cost_bta_dist_middle_process_factorization = getcost_bta_middle_dist_factorization(n_blocks_partition, diag_blocksize, arrowhead_blocksize)
            cost_bta_dist_reduced_system_solve = getcost_bta_seq_factorization(reduced_system_size, diag_blocksize, arrowhead_blocksize) + getcost_bta_seq_sinv(reduced_system_size, diag_blocksize, arrowhead_blocksize)
            cost_bta_dist_middle_process_sinv = getcost_bta_middle_dist_sinv(n_blocks_partition, diag_blocksize, arrowhead_blocksize)
            cost_bta_dist = sp.simplify(cost_bta_dist_middle_process_factorization + cost_bta_dist_reduced_system_solve + cost_bta_dist_middle_process_sinv)


            """ # Divid and conquer algorithm aproach for the reduced system inversion
            cost_bta_dist_middle_process_factorization = getcost_bta_middle_dist_factorization(n_blocks_partition, diag_blocksize, arrowhead_blocksize)
            
            blocks_per_process = 3
            n_processes_inverting_reduced_system = reduced_system_size // blocks_per_process
            reduced_system_sub_partition_size = get_reduced_system_size(n_processes_inverting_reduced_system)
            cost_bta_dist_reduced_system_solve = getcost_bta_middle_dist_factorization(blocks_per_process, diag_blocksize, arrowhead_blocksize) + getcost_bta_middle_dist_sinv(blocks_per_process, diag_blocksize, arrowhead_blocksize)
            cost_bta_dist_reduced_system_solve += getcost_bta_seq_factorization(reduced_system_sub_partition_size, diag_blocksize, arrowhead_blocksize) + getcost_bta_seq_sinv(reduced_system_sub_partition_size, diag_blocksize, arrowhead_blocksize)
            
            cost_bta_dist_middle_process_sinv = getcost_bta_middle_dist_sinv(n_blocks_partition, diag_blocksize, arrowhead_blocksize)
            
            cost_bta_dist = sp.simplify(cost_bta_dist_middle_process_factorization + cost_bta_dist_reduced_system_solve + cost_bta_dist_middle_process_sinv) """



            cost_matrix[n_partitions, n_blocks] = cost_bta_seq / cost_bta_dist


    print(f"Bench up to: {max_n_blocks} blocks and {max_partitions} partitions")
    print(f"diagonal_blocksize {diag_blocksize}, arrowhead_blocksize {arrowhead_blocksize}")



    # Extract the trend of the break-even point
    break_even_points = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    
    optimal_n_blocks = np.ndarray((len(break_even_points), max_partitions))
    
    for i, break_even_point in enumerate(break_even_points):
        for p in range(max_partitions):
            optimal_n_blocks[i, p] = np.argmax(cost_matrix[p, :] >= break_even_point)
        
    # Plot the break-even points, excluding zero values
    for i, break_even_point in enumerate(break_even_points):
        non_zero_indices = np.where(optimal_n_blocks[i] != 0)[0]  # Find indices of non-zero values
        plt.plot(non_zero_indices, optimal_n_blocks[i, non_zero_indices], label=f"break-even ratio >= {break_even_point}")
        
    plt.title("Number of blocks from which each processes of BTA-dist are doing less than the break-even ration work then BTA-seq")
    plt.xlabel("Number of processes")
    plt.ylabel("Number of Blocks")
    plt.legend()
    plt.show()
    

    
    # Create the figure and axes object
    fig, ax = plt.subplots()

    # Display the image
    im = ax.imshow(cost_matrix, cmap='inferno', interpolation='nearest')


    # Set labels and title
    ax.set_title(f"Ratio of the computational cost of BTA-seq over BTA-dist \n diag_blocksize = {diag_blocksize} \n arrowhead_blocksize = {arrowhead_blocksize}")
    
    ax.set_xlabel("Number of Blocks")
    ax.set_ylabel("Number of processes")
    
    # Add colorbar
    fig.colorbar(im)
    #fig.colorbar(im, label='Cost Ratio', ticks=[0.5, 1.0, 1.5], format='%0.2f')

    # Optional: Customize ticks (uncomment if needed)
    # ax.set_xticks(np.arange(0, max_n_blocks, 250))
    # ax.set_yticks(np.arange(0, max_partitions, 16))

    # Show the plot
    plt.show()


