import sympy as sp
import numpy as np
import matplotlib.pyplot as plt


# ----- BTA-sequential -----
def getmem_bta_seq_input(
    n_blocks, 
    diag_blocksize,
    arrowhead_blocksize
):
    diag_blk_mem = n_blocks * diag_blocksize**2
    lower_blk_mem = (n_blocks-1) * diag_blocksize**2
    upper_blk_mem = (n_blocks-1) * diag_blocksize**2
    arrow_lower_blk_mem = n_blocks * diag_blocksize * arrowhead_blocksize
    arrow_right_blk_mem = n_blocks * diag_blocksize * arrowhead_blocksize
    arrow_tip_blk_mem = arrowhead_blocksize**2

    bta_seq_input = diag_blk_mem + lower_blk_mem + upper_blk_mem + arrow_lower_blk_mem + arrow_right_blk_mem + arrow_tip_blk_mem

    return bta_seq_input


def getmem_bta_seq_factorization(
    n_blocks, 
    diag_blocksize,
    arrowhead_blocksize
):
    l_diag_blk_mem = n_blocks * diag_blocksize**2
    l_lower_blk_mem = (n_blocks-1) * diag_blocksize**2
    l_arrow_blk_mem = n_blocks * diag_blocksize * arrowhead_blocksize
    l_arrow_tip_blk_mem = arrowhead_blocksize**2
    
    u_diag_blk_mem = n_blocks * diag_blocksize**2
    u_lower_blk_mem = (n_blocks-1) * diag_blocksize**2
    u_arrow_blk_mem = n_blocks * diag_blocksize * arrowhead_blocksize
    u_arrow_tip_blk_mem = arrowhead_blocksize**2

    lu_inversion_buffers = 2 * diag_blocksize**2

    bta_seq_factorization = l_diag_blk_mem + l_lower_blk_mem + l_arrow_blk_mem + l_arrow_tip_blk_mem + u_diag_blk_mem + u_lower_blk_mem + u_arrow_blk_mem + u_arrow_tip_blk_mem + lu_inversion_buffers

    return bta_seq_factorization


def getmem_bta_seq_sinv(
    n_blocks, 
    diag_blocksize,
    arrowhead_blocksize
):
    diag_blk_mem = n_blocks * diag_blocksize**2
    lower_blk_mem = (n_blocks-1) * diag_blocksize**2
    upper_blk_mem = (n_blocks-1) * diag_blocksize**2
    arrow_lower_blk_mem = n_blocks * diag_blocksize * arrowhead_blocksize
    arrow_right_blk_mem = n_blocks * diag_blocksize * arrowhead_blocksize
    arrow_tip_blk_mem = arrowhead_blocksize**2
    
    lu_inversion_buffers = 2 * diag_blocksize**2
    
    bta_seq_sinv = diag_blk_mem + lower_blk_mem + upper_blk_mem + arrow_lower_blk_mem + arrow_right_blk_mem + arrow_tip_blk_mem + lu_inversion_buffers

    return bta_seq_sinv


# ----- BTA-distributed -----

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


def getmem_bta_dist_input(
    n_blocks_partition, 
    diag_blocksize,
    arrowhead_blocksize
):
    diag_blk_mem = n_blocks_partition * diag_blocksize**2
    lower_blk_mem = n_blocks_partition * diag_blocksize**2
    upper_blk_mem = n_blocks_partition * diag_blocksize**2
    arrow_lower_blk_mem = n_blocks_partition * diag_blocksize * arrowhead_blocksize
    arrow_right_blk_mem = n_blocks_partition * diag_blocksize * arrowhead_blocksize
    arrow_tip_blk_mem = arrowhead_blocksize**2

    bta_dist_input = diag_blk_mem + lower_blk_mem + upper_blk_mem + arrow_lower_blk_mem + arrow_right_blk_mem + arrow_tip_blk_mem

    return bta_dist_input


def getmem_bta_dist_factorization(
    n_blocks_partition, 
    diag_blocksize,
    arrowhead_blocksize
):
    l_diag_blk_mem = n_blocks_partition * diag_blocksize**2
    l_lower_blk_mem = (n_blocks_partition-1) * diag_blocksize**2
    l_arrow_blk_mem = n_blocks_partition * diag_blocksize * arrowhead_blocksize
    l_2sided_blk_mem = (n_blocks_partition - 1) * diag_blocksize**2
    
    u_diag_blk_mem = n_blocks_partition * diag_blocksize**2
    u_lower_blk_mem = (n_blocks_partition-1) * diag_blocksize**2
    u_arrow_blk_mem = n_blocks_partition * diag_blocksize * arrowhead_blocksize
    u_2sided_blk_mem = (n_blocks_partition - 1) * diag_blocksize**2
    
    arrow_tip_accumulate_blk_mem = arrowhead_blocksize**2
    lu_inversion_buffers = 2 * diag_blocksize**2

    bta_dist_factorization = l_diag_blk_mem + l_lower_blk_mem + l_arrow_blk_mem + l_2sided_blk_mem + u_diag_blk_mem + u_lower_blk_mem + u_arrow_blk_mem + u_2sided_blk_mem + arrow_tip_accumulate_blk_mem + lu_inversion_buffers

    return bta_dist_factorization


def getmem_bta_dist_sinv(
    n_blocks_partition, 
    diag_blocksize,
    arrowhead_blocksize
):
    diag_blk_mem = n_blocks_partition * diag_blocksize**2
    lower_blk_mem = n_blocks_partition * diag_blocksize**2
    upper_blk_mem = n_blocks_partition * diag_blocksize**2
    arrow_lower_blk_mem = n_blocks_partition * diag_blocksize * arrowhead_blocksize
    arrow_right_blk_mem = n_blocks_partition * diag_blocksize * arrowhead_blocksize
    arrow_tip_blk_mem = arrowhead_blocksize**2
    
    lu_inversion_buffers = 2 * diag_blocksize**2

    bta_dist_sinv = diag_blk_mem + lower_blk_mem + upper_blk_mem + arrow_lower_blk_mem + arrow_right_blk_mem + arrow_tip_blk_mem + lu_inversion_buffers

    return bta_dist_sinv


""" if __name__ == "__main__":
    # Define the symbols
    n_blocks = sp.symbols('n_blocks')
    diag_blocksize, arrowhead_blocksize = sp.symbols('diag_blocksize arrowhead_blocksize')
    n_partitions = sp.symbols('n_partitions')


    # Compute the cost of the BTA sequential algorithm
    mem_bta_seq_input = getmem_bta_seq_input(n_blocks, diag_blocksize, arrowhead_blocksize)
    mem_bta_seq_factorization = getmem_bta_seq_factorization(n_blocks, diag_blocksize, arrowhead_blocksize)
    mem_bta_seq_sinv = getmem_bta_seq_sinv(n_blocks, diag_blocksize, arrowhead_blocksize)

    mem_bta_seq = sp.simplify(mem_bta_seq_input + mem_bta_seq_factorization + mem_bta_seq_sinv)

    print("cost_bta_dist: \n", mem_bta_seq)


    # Compute the cost of the BTA distributed algorithm
    n_blocks_partition = get_partition_size(n_blocks, n_partitions)
    reduced_system_size = get_reduced_system_size(n_partitions)
    
    mem_bta_dist_input = getmem_bta_dist_input(n_blocks_partition, diag_blocksize, arrowhead_blocksize)
    mem_bta_dist_factorization = getmem_bta_dist_factorization(n_blocks_partition, diag_blocksize, arrowhead_blocksize)
    mem_reduced_system_solve = getmem_bta_seq_factorization(reduced_system_size, diag_blocksize, arrowhead_blocksize) + getmem_bta_seq_sinv(reduced_system_size, diag_blocksize, arrowhead_blocksize)
    mem_bta_dist_sinv = getmem_bta_dist_sinv(n_blocks_partition, diag_blocksize, arrowhead_blocksize)
    
    mem_bta_dist = sp.simplify(mem_bta_dist_input + mem_bta_dist_factorization + mem_reduced_system_solve + mem_bta_dist_sinv)

    print("cost_bta_dist: \n", mem_bta_dist) """



if __name__ == "__main__":
    n_blocks = 250
    diag_blocksize = 4000
    arrowhead_blocksize = 4

    max_partitions = 128

    l_mem_bta_seq = []
    l_mem_bta_dist = []
    l_mem_bta_seq_over_dist = []

    for n_partitions in range(1, max_partitions):
        # Memory cost of the BTA sequential algorithm
        mem_bta_seq_input = getmem_bta_seq_input(n_blocks, diag_blocksize, arrowhead_blocksize)
        mem_bta_seq_factorization = getmem_bta_seq_factorization(n_blocks, diag_blocksize, arrowhead_blocksize)
        mem_bta_seq_sinv = getmem_bta_seq_sinv(n_blocks, diag_blocksize, arrowhead_blocksize)

        mem_bta_seq = sp.simplify(mem_bta_seq_input + mem_bta_seq_factorization + mem_bta_seq_sinv)

        l_mem_bta_seq.append(mem_bta_seq)


        # Compute the cost of the BTA distributed algorithm
        n_blocks_partition = get_partition_size(n_blocks, n_partitions)
        reduced_system_size = get_reduced_system_size(n_partitions)

        mem_bta_dist_input = getmem_bta_dist_input(n_blocks_partition, diag_blocksize, arrowhead_blocksize)
        mem_bta_dist_factorization = getmem_bta_dist_factorization(n_blocks_partition, diag_blocksize, arrowhead_blocksize)
        mem_reduced_system_solve = getmem_bta_seq_factorization(reduced_system_size, diag_blocksize, arrowhead_blocksize) + getmem_bta_seq_sinv(reduced_system_size, diag_blocksize, arrowhead_blocksize)
        mem_bta_dist_sinv = getmem_bta_dist_sinv(n_blocks_partition, diag_blocksize, arrowhead_blocksize)
        
        mem_bta_dist = sp.simplify(mem_bta_dist_input + mem_bta_dist_factorization + mem_reduced_system_solve + mem_bta_dist_sinv)

        # print("cost_bta_dist: \n", cost_bta_dist)
        l_mem_bta_dist.append(mem_bta_dist)

        l_mem_bta_seq_over_dist.append(mem_bta_seq / mem_bta_dist)

    plt.plot(range(1, max_partitions), l_mem_bta_seq, label="BTA Sequential")
    plt.plot(range(1, max_partitions), l_mem_bta_dist, label="BTA Distributed")
    plt.xlabel("Number of Partitions")
    plt.ylabel("Memory usage (elements)")
    plt.title(f"BTA Sequential vs BTA Distributed\n diag_blocksize = {diag_blocksize}, arrowhead_blocksize = {arrowhead_blocksize}")
    plt.legend()
    plt.show()

    plt.plot(range(1, max_partitions), l_mem_bta_seq_over_dist, label="BTA Sequential / BTA Distributed")
    plt.xlabel("Number of Processes")
    plt.ylabel("Memory need ratio")
    plt.title(f"BTA Sequential vs BTA Distributed\n diag_blocksize = {diag_blocksize}, arrowhead_blocksize = {arrowhead_blocksize}")
    plt.legend()
    plt.show()


    
""" if __name__ == "__main__":
    diag_blocksize = 4000
    arrowhead_blocksize = 4

    max_n_blocks = 3000
    max_partitions = 128

    cost_matrix = np.zeros((max_partitions, max_n_blocks))

    for n_blocks in range(1, max_n_blocks):
        for n_partitions in range(1, max_partitions):
            # Compute the cost of the BTA sequential algorithm
            cost_bta_seq_factorization = getmem_bta_seq_factorization(n_blocks, diag_blocksize, arrowhead_blocksize)
            cost_bta_seq_sinv = getmem_bta_seq_sinv(n_blocks, diag_blocksize, arrowhead_blocksize)

            cost_bta_seq = sp.simplify(cost_bta_seq_factorization)


            # Compute the cost of the BTA distributed algorithm
            n_blocks_partition = get_partition_size(n_blocks, n_partitions)
            reduced_system_size = get_reduced_system_size(n_partitions)

            # Standard algorithm analysis
            cost_bta_dist_middle_process_factorization = getmem_bta_middle_dist_factorization(n_blocks_partition, diag_blocksize, arrowhead_blocksize)
            cost_bta_dist_reduced_system_solve = getmem_bta_seq_factorization(reduced_system_size, diag_blocksize, arrowhead_blocksize) + getmem_bta_seq_sinv(reduced_system_size, diag_blocksize, arrowhead_blocksize)
            cost_bta_dist_middle_process_sinv = getmem_bta_middle_dist_sinv(n_blocks_partition, diag_blocksize, arrowhead_blocksize)
            cost_bta_dist = sp.simplify(cost_bta_dist_middle_process_factorization + cost_bta_dist_reduced_system_solve + cost_bta_dist_middle_process_sinv)


            # # Divid and conquer algorithm aproach for the reduced system inversion
            # cost_bta_dist_middle_process_factorization = getmem_bta_middle_dist_factorization(n_blocks_partition, diag_blocksize, arrowhead_blocksize)
            
            # blocks_per_process = 3
            # n_processes_inverting_reduced_system = reduced_system_size // blocks_per_process
            # reduced_system_sub_partition_size = get_reduced_system_size(n_processes_inverting_reduced_system)
            # cost_bta_dist_reduced_system_solve = getmem_bta_middle_dist_factorization(blocks_per_process, diag_blocksize, arrowhead_blocksize) + getmem_bta_middle_dist_sinv(blocks_per_process, diag_blocksize, arrowhead_blocksize)
            # cost_bta_dist_reduced_system_solve += getmem_bta_seq_factorization(reduced_system_sub_partition_size, diag_blocksize, arrowhead_blocksize) + getmem_bta_seq_sinv(reduced_system_sub_partition_size, diag_blocksize, arrowhead_blocksize)
            
            # cost_bta_dist_middle_process_sinv = getmem_bta_middle_dist_sinv(n_blocks_partition, diag_blocksize, arrowhead_blocksize)
            
            # cost_bta_dist = sp.simplify(cost_bta_dist_middle_process_factorization + cost_bta_dist_reduced_system_solve + cost_bta_dist_middle_process_sinv)



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
    plt.show() """


