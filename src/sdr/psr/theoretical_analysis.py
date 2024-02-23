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



if __name__ == "__main__":
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
    plt.show()

    