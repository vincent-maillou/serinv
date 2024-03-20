# STRONG SCALING ON PARTITIONS SIZE
if __name__ == "__main__":
    # TODO: Check the biggest case we can run with the sequential code
    # use that as the baseline case for the strong scaling analysis
    total_number_of_blocks = 512
    diag_blocksize = 2000
    arrow_blocksize = 500
    matrix_size = total_number_of_blocks * diag_blocksize
    n_processes = [2, 4, 8, 16]

    print("\n")
    print("     --- STRONG SCALING ON PARTITIONS SIZE --- ")
    for n_process in n_processes:
        partition_size = total_number_of_blocks // n_process
        print(
            f"    n_processes: {n_process}, partition_size: {partition_size}, total_number_of_blocks: {total_number_of_blocks} diag_blocksize: {diag_blocksize}, arrow_blocksize: {arrow_blocksize}, matrix_size: {matrix_size}"
        )
    print("\n")


# WEAK SCALING ON PARTITIONS SIZE
if __name__ == "__main__":
    partitons_sizes = [128, 256]
    diag_blocksize = 2000
    arrow_blocksize = 500
    n_processes = [2, 4, 8, 16]

    print("     --- WEAK SCALING ON PARTITIONS SIZE --- ")
    for partition_size in partitons_sizes:
        for n_process in n_processes:
            total_number_of_blocks = partition_size * n_process
            matrix_size = partition_size * n_process * diag_blocksize
            print(
                f"    n_processes: {n_process}, partition_size: {partition_size}, total_number_of_blocks: {total_number_of_blocks} diag_blocksize: {diag_blocksize}, arrow_blocksize: {arrow_blocksize}, matrix_size: {matrix_size}"
            )
        print("\n")


# WEAK SCALING ON PARTITIONS RATIO
if __name__ == "__main__":
    partitons_ratios = [8, 16]
    diag_blocksize = 2000
    arrow_blocksize = 500
    n_processes = [2, 4, 8, 16]

    print("     --- WEAK SCALING ON PARTITIONS RATIO --- ")
    for partition_ratio in partitons_ratios:
        for n_process in n_processes:
            reduced_system_size = n_process * 2
            partition_size = partition_ratio * reduced_system_size
            total_number_of_blocks = partition_size * n_process
            matrix_size = total_number_of_blocks * diag_blocksize
            print(
                f"    n_processes: {n_process}, partition_size: {partition_size}, partition_ratio: {partition_ratio}, total_number_of_blocks: {total_number_of_blocks} diag_blocksize: {diag_blocksize}, arrow_blocksize: {arrow_blocksize}, matrix_size: {matrix_size}"
            )
        print("\n")
