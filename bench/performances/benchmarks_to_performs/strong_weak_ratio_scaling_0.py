# STRONG CALING RUNS
if __name__ == "__main__":
    # TODO: Check the biggest case we can run with the sequential code
    # use that as the baseline case for the strong scaling analysis
    NB = 256
    BS = 2000
    AS = 500
    N_PROCESSES = [2, 4, 8, 16]

    print("\n")
    print("         --- STRONG SCALING --- ")
    for p, i in enumerate(N_PROCESSES):
        partitons_size = NB // i
        matrix_size = NB * BS
        print(
            f"    N_PROCESSES: {i}, NB: {NB}, BS: {BS}, AS: {AS}     (partitons_size = {partitons_size}, matrix_size = {matrix_size})"
        )


# WEAK SCALING RUNS
if __name__ == "__main__":
    NB = [128, 256, 512, 1024]
    BS = 2000
    AS = 500
    N_PROCESSES = [2, 4, 8, 16]

    print("\n")
    print("         --- WEAK SCALING --- ")
    for p, i in enumerate(N_PROCESSES):
        for n, nb in enumerate(NB):
            partitons_size = nb // i
            matrix_size = nb * BS
            print(
                f"    N_PROCESSES: {i}, NB: {nb}, BS: {BS}, AS: {AS}     (partitons_size = {partitons_size}, matrix_size = {matrix_size})"
            )
        print("\n")

# RATIO SCALING RUNS
if __name__ == "__main__":
    RATIO = [8, 16, 32, 64]
    BS = 2000
    AS = 500
    N_PROCESSES = 2

    print("         --- RATIO SCALING --- ")
    for ratio in RATIO:
        partitons_size = N_PROCESSES * 2 * ratio
        nb = partitons_size * N_PROCESSES
        matrix_size = nb * BS
        print(
            f"    N_PROCESSES: {N_PROCESSES}, NB: {nb}, BS: {BS}, AS: {AS}     (partitons_size = {partitons_size}, ratio = {ratio}, matrix_size = {matrix_size})"
        )
    print("\n")

# RATIO SCALING RUNS
if __name__ == "__main__":
    RATIO = [8, 16, 32]
    BS = 2000
    AS = 500
    N_PROCESSES = 4

    print("         --- RATIO SCALING --- ")
    for ratio in RATIO:
        partitons_size = N_PROCESSES * 2 * ratio
        nb = partitons_size * N_PROCESSES
        matrix_size = nb * BS
        print(
            f"    N_PROCESSES: {N_PROCESSES}, NB: {nb}, BS: {BS}, AS: {AS}     (partitons_size = {partitons_size}, ratio = {ratio}, matrix_size = {matrix_size})"
        )
    print("\n")
