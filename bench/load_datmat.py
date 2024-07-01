import numpy as np
from scipy.sparse import csc_matrix


def read_sym_CSC(filename):
    with open(filename, "r") as f:
        n = int(f.readline().strip())
        n = int(f.readline().strip())
        nnz = int(f.readline().strip())

        inner_indices = np.zeros(nnz, dtype=int)
        outer_index_ptr = np.zeros(n + 1, dtype=int)
        values = np.zeros(nnz, dtype=float)

        for i in range(nnz):
            inner_indices[i] = int(f.readline().strip())

        for i in range(n + 1):
            outer_index_ptr[i] = int(f.readline().strip())

        for i in range(nnz):
            values[i] = float(f.readline().strip())

    # Create the lower triangular CSC matrix
    A = csc_matrix((values, inner_indices, outer_index_ptr), shape=(n, n))

    return A


def csc_to_dense_bta(
    A: csc_matrix, diagonal_blocksize: int, arrowhead_blocksize: int, n_diag_blocks: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    A_diagonal_blocks = np.zeros(
        (n_diag_blocks, diagonal_blocksize, diagonal_blocksize), dtype=A.dtype
    )
    A_lower_diagonal_blocks = np.zeros(
        (n_diag_blocks - 1, diagonal_blocksize, diagonal_blocksize), dtype=A.dtype
    )
    A_arrow_bottom_blocks = np.zeros(
        (n_diag_blocks, arrowhead_blocksize, diagonal_blocksize), dtype=A.dtype
    )
    A_arrow_tip_block = np.zeros(
        (arrowhead_blocksize, arrowhead_blocksize), dtype=A.dtype
    )

    for i in range(n_diag_blocks):
        A_diagonal_blocks[i, :, :] = A[
            i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
            i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
        ].toarray()

        if i < n_diag_blocks - 1:
            A_lower_diagonal_blocks[i, :, :] = A[
                (i + 1) * diagonal_blocksize : (i + 2) * diagonal_blocksize,
                i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
            ].toarray()

        A_arrow_bottom_blocks[i, :, :] = A[
            -arrowhead_blocksize:, i * diagonal_blocksize : (i + 1) * diagonal_blocksize
        ].toarray()

    A_arrow_tip_block[:, :] = A[-arrowhead_blocksize:, -arrowhead_blocksize:].toarray()

    return (
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_arrow_bottom_blocks,
        A_arrow_tip_block,
    )


def csc_to_sparse_bta(
    A: csc_matrix, diagonal_blocksize: int, arrowhead_blocksize: int, n_diag_blocks: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    A_diagonal_blocks_csc = [None] * n_diag_blocks
    A_lower_diagonal_blocks_csc = [None] * (n_diag_blocks - 1)
    A_arrow_bottom_blocks_csc = [None] * n_diag_blocks
    A_arrow_tip_block_csc = None

    for i in range(n_diag_blocks):
        A_diagonal_blocks_csc[i] = csc_matrix(
            A[
                i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
                i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
            ],
            dtype=A.dtype,
        )

        if i < n_diag_blocks - 1:
            A_lower_diagonal_blocks_csc[i] = csc_matrix(
                A[
                    (i + 1) * diagonal_blocksize : (i + 2) * diagonal_blocksize,
                    i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
                ],
                dtype=A.dtype,
            )

        A_arrow_bottom_blocks_csc[i] = csc_matrix(
            A[
                -arrowhead_blocksize:,
                i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
            ],
            dtype=A.dtype,
        )

    A_arrow_tip_block_csc = csc_matrix(
        A[-arrowhead_blocksize:, -arrowhead_blocksize:], dtype=A.dtype
    )

    return (
        A_diagonal_blocks_csc,
        A_lower_diagonal_blocks_csc,
        A_arrow_bottom_blocks_csc,
        A_arrow_tip_block_csc,
    )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Example usage
    filename = "Qxy_ns92_nt5_nb4_464.dat"
    # filename = "Qxy_ns92_nt50_nb4_4604.dat"

    A = read_sym_CSC(filename)

    plt.spy(A, markersize=0.1)
    # plt.matshow(A.toarray())

    diagonal_blocksize = 92
    arrowhead_blocksize = 4
    n_diag_blocks = 5

    (
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_arrow_bottom_blocks,
        A_arrow_tip_block,
    ) = csc_to_dense_bta(A, diagonal_blocksize, arrowhead_blocksize, n_diag_blocks)

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].matshow(A_diagonal_blocks[0])
    axs[0, 1].matshow(A_lower_diagonal_blocks[0])
    axs[1, 0].matshow(A_arrow_bottom_blocks[0])
    axs[1, 1].matshow(A_arrow_tip_block)

    (
        A_diagonal_blocks_csc,
        A_lower_diagonal_blocks_csc,
        A_arrow_bottom_blocks_csc,
        A_arrow_tip_block_csc,
    ) = csc_to_sparse_bta(A, diagonal_blocksize, arrowhead_blocksize, n_diag_blocks)

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].matshow(A_diagonal_blocks_csc[0].toarray())
    axs[0, 1].matshow(A_lower_diagonal_blocks_csc[0].toarray())
    axs[1, 0].matshow(A_arrow_bottom_blocks_csc[0].toarray())
    axs[1, 1].matshow(A_arrow_tip_block_csc.toarray())

    plt.show()
