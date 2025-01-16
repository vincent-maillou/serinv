import numpy as np
from scipy.sparse import csc_matrix, coo_matrix, lil_matrix

from serinv import ArrayLike, CUPY_AVAIL, _get_module_from_array, _get_module_from_str

SEED = 63

np.random.seed(SEED)


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


def read_CSC(filename):
    with open(filename, "r") as f:
        nrows = int(f.readline().strip())
        ncols = int(f.readline().strip())
        nnz = int(f.readline().strip())

        inner_indices = np.zeros(nnz, dtype=int)
        outer_index_ptr = np.zeros(ncols + 1, dtype=int)
        values = np.zeros(nnz, dtype=float)

        for i in range(nnz):
            inner_indices[i] = int(f.readline().strip())

        for i in range(ncols + 1):
            outer_index_ptr[i] = int(f.readline().strip())

        for i in range(nnz):
            values[i] = float(f.readline().strip())

    # Create CSC matrix
    A = csc_matrix((values, inner_indices, outer_index_ptr), shape=(nrows, ncols))

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

def bta_to_coo(
    A_diagonal_blocks,
    A_lower_diagonal_blocks,
    A_arrow_bottom_blocks,
    A_arrow_tip_block,
    banded: bool = False,
) -> coo_matrix:
    n_diagonal_blocks = A_diagonal_blocks.shape[0]
    diagonal_blocksize = A_diagonal_blocks.shape[1]
    arrow_blocksize = A_arrow_bottom_blocks.shape[1]

    n = n_diagonal_blocks * diagonal_blocksize + arrow_blocksize

    # initialize a zeros lil matrix

    A_lil = lil_matrix((n, n), dtype=A_diagonal_blocks.dtype)

    print(A_diagonal_blocks.shape)
    print(A_lower_diagonal_blocks.shape)
    print(A_arrow_bottom_blocks.shape)

    for i in range(n_diagonal_blocks):
        A_lil[
            i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
            i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
        ] = np.tril(A_diagonal_blocks[i])

        if i < n_diagonal_blocks - 1:
            if banded:
                A_lil[
                    (i + 1) * diagonal_blocksize : (i + 2) * diagonal_blocksize,
                    i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
                ] = np.triu(A_lower_diagonal_blocks[i])
            else:
                A_lil[
                    (i + 1) * diagonal_blocksize : (i + 2) * diagonal_blocksize,
                    i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
                ] = A_lower_diagonal_blocks[i]

        A_lil[
            -arrow_blocksize:,
            i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
        ] = A_arrow_bottom_blocks[i]

    A_lil[-arrow_blocksize:, -arrow_blocksize:] = np.tril(A_arrow_tip_block[:, :])

    return A_lil.tocoo()

def bta_dense_to_arrays(
    bta: ArrayLike,
    diagonal_blocksize: int,
    arrowhead_blocksize: int,
    n_diag_blocks: int,
):
    """Converts a block tridiagonal arrowhead matrix from a dense representation to arrays of blocks."""
    if CUPY_AVAIL:
        xp, _ = _get_module_from_array(arr=bta)
    else:
        xp = np

    A_diagonal_blocks = xp.zeros(
        (n_diag_blocks, diagonal_blocksize, diagonal_blocksize),
        dtype=bta.dtype,
    )

    A_lower_diagonal_blocks = xp.zeros(
        (n_diag_blocks - 1, diagonal_blocksize, diagonal_blocksize),
        dtype=bta.dtype,
    )
    A_upper_diagonal_blocks = xp.zeros(
        (n_diag_blocks - 1, diagonal_blocksize, diagonal_blocksize),
        dtype=bta.dtype,
    )

    A_arrow_bottom_blocks = xp.zeros(
        (n_diag_blocks, arrowhead_blocksize, diagonal_blocksize),
        dtype=bta.dtype,
    )

    A_arrow_right_blocks = xp.zeros(
        (n_diag_blocks, diagonal_blocksize, arrowhead_blocksize),
        dtype=bta.dtype,
    )

    for i in range(n_diag_blocks):
        A_diagonal_blocks[i, :, :] = bta[
            i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
            i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
        ]
        if i > 0:
            A_lower_diagonal_blocks[i - 1, :, :] = bta[
                i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
                (i - 1) * diagonal_blocksize : i * diagonal_blocksize,
            ]
        if i < n_diag_blocks - 1:
            A_upper_diagonal_blocks[i, :, :] = bta[
                i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
                (i + 1) * diagonal_blocksize : (i + 2) * diagonal_blocksize,
            ]

        A_arrow_bottom_blocks[i, :, :] = bta[
            -arrowhead_blocksize:,
            i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
        ]

        A_arrow_right_blocks[i, :, :] = bta[
            i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
            -arrowhead_blocksize:,
        ]

    A_arrow_tip_block = bta[-arrowhead_blocksize:, -arrowhead_blocksize:]

    return (
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_upper_diagonal_blocks,
        A_arrow_bottom_blocks,
        A_arrow_right_blocks,
        A_arrow_tip_block,
    )


def bta_arrays_to_dense(
    A_diagonal_blocks: ArrayLike,
    A_lower_diagonal_blocks: ArrayLike,
    A_upper_diagonal_blocks: ArrayLike,
    A_arrow_bottom_blocks: ArrayLike,
    A_arrow_right_blocks: ArrayLike,
    A_arrow_tip_block: ArrayLike,
):
    """Converts arrays of blocks to a block tridiagonal arrowhead matrix in a dense representation."""
    if CUPY_AVAIL:
        xp, _ = _get_module_from_array(arr=A_diagonal_blocks)
    else:
        xp = np

    diagonal_blocksize = A_diagonal_blocks.shape[1]
    arrowhead_blocksize = A_arrow_bottom_blocks.shape[1]
    n_diag_blocks = A_diagonal_blocks.shape[0]

    bta = xp.zeros(
        (
            diagonal_blocksize * n_diag_blocks + arrowhead_blocksize,
            diagonal_blocksize * n_diag_blocks + arrowhead_blocksize,
        ),
        dtype=A_diagonal_blocks.dtype,
    )

    for i in range(n_diag_blocks):
        bta[
            i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
            i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
        ] = A_diagonal_blocks[i, :, :]
        if i > 0:
            bta[
                i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
                (i - 1) * diagonal_blocksize : i * diagonal_blocksize,
            ] = A_lower_diagonal_blocks[i - 1, :, :]
        if i < n_diag_blocks - 1:
            bta[
                i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
                (i + 1) * diagonal_blocksize : (i + 2) * diagonal_blocksize,
            ] = A_upper_diagonal_blocks[i, :, :]

        bta[
            -arrowhead_blocksize:,
            i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
        ] = A_arrow_bottom_blocks[i, :, :]

        bta[
            i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
            -arrowhead_blocksize:,
        ] = A_arrow_right_blocks[i, :, :]

    bta[-arrowhead_blocksize:, -arrowhead_blocksize:] = A_arrow_tip_block

    return bta


def bta_symmetrize(
    bta: ArrayLike,
):
    """Symmetrizes a block tridiagonal arrowhead matrix."""

    return (bta + bta.conj().T) / 2


def dd_bta(
    diagonal_blocksize: int,
    arrowhead_blocksize: int,
    n_diag_blocks: int,
    device_array: bool,
    dtype: np.dtype,
):
    """Returns a random, diagonaly dominant general, block tridiagonal arrowhead matrix."""
    xp, _ = _get_module_from_str(device_array="cupy" if device_array else "numpy")

    DD_BTA = xp.zeros(
        (
            diagonal_blocksize * n_diag_blocks + arrowhead_blocksize,
            diagonal_blocksize * n_diag_blocks + arrowhead_blocksize,
        ),
        dtype=dtype,
    )

    rc = (1.0 + 1.0j) if dtype == np.complex128 else 1.0

    # Fill the lower arrowhead blocks
    DD_BTA[-arrowhead_blocksize:, :-arrowhead_blocksize] = rc * xp.random.rand(
        arrowhead_blocksize, n_diag_blocks * diagonal_blocksize
    )
    # Fill the right arrowhead blocks
    DD_BTA[:-arrowhead_blocksize, -arrowhead_blocksize:] = rc * xp.random.rand(
        n_diag_blocks * diagonal_blocksize, arrowhead_blocksize
    )

    # Fill the tip of the arrowhead
    DD_BTA[-arrowhead_blocksize:, -arrowhead_blocksize:] = rc * xp.random.rand(
        arrowhead_blocksize, arrowhead_blocksize
    )

    # Fill the diagonal blocks
    for i in range(n_diag_blocks):
        DD_BTA[
            i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
            i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
        ] = rc * xp.random.rand(diagonal_blocksize, diagonal_blocksize) + rc * xp.eye(
            diagonal_blocksize
        )

        # Fill the off-diagonal blocks
        if i > 0:
            DD_BTA[
                i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
                (i - 1) * diagonal_blocksize : i * diagonal_blocksize,
            ] = rc * xp.random.rand(diagonal_blocksize, diagonal_blocksize)

        if i < n_diag_blocks - 1:
            DD_BTA[
                i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
                (i + 1) * diagonal_blocksize : (i + 2) * diagonal_blocksize,
            ] = rc * xp.random.rand(diagonal_blocksize, diagonal_blocksize)

    # Make the matrix diagonally dominant
    for i in range(DD_BTA.shape[0]):
        DD_BTA[i, i] = 1 + xp.sum(DD_BTA[i, :])

    return DD_BTA


def rand_bta(
    diagonal_blocksize: int,
    arrowhead_blocksize: int,
    n_diag_blocks: int,
    device_array: bool,
    dtype: np.dtype,
):
    """Returns a random, diagonaly dominant general, block tridiagonal arrowhead matrix."""
    xp, _ = _get_module_from_str(device_array="cupy" if device_array else "numpy")

    RAND_BTA = xp.zeros(
        (
            diagonal_blocksize * n_diag_blocks + arrowhead_blocksize,
            diagonal_blocksize * n_diag_blocks + arrowhead_blocksize,
        ),
        dtype=dtype,
    )

    rc = (1.0 + 1.0j) if dtype == np.complex128 else 1.0

    # Fill the lower arrowhead blocks
    RAND_BTA[-arrowhead_blocksize:, :-arrowhead_blocksize] = rc * xp.random.rand(
        arrowhead_blocksize, n_diag_blocks * diagonal_blocksize
    )
    # Fill the right arrowhead blocks
    RAND_BTA[:-arrowhead_blocksize, -arrowhead_blocksize:] = rc * xp.random.rand(
        n_diag_blocks * diagonal_blocksize, arrowhead_blocksize
    )

    # Fill the tip of the arrowhead
    RAND_BTA[-arrowhead_blocksize:, -arrowhead_blocksize:] = rc * xp.random.rand(
        arrowhead_blocksize, arrowhead_blocksize
    )

    # Fill the diagonal blocks
    for i in range(n_diag_blocks):
        RAND_BTA[
            i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
            i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
        ] = rc * xp.random.rand(diagonal_blocksize, diagonal_blocksize) + rc * xp.eye(
            diagonal_blocksize
        )

        # Fill the off-diagonal blocks
        if i > 0:
            RAND_BTA[
                i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
                (i - 1) * diagonal_blocksize : i * diagonal_blocksize,
            ] = rc * xp.random.rand(diagonal_blocksize, diagonal_blocksize)

        if i < n_diag_blocks - 1:
            RAND_BTA[
                i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
                (i + 1) * diagonal_blocksize : (i + 2) * diagonal_blocksize,
            ] = rc * xp.random.rand(diagonal_blocksize, diagonal_blocksize)

    return RAND_BTA


def b_rhs(
    n_rhs: int,
    diagonal_blocksize: int,
    arrowhead_blocksize: int,
    n_diag_blocks: int,
    device_array: bool,
    dtype: np.dtype,
):
    """Returns a random right-hand side."""
    xp, _ = _get_module_from_str(device_array="cupy" if device_array else "numpy")

    rc = (1.0 + 1.0j) if dtype == np.complex128 else 1.0

    B = rc * xp.random.rand(
        diagonal_blocksize * n_diag_blocks + arrowhead_blocksize, n_rhs
    )

    return B
