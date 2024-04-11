import matplotlib.pyplot as plt
import numpy as np

from sdr.lu.lu_decompose import lu_dcmp_ndiags_arrowhead
from sdr.lu.lu_selected_inversion import lu_sinv_ndiags_arrowhead
from sdr.utils.matrix_generation_dense import generate_tridiag_arrowhead_dense
from sdr.utils.matrix_transform import cut_to_blocktridiag_arrowhead


def create_permutation_matrix(
    mat_size: int,
) -> np.ndarray:
    P = np.zeros((mat_size, mat_size))

    offset = 0
    half = mat_size // 2
    for i in range(mat_size):
        if i % 2 == 0:
            P[i, half + offset] = 1
            offset += 1
        else:
            P[i, half - offset] = 1

    return P


def create_permutation_matrix_for_arrowhead(
    n_blocks: int,
    blocksize: int,
) -> np.ndarray:
    P = np.zeros((n_blocks * blocksize, n_blocks * blocksize))

    I = np.eye(blocksize)

    offset = 0
    half = n_blocks // 2 - 1
    for i in range(n_blocks):
        if i % 2 == 1:
            P[
                i * blocksize : (i + 1) * blocksize,
                (half + offset) * blocksize : (half + offset + 1) * blocksize,
            ] = I
        else:
            P[
                i * blocksize : (i + 1) * blocksize,
                (half - offset) * blocksize : (half - offset + 1) * blocksize,
            ] = I
            offset += 1

    return P


def tridiag_matrix(
    mat_size: int,
) -> np.ndarray:
    A = np.zeros((mat_size, mat_size))

    for i in range(mat_size):
        A[i, i] = i + 1
        if i > 0:
            A[i, i - 1] = i + 1 - 0.1
        if i < mat_size - 1:
            A[i, i + 1] = i + 1 + 0.1

    return A


def arrow_matrix(
    mat_size: int,
) -> np.ndarray:
    A = np.zeros((mat_size, mat_size))

    for i in range(mat_size):
        A[i, i] = i + 1
        if i > 0:
            A[i, i - 1] = i + 1 - 0.1
        if i < mat_size - 1:
            A[i, i + 1] = i + 1 + 0.1

        if i < mat_size - 2:
            A[i, -1] = i + 1 - 0.2
            A[-1, i] = i + 1 + 0.2

    return A


def create_vector(
    mat_size: int,
    row_vector: bool = False,
) -> np.ndarray:
    if row_vector:
        v = np.zeros((1, mat_size))
        for i in range(mat_size):
            v[0, i] = i + 1
    else:
        v = np.zeros((mat_size, 1))
        for i in range(mat_size):
            v[i] = i + 1

    return v


if __name__ == "__main__":
    n_blocks = 10
    diag_blocksize = 3
    arrow_blocksize = 3
    symmetric = False
    diag_dom = True
    seed = 63

    P = create_permutation_matrix_for_arrowhead(n_blocks, diag_blocksize)
    Pt = P.T

    # A = tridiag_matrix(mat_size)

    A = generate_tridiag_arrowhead_dense(
        n_blocks, diag_blocksize, arrow_blocksize, symmetric, diag_dom, seed
    )

    ref_inverse = np.linalg.inv(A)
    ref_inverse = cut_to_blocktridiag_arrowhead(
        ref_inverse, diag_blocksize, arrow_blocksize
    )

    # v_right = create_vector(mat_size)
    # v_bottom = create_vector(mat_size, row_vector=True)

    PAPt = P @ A @ P.T
    PtPAPtP = P.T @ PAPt @ P
    # Pv_right = P @ v_right
    # Pv_bottom = v_bottom @ P.T

    ndiags = 5
    L_sdr, U_sdr = lu_dcmp_ndiags_arrowhead(
        PAPt, ndiags, diag_blocksize, arrow_blocksize
    )
    LU_sdr = L_sdr + U_sdr
    sdr_inverse = lu_sinv_ndiags_arrowhead(
        L_sdr, U_sdr, ndiags, diag_blocksize, arrow_blocksize
    )

    Ptsdr_inverseP = Pt @ sdr_inverse @ P
    Ptsdr_inverseP_cut = cut_to_blocktridiag_arrowhead(
        Ptsdr_inverseP, diag_blocksize, arrow_blocksize
    )

    # sdr_inv = lu_dcmp_ndiags_arrowhead(PAPt, )

    fig, axs = plt.subplots(2, 4)
    axs[0, 0].matshow(P)
    axs[0, 0].set_title("P")
    axs[0, 1].matshow(A)
    axs[0, 1].set_title("A")
    axs[0, 2].matshow(Pt)
    axs[0, 2].set_title("Pt")
    axs[0, 3].matshow(PAPt)
    axs[0, 3].set_title("PAPt")

    axs[1, 0].matshow(ref_inverse)
    axs[1, 0].set_title("ref_inverse")
    axs[1, 1].matshow(LU_sdr)
    axs[1, 1].set_title("LU_sdr")
    axs[1, 2].matshow(sdr_inverse)
    axs[1, 2].set_title("sdr_inverse")
    axs[1, 3].matshow(Ptsdr_inverseP_cut)
    axs[1, 3].set_title("Pt @ sdr_inverse_cut @ P")

    assert np.allclose(ref_inverse, Ptsdr_inverseP_cut)

    norme_ref = np.linalg.norm(ref_inverse)
    norme_sdr = np.linalg.norm(Ptsdr_inverseP_cut)

    norme_diff = np.linalg.norm(ref_inverse - Ptsdr_inverseP_cut)
    print(f"Norme ref: {norme_ref}")
    print(f"Norme sdr: {norme_sdr}")
    print(f"Norme diff: {norme_diff}")

    plt.show()
