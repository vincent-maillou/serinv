import numpy as np
import scipy.linalg as scla
import copy as cp
import matplotlib.pyplot as plt

from sdr.utils.matrix_generation import generate_tridiag_arrowhead_arrays, generate_tridiag_arrowhead_dense
from sdr.utils.matrix_transform import from_arrowhead_arrays_to_dense, from_dense_to_arrowhead_arrays, cut_to_blocktridiag_arrowhead
from sdr.lu.lu_factorize import lu_factorize_tridiag_arrowhead
from sdr.lu.lu_factorize_handle_pivoting import lu_factorize_handle_pivoting_tridiag_arrowhead, lu_factorize_inverse_tip_tridiag_arrowhead
from sdr.lu.lu_selected_inversion import lu_sinv_tridiag_arrowhead
from sdr.lu.lu_selected_inversion_handle_pivoting import lu_sinv_handle_pivoting_tridiag_arrowhead, lu_sinv_inverse_tip_tridiag_arrowhead


def load_from_npy(filename):
    matrix = np.load(filename)
    return matrix


if __name__ == '__main__':
    file_path = 'matrix.npy'
    A = load_from_npy(file_path)

    A_inv_tip = cp.deepcopy(A)
    A_pivot = cp.deepcopy(A)

    diag_blocksize = 336
    arrow_blocksize = 84
    n_diag_blocks = (A.shape[0] - arrow_blocksize) // diag_blocksize
    print(f"n_diag_blocks: {n_diag_blocks}")

    A = cut_to_blocktridiag_arrowhead(A, diag_blocksize, arrow_blocksize)
    X_ref = np.linalg.inv(A)
    X_ref = cut_to_blocktridiag_arrowhead(X_ref, diag_blocksize, arrow_blocksize)

    (
        X_ref_diagonal_blocks,
        X_ref_lower_diagonal_blocks,
        X_ref_upper_diagonal_blocks,
        X_ref_arrow_bottom_blocks,
        X_ref_arrow_right_blocks,
        X_ref_arrow_tip_block,
    ) = from_dense_to_arrowhead_arrays(X_ref, diag_blocksize, arrow_blocksize)

    (P, L, U) = scla.lu(A)

    fig, ax = plt.subplots(1, 3)
    ax[0].spy(P)
    ax[0].set_title('P')
    ax[1].spy(L)
    ax[1].set_title('L')
    ax[2].spy(U)
    ax[2].set_title('U')
    plt.show()


    # ----- Dense to Arrays -----
    (
        A_inv_tip_diagonal_blocks,
        A_inv_tip_lower_diagonal_blocks,
        A_inv_tip_upper_diagonal_blocks,
        A_inv_tip_arrow_bottom_blocks,
        A_inv_tip_arrow_right_blocks,
        A_inv_tip_arrow_tip_block,
    ) = from_dense_to_arrowhead_arrays(A_inv_tip, diag_blocksize, arrow_blocksize)

    (
        A_pivot_diagonal_blocks,
        A_pivot_lower_diagonal_blocks,
        A_pivot_upper_diagonal_blocks,
        A_pivot_arrow_bottom_blocks,
        A_pivot_arrow_right_blocks,
        A_pivot_arrow_tip_block,
    ) = from_dense_to_arrowhead_arrays(A_pivot, diag_blocksize, arrow_blocksize)


    # ----- LU Factorization -----
    (
        L_inv_tip_diagonal_blocks,
        L_inv_tip_lower_diagonal_blocks,
        L_inv_tip_arrow_bottom_blocks,
        U_inv_tip_diagonal_blocks,
        U_inv_tip_upper_diagonal_blocks,
        U_inv_tip_arrow_right_blocks,
    ) = lu_factorize_inverse_tip_tridiag_arrowhead(
        A_inv_tip_diagonal_blocks,
        A_inv_tip_lower_diagonal_blocks,
        A_inv_tip_upper_diagonal_blocks,
        A_inv_tip_arrow_bottom_blocks,
        A_inv_tip_arrow_right_blocks,
        A_inv_tip_arrow_tip_block,
    )

    (
        L_pivot_diagonal_blocks,
        L_pivot_lower_diagonal_blocks,
        L_pivot_arrow_bottom_blocks,
        U_pivot_diagonal_blocks,
        U_pivot_upper_diagonal_blocks,
        U_pivot_arrow_right_blocks,
        P,
    ) = lu_factorize_handle_pivoting_tridiag_arrowhead(
        A_pivot_diagonal_blocks,
        A_pivot_lower_diagonal_blocks,
        A_pivot_upper_diagonal_blocks,
        A_pivot_arrow_bottom_blocks,
        A_pivot_arrow_right_blocks,
        A_pivot_arrow_tip_block,
    )


    # ----- LU Selected Inversion -----
    (
        X_inv_tip_diagonal_blocks,
        X_inv_tip_lower_diagonal_blocks,
        X_inv_tip_upper_diagonal_blocks,
        X_inv_tip_arrow_bottom_blocks,
        X_inv_tip_arrow_right_blocks,
        X_inv_tip_arrow_tip_block,
    ) = lu_sinv_inverse_tip_tridiag_arrowhead(
        L_inv_tip_diagonal_blocks,
        L_inv_tip_lower_diagonal_blocks,
        L_inv_tip_arrow_bottom_blocks,
        U_inv_tip_diagonal_blocks,
        U_inv_tip_upper_diagonal_blocks,
        U_inv_tip_arrow_right_blocks,
    )

    (
        X_pivot_diagonal_blocks,
        X_pivot_lower_diagonal_blocks,
        X_pivot_upper_diagonal_blocks,
        X_pivot_arrow_bottom_blocks,
        X_pivot_arrow_right_blocks,
        X_pivot_arrow_tip_block,
    ) = lu_sinv_handle_pivoting_tridiag_arrowhead(
        L_pivot_diagonal_blocks,
        L_pivot_lower_diagonal_blocks,
        L_pivot_arrow_bottom_blocks,
        U_pivot_diagonal_blocks,
        U_pivot_upper_diagonal_blocks,
        U_pivot_arrow_right_blocks,
        P,
    )


    # ----- From Arrowhead Arrays to Dense -----
    X_inv_tip_SerinV = from_arrowhead_arrays_to_dense(
        X_inv_tip_diagonal_blocks,
        X_inv_tip_lower_diagonal_blocks,
        X_inv_tip_upper_diagonal_blocks,
        X_inv_tip_arrow_bottom_blocks,
        X_inv_tip_arrow_right_blocks,
        X_inv_tip_arrow_tip_block,
    )

    X_pivot_SerinV = from_arrowhead_arrays_to_dense(
        X_pivot_diagonal_blocks,
        X_pivot_lower_diagonal_blocks,
        X_pivot_upper_diagonal_blocks,
        X_pivot_arrow_bottom_blocks,
        X_pivot_arrow_right_blocks,
        X_pivot_arrow_tip_block,
    )


    # ----- Plot -----
    fig, ax = plt.subplots(1, 3)
    ax[0].spy(X_ref)
    ax[0].set_title('X_ref')
    ax[1].spy(X_inv_tip_SerinV)
    ax[1].set_title('X_inv_tip_SerinV')
    ax[2].spy(X_pivot_SerinV)
    ax[2].set_title('X_pivot_SerinV')
    plt.show()

    fig, ax = plt.subplots(1, 3)
    ax[0].matshow(X_ref.real)
    ax[0].set_title('X_ref (Real Part)')
    ax[1].matshow(X_inv_tip_SerinV.real)
    ax[1].set_title('X_inv_tip_SerinV')
    X_inv_tip_diff = X_ref - X_inv_tip_SerinV
    ax[2].matshow(X_inv_tip_diff.real)
    ax[2].set_title('X_inv_tip_diff')
    fig.colorbar(ax[2].matshow(X_inv_tip_diff.real), ax=ax[2], label="Relative error")
    plt.show()

    fig, ax = plt.subplots(1, 3)
    ax[0].matshow(X_ref.real)
    ax[0].set_title('X_ref (Real Part)')
    ax[1].matshow(X_pivot_SerinV.real)
    ax[1].set_title('X_pivot_SerinV')
    X_pivot_diff = X_ref - X_pivot_SerinV
    ax[2].matshow(X_pivot_diff.real)
    ax[2].set_title('X_pivot_diff')
    fig.colorbar(ax[2].matshow(X_pivot_diff.real), ax=ax[2], label="Relative error")
    plt.show()

