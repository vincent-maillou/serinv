import numpy as np
import copy as cp
import matplotlib.pyplot as plt

from sdr.utils.matrix_generation import generate_tridiag_arrowhead_arrays, generate_tridiag_arrowhead_dense
from sdr.utils.matrix_transform import from_arrowhead_arrays_to_dense, from_dense_to_arrowhead_arrays, cut_to_blocktridiag_arrowhead
from sdr.lu.lu_factorize import lu_factorize_tridiag_arrowhead
from sdr.lu.lu_factorize_handle_pivoting import lu_factorize_handle_pivoting_tridiag_arrowhead, lu_factorize_inverse_tip_tridiag_arrowhead
from sdr.lu.lu_selected_inversion import lu_sinv_tridiag_arrowhead
from sdr.lu.lu_selected_inversion_handle_pivoting import lu_sinv_handle_pivoting_tridiag_arrowhead, lu_sinv_inverse_tip_tridiag_arrowhead


if __name__ == '__main__':


    nblocks = 8
    diag_blocksize = 336
    arrow_blocksize = 84

    # nblocks = 10
    # diag_blocksize = 30
    # arrow_blocksize = 10

    # nblocks = 5
    # diag_blocksize = 3
    # arrow_blocksize = 2

    symmetric = False
    diagonal_dominant = True
    seed = 63


    A_diag_dom = generate_tridiag_arrowhead_dense(
        nblocks,
        diag_blocksize,
        arrow_blocksize,
        symmetric,
        diagonal_dominant,
        seed,
    )

    X_diag_dom_ref = np.linalg.inv(A_diag_dom)
    X_diag_dom_ref = cut_to_blocktridiag_arrowhead(X_diag_dom_ref, diag_blocksize, arrow_blocksize)


    A_need_pivot = cp.deepcopy(A_diag_dom)

    A_need_pivot[-arrow_blocksize:, :] = np.random.rand(arrow_blocksize, A_need_pivot.shape[1])
    A_need_pivot[:, -arrow_blocksize:] = np.random.rand(A_need_pivot.shape[0], arrow_blocksize)

    A_inverse_tip = cp.deepcopy(A_need_pivot)

    X_need_pivot_ref = np.linalg.inv(A_need_pivot)
    X_need_pivot_ref = cut_to_blocktridiag_arrowhead(X_need_pivot_ref, diag_blocksize, arrow_blocksize)



    # ----- Dense to Arrays -----
    (
        A_diag_dom_diagonal_blocks,
        A_diag_dom_lower_diagonal_blocks,
        A_diag_dom_upper_diagonal_blocks,
        A_diag_dom_arrow_bottom_blocks,
        A_diag_dom_arrow_right_blocks,
        A_diag_dom_arrow_tip_block,
    ) = from_dense_to_arrowhead_arrays(A_diag_dom, diag_blocksize, arrow_blocksize)

    (
        A_need_pivot_diagonal_blocks,
        A_need_pivot_lower_diagonal_blocks,
        A_need_pivot_upper_diagonal_blocks,
        A_need_pivot_arrow_bottom_blocks,
        A_need_pivot_arrow_right_blocks,
        A_need_pivot_arrow_tip_block,
    ) = from_dense_to_arrowhead_arrays(A_need_pivot, diag_blocksize, arrow_blocksize)

    (
        A_inverse_tip_diagonal_blocks,
        A_inverse_tip_lower_diagonal_blocks,
        A_inverse_tip_upper_diagonal_blocks,
        A_inverse_tip_arrow_bottom_blocks,
        A_inverse_tip_arrow_right_blocks,
        A_inverse_tip_arrow_tip_block,
    ) = from_dense_to_arrowhead_arrays(A_inverse_tip, diag_blocksize, arrow_blocksize)


    # ----- LU FACTORIZATION -----
    (
        L_diag_dom_diagonal_blocks,
        L_diag_dom_lower_diagonal_blocks,
        L_diag_dom_arrow_bottom_blocks,
        U_diag_dom_diagonal_blocks,
        U_diag_dom_upper_diagonal_blocks,
        U_diag_dom_arrow_right_blocks,
    ) = lu_factorize_tridiag_arrowhead(
        A_diag_dom_diagonal_blocks,
        A_diag_dom_lower_diagonal_blocks,
        A_diag_dom_upper_diagonal_blocks,
        A_diag_dom_arrow_bottom_blocks,
        A_diag_dom_arrow_right_blocks,
        A_diag_dom_arrow_tip_block,
    )

    (
        L_need_pivot_diagonal_blocks,
        L_need_pivot_lower_diagonal_blocks,
        L_need_pivot_arrow_bottom_blocks,
        U_need_pivot_diagonal_blocks,
        U_need_pivot_upper_diagonal_blocks,
        U_need_pivot_arrow_right_blocks,
        P_arrow_tip_block,
    ) = lu_factorize_handle_pivoting_tridiag_arrowhead(
        A_need_pivot_diagonal_blocks,
        A_need_pivot_lower_diagonal_blocks,
        A_need_pivot_upper_diagonal_blocks,
        A_need_pivot_arrow_bottom_blocks,
        A_need_pivot_arrow_right_blocks,
        A_need_pivot_arrow_tip_block,
    )

    (
        L_inverse_tip_diagonal_blocks,
        L_inverse_tip_lower_diagonal_blocks,
        L_inverse_tip_arrow_bottom_blocks,
        U_inverse_tip_diagonal_blocks,
        U_inverse_tip_upper_diagonal_blocks,
        U_inverse_tip_arrow_right_blocks,
    ) = lu_factorize_inverse_tip_tridiag_arrowhead(
        A_inverse_tip_diagonal_blocks,
        A_inverse_tip_lower_diagonal_blocks,
        A_inverse_tip_upper_diagonal_blocks,
        A_inverse_tip_arrow_bottom_blocks,
        A_inverse_tip_arrow_right_blocks,
        A_inverse_tip_arrow_tip_block,
    )


    # ----- SELECTED INVERSION -----
    (
        X_diag_dom_diagonal_blocks,
        X_diag_dom_lower_diagonal_blocks,
        X_diag_dom_upper_diagonal_blocks,
        X_diag_dom_arrow_bottom_blocks,
        X_diag_dom_arrow_right_blocks,
        X_diag_dom_arrow_tip_block,
    ) = lu_sinv_tridiag_arrowhead(
        L_diag_dom_diagonal_blocks,
        L_diag_dom_lower_diagonal_blocks,
        L_diag_dom_arrow_bottom_blocks,
        U_diag_dom_diagonal_blocks,
        U_diag_dom_upper_diagonal_blocks,
        U_diag_dom_arrow_right_blocks,
    )

    (
        X_need_pivot_diagonal_blocks,
        X_need_pivot_lower_diagonal_blocks,
        X_need_pivot_upper_diagonal_blocks,
        X_need_pivot_arrow_bottom_blocks,
        X_need_pivot_arrow_right_blocks,
        X_need_pivot_arrow_tip_block,
    ) = lu_sinv_handle_pivoting_tridiag_arrowhead(
        L_need_pivot_diagonal_blocks,
        L_need_pivot_lower_diagonal_blocks,
        L_need_pivot_arrow_bottom_blocks,
        U_need_pivot_diagonal_blocks,
        U_need_pivot_upper_diagonal_blocks,
        U_need_pivot_arrow_right_blocks,
        P_arrow_tip_block,
    )

    (
        X_inverse_tip_diagonal_blocks,
        X_inverse_tip_lower_diagonal_blocks,
        X_inverse_tip_upper_diagonal_blocks,
        X_inverse_tip_arrow_bottom_blocks,
        X_inverse_tip_arrow_right_blocks,
        X_inverse_tip_arrow_tip_block,
    ) = lu_sinv_inverse_tip_tridiag_arrowhead(
        L_inverse_tip_diagonal_blocks,
        L_inverse_tip_lower_diagonal_blocks,
        L_inverse_tip_arrow_bottom_blocks,
        U_inverse_tip_diagonal_blocks,
        U_inverse_tip_upper_diagonal_blocks,
        U_inverse_tip_arrow_right_blocks,
    )


    # ----- Arrays to Dense -----
    X_diag_dom_SerinV = from_arrowhead_arrays_to_dense(
        X_diag_dom_diagonal_blocks,
        X_diag_dom_lower_diagonal_blocks,
        X_diag_dom_upper_diagonal_blocks,
        X_diag_dom_arrow_bottom_blocks,
        X_diag_dom_arrow_right_blocks,
        X_diag_dom_arrow_tip_block,
    )

    X_need_pivot_SerinV = from_arrowhead_arrays_to_dense(
        X_need_pivot_diagonal_blocks,
        X_need_pivot_lower_diagonal_blocks,
        X_need_pivot_upper_diagonal_blocks,
        X_need_pivot_arrow_bottom_blocks,
        X_need_pivot_arrow_right_blocks,
        X_need_pivot_arrow_tip_block,
    )

    X_inverse_tip_SerinV = from_arrowhead_arrays_to_dense(
        X_inverse_tip_diagonal_blocks,
        X_inverse_tip_lower_diagonal_blocks,
        X_inverse_tip_upper_diagonal_blocks,
        X_inverse_tip_arrow_bottom_blocks,
        X_inverse_tip_arrow_right_blocks,
        X_inverse_tip_arrow_tip_block,
    )



    # fig, ax = plt.subplots(1, 3)
    # ax[0].matshow(X_diag_dom_ref)
    # ax[0].set_title('X_diag_dom_ref')
    # ax[1].matshow(X_diag_dom_SerinV)
    # ax[1].set_title('X_diag_dom_SerinV')
    # X_diag_dom_diff = X_diag_dom_ref - X_diag_dom_SerinV
    # ax[2].matshow(X_diag_dom_diff)
    # ax[2].set_title('X_diag_dom_diff')
    # fig.colorbar(ax[2].matshow(X_diag_dom_diff), ax=ax[2], label="Relative error")
    # plt.show()


    fig, ax = plt.subplots(1, 3)
    ax[0].matshow(X_need_pivot_ref)
    ax[0].set_title('X_need_pivot_ref')
    ax[1].matshow(X_need_pivot_SerinV)
    ax[1].set_title('X_need_pivot_SerinV')
    X_need_pivot_diff = X_need_pivot_ref - X_need_pivot_SerinV
    ax[2].matshow(X_need_pivot_diff)
    ax[2].set_title('X_need_pivot_diff')
    fig.colorbar(ax[2].matshow(X_need_pivot_diff), ax=ax[2], label="Relative error")
    plt.show()


    fig, ax = plt.subplots(1, 3)
    ax[0].matshow(X_need_pivot_ref)
    ax[0].set_title('X_need_pivot_ref')
    ax[1].matshow(X_inverse_tip_SerinV)
    ax[1].set_title('X_inverse_tip_SerinV')
    X_need_pivot_diff = X_need_pivot_ref - X_inverse_tip_SerinV
    ax[2].matshow(X_need_pivot_diff)
    ax[2].set_title('X_need_pivot_diff')
    fig.colorbar(ax[2].matshow(X_need_pivot_diff), ax=ax[2], label="Relative error")
    plt.show()