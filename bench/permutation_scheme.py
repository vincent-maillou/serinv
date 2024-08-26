import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = True

from matrix_utilities import (
    ColorScheme,
    make_pobta_symm_matrix,
    make_ddbta_matrix,
    annotate_matrix,
    hatch_matrix,
)

color_scheme = ColorScheme()


def make_bta_permutation_matrix(
    n: int,
):
    P_show = np.ones((n + 1, n + 1, 3)) * color_scheme.background
    P = np.zeros((n + 1, n + 1), dtype=bool)

    offset = 0
    for i in range(n):
        if i % 2 == 0:
            P[i, n // 2 + offset] = True
            P_show[i, n // 2 + offset] = color_scheme.P_elements
            offset += 1
        else:
            P[i, n // 2 - offset] = True
            P_show[i, n // 2 - offset] = color_scheme.P_elements

    P[-1:, -1:] = 1
    P_show[-1:, -1:] = color_scheme.P_elements

    return P, P_show


def get_fillin_hatching(A, P):
    # PAPt = P @ A @ P.T

    PAPt = np.einsum("ij,jkl->ikl", P, A)
    PAPt = np.einsum("ijk,jl->ilk", PAPt, P.T)

    n = PAPt.shape[0]
    hatching = np.zeros((n, n), dtype=bool)

    for i in range(n):
        fillin_lower = False
        fillin_upper = False
        for j in range(0, i):
            # if PAPt[i, j] > 0:
            if np.all(PAPt[i, j] != color_scheme.background):
                fillin_lower = True
            elif fillin_lower == True:
                hatching[i, j] = True

            # if PAPt[j, i] > 0:
            if np.all(PAPt[j, i] != color_scheme.background):
                fillin_upper = True
            elif fillin_upper == True:
                hatching[j, i] = True

    return hatching


def get_reordering_vector_from_permutation_matrix(P):
    return np.array([np.where(P[i] == 1)[0][0] for i in range(P.shape[0])])


def show_triptic(A, P, P_show, annotations=None, hatching=None):
    # PAPt = P @ A @ P.T

    PAPt = np.einsum("ij,jkl->ikl", P, A)
    PAPt = np.einsum("ijk,jl->ilk", PAPt, P.T)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(A)
    ax[0].set_title("$A$", fontsize=16)

    if annotations is not None:
        annotate_matrix(ax[0], A, annotations)

    ax[1].imshow(P_show)
    ax[1].set_title("$P$", fontsize=16)

    ax[2].imshow(PAPt)
    ax[2].set_title(f"$PAP^T$", fontsize=16)

    O = get_reordering_vector_from_permutation_matrix(P)

    annotations[:,] = annotations[O, :]
    annotations[:,] = annotations[:, O]

    if annotations is not None:
        annotate_matrix(ax[2], PAPt, annotations)

    if hatching is not None:
        hatch_matrix(ax[2], hatching)

    # Remove ticks
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
        a.set_xticklabels([])
        a.set_yticklabels([])

    # Add bottom figure annotations
    ax[0].text(0.5, -0.1, "a)", transform=ax[0].transAxes, ha="center", fontsize=12)
    ax[1].text(0.5, -0.1, "b)", transform=ax[1].transAxes, ha="center", fontsize=12)
    ax[2].text(0.5, -0.1, "c)", transform=ax[2].transAxes, ha="center", fontsize=12)


if __name__ == "__main__":
    n = 8
    save_fig = True

    P, P_show = make_bta_permutation_matrix(n)

    type = "Cholesky"

    if type == "Cholesky":
        # POBTA (CHOLESKY)
        A, annotations_A = make_pobta_symm_matrix(n)
        hatching_A = get_fillin_hatching(A, P)
        show_triptic(A, P, P_show, annotations_A, hatching_A)
        if save_fig:
            plt.savefig("pobta_permutation_scheme.png")
    elif type == "LU":
        # DDBTA (LU)
        B, annotations_B = make_ddbta_matrix(n)
        hatching_B = get_fillin_hatching(B, P)
        show_triptic(B, P, P_show, annotations_B, hatching_B)
        if save_fig:
            plt.savefig("ddbta_permutation_scheme.png")

    plt.show()
