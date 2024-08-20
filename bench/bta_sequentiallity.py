import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

plt.rcParams["text.usetex"] = True

from matrix_utilities import (
    make_pobta_matrix,
    make_ddbta_matrix,
    slice_matrix,
    annotate_matrix,
    hatch_matrix,
    get_section_sizes,
)

colormap = cm.get_cmap("viridis")
norm = Normalize(vmin=0, vmax=1)
background_color = colormap(norm(0))[:3]

factors_color = np.array([0.5, 0.5, 0.5])
# factors_color = colormap(0.5)[:3]

# boundary_color = colormap(0.8)[:3]
boundary_color = np.array([0.6, 0.4, 0.2])

inverse_color = np.array([6, 57, 112]) / 255


def d_pobtaf_matrix(A, n_processes):
    n = A.shape[0] - 1

    A_factorized = np.ones((n + 1, n + 1, 3)) * background_color
    annotations = np.zeros((n + 1, n + 1), dtype="U20")

    for i in range(n):
        A_factorized[i, i] = factors_color
        annotations[i, i] = f"$D_{{{i+1},{i+1}}}$"

        A_factorized[-1:, i] = factors_color
        annotations[n, i] = f"$L_{{{n+1},{i+1}}}$"

        if np.all(A[i, -1:] != background_color):
            A_factorized[i, -1:] = factors_color
            annotations[i, n] = f"$U_{{{i+1},{n+1}}}$"

        if i < n - 1:
            A_factorized[(i + 1), i] = factors_color
            annotations[i + 1, i] = f"$L_{{{i+2},{i+1}}}$"

            if np.all(A[i, (i + 1)] != background_color):
                A_factorized[i, (i + 1)] = factors_color
                annotations[i, i + 1] = f"$U_{{{i+1},{i+2}}}$"

    A_factorized[-1:, -1:] = factors_color
    annotations[-1, -1] = f"$D_{{{n+1},{n+1}}}$"

    section_sizes = get_section_sizes(n, n_processes)

    for i in range(len(section_sizes)):
        boundary = np.sum(section_sizes[: i + 1])

        # Diagonal boundary blocks
        A_factorized[boundary - 1, boundary - 1] = boundary_color
        annotations[boundary - 1, boundary - 1] = f"$R_{{{boundary},{boundary}}}$"

        A_factorized[boundary, boundary] = boundary_color
        annotations[boundary, boundary] = f"$R_{{{boundary+1},{boundary+1}}}$"

        # Lower/Upper boundary blocks
        A_factorized[boundary, boundary - 1] = boundary_color
        annotations[boundary, boundary - 1] = f"$R_{{{boundary+1},{boundary}}}$"

        # if A[boundary - 1, boundary] > 0:
        if np.all(A[boundary - 1, boundary] != background_color):
            A_factorized[boundary - 1, boundary] = boundary_color
            annotations[boundary - 1, boundary] = f"$R_{{{boundary},{boundary+1}}}$"

        # Arrowhead blocks
        A_factorized[n, boundary - 1] = boundary_color
        annotations[n, boundary - 1] = f"$R_{{{n+1},{boundary}}}$"

        A_factorized[n, boundary] = boundary_color
        annotations[n, boundary] = f"$R_{{{n+1},{boundary+1}}}$"

        # if A[boundary - 1, n] > 0:
        if np.all(A[boundary - 1, n] != background_color):
            A_factorized[boundary - 1, n] = boundary_color
            annotations[boundary - 1, n] = f"$R_{{{boundary},{n+1}}}$"

            A_factorized[boundary, n] = boundary_color
            annotations[boundary, n] = f"$R_{{{boundary+1},{n+1}}}$"

        # Lower/Upper buffer blocks
        if i < len(section_sizes) - 1:
            boundary_p1 = np.sum(section_sizes[: i + 2]) - 1

            A_factorized[boundary_p1, boundary] = boundary_color
            annotations[boundary_p1, boundary] = f"$R_{{{boundary_p1+1},{boundary+1}}}$"

            for i in range(boundary + 1, boundary_p1):
                A_factorized[i, boundary] = factors_color
                annotations[i, boundary] = f"$L_{{{i+1},{boundary+1}}}$"

            # if A[boundary - 1, n] > 0:
            if np.all(A[boundary - 1, n] != background_color):
                A_factorized[boundary, boundary_p1] = boundary_color
                annotations[boundary, boundary_p1] = (
                    f"$R_{{{boundary+1},{boundary_p1+1}}}$"
                )

                for i in range(boundary + 1, boundary_p1):
                    A_factorized[boundary, i] = factors_color
                    annotations[boundary, i] = f"$U_{{{boundary+1},{i+1}}}$"

    return A_factorized, annotations


def get_d_pobtaf_hatching(A, P):
    n = A.shape[0] - 1
    hatching = np.zeros((n + 1, n + 1), dtype=bool)

    section_sizes = get_section_sizes(n, n_processes)

    for i in range(len(section_sizes)):
        boundary = np.sum(section_sizes[: i + 1])

        if i < len(section_sizes) - 1:
            boundary_p1 = np.sum(section_sizes[: i + 2]) - 1

            hatching[boundary + 1 : boundary_p1 + 1, boundary] = True

            if np.all(A[0, 1] != background_color):
                hatching[boundary, boundary + 1 : boundary_p1 + 1] = True

    return hatching


def make_reduce_system(
    A_factorized,
    annotations_factorized,
    n_processes,
):
    n = A_factorized.shape[0] - 1
    r_size = 2 * n_processes
    R = np.ones((r_size, r_size, 3)) * background_color
    annotations = np.zeros((r_size, r_size), dtype="U20")

    section_sizes = get_section_sizes(n, n_processes)

    for i in range(len(section_sizes)):
        boundary = np.sum(section_sizes[: i + 1])
        j = 2 * i

        # Diagonal blocks
        R[j, j] = A_factorized[boundary - 1, boundary - 1]
        annotations[j, j] = annotations_factorized[boundary - 1, boundary - 1]

        R[j + 1, j + 1] = A_factorized[boundary, boundary]
        annotations[j + 1, j + 1] = annotations_factorized[boundary, boundary]

        # Lower/Upper blocks
        R[j + 1, j] = A_factorized[boundary, boundary - 1]
        annotations[j + 1, j] = annotations_factorized[boundary, boundary - 1]

        R[j, j + 1] = A_factorized[boundary - 1, boundary]
        annotations[j, j + 1] = annotations_factorized[boundary - 1, boundary]

        # Lower arrowhead blocks
        R[r_size - 1, j] = A_factorized[n, boundary - 1]
        annotations[r_size - 1, j] = annotations_factorized[n, boundary - 1]

        R[r_size - 1, j + 1] = A_factorized[n, boundary]
        annotations[r_size - 1, j + 1] = annotations_factorized[n, boundary]

        # Upper arrowhead blocks
        R[j, r_size - 1] = A_factorized[boundary - 1, n]
        annotations[j, r_size - 1] = annotations_factorized[boundary - 1, n]

        R[j + 1, r_size - 1] = A_factorized[boundary, n]
        annotations[j + 1, r_size - 1] = annotations_factorized[boundary, n]

        # Lower/Upper buffer blocks
        if i < len(section_sizes) - 1:
            boundary_p1 = np.sum(section_sizes[: i + 2]) - 1
            j_p1 = 2 * (i + 1)

            R[j_p1, j + 1] = A_factorized[boundary_p1, boundary]
            annotations[j_p1, j + 1] = annotations_factorized[boundary_p1, boundary]

            R[j + 1, j_p1] = A_factorized[boundary, boundary_p1]
            annotations[j + 1, j_p1] = annotations_factorized[boundary, boundary_p1]

    return R, annotations


def get_R_hatching(A, R, n_processes):
    r_size = 2 * n_processes
    hatching = np.zeros((r_size, r_size), dtype=bool)

    n = A_factorized.shape[0] - 1

    section_sizes = get_section_sizes(n, n_processes)

    for i in range(len(section_sizes)):
        j = 2 * i
        if i < len(section_sizes) - 1:
            j_p1 = 2 * (i + 1)

            hatching[j_p1, j + 1] = True

            if np.all(A[0, 1] != background_color):
                hatching[j + 1, j_p1] = True

    return hatching


def inverse_reduce_system(
    R,
    A,
):
    n = A.shape[0] - 1
    Xr = np.ones(R.shape) * background_color
    annotations = np.zeros((n + 1, n + 1), dtype="U20")

    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if np.all(R[i, j] != background_color):
                Xr[i, j] = inverse_color
                annotations[i, j] = f"$Xr_{{{i+1},{j+1}}}$"

    return Xr, annotations


def initialize_inverse_matrix(
    A_factorize,
    annotations_factorized,
    Xr,
):
    Xinit = A_factorize.copy()
    annotations_Xinit = annotations_factorized.copy()

    for i in range(Xinit.shape[0]):
        for j in range(Xinit.shape[1]):
            if np.all(Xinit[i, j] == boundary_color):
                Xinit[i, j] = inverse_color
                annotations_Xinit[i, j] = f"$X_{{{i+1},{j+1}}}$"

    return Xinit, annotations_Xinit


def invert_matrix(Xinit, annotations_Xinit):
    X = Xinit.copy()
    annotations_X = annotations_Xinit.copy()

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if np.all(X[i, j] != background_color):
                X[i, j] = inverse_color
                annotations_X[i, j] = f"$X_{{{i+1},{j+1}}}$"

    return X, annotations_X


if __name__ == "__main__":
    n = 11
    n_processes = 3
    save_fig = True

    type = "LU"

    if type == "Cholesky":
        A, annotations_A = make_pobta_matrix(n)
        hatching_A = get_d_pobtaf_hatching(A, n_processes)
    elif type == "LU":
        A, annotations_A = make_ddbta_matrix(n)
        hatching_A = get_d_pobtaf_hatching(A, n_processes)

    # Print initial matrix
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(A)
    ax.set_title("$A$", fontsize=16)
    ax.text(0.5, -0.1, "a)", transform=ax.transAxes, ha="center", fontsize=12)
    slice_matrix(ax, n_processes, A)
    annotate_matrix(ax, A, annotations_A)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    fig.tight_layout()

    if save_fig:
        plt.savefig(f"{type}_sequentiallity_matrix_a.png")

    # Print factorized matrix
    A_factorized, annotations_factorized = d_pobtaf_matrix(A, n_processes)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(A_factorized)
    ax.set_title(type, fontsize=16)
    ax.text(0.5, -0.1, "b)", transform=ax.transAxes, ha="center", fontsize=12)
    slice_matrix(ax, n_processes, A_factorized)
    annotate_matrix(ax, A_factorized, annotations_factorized)
    hatch_matrix(ax, hatching_A)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    fig.tight_layout()

    if save_fig:
        plt.savefig(f"{type}_sequentiallity_matrix_b.png")

    # Print reduced system
    R, annotations_R = make_reduce_system(
        A_factorized, annotations_factorized, n_processes
    )
    hatching_R = get_R_hatching(A, R, n_processes)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.imshow(R)
    ax.set_title("$R$", fontsize=16)
    ax.text(1.1, 0.5, "c)", transform=ax.transAxes, ha="center", fontsize=12)
    annotate_matrix(ax, R, annotations_R)
    hatch_matrix(ax, hatching_R)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    fig.tight_layout()

    if save_fig:
        plt.savefig(f"{type}_sequentiallity_matrix_c.png")

    # Print inverse reduced system
    Xr, annotations_Xr = inverse_reduce_system(R, A)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.imshow(Xr)
    ax.set_title("$X_r$", fontsize=16)
    ax.text(1.1, 0.5, "d)", transform=ax.transAxes, ha="center", fontsize=12)
    annotate_matrix(ax, Xr, annotations_Xr)
    hatch_matrix(ax, hatching_R)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    fig.tight_layout()

    if save_fig:
        plt.savefig(f"{type}_sequentiallity_matrix_d.png")

    # Print initialization of inverse matrix
    Xinit, annotation_Xinit = initialize_inverse_matrix(
        A_factorized, annotations_factorized, Xr
    )

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(Xinit)
    ax.set_title("$X_{init}$", fontsize=16)
    ax.text(0.5, -0.1, "e)", transform=ax.transAxes, ha="center", fontsize=12)
    slice_matrix(ax, n_processes, Xinit)
    annotate_matrix(ax, Xinit, annotation_Xinit)
    hatch_matrix(ax, hatching_A)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    fig.tight_layout()

    if save_fig:
        plt.savefig(f"{type}_sequentiallity_matrix_e.png")

    # Print final inverse matrix
    X, annotation_X = invert_matrix(Xinit, annotation_Xinit)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(X)
    ax.set_title("$X$", fontsize=16)
    ax.text(0.5, -0.1, "f)", transform=ax.transAxes, ha="center", fontsize=12)
    slice_matrix(ax, n_processes, X)
    annotate_matrix(ax, X, annotation_X)
    hatch_matrix(ax, hatching_A)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    fig.tight_layout()

    if save_fig:
        plt.savefig(f"{type}_sequentiallity_matrix_f.png")

    plt.show()
