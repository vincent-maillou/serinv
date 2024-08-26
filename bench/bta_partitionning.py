import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = True

from matrix_utilities import (
    make_pobta_symm_matrix,
    make_ddbta_matrix,
    slice_matrix,
    annotate_matrix,
)


def show_partitionning(
    A,
    n_processes,
    annotations=None,
):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(A)
    ax.set_title("$A$", fontsize=16)

    slice_matrix(ax, n_processes, A)

    if annotations is not None:
        annotate_matrix(ax, A, annotations)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    fig.tight_layout()


if __name__ == "__main__":
    n = 11
    n_processes = 3
    save_fig = True

    type = "Cholesky"

    if type == "Cholesky":
        A, annotations_A = make_pobta_symm_matrix(n)
        show_partitionning(A, n_processes, annotations_A)
        if save_fig:
            plt.savefig("pobta_partitionning.png")
    elif type == "LU":
        B, annotations_B = make_ddbta_matrix(n)
        show_partitionning(B, n_processes, annotations_B)
        if save_fig:
            plt.savefig("ddbta_partitionning.png")

    plt.show()
