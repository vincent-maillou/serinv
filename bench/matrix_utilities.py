import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

plt.rcParams["text.usetex"] = True


colormap = cm.get_cmap("viridis")
norm = Normalize(vmin=0, vmax=1)
background_color = colormap(norm(0))[:3]

# initial_color = np.array([224, 123, 57]) / 255
initial_color = np.array([0, 143, 140]) / 255


def make_pobta_matrix(
    n: int,
):
    A = np.ones((n + 1, n + 1, 3)) * background_color
    annotations = np.zeros((n + 1, n + 1), dtype="U20")

    for i in range(n):
        A[i, i] = initial_color
        annotations[i, i] = f"$A_{{{i+1}, {i+1}}}$"

        A[-1:, i] = initial_color
        annotations[n, i] = f"$A_{{{n+1}, {i+1}}}$"

        if i < n - 1:
            A[(i + 1), i] = initial_color
            annotations[i + 1, i] = f"$A_{{{i+2}, {i+1}}}$"

    A[-1:, -1:] = initial_color
    annotations[-1, -1] = f"$A_{{{n+1}, {n+1}}}$"

    return A, annotations


def make_ddbta_matrix(
    n: int,
):
    A = np.ones((n + 1, n + 1, 3)) * background_color
    annotations = np.zeros((n + 1, n + 1), dtype="U20")

    for i in range(n):
        A[i, i] = initial_color
        annotations[i, i] = f"$A_{{{i+1},{i+1}}}$"

        A[-1:, i] = initial_color
        annotations[n, i] = f"$A_{{{n+1},{i+1}}}$"

        A[i, -1:] = initial_color
        annotations[i, n] = f"$A_{{{i+1},{n+1}}}$"

        if i < n - 1:
            A[(i + 1), i] = initial_color
            annotations[i + 1, i] = f"$A_{{{i+2},{i+1}}}$"

            A[i, (i + 1)] = initial_color
            annotations[i, i + 1] = f"$A_{{{i+1},{i+2}}}$"

    A[-1:, -1:] = initial_color
    annotations[-1, -1] = f"$A_{{{n+1},{n+1}}}$"

    return A, annotations


def annotate_matrix(ax, matrix, annotations):
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                annotations[i, j],
                va="center",
                ha="center",
                color="white",
                fontsize=12,
            )


def hatch_matrix(ax, hatching):
    for i in range(hatching.shape[0]):
        for j in range(hatching.shape[1]):
            if hatching[i, j]:
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        fill=False,
                        hatch="//",
                        edgecolor="red",
                    )
                )


def get_section_sizes(n, n_processes):
    Neach_section, extras = divmod(n, n_processes)
    section_sizes = (n_processes - extras) * [Neach_section] + extras * [
        Neach_section + 1
    ]
    return section_sizes


def slice_matrix(
    ax,
    n_processes,
    matrix,
):
    n = matrix.shape[0] - 1
    linewidth = 1.5
    color = "white"
    # color = "black"

    section_sizes = get_section_sizes(n, n_processes)

    ax.plot(
        [
            -0.49,
            -0.49,
        ],
        [
            -0.49,
            matrix.shape[0] - 0.5,
        ],
        color=color,
        linestyle="--",
        linewidth=linewidth,
    )

    ax.plot(
        [
            -0.49,
            matrix.shape[0] - 0.5,
        ],
        [
            -0.49,
            -0.49,
        ],
        color=color,
        linestyle="--",
        linewidth=linewidth,
    )

    for i in range(len(section_sizes)):
        ax.plot(
            [
                np.sum(section_sizes[: i + 1]) - 0.5,
                np.sum(section_sizes[: i + 1]) - 0.5,
            ],
            [
                np.sum(section_sizes[: i + 1]) - 0.5,
                matrix.shape[0] - 0.5,
            ],
            color=color,
            linestyle="--",
            linewidth=linewidth,
        )

        ax.plot(
            [
                np.sum(section_sizes[: i + 1]) - 0.5,
                matrix.shape[0] - 0.5,
            ],
            [
                np.sum(section_sizes[: i + 1]) - 0.5,
                np.sum(section_sizes[: i + 1]) - 0.5,
            ],
            color=color,
            linestyle="--",
            linewidth=linewidth,
        )

        ax.text(
            np.sum(section_sizes[:i]) + section_sizes[i] / 2 - 0.5,
            matrix.shape[0],
            f"$P_{i}$",
            ha="center",
            fontsize=12,
        )

    ax.text(
        matrix.shape[0] - 1,
        matrix.shape[0],
        f"$ALL$",
        ha="center",
        fontsize=12,
    )
