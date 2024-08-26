import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from numpy.typing import ArrayLike

from dataclasses import dataclass, field

plt.rcParams["text.usetex"] = True


colormap = cm.get_cmap("viridis")
norm = Normalize(vmin=0, vmax=1)


@dataclass
class ColorScheme:
    """Dataclass that defines a set of colors for the
    matrix utilities functions. Colors are defined as
    RGB np.array().

    Attributes:
        background (np.ndarray): Background color of the matrix.
        A_elements (np.ndarray): Color of the A (Initial) matrix elements.
        P_elements (np.ndarray): Color of the P (permutation) matrix elements.
        fillin_hatching (np.ndarray): Hatch color for the fill-in elements.
        factor_elements (np.ndarray): Color of the factors elements.
        boundary_elements (np.ndarray): Color of the boundary elements.
        inverse_elements (np.ndarray): Color of the inverse elements.
    """

    # hex colors: #CE7727 #000F42 #D4BEAA #166A8C #02946A
    background: ArrayLike = field(default_factory=lambda: np.array([212, 190, 170]) / 255)
    A_elements: ArrayLike = field(default_factory=lambda: np.array([22, 106, 140]) / 255)
    P_elements: ArrayLike = field(default_factory=lambda: np.array([0, 77, 64]) / 255)
    fillin_hatching: ArrayLike = field(
        default_factory=lambda: np.array([255, 0, 0]) / 255
    )
    factor_elements: ArrayLike = field(
        default_factory=lambda: np.array([0, 123, 109]) / 255
    )
    boundary_elements: ArrayLike = field(
        default_factory=lambda: np.array([206, 119, 39]) / 255
    )
    inverse_elements: ArrayLike = field(
        default_factory=lambda: np.array([0, 15, 66]) / 255
    )

color_scheme = ColorScheme()


def make_pobta_lower_matrix(
    n: int,
):
    A = np.ones((n + 1, n + 1, 3)) * color_scheme.background
    annotations = np.zeros((n + 1, n + 1), dtype="U20")

    for i in range(n):
        A[i, i] = color_scheme.A_elements
        annotations[i, i] = f"$A_{{{i+1}, {i+1}}}$"

        A[-1:, i] = color_scheme.A_elements
        annotations[n, i] = f"$A_{{{n+1}, {i+1}}}$"

        if i < n - 1:
            A[(i + 1), i] = color_scheme.A_elements
            annotations[i + 1, i] = f"$A_{{{i+2}, {i+1}}}$"

    A[-1:, -1:] = color_scheme.A_elements
    annotations[-1, -1] = f"$A_{{{n+1}, {n+1}}}$"

    return A, annotations


def show_lower_diag_elements(
    ax,
    diag_colors,
):
    n = diag_colors.shape[0]
    offset = 0.5
    for i in range(n):
        ax.add_patch(
            plt.Polygon(
                [
                    [i - offset, i - offset],
                    [i - offset, i - offset + 1],
                    [i - offset + 1, i - offset + 1],
                ],
                closed=True,
                fill=True,
                edgecolor="none",
                facecolor=diag_colors[i],
            )
        )


def make_pobta_symm_matrix(
    n: int,
):
    A = np.ones((n + 1, n + 1, 3)) * color_scheme.background
    annotations = np.zeros((n + 1, n + 1), dtype="U20")

    for i in range(n):
        A[i, i] = color_scheme.A_elements
        annotations[i, i] = f"$A_{{{i+1},{i+1}}}$"

        A[-1:, i] = color_scheme.A_elements
        annotations[n, i] = f"$A_{{{n+1},{i+1}}}$"

        A[i, -1:] = color_scheme.A_elements
        annotations[i, n] = f"$A_{{{n+1},{i+1}}}^{{T}}$"

        if i < n - 1:
            A[(i + 1), i] = color_scheme.A_elements
            annotations[i + 1, i] = f"$A_{{{i+2},{i+1}}}$"

            A[i, (i + 1)] = color_scheme.A_elements
            annotations[i, i + 1] = f"$A_{{{i+1},{i+2}}}^{{T}}$"

    A[-1:, -1:] = color_scheme.A_elements
    annotations[-1, -1] = f"$A_{{{n+1},{n+1}}}$"

    return A, annotations


def make_ddbta_matrix(
    n: int,
):
    A = np.ones((n + 1, n + 1, 3)) * color_scheme.background
    annotations = np.zeros((n + 1, n + 1), dtype="U20")

    for i in range(n):
        A[i, i] = color_scheme.A_elements
        annotations[i, i] = f"$A_{{{i+1},{i+1}}}$"

        A[-1:, i] = color_scheme.A_elements
        annotations[n, i] = f"$A_{{{n+1},{i+1}}}$"

        A[i, -1:] = color_scheme.A_elements
        annotations[i, n] = f"$A_{{{i+1},{n+1}}}$"

        if i < n - 1:
            A[(i + 1), i] = color_scheme.A_elements
            annotations[i + 1, i] = f"$A_{{{i+2},{i+1}}}$"

            A[i, (i + 1)] = color_scheme.A_elements
            annotations[i, i + 1] = f"$A_{{{i+1},{i+2}}}$"

    A[-1:, -1:] = color_scheme.A_elements
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
                #color="black",
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
                        # edgecolor="red",
                        edgecolor=color_scheme.fillin_hatching,
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
    # linewidth = 1
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
