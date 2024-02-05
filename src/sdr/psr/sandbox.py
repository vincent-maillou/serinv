import numpy as np
import matplotlib.pyplot as plt


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
    mat_size = 10

    P = create_permutation_matrix(mat_size)

    # A = tridiag_matrix(mat_size)

    A = arrow_matrix(mat_size)

    v_right = create_vector(mat_size)
    v_bottom = create_vector(mat_size, row_vector=True)

    plt.matshow(P)
    plt.title("P")

    plt.matshow(A)
    plt.title("A")

    PAP = P @ A @ P.T
    Pv_right = P @ v_right
    Pv_bottom = v_bottom @ P.T

    plt.matshow(PAP)
    plt.title("PAP")

    # plt.matshow(v_right)
    # plt.title("v_right")

    # plt.matshow(v_bottom)
    # plt.title("v_bottom")

    # plt.matshow(Pv_right)
    # plt.title("Pv_right")

    # plt.matshow(Pv_bottom)
    # plt.title("Pv_bottom")

    plt.show()
