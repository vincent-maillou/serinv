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


def create_vector(
    mat_size: int,
) -> np.ndarray:
    v = np.zeros((mat_size, 1))

    for i in range(mat_size):
        v[i] = i + 1

    return v


if __name__ == "__main__":
    mat_size = 10

    P = create_permutation_matrix(mat_size)

    A = tridiag_matrix(mat_size)

    plt.matshow(P)
    plt.title("P")

    plt.matshow(A)
    plt.title("A")

    PAP = P @ A @ P.T

    plt.matshow(PAP)
    plt.title("PAP")

    plt.show()
