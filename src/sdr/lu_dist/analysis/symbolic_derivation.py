import matplotlib.pyplot as plt
import numpy as np
import sympy as sp


def create_permutation_matrix_for_arrowhead(
    n_blocks: int,
) -> np.ndarray:
    P = np.zeros((n_blocks, n_blocks))

    offset = 0
    half = n_blocks // 2 - 1
    for i in range(n_blocks):
        if i % 2 == 1:
            P[i, (half + offset)] = 1
        else:
            P[i, (half - offset)] = 1
            offset += 1

    return P


def arrow_matrix(
    n_blocks: int,
) -> sp.Matrix:
    A = sp.zeros(n_blocks, n_blocks)

    for i in range(n_blocks):
        A[i, i] = sp.symbols(f"A_{i}_{i}")
        if i > 0:
            A[i, i - 1] = sp.symbols(f"A_{i}_{i-1}")
        if i < n_blocks - 1:
            A[i, i + 1] = sp.symbols(f"A_{i}_{i+1}")

        if i < n_blocks - 2:
            A[i, -1] = sp.symbols(f"A_{i}_{n_blocks-1}")
            A[-1, i] = sp.symbols(f"A_{n_blocks-1}_{i}")

    return A


if __name__ == "__main__":
    n_blocks = 10

    P = create_permutation_matrix_for_arrowhead(n_blocks)
    A = arrow_matrix(n_blocks)

    PAPt = P @ A @ P.T

    # Convert the sympy matrix to a numpy array of strings
    A_str = np.vectorize(str)(np.array(PAPt.tolist()))

    # Create a new figure
    fig, ax = plt.subplots()

    # Hide axes
    ax.axis("off")

    # Create a table
    table = plt.table(cellText=A_str, loc="center")

    # Show the plot
    plt.show()

    """ # 1. Create a matrix "A" of size 6*6
    A = sp.Matrix(6, 6, lambda i, j: sp.symbols(f"A_{i+1}_{j+1}"))

    # 2. Create an identity matrix of size 6*6
    I = sp.eye(6)

    # 3. Multiply A and I
    AI = A * I

    # Create an empty matrix of size 6*6
    empty_matrix = sp.zeros(6, 6)

    print(A)

    print(I)

    print(empty_matrix)

    # 4. Verify that A = A*I
    if A == AI:
        print("A = A * I")
    else:
        print("A != A * I") """

    """ # Create a sympy matrix
    A = sp.Matrix(6, 6, lambda i, j: sp.symbols(f"A_{i+1}_{j+1}"))

    # Convert the sympy matrix to a numpy array of strings
    A_str = np.vectorize(str)(np.array(A.tolist()))

    # Create a new figure
    fig, ax = plt.subplots()

    # Hide axes
    ax.axis('off')

    # Create a table
    table = plt.table(cellText=A_str, loc='center')

    # Show the plot
    plt.show() """
