from sdr.lu.lu_selected_inversion import sinv_ndiags_greg


import numpy as np
import pytest

seed = 10

def cut_to_banded(
    A: np.ndarray,
    ndiags : int
):
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if abs(i-j) > ndiags:
                A[i, j] = 0
    return A


def create_banded_matrix(
        matrix_size: int,
        ndiags: int
):
    A = np.zeros((matrix_size, matrix_size), dtype=np.float32)
    assert matrix_size >= 2*ndiags+1
    tmp = np.random.randint(1,10, size=(matrix_size,  2*ndiags+1)) #+ 1j * np.random.rand(matrix_size, 2*ndiags+1)
    for i in range(matrix_size):
        for j in range(max(0,i-ndiags), min(matrix_size, i+ndiags+1)):
            A[i, j] = tmp[i, j-i+ndiags]
    np.fill_diagonal(A, np.sum(np.abs(A), axis=1)+10)
    return A

np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))

@pytest.mark.parametrize(
        "matrix_size, ndiags",
        [(1, 0),
         (2, 0),
         (3, 0),
         (4, 0),
         (5, 0),
         (3, 1),
         (4, 1),
         (5, 1),
         (5, 2),
         (6, 2),
         (7, 2),
         (8, 2),
         (9, 3),
         (10, 2),
         (10, 4),
         (11, 4),
         (12, 4),
         (20, 5),
         (21, 5),
         (23, 5),
         (25, 5),
         (20, 6),
         (21, 6),
         (23, 6),
         (25, 6),
        ]
)
def test_sinv_ndiags_greg(
        matrix_size: int,
        ndiags: int
):
    matrix_size = 6
    ndiags = 1
    A = create_banded_matrix(matrix_size, ndiags)
    reference_inverse = np.linalg.inv(A)
    # assert(np.allclose(reference_inverse @ A, np.eye(matrix_size), atol=1e-07))
    assert np.linalg.norm(reference_inverse @ A- np.eye(matrix_size))/np.linalg.norm(A) < 1e-7
    
    # test_inverse = sinv_ndiags_greg(A, 2*ndiags)
    test_inverse = sinv_ndiags_greg(A, matrix_size)
    cut_to_banded(reference_inverse, ndiags)
    cut_to_banded(test_inverse, ndiags)
    assert np.allclose(test_inverse, reference_inverse)


if __name__ == "__main__":
    params = [(1, 0),
         (2, 0),
         (3, 0),
         (4, 0),
         (5, 0),
         (3, 1),
         (4, 1),
         (5, 1),
         (5, 2),
         (6, 2),
         (7, 2),
         (8, 2),
         (9, 3),
         (10, 2),
         (10, 4),
         (11, 4),
         (12, 4),
         (20, 5),
         (21, 5),
         (23, 5),
         (25, 5),
         (20, 6),
         (21, 6),
         (23, 6),
         (25, 6),
        ]
    for param in params:
        test_sinv_ndiags_greg(*param)