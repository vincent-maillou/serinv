# import cupy as cp

# # Define your matrices as CuPy arrays
# A = cp.array([[1, 2, 3], [4, 5, 6]])
# B = cp.array([[7, 8], [9, 10], [11, 12]])

# # Perform matrix multiplication
# C = cp.matmul(A, B)

# # Print the resulting matrix
# print(C)


import cupy as cp
import cupyx.scipy.linalg as cpla

# Define your matrices as CuPy arrays
A = cp.array([[5, 1, 1], [1, 5, 1], [1, 1, 5]], dtype=cp.float32)

P_A, L_A, U_A = cpla.lu(A)

# lu, piv = cpla.lu_factor(A)


# Print the resulting matrix
print(P_A)
print(L_A)
print(U_A)
