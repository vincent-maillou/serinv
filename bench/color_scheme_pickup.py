import numpy as np
import matplotlib.pyplot as plt

# Define the dimensions of the matrix
colors, nuances = 5, 4

# Create a matrix of RGBA colors
# Each color is represented as [R, G, B, A] where values range from 0 to 1
color_matrix = np.zeros((colors, nuances, 4))

# CE7727 [206,119,39]
# 000F42 [0,15,66]
# D4BEAA [212,190,170]
# 166A8C [22,106,140]
# 02946A [2,148,106]

c0 = np.array([206, 119, 39, 255]) / 255
c1 = np.array([0, 15, 66, 255]) / 255
c2 = np.array([212, 190, 170, 255]) / 255
c3 = np.array([22, 106, 140, 255]) / 255
c4 = np.array([2, 148, 106, 255]) / 255

color_matrix[0, :] = c0
color_matrix[1, :] = c1
color_matrix[2, :] = c2
color_matrix[3, :] = c3
color_matrix[4, :] = c4

alpha = [1, 0.75, 0.5, 0.25]

# Fill the matrix with random colors
for c in range(colors):
    for n in range(nuances):
        color_matrix[c, n, 3] = alpha[n]

# Plot the matrix
fig, ax = plt.subplots()
ax.imshow(color_matrix, aspect="auto")

# Annotate each cell with the corresponding color
for c in range(colors):
    for n in range(nuances):
        ax.text(
            n,
            c,
            f"({color_matrix[c, n, 0]:.2f}, {color_matrix[c, n, 1]:.2f}, {color_matrix[c, n, 2]:.2f}, {color_matrix[c, n, 3]:.2f})",
            ha="center",
            va="center",
            color="black" if color_matrix[c, n, 3] <= 0.5 else "white",
        )

# Remove the axes for better visualization
ax.axis("off")

plt.show()
