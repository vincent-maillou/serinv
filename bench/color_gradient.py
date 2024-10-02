import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

colors = [(0, "#fe0000"), (0.30, "#93990c"), (1, "#00ff31")]  # Positions and colors
cmap = LinearSegmentedColormap.from_list("custom_gradient", colors)


def get_color_from_gradient(cmap, param):
    return cmap(param)


# No nested solving
# params = [0.7057, 0.7048, 0.7043, 0.713, 0.7129]
# params = [0.4043, 0.4796, 0.528, 0.5558, 0.565]
# params = [0.1905, 0.2892, 0.3677, 0.4249, 0.4606]
# params = [0.1083, 0.1744, 0.2593, 0.3341]
# params = [0.058, 0.1028, 0.1668]

# Nested solving
# params = [0.7057, 0.7048, 0.7043, 0.713, 0.7129]
# params = [0.4043, 0.4796, 0.528, 0.5558, 0.565]
params = [0.2085, 0.3092, 0.3834, 0.4352, 0.4666]
# params = [0.1478, 0.2218, 0.3082, 0.3721]
# params = [0.1109, 0.1775, 0.2530]

colors = [get_color_from_gradient(cmap, p) for p in params]

for i, color in enumerate(colors):
    # print(f"Param: {params[i]}, Color: {color}")
    # transofrm the rgb color in hex
    colors[i] = "#{:02x}{:02x}{:02x}".format(
        int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
    )
    print(f"Param: {params[i]}, Color: {colors[i]}")

# Plotting the colors to visualize the gradient
plt.figure(figsize=(8, 2))
for i, color in enumerate(colors):
    plt.plot([i, i + 1], [1, 1], color=color, linewidth=15)
# plt.xlim(0, 10)
plt.axis("off")
plt.show()
