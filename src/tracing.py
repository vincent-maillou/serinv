from dataclasses import dataclass, field
from collections import Counter, defaultdict
import numpy as np
import networkx as nx


@dataclass
class Vertex:
    arr: str
    ind: tuple[int, ...]
    version: int
    is_zero: bool = False

    def __init__(
        self, arr: str, arr_obj: np.ndarray, ind: tuple[int, ...], output: bool = False
    ):
        self.arr = arr
        self.ind = ind
        if np.isclose(arr_obj[ind], 0):
            self.is_zero = True
        if output:
            vertex_versions[self.base] = vertex_versions[self.base] + 1

        self.version = vertex_versions[self.base]

    # def __eq__(self, __value: object) -> bool:
    #     return isinstance(__value, Vertex) and self.arr == __value.arr and self.ind == __value.ind

    @property
    def base(self) -> str:
        return hash((self.arr, self.ind))

    @property
    def ssa(self) -> str:
        return hash((self.arr, self.ind, self.version))

    @property
    def name(self) -> str:
        return f"{self.arr}[{self.ind}, {self.version}]"


@dataclass
class Edge:
    u: Vertex
    v: Vertex


vertex_versions = Counter()


@dataclass
class CDAG:
    vertices: list[Vertex] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    in_neighs: defaultdict[Vertex, list[Vertex]] = field(
        default_factory=lambda: defaultdict(list)
    )
    out_neighs: defaultdict[Vertex, list[Vertex]] = field(
        default_factory=lambda: defaultdict(list)
    )

    """
    Implement method to add all edges to all pairs of u's to v
    """

    def add(self, us: list[Vertex], v: Vertex):
        for u in us:
            if u.is_zero:
                continue
            if u not in self.vertices:
                self.vertices.append(u)
        if v not in self.vertices:
            self.vertices.append(v)

        for u in us:
            if u.is_zero:
                continue
            self.edges.append(Edge(u, v))
            self.out_neighs[u.ssa].append(v)
            self.in_neighs[v.ssa].append(u)

    """
    Plot the directed acyclic graph in a 3D space. X and Y coordinates of vertices
    are i and j indices of the matrix, and Z coordinate is the version of the vertex.
    """

    def plot(self):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib.patches import FancyArrowPatch
        from mpl_toolkits.mplot3d.proj3d import proj_transform

        class Arrow3D(FancyArrowPatch):
            def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
                super().__init__((0, 0), (0, 0), *args, **kwargs)
                self._xyz = (x, y, z)
                self._dxdydz = (dx, dy, dz)

            def draw(self, renderer):
                x1, y1, z1 = self._xyz
                dx, dy, dz = self._dxdydz
                x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

                xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
                self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
                super().draw(renderer)

            def do_3d_projection(self, renderer=None):
                x1, y1, z1 = self._xyz
                dx, dy, dz = self._dxdydz
                x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

                xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
                self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
                return np.min(zs)

        def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
            """Add an 3d arrow to an `Axes3D` instance."""
            arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
            ax.add_artist(arrow)

        setattr(Axes3D, "arrow3D", _arrow3D)

        "Get the set of all arrays accessed by the program"
        arrs = {
            a: i for (i, a) in enumerate(sorted(set([v.arr for v in self.vertices])))
        }

        "Get the maximum array dimension"
        max_dim = max([max(v.ind) for v in self.vertices]) + 2

        fig = plt.figure()
        # ax = Axes3D(fig)
        ax = fig.add_subplot(projection="3d")
        for v in self.vertices:
            # if v.arr != 'A':
            #     continue
            x_offset = arrs[v.arr] * max_dim
            ax.scatter(v.ind[0] + x_offset, v.ind[1], v.version, color="b")
        for e in self.edges:
            xu_offset = arrs[e.u.arr] * max_dim
            xv_offset = arrs[e.v.arr] * max_dim
            ax.arrow3D(
                e.u.ind[0] + xu_offset,
                e.u.ind[1],
                e.u.version,
                e.v.ind[0] + xv_offset - e.u.ind[0] - xu_offset,
                e.v.ind[1] - e.u.ind[1],
                e.v.version - e.u.version,
                mutation_scale=10,
                ec="green",
                fc="red",
            )

        plt.show()
        self.find_redundant_vertices()
        work, depth = self.workDepth()
        a = 1

    def find_redundant_vertices(self):
        """
        Find vertices that are redundant, i.e., they are not used in the computation of the output array
        """
        edges = [(e.u.name, e.v.name) for e in self.edges]
        G = nx.DiGraph(edges)

        arrs = {a: i for (i, a) in enumerate(list(set([v.arr for v in self.vertices])))}
        if len(arrs) > 1:
            non_output_arrays = sorted(arrs.keys())[:-1]
            output_arr = sorted(arrs.keys())[-1]
        else:
            non_output_arrays = []
            output_arr = sorted(arrs.keys())[0]

        out_vertices = [
            v
            for v in self.vertices
            if v.arr == output_arr and v.version == vertex_versions[v.base]
        ]

        redundant_vertices = []
        for v in self.vertices:
            if not nx.algorithms.has_path(G, v.name, out_vertices[0].name):
                redundant_vertices.append(v)
        return redundant_vertices

    def workDepth(self) -> (int, int):
        """
        Compute the work depth of the CDAG
        """
        # edge list
        edges = [(e.u.name, e.v.name) for e in self.edges]
        G = nx.DiGraph(edges)

        # work: total number of non-input vertices
        work = len([v for v in G.nodes if G.in_degree(v) > 0])

        # depth:
        depth = nx.dag_longest_path_length(G)

        return (work, depth)


def tmp():
    import matplotlib.pyplot as plt
    import numpy as np

    # Fixing random state for reproducibility
    np.random.seed(19680801)

    def randrange(n, vmin, vmax):
        """
        Helper function to make an array of random numbers having shape (n, )
        with each number distributed Uniform(vmin, vmax).
        """
        return (vmax - vmin) * np.random.rand(n) + vmin

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    n = 100

    # For each set of style and range settings, plot n random points in the box
    # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
    for m, zlow, zhigh in [("o", -50, -25), ("^", -30, -5)]:
        xs = randrange(n, 23, 32)
        ys = randrange(n, 0, 100)
        zs = randrange(n, zlow, zhigh)
        ax.scatter(xs, ys, zs, marker=m)

    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")

    plt.show()


sinv_cdag = CDAG()
