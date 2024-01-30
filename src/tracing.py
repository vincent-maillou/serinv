
import copy
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

    def __init__(self, arr:str, arr_obj: np.ndarray, ind: tuple[int, ...], output: bool = False, version: int = -1, is_zero: bool = None):
        self.arr = arr
        self.ind = ind
        if arr_obj is not None and is_zero is None and np.isclose(arr_obj[ind], 0):
            self.is_zero = True
        if output:
            vertex_versions[self.base] = vertex_versions[self.base] + 1
        
        if version == -1:
            self.version = vertex_versions[self.base]
        else:
            self.version = version

    @classmethod
    def new(cls, arr:str, ind: tuple[int, ...]):
        return cls(arr, None, ind[:-1], output=False, version= ind[-1])

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
    
    def __hash__(self) -> int:
        return hash((self.arr, self.ind, self.version))
    



@dataclass
class Edge:
    u: Vertex
    v: Vertex

vertex_versions = Counter()

@dataclass
class CDAG:
    vertices: list[Vertex] = field(default_factory=list)
    edges: list[Edge]  = field(default_factory=list)
    in_neighs: defaultdict[Vertex, list[Vertex]] = field(default_factory=lambda: defaultdict(list))
    out_neighs: defaultdict[Vertex, list[Vertex]] = field(default_factory=lambda: defaultdict(list))    

    '''
    Implement method to add all edges to all pairs of u's to v
    '''
    def add(self, us: list[Vertex], v: Vertex, ignore_zeros: bool = True, tracing: bool = True):
        if not tracing:
            return
        for u in us:
            if ignore_zeros and u.is_zero:
                continue
            if u not in self.vertices:
                self.vertices.append(u) 
        if v not in self.vertices:
            self.vertices.append(v)

        for u in us:
            if ignore_zeros and u.is_zero:
                continue
            self.edges.append(Edge(u, v))
            self.out_neighs[u.ssa].append(v)
            self.in_neighs[v.ssa].append(u)
        
        print(f"{v.name} = ({', '.join(u.name for u in us)}))")

    """
    Plot the directed acyclic graph in a 3D space. X and Y coordinates of vertices
    are i and j indices of the matrix, and Z coordinate is the version of the vertex.
    """
    def plot(self, input_array_only: bool = True):
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
            '''Add an 3d arrow to an `Axes3D` instance.'''
            arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
            ax.add_artist(arrow)

        setattr(Axes3D, 'arrow3D', _arrow3D)

        "Get the set of all arrays accessed by the program"
        arrs = {a:i for (i,a) in enumerate(sorted(set([v.arr for v in self.vertices])))}

        "Get the maximum array dimension"
        max_dim = max([max(v.ind) for v in self.vertices]) + 2

        fig = plt.figure()
        # ax = Axes3D(fig)
        ax = fig.add_subplot(projection='3d')
        for v in self.vertices:
            if input_array_only and v.arr != 'A':
                continue
            x_offset = arrs[v.arr] * max_dim
            ax.scatter(v.ind[0] + x_offset, v.ind[1], v.version, color='b')
        for e in self.edges:
            if input_array_only and (e.u.arr != 'A' or e.v.arr != 'A'):
                continue
            xu_offset = arrs[e.u.arr] * max_dim
            xv_offset = arrs[e.v.arr] * max_dim
            ax.arrow3D(e.u.ind[0] + xu_offset, e.u.ind[1], e.u.version,
                e.v.ind[0] + xv_offset - e.u.ind[0] - xu_offset, e.v.ind[1] - e.u.ind[1], e.v.version - e.u.version,
                mutation_scale=10,
                ec ='green',
                fc='red')

        plt.show()
        self.find_redundant_vertices()
        work, depth = self.workDepth()
        a = 1
        
    def find_redundant_vertices(self):
        """
        Find vertices that are redundant, i.e., they are not used in the computation of the output array
        """
        # edges = [(e.u.name, e.v.name) for e in self.edges]
        edges = [(e.u, e.v) for e in self.edges]
        G = nx.DiGraph(edges)

        arrs = {a:i for (i,a) in enumerate(list(set([v.arr for v in self.vertices])))}
        if len(arrs) > 1:
            non_output_arrays = sorted(arrs.keys())[:-1]
            output_arr = sorted(arrs.keys())[-1]
        else:
            non_output_arrays = []
            output_arr = sorted(arrs.keys())[0]
        
        out_vertices = [v for v in self.vertices if v.arr == output_arr and v.version == vertex_versions[v.base]]

        redundant_vertices = []
        for u in self.vertices:
            if u not in G.nodes:
                continue
            if not any(nx.algorithms.has_path(G, u, v) for v in out_vertices):
                redundant_vertices.append(u)
        return redundant_vertices

    def workDepth(self) -> (int, int):
        '''
        Compute the work depth of the CDAG
        '''
        # edge list
        edges = [(e.u.name, e.v.name) for e in self.edges]
        G = nx.DiGraph(edges)

        # work: total number of non-input vertices
        work = len([v for v in G.nodes if G.in_degree(v) > 0])

        # depth: 
        depth = nx.dag_longest_path_length(G)

        return (work, depth)



def output_descendants(G: nx.DiGraph, v: Vertex):
    desc = nx.algorithms.descendants(G, v)
    return [d for d in desc if d.version == vertex_versions[d.base]]




# -------------------------------------------------------------------------------- #
# ------------------explicit LA algorithms for tracing --------------------------- #
# -------------------------------------------------------------------------------- #

def invert_LU_explicit(in_LU: np.ndarray, m_name: str = "A") -> np.ndarray:
    """ Invert an LU matrix (matrix containing the L and U factors of a triangular matrix
    in its lower and upper parts, respectively)

    The method adds the CDAG vertex tracing.
    
    Parameters
    ----------
    LU : np.ndarray
    m_name: str
        The name of the matrix to be inverted. It is used for naming the vertices in the CDAG.
    
    Returns
    -------
    LU_inv : np.ndarray
        The inverted LU matrix.
    """
    LU = copy.deepcopy(in_LU)
    n = LU.shape[0]
    
    for i in range(n):
        sinv_cdag.add([Vertex(m_name, LU, (i, i), output=False)], Vertex(m_name, LU, (i, i), output=True))
        LU[i, i] = 1 / LU[i, i]
        for j in range(i):
            sinv_cdag.add([Vertex(m_name, LU, (j, i), output=False), 
                           Vertex(m_name, LU, (j, j), output=False)],
                             Vertex(m_name, LU, (j, i), output=True))
            LU[j,i] = LU[j, i] * LU[j, j]
            for k in range(j+1, i):
                sinv_cdag.add([Vertex(m_name, LU, (j, k), output=False),
                                 Vertex(m_name, LU, (k, i), output=False),
                                 Vertex(m_name, LU, (j, i), output=False)],
                                 Vertex(m_name, LU, (j, i), output=True))
                LU[j,i] += LU[k, i] * LU[j, k]
            sinv_cdag.add([Vertex(m_name, LU, (j, i), output=False),
                             Vertex(m_name, LU, (i, i), output=False)],
                             Vertex(m_name, LU, (j, i), output=True))
            LU[j,i] = - LU[j,i] * LU[i, i]
    
    for i in range(n):
        for j in range(i):
            for k in range(j+1, i):
                sinv_cdag.add([Vertex(m_name, LU, (i, k), output=False),
                                 Vertex(m_name, LU, (k, j), output=False),
                                 Vertex(m_name, LU, (i, j), output=False)],
                                 Vertex(m_name, LU, (i, j), output=True))
                LU[i, j] += LU[i, k] * LU[k, j]
            sinv_cdag.add([Vertex(m_name, LU, (i, j), output=False),
                                Vertex(m_name, LU, (j, j), output=False)],
                                Vertex(m_name, LU, (i, j), output=True))
            LU[i,j] = - LU[i, j]


    L = np.tril(LU, k=-1) + np.eye(n, dtype=LU.dtype)
    U = np.triu(LU, k=0)    

    # return U @ L
    return L, U  


def lu_dcmp_explicit(in_A: np.ndarray) -> (np.ndarray, np.ndarray):
    A = copy.deepcopy(in_A)
    n = A.shape[0]
    for k in range(n):
        # BEGIN in-place LU decomposition without pivoting
        sinv_cdag.add([Vertex("A", A, (k, k), output=False)], Vertex("A", A, (k, k), output=True))

        for i in range(k+1, n): #(n-1, k, -1)
            sinv_cdag.add([Vertex("A", A, (i, k), output=False), Vertex("A", A, (k, k), output=False)], Vertex("A", A, (i, k), output=True))
            A[i, k] = A[i,k] / A[k,k]
            # for j in range(k+1,min(n, n)): #range(n-1, k, -1): 
            for j in range(k+1,  n): #range(n-1, k, -1): 
                sinv_cdag.add([Vertex("A", A, (i, j), output=False), 
                               Vertex("A", A, (i, k), output=False), 
                               Vertex("A", A, (k, j), output=False)], 
                               Vertex("A", A, (i, j), output=True))
                A[i,j]  -= A[i,k]*A[k,j]      
    L = np.tril(A, k=-1) + np.eye(n, dtype=A.dtype)
    U = np.triu(A, k=0)
    LU = np.triu(A, k=0) + np.tril(A, k=-1)
    return L, U
    # END in-place LU decomposition without pivoting


def MMM_explicit(A: np.ndarray, B: np.ndarray, 
        a_name:str = "A",
        b_name:str = "B",
        c_name:str = "C") -> np.ndarray:
    """
    explicit Matrix-matrix multiplication with tracing
    """
    C = np.zeros((A.shape[0], B.shape[1]), dtype=A.dtype)
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for k in range(A.shape[1]):
                sinv_cdag.add([Vertex(a_name, A, (i, k), output=False), Vertex(b_name, B, (k, j), output=False)], Vertex(c_name, None, (i, j), output=True))
                C[i,j] += A[i,k] * B[k,j]
    return C

def subs_explicit(A:np.ndarray, B:np.ndarray,
         a_name: str = "A",
         b_name: str = "B",
         c_name:str = "C") -> np.ndarray:
    """
    explicit matrix subtraction with tracing
    """
    C = np.zeros(A.shape, dtype=A.dtype)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            sinv_cdag.add([Vertex(a_name, A, (i, j), output=False), Vertex(b_name, B, (i, j), output=False)], Vertex(c_name, None, (i, j), output=True))
            C[i,j] = A[i,j] - B[i,j]
    return C



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
        return (vmax - vmin)*np.random.rand(n) + vmin

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    n = 100

    # For each set of style and range settings, plot n random points in the box
    # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
    for m, zlow, zhigh in [('o', -50, -25), ('^', -30, -5)]:
        xs = randrange(n, 23, 32)
        ys = randrange(n, 0, 100)
        zs = randrange(n, zlow, zhigh)
        ax.scatter(xs, ys, zs, marker=m)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

sinv_cdag = CDAG()
