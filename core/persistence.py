from collections import defaultdict
from itertools import combinations
from scipy.spatial.distance import cdist
import numpy as np
import seaborn as sns
sns.set_theme(style="whitegrid")
palette = sns.color_palette("bright")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from ripser import ripser

try:
    from core.topology import Simplex, SimplicialComplex
    from core.linear_algebra import reduce_boundary_matrix
except ModuleNotFoundError:
    from topology import Simplex, SimplicialComplex
    from linear_algebra import reduce_boundary_matrix


# F - filtration class, RF - rips filtration, ST - star filtration, P - persistence, ripser - library
TESTING_F = False
TESTING_RF = False
TESTING_ST = False
TESTING_P = False
TESTING_ripser = False


class Filtration:
    """
    Represents a filtered simplicial complex as a sequence of simplices ordered by their birth times.
    Provides methods to sort the filtration, build boundary matrices, and extract persistence pairs.
    """
    def __init__(self):
        """Initializes an empty filtration."""
        self.filtration = dict()

    def add(self, simplex, value):
        """Adds a simplex with a given filtration value, keeping the minimal value if already present."""
        if simplex in self.filtration:
            if value < self.filtration[simplex]:
                self.filtration[simplex] = value
        else:
            self.filtration[simplex] = value

    def sort(self):
        """Sorts the filtration by birth time, resolving ties by simplex dimension to ensure validity."""
        grouped_dict = defaultdict(list)

        for key, value in self.filtration.items():
            grouped_dict[value].append(key)

        grouped_dict = dict(grouped_dict)
        sorted_grouped_dict = dict(sorted(grouped_dict.items(), key=lambda item: item[0]))

        for key in sorted_grouped_dict.keys():
            sorted_values = sorted(sorted_grouped_dict[key], key=lambda x: x.dim())
            sorted_grouped_dict[key] = sorted_values

        sorted_dict = dict()
        for key, value in sorted_grouped_dict.items():
            for v in value:
                sorted_dict[v] = key
        
        self.filtration = sorted_dict

    def get_complex(self, k):
        """Returns the simplicial complex at a given filtration threshold value k."""
        simplices_list = list()

        for key, value in self.filtration.items():
            if value <= k:
                simplices_list.append(key)
        
        K = SimplicialComplex()
        K.add_simplices(simplices_list)

        return K

    def filtration_values(self):
        """Returns a sorted list of all unique filtration values in the filtration."""
        return sorted(list(set(v for v in self.filtration.values())))
    
    def is_valid(self):
        """Verifies if the filtration is valid by checking face and coface birth values."""
        for simplex, value in self.filtration.items():
            faces = simplex.faces()
            for face in faces:
                if face not in self.filtration:
                    return False
                if self.filtration[face] > value:
                    return False
        return True

    def build_filtration_boundary_matrix(self):
        """Builds and returns the boundary matrix of the filtered simplicial complex."""
        self.sort()
        n = len(self.filtration)
        bd_matrix = np.zeros((n,n), dtype=int)
        index = {s: i for i, s in enumerate(self.filtration)}

        for j, simplex in enumerate(self.filtration.keys()):
            boundary = simplex.boundary()
            for bd in boundary:
                i = index[bd]
                bd_matrix[i, j] = 1

        return bd_matrix
    
    def extract_pairs(self):
        """Reduces the boundary matrix and returns a list of birth-death persistence pairs."""
        bd_matrix = self.build_filtration_boundary_matrix()
        reduced_matrix = reduce_boundary_matrix(bd_matrix)
        simplex_list = list(self.filtration.keys())

        pairs = {} 
        
        for j in range(np.shape(reduced_matrix)[1]):
            col = reduced_matrix[:,j]
            if np.count_nonzero(col) == 0:
                simplex = simplex_list[j]
                pair = PersistencePair(simplex.dim(), self.filtration[simplex], float("inf"))
                pairs[j] = pair
                
            else:
                i = np.where(col == 1)[0][-1]
                birth_simplex = simplex_list[i]
                death_simplex = simplex_list[j]
                pair = PersistencePair(
                    birth_simplex.dim(),
                    self.filtration[birth_simplex],
                    self.filtration[death_simplex]
                )
                pairs[i] = pair

        # Filter out trivial features with zero lifetime
        return [p for p in pairs.values() if p.lifetime() != 0.0]

    def __repr__(self):
        """Returns the string representation of the filtration mapping."""
        return f"{self.filtration}"


class RipserFiltration(Filtration):
    """
    A filtration wrapper that delegates the persistent homology computation to the
    optimized `ripser` library, returning standard `PersistencePair` objects.
    """
    def __init__(self, points, max_dim):
        super().__init__()
        self.points = points
        self.max_dim = max_dim

    def extract_pairs(self):
        result = ripser(np.array(self.points), maxdim=self.max_dim)
        dgms = result['dgms']

        pairs = []
        for dim, dgm in enumerate(dgms):
            for birth, death in dgm:
                pairs.append(PersistencePair(dim, float(birth), float(death)))
                
        return pairs


def rips_filtration(points, max_dim=2, use_ripser=False):
    """
    Constructs and returns a Vietoris-Rips filtration of a point cloud.
    If use_ripser is True, delegates to RipserFiltration for fast execution.
    Otherwise, builds the filtration manually up to dimension max_dim + 1.
    """
    if use_ripser:
        return RipserFiltration(points, max_dim=max_dim)

    F = Filtration()
    distance_matrix = cdist(points, points, metric='euclidean')
    n = len(points)

    for i in range(n):
        F.add(Simplex([i]), 0.0)

    if max_dim >= 0:
        for i, j in combinations(range(n), 2):
            F.add(Simplex([i,j]), float(distance_matrix[i, j]))

    if max_dim >= 1:
        for i, j, k in combinations(range(n), 3):
            value = max(distance_matrix[i, j], distance_matrix[i, k], distance_matrix[j, k])
            F.add(Simplex([i,j,k]), float(value))

    if max_dim >= 2:
        for i, j, k, l in combinations(range(n), 4):
            value = max(distance_matrix[i, j], distance_matrix[i, k], distance_matrix[i, l],
                        distance_matrix[j, k], distance_matrix[j, l], distance_matrix[k, l])
            F.add(Simplex([i,j,k,l]), float(value))

    return F
    

def star_filtration(sc, vertex_values):
    """Constructs and returns the star filtration of a simplicial complex given values on its vertices."""

    def simplex_value(simplex, vertex_values=vertex_values):
        """Computes the filtration value of a simplex as the maximum of its vertex values."""
        faces = simplex.faces()
        vertices = list()
        for face in faces:
            if face.dim() == 0:
                vertices.append(face)

        if vertices:
            maximum = vertex_values[vertices[0].return_values()]
            for vertex in vertices:
                key = vertex.return_values()
                if vertex_values[key] > maximum:
                    maximum = vertex_values[key]
        
        return maximum

    F = Filtration()
    for simplex in sc.simplices:
        val = simplex_value(simplex)
        F.add(simplex, val)
    F.sort()

    return F


class PersistencePair:
    """
    Represents a persistence pair (homology class birth and death) in topological data analysis.
    """
    def __init__(self, dim, birth, death):
        """Initializes a persistence pair with a dimension, birth time, and death time."""
        self.dim = dim
        self.birth = birth
        self.death = death

    def lifetime(self):
        """Returns the lifetime of the homology class (death minus birth, or 'inf' for infinite classes)."""
        if self.death == "inf":
            return self.death
        return self.death - self.birth
    
    def is_infinite(self):
        """Returns True if the homology class never dies, and False otherwise."""
        return float(self.death) == float("inf")

    def __repr__(self):
        """Returns the string representation of the persistence pair."""
        return f"{self.dim, self.birth, self.death}"


class PersistenceDiagram:
    """
    Represents a collection of persistence pairs, providing methods for barcode output and diagram plotting.
    """
    def __init__(self):
        """Initializes an empty persistence diagram."""
        self.all_pairs = list()

    def add_pair(self, pair):
        """Adds a persistence pair to the diagram."""
        self.all_pairs.append(pair)

    def pairs_by_dim(self, k):
        """Returns all persistence pairs of dimension k."""
        k_pairs = list()
        for pair in self.all_pairs:
            if pair.dim == k:
                k_pairs.append(pair)
        return k_pairs
    
    def infinite_pairs(self):
        """Returns a list of all infinite persistence pairs in the diagram."""
        inf_pairs = list()
        for pair in self.all_pairs:
            if pair.is_infinite():
                inf_pairs.append(pair)
        return inf_pairs
    
    def barcode(self):
        """Prints the barcode representation of the persistence diagram."""
        h_dict = dict()
        dims = set(pair.dim for pair in self.all_pairs)
        for n in sorted(dims):
            pairs = self.pairs_by_dim(n)
            h_dict[n] = pairs
        
        for k, val in h_dict.items():
            print(f"H{k}:")
            for v in val:
                print(f"[{v.birth}, {v.death})")

    def plot_diagram(self):
        """Plots the persistence diagram scatterplot and barcode side by side."""
        births, deaths, dims = [], [], []
        bar_data = []

        finite_deaths = [p.death for p in self.all_pairs if not p.is_infinite()]
        max_finite = max(finite_deaths) if finite_deaths else 1
        inf_val = max_finite + 0.5

        for p in self.all_pairs:
            dim_str = f"H{p.dim}"
            if not p.is_infinite():
                births.append(p.birth)
                deaths.append(p.death)
                dims.append(dim_str)
                bar_data.append((dim_str, p.birth, p.death, False))
            else:
                bar_data.append((dim_str, p.birth, inf_val, True))

        unique_dims = {0, 1}
        for p in self.all_pairs:
            unique_dims.add(p.dim)
        unique_dims = sorted(list(unique_dims))

        colors = {f"H{d}": palette[i] for i, d in enumerate(unique_dims)}

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        sns.scatterplot(x=births, y=deaths, hue=dims, s=60, ax=ax1, palette=colors)

        min_val = min(births + deaths) if births else 0
        max_val = max(births + deaths) if births else 1

        ax1.plot([min_val, max_val], [min_val, max_val], "--", linewidth=1, color="gray")

        ax1.set_title("Persistence Diagram", fontsize=18, pad=15)
        ax1.set_xlabel("Birth", fontsize=15, labelpad=15)
        ax1.set_ylabel("Death", fontsize=15, labelpad=15)

        handles = [
            Line2D([0], [0],
                marker='o',
                linestyle='',
                color=colors[f"H{d}"],
                label=f"H{d}",
                markersize=8)
            for d in unique_dims
        ]

        ax1.legend(handles=handles, title="", fontsize=13)

        bar_data.sort(key=lambda x: (x[0], x[1]))
        for i, (dim, b, d, inf) in enumerate(bar_data):
            ax2.hlines(y=i, xmin=b, xmax=d, color=colors[dim], linewidth=4)
            if inf:
                ax2.annotate('', xy=(d + 0.1, i), xytext=(d, i),
                             arrowprops=dict(arrowstyle="->", color=colors[dim], lw=2))

        ax2.set_title("Persistence Barcode", fontsize=18, pad=15)
        ax2.set_xlabel("Filtration Value", fontsize=15, labelpad=15)
        ax2.set_yticks([])

        plt.tight_layout()
        plt.show()
    
    def __repr__(self):
        return f"{self.all_pairs}"


if __name__ == "__main__":
    if TESTING_F:
        s1 = Simplex([0,1])
        s2 = Simplex([1])
        s3 = Simplex([0])
        F = Filtration()
        F.add(s1, 0.4)
        F.add(s2, 0.2)
        F.add(s3, 0.1)
        print(F.filtration)
        F.sort()
        print(F.filtration)
        K = F.get_complex(0.3)
        print(K)
        print(F.filtration_values())
        print("Expected True |", F.is_valid())

        print()
        print("Persistence pairing logic")
        F.add(Simplex([2]), 0.3)
        F.add(Simplex([0,2]), 0.6)
        F.add(Simplex([1,2]), 0.5)
        F.add(Simplex([0,1,2]), 0.7)
        print(reduce_boundary_matrix(F.build_filtration_boundary_matrix()))
        print(F.extract_pairs())


    if TESTING_RF:
        points = [(0,0), (0,1), (1,0), (1,1)]
        rf = rips_filtration(points)
        print(rf.filtration)
        print("Expected True |", rf.is_valid())


    if TESTING_ST:
        sc = SimplicialComplex()
        sc.add_simplex(Simplex([0,1,2]))
        vertex_values = {0: 0, 1: 2, 2: 1}
        F = star_filtration(sc, vertex_values)
        print(F)
        print(F.extract_pairs())
    

    if TESTING_P:
        p1 = PersistencePair(0, 0.0, float("inf"))
        p2 = PersistencePair(1, 1.2, 3.8)
        print("Lifetimes", p1.lifetime(), p2.lifetime(), sep=" | ")
        print("Is infinite", p1.is_infinite(), p2.is_infinite(), sep=" | ")

        pd = PersistenceDiagram()
        pd.add_pair(p1)
        pd.add_pair(p2)

        print("Pairs by dimention", pd.pairs_by_dim(0), pd.pairs_by_dim(1), 
              pd.pairs_by_dim(2), sep=" | ")
        print("Infinite pairs", pd.infinite_pairs(), sep=" | ")
        pd.barcode()
        pd.plot_diagram()


    if TESTING_ripser:
        points = [(0,0), (0,1), (1,0), (1,1)]
        rf = rips_filtration(points)
        rf_ripser = rips_filtration(points, use_ripser=True)

        print("Expected True |", rf.is_valid())
        print("Expected True |", rf_ripser.is_valid())

        print(rf.extract_pairs())
        print(rf_ripser.extract_pairs())

        
        pd = PersistenceDiagram()
        for pair in rf.extract_pairs():
            pd.add_pair(pair)

        pd.barcode()
        pd.plot_diagram()
        
        pd = PersistenceDiagram()
        for pair in rf_ripser.extract_pairs():
            pd.add_pair(pair)

        pd.barcode()
        pd.plot_diagram()