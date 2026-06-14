from collections import defaultdict
from itertools import combinations
from scipy.spatial.distance import cdist
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from topology import Simplex, SimplicialComplex
from linear_algebra import reduce_boundary_matrix


# F - filtration class, RF - rips filtration, P - persistence
TESTING_F = False
TESTING_RF = False
TESTING_P = True


class Filtration:
    def __init__(self):
        self.filtration = dict()

    def add(self, simplex, value):
        if simplex in self.filtration:
            if value < self.filtration[simplex]:
                self.filtration[simplex] = value
        else:
            self.filtration[simplex] = value

    def sort(self):
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
        simplices_list = list()

        for key, value in self.filtration.items():
            if value <= k:
                simplices_list.append(key)
        
        K = SimplicialComplex()
        K.add_simplices(simplices_list)

        return K

    def filtration_values(self):
        return sorted(list(set(v for v in self.filtration.values())))
    
    def is_valid(self):
        for simplex, value in self.filtration.items():
            faces = simplex.faces()
            for face in faces:
                if face not in self.filtration:
                    return False
                if self.filtration[face] > value:
                    return False
        return True

    def build_filtration_boundary_matrix(self):
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

        return list(pairs.values())


def rips_filtration(points):
    F = Filtration()
    distance_matrix = cdist(points, points, metric='euclidean')
    n = len(points)

    for i in range(n):
        F.add(Simplex([i]), 0.0)

    for i, j in combinations(range(n), 2):
        F.add(Simplex([i,j]), float(distance_matrix[i, j]))

    for i, j, k in combinations(range(n), 3):
        value = max(distance_matrix[i, j], distance_matrix[i, k], distance_matrix[j, k])
        F.add(Simplex([i,j,k]), float(value))

    return F
    

class PersistencePair:
    def __init__(self, dim, birth, death):
        self.dim = dim
        self.birth = birth
        self.death = death

    def lifetime(self):
        if self.death == "inf":
            return self.death
        return self.death - self.birth
    
    def is_infinite(self):
        return float(self.death) == float("inf")

    def __repr__(self):
        return f"{self.dim, self.birth, self.death}"


class PersistenceDiagram:
    def __init__(self):
        self.all_pairs = list()

    def add_pair(self, pair):
        self.all_pairs.append(pair)

    def pairs_by_dim(self, k):
        k_pairs = list()
        for pair in self.all_pairs:
            if pair.dim == k:
                k_pairs.append(pair)
        return k_pairs
    
    def infinite_pairs(self):
        inf_pairs = list()
        for pair in self.all_pairs:
            if pair.is_infinite():
                inf_pairs.append(pair)
        return inf_pairs
    
    def barcode(self):
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
        """ Drops infinite pairs """
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

        sns.set_theme(style="whitegrid")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        sns.scatterplot(x=births, y=deaths, hue=dims, s=60, ax=ax1)

        min_val = min(births + deaths) if births else 0
        max_val = max(births + deaths) if births else 1

        ax1.plot([min_val, max_val], [min_val, max_val], "--", linewidth=1, color="gray")

        ax1.set_xlabel("Birth")
        ax1.set_ylabel("Death")
        ax1.grid(False)
        ax1.legend(title="Homology", loc="lower right")

        bar_data.sort(key=lambda x: (x[0], x[1]))
        unique_dims = sorted(list(set(d[0] for d in bar_data)))
        colors = {d: sns.color_palette()[i] for i, d in enumerate(unique_dims)}

        for i, (dim, b, d, inf) in enumerate(bar_data):
            ax2.hlines(y=i, xmin=b, xmax=d, color=colors[dim], linewidth=4)
            if inf:
                ax2.annotate('', xy=(d + 0.1, i), xytext=(d, i), arrowprops=dict(arrowstyle="->", color=colors[dim], lw=2))

        ax2.set_xlabel("Filtration Value")
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
        print(F.is_valid())

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
        print(rf.is_valid())
    

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