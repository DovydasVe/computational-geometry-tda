# Computational Topological Data Analysis (TDA) Core

A lightweight Python implementation of Topological Data Analysis (TDA) core algorithms from scratch. This project implements simplicial complexes, boundary operators, and persistent homology computation from first principles using NumPy and SciPy.

---

## Features & Implementation

### 1. Mathematical & Algorithmic Theory
* **Simplex Representation**: Class representing $d$-simplices with canonical sorting, face extraction, and boundary operators.
* **Simplicial Complexes**: Data structure closed under the face relation, supporting $k$-skeleton queries, validity verification, and boundary matrix generation.
* **Complex Constructions**:
  * **Vietoris-Rips Complex**: Construct complexes from point cloud distance matrices given threshold $\epsilon$.
  * **Nerve Complex**: Construct nerves from a family of intersecting sets.
* **Filtration**: Support for ordered sequences of complexes (e.g., Vietoris-Rips filtrations, Star filtrations).
* **Matrix Reduction modulo 2**: Persistence-compatible column-reduction algorithm over $\mathbb{Z}_2$ running in $O(n^2)$ additions.
* **Homology Computation**: Calculation of birth-death persistence pairs and Betti numbers.

### 2. Applications & Case Studies
* **Mathematical Structures**: Computation of Betti numbers, boundary operators, and persistence diagrams for standard topological spaces (e.g., Disk $B^2$, Circle $S^1$, Tetrahedron, Torus $T^2$).
* **Synthetic Datasets**: Noise analysis and persistent homology diagram generation (scatterplot & barcodes) for noisy point cloud circles. Output compared against the Ripser library.

---

## Reproduced Papers

*(Section for reproduced computational topology and TDA papers—to be added).*
