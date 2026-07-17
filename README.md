A lightweight Python implementation of Topological Data Analysis (TDA) core algorithms from scratch. This project implements simplicial complexes, boundary operators, and persistent homology computation from first principles using NumPy and SciPy. The project also includes thorough case studies with TDA applications & interpretations, as well as reproductions of foundational TDA-related research papers.

---

## Core Modules
* **Simplicial Complexes (`core/topology.py`)**: Representation of $d$-simplices with canonical sorting, face extraction, and boundary operators. Supports simplicial complexes, $k$-skeleton queries, complex validity verification, boundary matrix generation, Vietoris-Rips complexes, and Nerve complexes.
* **Persistent Homology (`core/persistence.py`)**: Support for ordered filtrations (Vietoris-Rips and Star filtrations). Implements the persistence-compatible boundary matrix reduction modulo 2 to extract birth-death persistence pairs and Betti numbers. Supports diagram visualization (scatterplots and barcodes) with consistent color mappings.
* **Linear Algebra Helper (`core/linear_algebra.py`)**: Optimized matrix reduction over $\mathbb{Z}_2$ running in $\mathcal{O}(n^2)$ additions.
* **Vectorization & Representations (`core/representations.py`)**: Vector representations of persistent homology to integrate TDA with standard machine learning. Includes the implementation of **Persistence Images (Adams et al., 2017)**.

## Case Studies & Applications
* **Mathematical Structures (`notebooks/case_study_mathematical_structures.ipynb`)**: Demonstrates Betti numbers, boundary matrices, and persistent homology computation on standard geometric shapes: the solid disk $B^2$, circle $S^1$, hollow 2-sphere $S^2$ (tetrahedron boundary), and 2-torus $T^2$.
* **Noisy Synthetic Circle (`notebooks/case_study_synthetic_dataset.ipynb`)**: Analyzes noise sensitivity and extracts persistent homology signatures for a noisy point cloud circle, validating custom implementations against `ripser`.
* **NYC Temperature Dataset (`notebooks/case_study_real_dataset.ipynb`)**: Applies 2D time-delay embeddings to monthly average temperatures in Central Park, NYC, to analyze seasonal cycles as topological loops.

## Reproduced Papers
#### Persistence Images: A Stable Vector Representation of Persistent Homology
* **Citation**: Adams, H., Emerson, T., Kirby, M., Neville, R., Peterson, C., Shipman, P., Chepushtanova, S., Hanson, E., Motta, F., & Ziegelmeier, L. (2017). *Persistence Images: A Stable Vector Representation of Persistent Homology*. Journal of Machine Learning Research, 18(8), 1-35.
* **Replication Scope**: Implementation of the `PersistenceImage` class in `core/representations.py`. Replication of the classification experiments with additional non-topological baseline can be found in the reproduction file at `reproductions/adams2017_persistence_images/reproduction_persistence_images.ipynb`.
