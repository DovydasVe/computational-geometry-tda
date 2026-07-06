# Replication Notes: Persistence Images


### General Info
* **Paper**: Adams et al. (2017)
* **Title**: *Persistence Images: A Stable Vector Representation of Persistent Homology*
* **Journal**: Journal of Machine Learning Research (JMLR), 18(8), 1-35
* **Link**: [JMLR Publication](https://jmlr.org/papers/v18/16-337.html) (Originally arXiv:1507.06217)


### Reproduction Strategy
We will implement the Persistence Image vectorization algorithm in `representations.py` using weighted 2D Gaussians integrated over a discretized grid. Correctness is benchmarked on Section 6.1 from the paper by generating point clouds from 6 classic topological spaces, using core modules to compute $H_0$, $H_1$, and $H_2$ persistent homology, and evaluating classification accuracy using `scikit-learn` classifiers on the resulting image vectors.
