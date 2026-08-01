# Replication Notes: TDA on time-series

### General Info
* **Paper**: Gidea & Katz (2018)
* **Title**: *Topological data analysis of financial time series: Landscapes of crashes*
* **Journal**: Physica A: Statistical Mechanics and its Applications, 491, 820–834
* **Link**: [Physica A Publication](https://doi.org/10.1016/j.physa.2017.09.028)

### Reproduction Strategy
We evaluate Persistence Landscape $L^1$ and $L^2$ norms as quantitative indicators of point cloud dispersion and phase shifts in financial time series. Synthetic experiments benchmark the linear dependency of PL norms on 4D white noise variance (Section 3.3) and their sensitivity to dynamic state transitions using Gamma-distributed variance shifts (Section 3.4). We then construct sliding-window point cloud embeddings from historical US financial market index time series (Section 4) to analyze topological shifts preceding major market crashes (the Dot-Com bubble crash and the 2008 Lehman Brothers collapse), accompanied by Early Warning Signal (EWS) analysis (variance, autocorrelation, spectral density, and Kendall's $\tau$). Furthermore, we take a detour by evaluating Persistence Images (PIs) as an alternative vector representation to Persistence Landscapes for financial crash detection.
