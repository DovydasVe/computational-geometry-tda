import numpy as np
from scipy.stats import norm
from scipy.integrate import simpson
import matplotlib.pyplot as plt
try:
    from core.persistence import rips_filtration, PersistenceDiagram
except ModuleNotFoundError:
    from persistence import rips_filtration, PersistenceDiagram

# PersistenceImage functionality tested in reproductions/adams_2017
# PL - PersistenceLandscapes
TESTING_PL = True

class PersistenceImage:
    '''
    Parameters:
    pd : PersistenceDiagram
        The input persistence diagram object containing birth-death pairs.
    sigma : float
        The standard deviation of the Gaussian kernel representing the spread of each point.
    resolution : tuple of int (res_x, res_y)
        The grid resolution (number of columns and rows of pixels in the output image).
    homology_dim : int, default=1
        The Betti dimension (0, 1, 2) of the pairs to extract from the diagram.
    bounds : tuple of tuples, optional
        The bounding box for the image as ((x_min, x_max), (y_min, y_max)). 
        If None, automatically computed from the diagram's birth-persistence range.
    max_b : float, optional
        The persistence scale value at which the linear weighting ramp reaches 1.0.
        If None, defaults to the maximum persistence bound.

    Methods:
    norm(p=2, mode='discrete') :
        Computes the L^p norm of the persistence image. Supports 'discrete' (vector norm) and 'continuous' (surface integral) modes.
    flatten() :
        Flattens the 2D persistence image matrix into a 1D NumPy array for machine learning compatibility.
    plot(ax=None, vmin=None, vmax=None) :
        Plots the 2D persistence image using a magma colormap with bicubic interpolation.
    '''
    def __init__(self, pd, sigma, resolution, homology_dim=1, bounds=None, max_b=None):
        self.sigma = sigma
        self.resolution = resolution
        self.points = []
        for pair in pd.pairs_by_dim(homology_dim):
            if pair.is_infinite():
                continue
            self.points.append((pair.birth, pair.death - pair.birth))
                        
        if not bounds and len(self.points) > 0:
            b_vals = [p[0] for p in self.points]
            p_vals = [p[1] for p in self.points]
            self.bounds = (
                (min(b_vals) - sigma, max(b_vals) + sigma),
                (0, max(p_vals) + sigma)
            )
        else:
            self.bounds = bounds or ((0, 1), (0, 1))
            
        self.max_b = max_b if max_b is not None else self.bounds[1][1]
        self.image = self._generate_image()

    def _weighting_function(self, y):
        if y <= 0:
            return 0
        elif y < self.max_b:
            return y / self.max_b
        else:
            return 1

    def _generate_image(self):
        res_x, res_y = self.resolution
        (x_min, x_max), (y_min, y_max) = self.bounds
        
        x_edges = np.linspace(x_min, x_max, res_x + 1)
        y_edges = np.linspace(y_min, y_max, res_y + 1)
        
        image = np.zeros((res_y, res_x))
        for u_x, u_y in self.points:
            weight = self._weighting_function(u_y)
            if weight == 0:
                continue

            cdf_x = norm.cdf(x_edges, loc=u_x, scale=self.sigma)
            cdf_y = norm.cdf(y_edges, loc=u_y, scale=self.sigma)
            prob_x = np.diff(cdf_x)
            prob_y = np.diff(cdf_y)
            
            pixel_integrals = np.outer(prob_y, prob_x)
            image += weight * pixel_integrals
            
        return image

    def norm(self, p=2, mode='discrete'):
        """
        Computes the L^p norm of the persistence image.
        """
        if p == np.inf or p == 'inf':
            return float(np.max(np.abs(self.image)))

        p = float(p)
        if p <= 0:
            raise ValueError("Norm order p must be positive (> 0).")

        discrete_norm = float(np.sum(np.abs(self.image) ** p) ** (1.0 / p))

        if mode == 'continuous':
            (x_min, x_max), (y_min, y_max) = self.bounds
            res_x, res_y = self.resolution
            dx = (x_max - x_min) / res_x
            dy = (y_max - y_min) / res_y
            pixel_area = dx * dy
            return discrete_norm * (pixel_area ** (1.0 / p))
        else:
            return discrete_norm
      
    def flatten(self):
        return self.image.flatten()
    
    def plot(self, ax=None, vmin=None, vmax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        (x_min, x_max), (y_min, y_max) = self.bounds
        
        c = ax.imshow(
            self.image, 
            origin='lower', 
            extent=[x_min, x_max, y_min, y_max], 
            cmap='magma',
            interpolation='bicubic',
            aspect='auto',
            vmin=vmin,
            vmax=vmax
        )
        ax.grid(False)
        ax.set_xlabel("Birth", fontsize=15, labelpad=15)
        ax.set_ylabel("Persistence", fontsize=15, labelpad=15)
        return ax


class PersistenceLandscape:
    '''
    Parameters:
    pd : PersistenceDiagram
        The input persistence diagram containing birth-death pairs.
    t_interval : array-like
        1D array or sequence of values t where landscape functions are evaluated.
    homology_dim : int, default=1
        The Betti dimension (0, 1, 2) of the pairs to extract from the diagram.
    num_layers : int, optional
        Maximum number of landscape layers (k) to retain. If None, retains all finite layers.

    Methods:
    kth_layer(k) :
        Returns the 1-indexed k-th landscape layer function evaluated across t_interval.
    norm(p=2, mode='global') :
        Computes the L^p norm using numerical integration. Supports mode='global' (Bubenik 2015) and mode='cross_sectional' (mixed Bochner L^1_t(l^p_k)).
    distance(other, p=2, mode='global') :
        Computes the L^p distance between self and another PersistenceLandscape instance. Supports mode='global' and mode='cross_sectional'.
    flatten() :
        Flattens the 2D landscape matrix into a 1D NumPy array for machine learning compatibility.
    plot(max_k=5, ax=None) :
        Plots the top max_k landscape layers.
    '''
    def __init__(self, pd, t_interval, homology_dim=1, num_layers=None):
        self.t_interval = np.asarray(t_interval, dtype=float)
        self.homology_dim = homology_dim
        
        pairs = []
        for p in pd.pairs_by_dim(homology_dim):
            if not p.is_infinite():
                pairs.append((p.birth, p.death))
                
        if len(pairs) == 0:
            self.landscapes = np.zeros((0, len(self.t_interval)))
            return
        pairs_arr = np.array(pairs)
        births = pairs_arr[:, 0, np.newaxis]
        deaths = pairs_arr[:, 1, np.newaxis]
        t = self.t_interval[np.newaxis, :]

        tents = np.maximum(0.0, np.minimum(t - births, deaths - t))
        sorted_tents = np.sort(tents, axis=0)[::-1]
        
        if num_layers is not None:
            sorted_tents = sorted_tents[:num_layers]
            
        self.landscapes = sorted_tents

    def kth_layer(self, k):
        """
        Returns the k-th landscape function (1-indexed) evaluated across t_interval.
        """
        if k < 1 or k > len(self.landscapes):
            return np.zeros_like(self.t_interval)
        return self.landscapes[k - 1]

    def norm(self, p=2, mode='global'):
        """
        Computes the L^p norm of the persistence landscape using Simpson's rule numerical integration.
        mode : str, default='global'
            - 'global': Standard Bubenik (2015) L^p norm on L^p(N x R): (sum_k int |lambda_k(t)|^p dt)^(1/p).
            - 'cross_sectional': Mixed Bochner norm L^1_t(l^p_k): int (sum_k |lambda_k(t)|^p)^(1/p) dt.
        """
        if len(self.landscapes) == 0:
            return 0.0

        if p == np.inf:
            return float(np.max(np.abs(self.landscapes)))

        if mode == 'cross_sectional':
            cross_p = np.sum(np.abs(self.landscapes) ** p, axis=0) ** (1.0 / p)
            return float(simpson(cross_p, x=self.t_interval))
        else:
            abs_pow = np.abs(self.landscapes) ** p
            integrals = simpson(abs_pow, x=self.t_interval, axis=1)
            total = np.sum(integrals)
            return float(total ** (1.0 / p))
        
    def distance(self, other, p=2, mode='global'):
        """
        Computes the L^p distance between self and another PersistenceLandscape instance.
        mode : str, default='global'
            - 'global': Standard Bubenik (2015) L^p distance on L^p(N x R).
            - 'cross_sectional': Mixed Bochner L^1_t(l^p_k) distance used in time-series analysis.
        """
        if not np.array_equal(self.t_interval, other.t_interval):
            raise ValueError("Landscapes must be evaluated on identical t_interval grids.")

        k1, k2 = len(self.landscapes), len(other.landscapes)

        max_k = max(k1, k2)
        if max_k == 0:
            return 0.0

        l1_pad = np.pad(self.landscapes, ((0, max_k - k1), (0, 0))) if k1 < max_k else self.landscapes
        l2_pad = np.pad(other.landscapes, ((0, max_k - k2), (0, 0))) if k2 < max_k else other.landscapes

        diff = np.abs(l1_pad - l2_pad)
        if p == np.inf:
            return float(np.max(diff))

        if mode == 'cross_sectional':
            cross_p = np.sum(diff ** p, axis=0) ** (1.0 / p)
            return float(simpson(cross_p, x=self.t_interval))
        else:
            diff_pow = diff ** p
            integrals = simpson(diff_pow, x=self.t_interval, axis=1)
            total = np.sum(integrals)
            return float(total ** (1.0 / p))

    def flatten(self):
        """
        Flattens the 2D landscape matrix into a 1D NumPy array for machine learning compatibility.
        """
        return self.landscapes.flatten()

    def plot(self, max_k=5, ax=None):
        """
        Plots the top max_k landscape layers.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        
        num_to_plot = min(max_k, len(self.landscapes))
        for k in range(1, num_to_plot + 1):
            ax.plot(self.t_interval, self.kth_layer(k), label=f"$\\lambda_{{{k}}}$")
            
        ax.set_xlabel("t", fontsize=12)
        ax.set_ylabel("Persistence Landscape", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        return ax
            

def pc_image_norms(pc, sigma, resolution, bounds, dim=1, max_b=None):
    rips = rips_filtration(pc, max_dim=dim, use_ripser=True)
    
    pd = PersistenceDiagram()
    for pair in rips.extract_pairs():
        if pair.dim == dim:
            pd.add_pair(pair)

    pi = PersistenceImage(pd, sigma=sigma, resolution=resolution, homology_dim=dim, bounds=bounds)
    p1 = pi.norm(1)
    pinf = pi.norm(np.inf)

    return p1, pinf


def pc_landscape_norms(pc, t_interval, dim=1, num_layers=None, mode="cross_sectional", scaled=True):
    rips = rips_filtration(pc, max_dim=dim, use_ripser=True)
    
    pd = PersistenceDiagram()
    for pair in rips.extract_pairs():
        if pair.dim == dim:
            pd.add_pair(pair)

    pl = PersistenceLandscape(pd, t_interval, num_layers=num_layers)
    if scaled:
        p1 = pl.norm(1, mode=mode) / len(t_interval)
        p2 = pl.norm(2, mode=mode) / len(t_interval)
    else:
        p1 = pl.norm(1, mode=mode)
        p2 = pl.norm(2, mode=mode)

    return p1, p2


if __name__ == "__main__":
    if TESTING_PL:
        from persistence import PersistenceDiagram, PersistencePair
        pd = PersistenceDiagram()
        pd.add_pair(PersistencePair(dim=1, birth=1.0, death=4.0))
        pd.add_pair(PersistencePair(dim=1, birth=2.0, death=6.0))
        pd.add_pair(PersistencePair(dim=1, birth=3.0, death=5.0))
        pd.add_pair(PersistencePair(dim=1, birth=0.0, death=float('inf')))
        t_grid = np.linspace(0, 7, 200)

        pl = PersistenceLandscape(pd, t_grid, homology_dim=1)
        
        print(f"Total finite landscape layers computed: {len(pl.landscapes)}")
        
        lambda_1 = pl.kth_layer(1)
        lambda_2 = pl.kth_layer(2)
        lambda_3 = pl.kth_layer(3)
        print(f"Layer 1 Max Value: {np.max(lambda_1):.4f}")
        print(f"Layer 2 Max Value: {np.max(lambda_2):.4f}")
        print(f"Layer 3 Max Value: {np.max(lambda_3):.4f}")
        
        fig, ax = plt.subplots(figsize=(8, 4))
        pl.plot(max_k=3, ax=ax)
        ax.set_title("Persistence Landscape $(\\lambda_1, \\lambda_2, \\lambda_3)$", fontsize=14)
        plt.tight_layout()
        plt.show()