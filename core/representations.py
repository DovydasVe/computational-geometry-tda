import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# PersistenceImage functionality tested in reproductions/adams_2017
# PL - PersistenceLandscapes
TESTING_PL = False

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
    """
    Computes the persistence landscape representations from a persistence diagram.
    
    Parameters:
    -----------
    pd : PersistenceDiagram
        The input persistence diagram containing birth-death pairs.
    t_interval : array-like
        1D array or sequence of values where landscape functions are evaluated.
    homology_dim : int, default=1
        The Betti dimension (0, 1, 2) of the pairs to extract.
    num_layers : int, optional
        Maximum number of landscape layers (k) to retain. If None, retains all finite layers.
    """
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