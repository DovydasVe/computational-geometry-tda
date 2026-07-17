import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

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