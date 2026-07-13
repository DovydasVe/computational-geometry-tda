import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

class PersistenceImage:
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
            aspect='auto',
            vmin=vmin,
            vmax=vmax
        )
        ax.grid(False)
        ax.set_xlabel("Birth", fontsize=15, labelpad=15)
        ax.set_ylabel("Persistence", fontsize=15, labelpad=15)
        return ax