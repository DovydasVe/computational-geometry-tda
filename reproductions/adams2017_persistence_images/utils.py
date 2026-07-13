import numpy as np
from sklearn_extra.cluster import KMedoids
from scipy.spatial.distance import pdist


def generate_circle(n_points=500, noise=0.05, random_state=None):
    rng = np.random.default_rng(random_state)
    theta = rng.uniform(0, 2*np.pi, n_points)
    x = 0.25 * np.cos(theta) + 0.5
    y = 0.25 * np.sin(theta) + 0.5
    pts = np.column_stack((x, y, np.zeros(n_points)))
    pts += rng.normal(0, noise, pts.shape)
    return pts

def generate_sphere(n_points=500, noise=0.05, random_state=None):
    rng = np.random.default_rng(random_state)
    X = rng.normal(0, 1, (n_points, 3))
    S = np.linalg.norm(X, axis=1, keepdims=True)
    X = 0.25 * (X / S) + 0.5
    X += rng.normal(0, noise, X.shape)
    return X

def generate_torus(n_points=500, noise=0.05, random_state=None):
    rng = np.random.default_rng(random_state)
    thetas = []
    while len(thetas) < n_points:
        batch_size = (n_points - len(thetas)) * 4
        x = rng.uniform(0, 2*np.pi, batch_size)
        y = rng.uniform(0, 1/np.pi, batch_size)
        fx = (1 + 0.5 * np.cos(x)) / (2*np.pi)
        accepted = x[y < fx]
        thetas.extend(accepted)
    theta = np.array(thetas[:n_points])
    phi = rng.uniform(0, 2*np.pi, n_points)
    x = (0.35 + 0.15 * np.cos(theta)) * np.cos(phi) + 0.5
    y = (0.35 + 0.15 * np.cos(theta)) * np.sin(phi) + 0.5
    z = np.sin(theta) + 0.5
    pts = np.column_stack((x, y, z))
    pts += rng.normal(0, noise, pts.shape)
    return pts

def generate_cube(n_points=500, noise=0.05, random_state=None):
    rng = np.random.default_rng(random_state)
    return rng.uniform(0, 1, (n_points, 3))

def generate_three_clusters(n_points=500, noise=0.05, random_state=None):
    rng = np.random.default_rng(random_state)
    centers = rng.uniform(0, 1, (3, 3))
    a = max(rng.integers(1, int(0.45 * n_points) + 1), int(0.1 * n_points))
    b = max(rng.integers(1, int(0.45 * (n_points - a)) + 1), int((n_points - a) * 0.1))
    c = n_points - a - b
    cluster_ids = np.concatenate([np.repeat(0, a), np.repeat(1, b), np.repeat(2, c)])
    points = centers[cluster_ids] + rng.normal(0, 0.05, (n_points, 3))
    return points

def generate_clusters_within_clusters(n_points=500, noise=0.05, random_state=None):
    rng = np.random.default_rng(random_state)
    centers_large = rng.uniform(0, 1, (3, 3))
    
    a = max(rng.integers(1, int(0.45 * n_points) + 1), int(0.1 * n_points))
    b = max(rng.integers(1, int(0.45 * (n_points - a)) + 1), int((n_points - a) * 0.1))
    c = n_points - a - b
    
    Centers = np.column_stack([np.repeat(centers_large[:, 0:1], 3, axis=1),
                               np.repeat(centers_large[:, 1:2], 3, axis=1),
                               np.repeat(centers_large[:, 2:3], 3, axis=1)])
    Centers += rng.normal(0, 0.05, (3, 9))
    
    a1 = max(rng.integers(1, int(0.45 * a) + 1), int(0.1 * a))
    a2 = max(rng.integers(1, int(0.45 * (a - a1)) + 1), int((a - a1) * 0.1))
    a3 = a - a1 - a2
    
    b1 = max(rng.integers(1, int(0.45 * b) + 1), int(0.1 * b))
    b2 = max(rng.integers(1, int(0.45 * (b - b1)) + 1), int((b - b1) * 0.1))
    b3 = b - b1 - b2
    
    c1 = max(rng.integers(1, int(0.45 * c) + 1), int(0.1 * c))
    c2 = max(rng.integers(1, int(0.45 * (c - c1)) + 1), int((c - c1) * 0.1))
    c3 = c - c1 - c2
    
    subcounts = [a1, a2, a3, b1, b2, b3, c1, c2, c3]
    points = []
    for idx, count in enumerate(subcounts):
        points.append(np.repeat(Centers[:, idx:idx+1], count, axis=1))
    X = np.column_stack(points).T
    X += rng.normal(0, 0.02, X.shape)
    return X

def keep_top_k(dgm, k=50):
    if len(dgm) <= k:
        return dgm
    persistence = dgm[:, 1] - dgm[:, 0]
    idx = np.argsort(persistence)[-k:]
    return dgm[idx]

def get_best_kmedoids_accuracy(matrix, true_labels, n_clusters=6, n_init=1000):
    best_inertia = np.inf
    best_labels = None
    best_medoids = None
    
    for seed in range(n_init):
        kmed = KMedoids(
            n_clusters=n_clusters, 
            metric='precomputed', 
            method='alternate', 
            init='random', 
            random_state=seed
        )
        labels = kmed.fit_predict(matrix)
        
        if kmed.inertia_ < best_inertia:
            best_inertia = kmed.inertia_
            best_labels = labels
            best_medoids = kmed.medoid_indices_
            
    cluster_classes = [true_labels[idx] for idx in best_medoids]
    pred_classes = np.array([cluster_classes[label] for label in best_labels])
    return np.mean(pred_classes == true_labels)

def compute_d2_histograms(dataset, n_bins=100, max_dist=2.0):
    histograms = []
    bin_edges = np.linspace(0, max_dist, n_bins + 1)
    
    for pts, _ in dataset:
        dists = pdist(pts)
        hist, _ = np.histogram(dists, bins=bin_edges)
        hist_norm = hist / hist.sum()
        histograms.append(hist_norm)
        
    return np.array(histograms)