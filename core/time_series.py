import numpy as np
from collections.abc import Iterable
try:
    from core.representations import pc_landscape_norms, PersistenceImage
    from core.persistence import rips_filtration, PersistenceDiagram
except ModuleNotFoundError:
    from representations import pc_landscape_norms, PersistenceImage
    from persistence import rips_filtration, PersistenceDiagram


def compute_time_series_image_norms(data, window_size, sigma, resolution, dim=1):
    if isinstance(window_size, int):
        window_size = [window_size]
        single_int_input = True
    elif isinstance(window_size, Iterable) and not isinstance(window_size, (str, bytes)):
        window_size = list(window_size)
        single_int_input = False
    else:
        raise TypeError("Expected an int or an iterable of ints")
    
    rows = data.shape[0]
    norm_series = []
    for w in window_size:
        size = rows - w
        
        pds = []
        all_births = []
        all_persistences = []
        for i in range(size):
            pc = data[i:(i + w), :]
            rips = rips_filtration(pc, max_dim=dim, use_ripser=True)
            pd = PersistenceDiagram()
            
            for pair in rips.extract_pairs():
                if pair.dim == dim:
                    pd.add_pair(pair)
                    all_births.append(pair.birth)
                    all_persistences.append(pair.death - pair.birth)
            pds.append(pd)

        global_bounds = (
            (min(all_births) - sigma, max(all_births) + sigma),
            (0, max(all_persistences) + sigma)
        )
        max_persistence = max(all_persistences)

        p1_series = np.zeros(size)
        pinf_series = np.zeros(size)
        for i, pd in enumerate(pds):
            pi = PersistenceImage(pd, sigma=sigma, resolution=resolution, homology_dim=dim,
                                 bounds=global_bounds, max_b=max_persistence)
            p1_series[i] = pi.norm(1)
            pinf_series[i] = pi.norm(np.inf)
        norm_series.append((p1_series, pinf_series))

    return norm_series[0] if single_int_input else norm_series


def compute_time_series_landscape_norms(data, window_size, t_interval, 
                                        dim=1, num_layers=None, mode="cross_sectional", scaled=False):
    if isinstance(window_size, int):
        window_size = [window_size]
        single_int_input = True
    elif isinstance(window_size, Iterable) and not isinstance(window_size, (str, bytes)):
        window_size = list(window_size)
        single_int_input = False
    else:
        raise TypeError("Expected an int or an iterable of ints")

    rows = data.shape[0]

    norm_series = list()
    for w in window_size:
        size = rows - w
        p1_series = np.zeros(size)
        p2_series = np.zeros(size)

        for i in range(size):
            arr = data[i:(i+w), :]
            p1, p2 = pc_landscape_norms(arr, t_interval,
                                            dim=dim, num_layers=num_layers, mode=mode, scaled=scaled)
            p1_series[i] = p1
            p2_series[i] = p2

        norm_series.append((p1_series, p2_series))

    return norm_series[0] if single_int_input else norm_series

