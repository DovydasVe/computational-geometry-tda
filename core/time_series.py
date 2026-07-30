import numpy as np
from collections.abc import Iterable
try:
    from core.representations import pc_landscape_norms
except ModuleNotFoundError:
    from representations import pc_landscape_norms


def compute_time_series_landscape_norms(data, window_size, t_interval, 
                                        dim=1, num_layers=None, mode="cross_sectional", scaled=False):
    if isinstance(window_size, int):
        window_size = [window_size]
        single_int_input = True
    elif isinstance(window_size, Iterable) and not isinstance(window_size, (str, bytes)):
        window_sizes = list(window_size)
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

