import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from scipy.signal import detrend, welch

def generate_4d_white_noise_point_cloud(sigma=1.0, num_points=100, delta_range=(-0.1, 0.1), seed=None):
    if seed is not None:
        np.random.seed(seed)
        
    deltas = np.random.uniform(low=delta_range[0], high=delta_range[1], size=4)
    std_devs = sigma + deltas
    point_cloud = np.random.normal(loc=0.0, scale=std_devs, size=(num_points, 4))
    
    return point_cloud

def generate_white_noise_dataset(sigma_values, num_realizations=10, num_points=100,
                                  delta_range=(-0.1, 0.1), seed=None):
    if seed is not None:
        np.random.seed(seed)
        
    dataset = {}
    for sigma in sigma_values:
        realizations = []
        for _ in range(num_realizations):
            pc = generate_4d_white_noise_point_cloud(
                sigma=sigma, 
                num_points=num_points, 
                delta_range=delta_range
            )
            realizations.append(pc)
        dataset[sigma] = realizations
        
    return dataset

def generate_gamma_transition_dataset(num_realizations=50, num_steps=100, num_points=100, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    dim = 4
    beta = 1.0
    
    alpha_schedule = np.full(num_steps, 8.0)
    for t in range(75, num_steps):
        alpha_schedule[t] = 8.0 - 0.25 * (t - 74)
        
    realizations = []
    for _ in range(num_realizations):
        time_series = []
        for step in range(num_steps):
            alpha = alpha_schedule[step]
            gammas = np.maximum(np.random.gamma(shape=alpha, scale=1.0 / beta, size=dim), 1e-8)
            std_devs = 1.0 / np.sqrt(gammas)
            pc = np.random.normal(loc=0.0, scale=std_devs, size=(num_points, dim))
            time_series.append(pc)
        realizations.append(time_series)
        
    return realizations

def min_max_scale(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def get_1000_days_prior(plot_dates, crash_date_str):
    idx_crash = plot_dates.get_indexer(
        [pd.Timestamp(crash_date_str)], method='nearest'
    )[0]
    return slice(max(0, idx_crash - 1000), idx_crash + 1)

def kendall_report(name, vals):
    tau, p = kendalltau(np.arange(len(vals)), vals)
    print(f"{name:20s}  τ = {tau:+.4f}  (p = {p:.4f})")

def compute_indicators(series, dates, crash_date_str,
                       stat_window=500, eval_days=250,
                       detrend_win=False):
    """
    detrend_win = True: linear detrend inside each 500-day window
    detrend_win = False: raw window
    """
    idx_crash = dates.get_indexer([pd.Timestamp(crash_date_str)], method='nearest')[0]
    eval_indices = range(idx_crash - eval_days + 1, idx_crash + 1)
    eval_dates = dates[eval_indices]

    variances, acfs, specs = [], [], []

    for idx in eval_indices:
        win = series[max(0, idx - stat_window + 1) : idx + 1]

        if detrend_win:
            win_proc = detrend(win, type='linear')
        else:
            win_proc = win

        variances.append(np.var(win_proc))

        acfs.append(pd.Series(win_proc).autocorr(lag=1))

        freqs, psd = welch(win_proc, nperseg=min(128, len(win_proc)))
        mask = freqs <= (0.1 * np.max(freqs))
        specs.append(np.mean(psd[mask]))

    return eval_dates, np.array(variances), np.array(acfs), np.array(specs)