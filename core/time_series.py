try:
    from core.persistence import rips_filtration, PersistenceDiagram
    from core.representations import PersistenceLandscape
except ModuleNotFoundError:
    from persistence import rips_filtration, PersistenceDiagram
    from representations import PersistenceLandscape


def window_landscape_norms(pc, t_interval, dim=1, num_layers=None, mode="cross_sectional", scaled=True):
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