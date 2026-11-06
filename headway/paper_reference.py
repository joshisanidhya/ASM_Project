"""Values published in Leungbootnak et al. (Paper.pdf), for side-by-side comparison with our runs.

Table 3 = fitted parameters, Table 5 = KL divergence and Wasserstein distance. Parameter order
matches headway.distributions.DISTS[...].params.
"""

PAPER_TABLE3_PARAMS = {
    "highd": {
        "Proposed": (0.936, 0.540),
        "Shifted Log-normal": (0.233, 0.899, 0.377),
        "Weibull": (1.481, 2.473),
        "Log-logistic": (2.574, 1.719),
        "Gamma": (2.335, 1.055),
        "Burr": (3.199, 0.602, 1.296),
        "Shifted Exponential": (0.584, 0.500),
    },
    "ngsim": {
        "Proposed": (2.277, 0.481),
        "Shifted Log-normal": (0.683, 0.594, 0.528),
        "Weibull": (1.744, 3.305),
        "Log-logistic": (3.910, 2.515),
        "Gamma": (4.175, 1.428),
        "Burr": (5.237, 0.524, 2.021),
        "Shifted Exponential": (0.423, 0.558),
    },
}

PAPER_TABLE5_KL = {
    "highd": {"Proposed": 0.0032, "Shifted Log-normal": 0.0111, "Weibull": 0.0534, "Log-logistic": 0.0287,
              "Gamma": 0.0454, "Burr": 0.0332, "Shifted Exponential": 0.0065},
    "ngsim": {"Proposed": 0.0805, "Shifted Log-normal": 0.0157, "Weibull": 0.2098, "Log-logistic": 0.0173,
              "Gamma": 0.1182, "Burr": 0.0027, "Shifted Exponential": 0.3024},
}

PAPER_TABLE5_W = {
    "highd": {"Proposed": 0.0011, "Shifted Log-normal": 0.0016, "Weibull": 0.0052, "Log-logistic": 0.0026,
              "Gamma": 0.0049, "Burr": 0.0032, "Shifted Exponential": 0.0018},
    "ngsim": {"Proposed": 0.0040, "Shifted Log-normal": 0.0021, "Weibull": 0.0088, "Log-logistic": 0.0020,
              "Gamma": 0.0173, "Burr": 0.0011, "Shifted Exponential": 0.0047},
}


def compare_with_paper(dataset, counts, params, metrics, diagnostics):
    """Compare refits and fixed Table 3 parameters on the SAME observations.

    Published results are reference columns only; they never enter the posterior.
    Differences with fixed parameters also reflect data/evaluation differences.
    """
    import numpy as np
    from .distributions import DISTS
    from .metrics import histogram_distances

    fitted = {row["distribution"]: row for row in metrics}
    converged = {row["distribution"]: row["converged"] for row in diagnostics}
    rows = []
    for name, dist in DISTS.items():
        reference = PAPER_TABLE3_PARAMS[dataset][name]
        fixed = histogram_distances(counts, dist.bin_probs(reference))
        current = fitted[name]
        # Sorting cannot increase L1 distance; Pinsker gives W_bin <= sqrt(2*KL)/49.
        proxy_bound = float(np.sqrt(2 * PAPER_TABLE5_KL[dataset][name]) / len(counts))
        rows.append({
            "distribution": name, "converged": converged[name],
            "fitted_params": repr(dict(zip(dist.params, map(float, params[name])))),
            "table3_params": repr(dict(zip(dist.params, reference))),
            "fitted_kl": current["kl"], "table5_kl": PAPER_TABLE5_KL[dataset][name],
            "kl_difference": current["kl"] - PAPER_TABLE5_KL[dataset][name],
            "table3_on_current_data_kl": fixed["kl"],
            "fitted_w_bin_proxy": current["wasserstein"], "table5_w": PAPER_TABLE5_W[dataset][name],
            "table3_on_current_data_w_bin_proxy": fixed["wasserstein"],
            "fitted_w1_seconds": current["wasserstein_seconds"],
            "fitted_mass_in_window": float(dist.raw_bin_probs(params[name]).sum()),
            "table3_mass_in_window": float(dist.raw_bin_probs(reference).sum()),
            "table5_proxy_upper_bound_from_kl": proxy_bound,
            "table5_w_exceeds_proxy_bound": PAPER_TABLE5_W[dataset][name] > proxy_bound,
        })
    return rows
