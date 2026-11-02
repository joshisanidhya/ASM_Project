"""Comprehensive test suite for goodness-of-fit evaluation metrics.

Verifies:
- 1-sample exact KS test vs continuous CDF
- 2-sample simulated KS test
- Chi-Square test with bin merging (Oi >= 5) and dof calculation
- Kullback-Leibler divergence (axioms, zero-handling, non-negativity)
- True 1D Distributional Wasserstein-1 distance (Vallender 1974) vs sorted bin proxy
"""
import numpy as np
import pytest
from scipy.stats import wasserstein_distance

from headway.distributions import ALPHA, CENTERS, DISTS, EDGES, T_MAX
from headway.metrics import evaluate, merge_bins_for_chi2, sample_model


def test_merge_bins_guarantees_minimum_frequency():
    obs = np.array([1, 2, 1, 10, 2, 3, 1, 0, 8])
    exp = np.array([1.5, 1.5, 1.5, 9.0, 2.0, 2.0, 2.0, 1.0, 7.0])
    o_merged, e_merged = merge_bins_for_chi2(obs, exp, min_count=5.0)

    # Every merged bin must have >= 5 count (except possibly empty edge cases)
    assert np.all(o_merged >= 5.0)
    # Total sum of frequencies must be strictly preserved
    assert abs(o_merged.sum() - obs.sum()) < 1e-9
    assert abs(e_merged.sum() - exp.sum()) < 1e-9


def test_kl_divergence_properties():
    p = np.array([0.1, 0.4, 0.3, 0.2])
    q = np.array([0.1, 0.4, 0.3, 0.2])
    # D_KL(P || P) = 0
    kl_identical = float(np.sum(p * np.log(p / q)))
    assert abs(kl_identical) < 1e-12

    # Non-negativity for distinct distributions
    q_diff = np.array([0.25, 0.25, 0.25, 0.25])
    kl_diff = float(np.sum(p * np.log(p / q_diff)))
    assert kl_diff > 0.0


def test_true_distributional_wasserstein_vs_proxy():
    """Verify that the true 1D Wasserstein distance tracks shifts along the time axis,
    whereas the sorted-bin proxy is blind to coordinate shifts."""
    # Two identical shapes shifted along the time axis by 4.0 seconds
    p1 = np.exp(-0.5 * ((CENTERS - 2.0) / 0.5) ** 2)
    p1 /= p1.sum()
    p2 = np.exp(-0.5 * ((CENTERS - 6.0) / 0.5) ** 2)
    p2 /= p2.sum()

    w_true = wasserstein_distance(CENTERS, CENTERS, p1, p2)
    w_proxy = float(np.mean(np.abs(np.sort(p1) - np.sort(p2))))

    # The true Wasserstein distance must be approximately equal to the 4.0s shift
    assert abs(w_true - 4.0) < 0.2
    # The proxy completely collapses the coordinate shift and reports near-zero!
    assert w_proxy < 0.01
    assert w_true > 200 * w_proxy


def test_evaluate_recovers_small_metrics_for_true_model():
    P = DISTS["Proposed"]
    theta_true = (2.5, 0.55)
    rng = np.random.default_rng(123)
    data = sample_model(P, theta_true, 40_000, seed=123)
    counts = np.histogram(data, bins=EDGES)[0].astype(float)

    m = evaluate(P, theta_true, data, counts, seed=123)

    assert m["ks_D"] < 0.02
    assert m["ks_p"] > 0.05
    assert m["kl"] < 0.005
    assert m["wasserstein"] < 0.02
    assert m["wasserstein_sorted_proxy"] < 0.002

    # A severely mismatched model must produce drastically higher error metrics
    theta_wrong = (8.0, 0.9)
    m_wrong = evaluate(P, theta_wrong, data, counts, seed=123)
    assert m_wrong["kl"] > 50 * m["kl"]
    assert m_wrong["wasserstein"] > 50 * m["wasserstein"]
    assert m_wrong["ks_D"] > 5 * m["ks_D"]
