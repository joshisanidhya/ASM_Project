import numpy as np

from headway.diagnostics import split_rhat
from headway.distributions import DISTS, EDGES
from headway.metrics import evaluate, sample_model


def test_metrics_are_small_for_data_drawn_from_the_model():
    P = DISTS["Proposed"]
    data = sample_model(P, (2.5, 0.55), 50_000, seed=3)
    counts = np.histogram(data, bins=EDGES)[0].astype(float)
    m = evaluate(P, (2.5, 0.55), data, counts)
    assert m["ks_D"] < 0.02 and m["kl"] < 0.01 and m["wasserstein"] < 0.02, m
    assert m["wasserstein_sorted_proxy"] < 0.002, m
    wrong = evaluate(P, (6.0, 0.8), data, counts)  # a clearly wrong model must score worse
    assert wrong["kl"] > 10 * m["kl"] and wrong["ks_D"] > m["ks_D"] and wrong["wasserstein"] > m["wasserstein"]


def test_rhat_near_one_for_agreeing_chains_and_large_for_disjoint_ones():
    rng = np.random.default_rng(0)
    same = rng.normal(size=(2, 4000, 1))
    assert abs(split_rhat(same)[0] - 1) < 0.01
    apart = same + np.array([0.0, 5.0])[:, None, None]  # chain 2 sits elsewhere
    assert split_rhat(apart)[0] > 1.5
