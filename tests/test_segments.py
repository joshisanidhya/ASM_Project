"""Vehicle-run segmentation, run-aware Markov/autocorrelation estimates, and the two fitter fixes."""
import numpy as np
import pandas as pd

from headway.datasets.common import segment_ids
from headway.datasets.highd import load_highd
from headway.datasets.ngsim import load_ngsim
from headway.diagnostics import split_rhat
from headway.distributions import DISTS, EDGES
from headway.fit import fit_all
from headway.metrics import sample_model
from headway.stochastic_process import TrafficRegimeMarkovChain
from headway.validation import independence_check, markov_validation


def test_segment_ids_break_on_vehicle_change_and_on_a_missing_second():
    vehicle = np.array([7, 7, 7, 7, 8, 8, 9])
    second = np.array([1, 2, 4, 5, 5, 6, 6])  # vehicle 7 skips second 3
    assert segment_ids(vehicle, second).tolist() == [0, 0, 1, 1, 2, 2, 3]


def test_highd_groups_are_runs_of_one_vehicle(tmp_path):
    pd.DataFrame({"frameRate": [25]}).to_csv(tmp_path / "01_recordingMeta.csv", index=False)
    rows = []  # (id, second, thw): vehicle 2 listed first to check the loader sorts by vehicle and time
    rows += [(2, s, 3.0) for s in (1, 2)]
    rows += [(1, s, 2.0) for s in (1, 2, 3, 5)] + [(1, 4, 0.2)]  # second 4 is out of range, so 3 and 5 are not adjacent
    pd.DataFrame([{"frame": s * 25, "id": i, "precedingId": 9, "thw": h} for i, s, h in rows]).to_csv(
        tmp_path / "01_tracks.csv", index=False)
    h, g = load_highd(tmp_path, workers=1, return_groups=True)
    assert h.tolist() == [2.0, 2.0, 2.0, 2.0, 3.0, 3.0]
    assert g.tolist() == [0, 0, 0, 1, 2, 2]
    assert load_highd(tmp_path, workers=1).tolist() == h.tolist()


def test_ngsim_groups_are_runs_of_one_vehicle(tmp_path):
    rows = [(2, 10, 3.0), (1, 10, 1.0), (1, 20, 1.5), (2, 20, 3.5), (1, 40, 1.2)]  # vehicle 1 skips second 3
    p = tmp_path / "e.csv"
    p.write_text("Vehicle_ID,Frame_ID,Preceding,Time_Headway,Location\n"
                 + "\n".join(f"{v},{f},9,{h},i-80" for v, f, h in rows))
    h, g = load_ngsim(p, return_groups=True)
    assert h.tolist() == [1.0, 1.5, 1.2, 3.0, 3.5] and g.tolist() == [0, 0, 1, 2, 2]


def test_markov_transitions_never_cross_a_vehicle_boundary():
    # every vehicle holds one regime for 10 seconds; the next vehicle holds another
    states = np.repeat([0, 2, 0, 1], 10)
    h = np.array([1.0, 2.5, 6.0])[states]
    groups = np.repeat(np.arange(4), 10)
    naive = TrafficRegimeMarkovChain.from_headways(h).P
    aware = TrafficRegimeMarkovChain.from_headways(h, groups).P
    assert naive[0, 2] > 0 and naive[2, 0] > 0 or naive[0, 1] > 0        # spurious cross-vehicle jumps
    assert np.allclose(np.diag(aware)[[0, 1, 2]], 1.0)                   # within a vehicle the regime never changes
    v = markov_validation(h, n_sim=2000, groups=groups)
    assert v["n_transitions"] == 36                                       # 40 samples, 4 runs of 10 -> 4 * 9


def test_autocorrelation_counts_only_pairs_inside_a_run():
    x = np.random.default_rng(0).normal(size=1000)
    groups = np.repeat(np.arange(10), 100)
    r = independence_check(x, 3, groups=groups)
    assert r.n_pairs.tolist() == [990, 980, 970]  # lag k loses k pairs per boundary


def test_shifted_exponential_posterior_converges_on_highd_sized_data():
    """Regression: the threshold ridge gave split R-hat of 2 to 35 here before the log-gap sampling."""
    data = sample_model(DISTS["Proposed"], (0.936, 0.540), 1_225_460, seed=1)
    counts = np.histogram(data, bins=EDGES)[0].astype(float)
    fit = fit_all(counts, float(data.min()), names=["Shifted Exponential"], seed=42, workers=1)["Shifted Exponential"]
    assert (split_rhat(fit.draws) < 1.05).all(), split_rhat(fit.draws)
    assert (fit.draws[..., 1] < data.min()).all()          # gamma stays below the smallest observation
    assert abs(fit.theta[0] - 0.5864) < 0.002              # the true mode found by a high-precision optimiser


def test_truncated_likelihood_recovers_parameters_of_a_truncated_sample_better_than_the_paper_one():
    weibull = DISTS["Weibull"]
    true = (1.5, 3.0)
    data = sample_model(weibull, true, 200_000, seed=3)  # drawn from the model conditioned on [0.5, 25] s
    counts = np.histogram(data, bins=EDGES)[0].astype(float)
    err = {}
    for lik in ("paper", "truncated"):
        theta = fit_all(counts, float(data.min()), names=["Weibull"], seed=1, workers=1, likelihood=lik)["Weibull"].theta
        err[lik] = np.abs(theta - np.array(true)).max()
    assert err["truncated"] < 0.05 and err["truncated"] < err["paper"], err
