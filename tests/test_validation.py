import numpy as np

from headway.plots import plot_event_count_distribution, plot_markov_state_diagram
from headway.validation import (independence_check, markov_validation, regime_states,
                                renewal_validation, simulate_counts)

REPRESENTATIVE = np.array([1.0, 2.5, 6.0])  # one headway inside each regime


def _chain(P, n, seed):
    rng = np.random.default_rng(seed)
    cum, u = np.cumsum(P, axis=1), rng.random(n)
    s = np.zeros(n, dtype=int)
    for i in range(1, n):
        s[i] = min(int(np.searchsorted(cum[s[i - 1]], u[i], side="right")), 2)
    return s


def test_renewal_simulation_matches_theory_and_is_not_poisson():
    h = np.random.default_rng(0).gamma(6.0, 0.5, 50_000)  # mean 3 s, CV^2 = 1/6: much more regular than Poisson
    df, counts = renewal_validation(h, times=(60, 300), n_paths=3000, seed=1)
    assert counts.shape == (3000, 2)
    assert (df.mean_rel_err < 0.02).all(), df
    assert (df.var_rel_err < 0.12).all(), df
    assert (df.fano_theory < 0.2).all() and (df.fano_sim < 0.3).all()  # Poisson would be 1
    assert (df.ks_poisson > 3 * df.ks_normal).all(), df  # Normal(m, v) describes N(t) far better


def test_simulated_counts_are_monotone_in_t():
    c = simulate_counts(np.full(100, 2.0), times=(10, 20, 40), n_paths=5, seed=0)
    assert (np.diff(c, axis=1) >= 0).all() and (c[:, 0] == 5).all()  # deterministic 2 s headways


def test_independence_check_flags_dependent_sequences_only():
    rng = np.random.default_rng(0)
    iid = rng.normal(size=20_000)
    ar = np.empty(20_000)
    ar[0] = 0
    for i in range(1, len(ar)):
        ar[i] = 0.8 * ar[i - 1] + rng.normal()
    assert (independence_check(iid, 3).autocorr.abs() < 0.03).all()
    assert independence_check(ar, 3).autocorr.iloc[0] > 0.7


def test_markov_validation_accepts_a_true_first_order_chain():
    P = np.array([[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6]])
    s = _chain(P, 200_000, 4)
    h = REPRESENTATIVE[s]
    assert (regime_states(h) == s).all()
    v = markov_validation(h, n_sim=50_000, seed=1)
    assert v["chapman_kolmogorov_2_max_err"] < 0.01 and v["chapman_kolmogorov_5_max_err"] < 0.01
    assert v["stationary_tv_distance"] < 0.005 and v["sim_P_reestimate_max_err"] < 0.02
    assert v["second_order_p"] > 0.001 and v["second_order_effect_tv"] < 0.01


def test_markov_validation_rejects_a_second_order_chain():
    rng = np.random.default_rng(2)
    s = np.zeros(200_000, dtype=int)
    for i in range(2, len(s)):  # next state equals the state two steps back 80% of the time
        s[i] = s[i - 2] if rng.random() < 0.8 else rng.integers(3)
    v = markov_validation(REPRESENTATIVE[s], n_sim=10_000)
    assert v["second_order_p"] < 1e-6 and v["second_order_effect_tv"] > 0.2


def test_new_plots_write_files(tmp_path):
    P = np.array([[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6]])
    plot_markov_state_diagram(P, ["Close", "Platoon", "Free"], "T", tmp_path / "d.png")
    c = np.random.default_rng(0).poisson(20, 500)
    plot_event_count_distribution(c, 60, 20.0, 18.0, 0.33, "T", tmp_path / "c.png")
    assert (tmp_path / "d.png").stat().st_size > 5_000 and (tmp_path / "c.png").stat().st_size > 5_000
