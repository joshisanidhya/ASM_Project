"""Unit tests for Traffic Regime Markov Chain and Vehicle Arrival Renewal Process.

Tests specific ASM course requirements from ASM Project 2026M_instructions.pdf:
- Transition probability matrix properties (row-stochasticity, non-negativity)
- n-step transition matrix powers P^n
- Stationary / long-run distribution pi * P = pi
- Mean recurrence times mu_i = 1 / pi_i
- Transient state distribution convergence
- Renewal counting process N(t) and multiple sample path generation
"""
import numpy as np
import pytest

from headway.stochastic_process import (
    REGIME_NAMES,
    TrafficRegimeMarkovChain,
    VehicleArrivalRenewalProcess,
    analyze_stochastic_processes,
)


def test_markov_chain_row_stochasticity():
    # Construct Markov Chain from synthetic headway sequence
    rng = np.random.default_rng(42)
    headways = rng.gamma(2.0, 1.2, size=3000)
    mc = TrafficRegimeMarkovChain.from_headways(headways)

    P = mc.P
    assert P.shape == (3, 3)
    assert np.all(P >= 0.0)
    assert np.all(P <= 1.0)
    # Each row must sum strictly to 1.0
    assert np.allclose(P.sum(axis=1), 1.0)


def test_markov_chain_n_step_transitions():
    P_init = np.array([
        [0.4, 0.4, 0.2],
        [0.2, 0.6, 0.2],
        [0.1, 0.3, 0.6],
    ])
    mc = TrafficRegimeMarkovChain(P_init)

    P_2 = mc.n_step_transition(2)
    assert np.allclose(P_2, P_init @ P_init)
    assert np.allclose(P_2.sum(axis=1), 1.0)

    P_10 = mc.n_step_transition(10)
    assert np.allclose(P_10.sum(axis=1), 1.0)


def test_stationary_distribution_satisfies_invariance():
    P_init = np.array([
        [0.4, 0.4, 0.2],
        [0.2, 0.6, 0.2],
        [0.1, 0.3, 0.6],
    ])
    mc = TrafficRegimeMarkovChain(P_init)
    pi = mc.stationary_distribution()

    assert len(pi) == 3
    assert np.all(pi >= 0.0)
    assert abs(pi.sum() - 1.0) < 1e-10
    # Invariance condition: pi * P = pi
    assert np.allclose(pi @ P_init, pi)


def test_mean_recurrence_times():
    P_init = np.array([
        [0.5, 0.5, 0.0],
        [0.25, 0.5, 0.25],
        [0.0, 0.5, 0.5],
    ])
    mc = TrafficRegimeMarkovChain(P_init)
    pi = mc.stationary_distribution()
    mu = mc.mean_recurrence_times()

    assert np.all(mu > 1.0)
    assert np.allclose(mu, 1.0 / pi)


def test_transient_distribution_convergence():
    P_init = np.array([
        [0.5, 0.3, 0.2],
        [0.2, 0.5, 0.3],
        [0.3, 0.3, 0.4],
    ])
    mc = TrafficRegimeMarkovChain(P_init)
    pi = mc.stationary_distribution()

    # Start entirely in State 0 (Close-Following)
    trajectory = mc.transient_distribution([1.0, 0.0, 0.0], steps=30)
    assert trajectory.shape == (31, 3)
    # The transient distribution must converge to the stationary distribution
    assert np.allclose(trajectory[-1], pi, atol=1e-3)


def test_markov_sample_paths_simulation():
    rng = np.random.default_rng(42)
    headways = rng.gamma(2.5, 1.0, size=2000)
    mc = TrafficRegimeMarkovChain.from_headways(headways)

    paths = mc.simulate_sample_paths(n_steps=50, n_paths=4, seed=42)
    assert paths.shape == (4, 50)
    assert np.all(np.isin(paths, [0, 1, 2]))


def test_renewal_process_and_sample_paths():
    rng = np.random.default_rng(42)
    # Mean headway 2.0s -> flow rate 0.5 veh/s
    headways = rng.exponential(scale=2.0, size=5000) + 0.5  # shifted exponential
    rp = VehicleArrivalRenewalProcess(headways)

    assert abs(rp.mean_h - (2.0 + 0.5)) < 0.1
    assert rp.rate > 0.0
    assert abs(rp.hourly_flow_rate - 3600.0 * rp.rate) < 1e-6

    # Simulate 5 realization paths over 100 seconds
    t_grid, path_counts, exp_counts = rp.simulate_sample_paths(t_max=100.0, n_paths=5, seed=42)
    assert len(t_grid) == 300
    assert path_counts.shape == (5, 300)
    assert len(exp_counts) == 300
    # Counting process must be non-decreasing along each sample path
    for p in range(5):
        assert np.all(np.diff(path_counts[p]) >= 0)
    # Final count at t=100s should be approximately lambda * 100s
    expected_approx = rp.rate * 100.0
    for p in range(5):
        assert abs(path_counts[p, -1] - expected_approx) < 20


def test_analyze_stochastic_processes_end_to_end():
    rng = np.random.default_rng(123)
    headways = rng.gamma(2.0, 1.1, size=4000)
    mc_res, rp_res = analyze_stochastic_processes(headways, n_sample_paths=3, t_simulation=150.0)

    mc_dict = mc_res.to_dict()
    assert "p_close_close" in mc_dict
    assert "pi_close" in mc_dict
    assert "recurrence_close" in mc_dict

    rp_dict = rp_res.to_dict()
    assert "mean_headway_s" in rp_dict
    assert "arrival_rate_veh_s" in rp_dict
    assert "capacity_flow_veh_h" in rp_dict
