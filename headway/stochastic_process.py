"""Stochastic processes for microscopic and macroscopic traffic flow modeling.

Implements the core stochastic processes required by the Applied Stochastic Modeling (ASM)
course specification (ASM Project 2026M_instructions.pdf):
1. Vehicle Arrival Renewal Process:
   - Microscopic headways as inter-arrival times H_i ~ f(t)
   - Counting process N(t) = max{n : sum_{i=1}^n H_i <= t}
   - Renewal function m(t) = E[N(t)], asymptotic flow rate lambda = 1 / E[H]
   - Multiple stochastic sample path simulation N_k(t)
2. Traffic Congestion / Car-Following Regime Markov Chain:
   - State space: S = {0: Close-Following (0.5 <= h < 1.5s), 1: Stable Platoon (1.5 <= h <= 4.0s), 2: Free-Flow (h > 4.0s)}
   - Transition probability matrix P = (p_ij)
   - n-step transition probabilities P^n
   - Stationary / long-run distribution pi = pi P
   - Mean recurrence times mu_i = 1 / pi_i
   - Transient state evolution pi(n) = pi(0) P^n
   - Multiple sample path simulation of car-following regimes
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .distributions import ALPHA, T_MAX, Dist
from .simulation import sample_distribution

REGIME_NAMES = ["Close-Following", "Stable Platoon", "Free-Flow"]
REGIME_BOUNDS = [0.5, 1.5, 4.0, 25.0]


@dataclass
class MarkovChainResults:
    transition_matrix: np.ndarray       # (3, 3) matrix P
    stationary_distribution: np.ndarray # (3,) stationary vector pi
    mean_recurrence_times: np.ndarray   # (3,) expected return steps
    regime_proportions: np.ndarray      # (3,) empirical state fractions
    sample_paths: np.ndarray            # (n_paths, n_steps) state trajectories

    def to_dict(self) -> Dict[str, Any]:
        return {
            "p_close_close": float(self.transition_matrix[0, 0]),
            "p_close_platoon": float(self.transition_matrix[0, 1]),
            "p_close_free": float(self.transition_matrix[0, 2]),
            "p_platoon_close": float(self.transition_matrix[1, 0]),
            "p_platoon_platoon": float(self.transition_matrix[1, 1]),
            "p_platoon_free": float(self.transition_matrix[1, 2]),
            "p_free_close": float(self.transition_matrix[2, 0]),
            "p_free_platoon": float(self.transition_matrix[2, 1]),
            "p_free_free": float(self.transition_matrix[2, 2]),
            "pi_close": float(self.stationary_distribution[0]),
            "pi_platoon": float(self.stationary_distribution[1]),
            "pi_free": float(self.stationary_distribution[2]),
            "recurrence_close": float(self.mean_recurrence_times[0]),
            "recurrence_platoon": float(self.mean_recurrence_times[1]),
            "recurrence_free": float(self.mean_recurrence_times[2]),
        }


class TrafficRegimeMarkovChain:
    """Discrete-Time Markov Chain modeling car-following regime transitions."""

    def __init__(self, transition_matrix: np.ndarray):
        P = np.asarray(transition_matrix, dtype=float)
        if P.shape != (3, 3):
            raise ValueError(f"Transition matrix must be 3x3, got {P.shape}")
        # Ensure row-stochasticity
        row_sums = P.sum(axis=1, keepdims=True)
        self.P = np.where(row_sums > 0, P / row_sums, np.eye(3))

    @classmethod
    def from_headways(cls, headways: np.ndarray) -> "TrafficRegimeMarkovChain":
        """Construct Markov Chain from consecutive empirical or simulated headway observations."""
        h = np.asarray(headways, dtype=float)
        # Discretize into 3 physical car-following regimes
        # 0: Close-Following [0.5, 1.5), 1: Stable Platoon [1.5, 4.0), 2: Free-Flow [4.0, inf)
        states = np.digitize(h, [1.5, 4.0])  # returns 0, 1, 2
        states = np.clip(states, 0, 2)

        # Count 1-step transitions
        n_states = 3
        counts = np.zeros((n_states, n_states), dtype=float)
        for i in range(len(states) - 1):
            counts[states[i], states[i + 1]] += 1.0

        # Regularize zero rows with uniform transitions
        row_sums = counts.sum(axis=1, keepdims=True)
        P = np.where(row_sums > 0, counts / row_sums, np.eye(n_states))
        return cls(P)

    def n_step_transition(self, n: int) -> np.ndarray:
        """Compute n-step transition probability matrix P^n."""
        return np.linalg.matrix_power(self.P, max(1, int(n)))

    def stationary_distribution(self) -> np.ndarray:
        """Compute exact stationary distribution pi satisfying pi * P = pi and sum(pi) = 1.

        Solves the linear system:
            (P^T - I) * pi^T = 0
            sum(pi) = 1
        """
        n = self.P.shape[0]
        # Form augmented system A * pi^T = b
        A = np.vstack([self.P.T - np.eye(n), np.ones(n)])
        b = np.append(np.zeros(n), 1.0)
        # Least squares solution is robust to numerical floating point imprecision
        pi, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        pi = np.clip(pi, 0.0, 1.0)
        return pi / pi.sum()

    def mean_recurrence_times(self) -> np.ndarray:
        """Expected number of transitions to return to state i: mu_i = 1 / pi_i."""
        pi = self.stationary_distribution()
        return np.where(pi > 1e-12, 1.0 / pi, np.inf)

    def transient_distribution(self, initial_dist: Sequence[float], steps: int = 20) -> np.ndarray:
        """Compute transient state probabilities pi(n) = pi(0) * P^n for n = 0, ..., steps.

        Returns array of shape (steps + 1, 3).
        """
        pi_0 = np.asarray(initial_dist, dtype=float)
        pi_0 = pi_0 / pi_0.sum()
        trajectory = np.empty((steps + 1, len(pi_0)), dtype=float)
        trajectory[0] = pi_0
        current = pi_0
        for s in range(1, steps + 1):
            current = current @ self.P
            trajectory[s] = current
        return trajectory

    def simulate_sample_paths(
        self,
        n_steps: int = 100,
        n_paths: int = 5,
        initial_state: int = 0,
        seed: Optional[int] = 42,
    ) -> np.ndarray:
        """Simulate multiple realization paths of car-following regimes over n successive vehicles.

        Returns array of shape (n_paths, n_steps).
        """
        rng = np.random.default_rng(seed)
        paths = np.empty((n_paths, n_steps), dtype=int)
        for p in range(n_paths):
            state = initial_state
            paths[p, 0] = state
            for step in range(1, n_steps):
                probs = self.P[state]
                state = int(rng.choice(3, p=probs))
                paths[p, step] = state
        return paths


@dataclass
class RenewalProcessResults:
    mean_headway: float             # E[H] [seconds]
    var_headway: float              # Var(H) [seconds^2]
    asymptotic_rate: float          # lambda = 1 / E[H] [veh/s]
    hourly_flow_rate: float         # q = 3600 * lambda [veh/h/lane]
    fano_factor: float              # Var(H) / E[H]^2 (dispersion index)
    sample_paths_time: np.ndarray   # Grid of time points t [s]
    sample_paths_counts: np.ndarray # (n_paths, len(time)) counts N(t)
    expected_counts: np.ndarray     # Renewal function m(t) = E[N(t)]

    def to_dict(self) -> Dict[str, float]:
        return {
            "mean_headway_s": self.mean_headway,
            "var_headway_s2": self.var_headway,
            "arrival_rate_veh_s": self.asymptotic_rate,
            "capacity_flow_veh_h": self.hourly_flow_rate,
            "fano_factor": self.fano_factor,
        }


class VehicleArrivalRenewalProcess:
    """Stochastic Renewal Counting Process modeling vehicle arrivals at a roadway detector."""

    def __init__(self, headways: np.ndarray):
        h = np.asarray(headways, dtype=float)
        if len(h) == 0:
            raise ValueError("Headway sample must not be empty.")
        self.headways = h
        self.mean_h = float(np.mean(h))
        self.var_h = float(np.var(h, ddof=1)) if len(h) > 1 else 0.0
        self.rate = float(1.0 / self.mean_h) if self.mean_h > 0 else 0.0
        self.hourly_flow_rate = self.rate * 3600.0

    @classmethod
    def from_distribution(
        cls,
        dist: Dist,
        theta: Sequence[float],
        sample_size: int = 20000,
        seed: int = 42,
    ) -> "VehicleArrivalRenewalProcess":
        """Construct renewal process from a theoretical fitted headway distribution."""
        samples = sample_distribution(dist, theta, size=sample_size, seed=seed)
        return cls(samples)

    def renewal_function(self, t_grid: np.ndarray) -> np.ndarray:
        """Asymptotic renewal function m(t) = E[N(t)] for large t:

        m(t) approx lambda * t + (sigma^2 - mu^2) / (2 * mu^2)
        """
        t = np.asarray(t_grid, dtype=float)
        mu = self.mean_h
        sigma2 = self.var_h
        asymp_const = (sigma2 - mu ** 2) / (2.0 * mu ** 2) if mu > 0 else 0.0
        return np.maximum(0.0, self.rate * t + asymp_const)

    def simulate_sample_paths(
        self,
        t_max: float = 300.0,
        n_paths: int = 5,
        grid_points: int = 300,
        seed: Optional[int] = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate multiple stochastic realization paths of cumulative vehicle arrivals N(t).

        Returns:
            t_grid: 1D array of evaluation time points [s]
            path_counts: (n_paths, grid_points) array of arrival counts N(t)
            expected_counts: (grid_points,) theoretical renewal function m(t)
        """
        rng = np.random.default_rng(seed)
        t_grid = np.linspace(0.0, t_max, grid_points)
        path_counts = np.zeros((n_paths, grid_points), dtype=int)

        # Estimate max arrivals needed with safety buffer
        est_events = int(np.ceil((t_max / max(self.mean_h, 0.1)) * 2.0))

        for p in range(n_paths):
            # Bootstrap sample headways from the empirical/fitted distribution
            drawn_h = rng.choice(self.headways, size=est_events, replace=True)
            arrival_times = np.cumsum(drawn_h)
            # N(t) = number of arrivals occurring at or before t
            path_counts[p] = np.searchsorted(arrival_times, t_grid, side="right")

        expected_counts = self.renewal_function(t_grid)
        return t_grid, path_counts, expected_counts


def analyze_stochastic_processes(
    headways: np.ndarray,
    n_sample_paths: int = 5,
    t_simulation: float = 300.0,
    seed: int = 42,
) -> Tuple[MarkovChainResults, RenewalProcessResults]:
    """Execute complete stochastic process analysis (Markov Chain + Renewal Process)."""
    # 1. Markov Chain Analysis
    mc = TrafficRegimeMarkovChain.from_headways(headways)
    P = mc.P
    stat_dist = mc.stationary_distribution()
    rec_times = mc.mean_recurrence_times()
    mc_paths = mc.simulate_sample_paths(n_steps=100, n_paths=n_sample_paths, seed=seed)

    # Empirical state proportions
    states = np.clip(np.digitize(headways, [1.5, 4.0]), 0, 2)
    regime_props = np.bincount(states, minlength=3) / float(len(states))

    mc_results = MarkovChainResults(
        transition_matrix=P,
        stationary_distribution=stat_dist,
        mean_recurrence_times=rec_times,
        regime_proportions=regime_props,
        sample_paths=mc_paths,
    )

    # 2. Renewal Counting Process Analysis
    rp = VehicleArrivalRenewalProcess(headways)
    t_grid, path_counts, exp_counts = rp.simulate_sample_paths(
        t_max=t_simulation,
        n_paths=n_sample_paths,
        seed=seed,
    )

    fano = rp.var_h / (rp.mean_h ** 2) if rp.mean_h > 0 else 1.0

    rp_results = RenewalProcessResults(
        mean_headway=rp.mean_h,
        var_headway=rp.var_h,
        asymptotic_rate=rp.rate,
        hourly_flow_rate=rp.rate * 3600.0,
        fano_factor=fano,
        sample_paths_time=t_grid,
        sample_paths_counts=path_counts,
        expected_counts=exp_counts,
    )

    return mc_results, rp_results
