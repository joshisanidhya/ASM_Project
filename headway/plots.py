"""Publication-quality visualization suite for headway distributions, MCMC diagnostics, and MLE extensions.

Produces figures corresponding to:
- Frequency histogram with fitted curves (paper Figure 5)
- MCMC trace plots and posterior distributions (paper Figure 4)
- Empirical vs theoretical CDF comparisons
- MCMC vs MLE parameter, runtime, and goodness-of-fit comparisons
- Stochastic simulation validation figures
"""
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .distributions import ALPHA, CENTERS, DISTS, EDGES, T_MAX

# Styling defaults for publication-grade scientific graphics
PALETTE = {
    "Proposed": "#D62728",           # Distinct Crimson
    "Shifted Log-normal": "#1F77B4", # Blue
    "Weibull": "#2CA02C",            # Green
    "Log-logistic": "#FF7F0E",       # Orange
    "Gamma": "#9467BD",              # Purple
    "Burr": "#8C564B",               # Brown
    "Shifted Exponential": "#E377C2",# Pink
}
CHAIN_COLORS = ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728"]


def plot_fitted_distributions(
    title: str,
    counts: np.ndarray,
    params: Dict[str, Sequence[float]],
    path: Union[str, Path],
    figsize: Tuple[float, float] = (10, 6),
) -> None:
    """Plot empirical headway frequency histogram with fitted distributions overlaid (paper Figure 5)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    n = float(counts.sum())

    # Empirical histogram
    ax.bar(
        CENTERS,
        counts,
        width=0.48,
        color="#D3D3D3",
        edgecolor="#808080",
        alpha=0.8,
        label="Observed Data",
    )

    # Plot fitted model curves
    fine_grid = np.linspace(ALPHA, T_MAX, 1000)
    for dname, theta in params.items():
        if dname not in DISTS:
            continue
        dist = DISTS[dname]
        color = PALETTE.get(dname, "#333333")
        is_proposed = (dname == "Proposed")
        lw = 2.4 if is_proposed else 1.4
        ls = "-" if is_proposed else "--"
        # Binned height expectation
        expected_counts = n * dist.bin_probs(theta)
        ax.plot(CENTERS, expected_counts, label=dname, color=color, lw=lw, linestyle=ls)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Headway [sec]", fontsize=12, labelpad=8)
    ax.set_ylabel("Frequency", fontsize=12, labelpad=8)
    ax.set_xlim(0.0, 25.5)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, facecolor="white", edgecolor="#D0D0D0", fontsize=10)
    fig.tight_layout()

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_empirical_cdf(
    title: str,
    data: np.ndarray,
    params: Dict[str, Sequence[float]],
    path: Union[str, Path],
) -> None:
    """Plot empirical cumulative distribution against theoretical model CDFs."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sorted_data = np.sort(data)
    emp_cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

    # Empirical step CDF
    ax.step(sorted_data, emp_cdf, label="Empirical CDF", color="#333333", lw=2.0, where="post")

    # Theoretical CDF curves
    fine_grid = np.linspace(ALPHA, T_MAX, 1000)
    for dname, theta in params.items():
        if dname not in DISTS:
            continue
        dist = DISTS[dname]
        color = PALETTE.get(dname, "#333333")
        is_proposed = (dname == "Proposed")
        lw = 2.2 if is_proposed else 1.3
        ls = "-" if is_proposed else "--"
        cdf_vals = dist.trunc_cdf(fine_grid, theta)
        ax.plot(fine_grid, cdf_vals, label=dname, color=color, lw=lw, linestyle=ls)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Headway [sec]", fontsize=12, labelpad=8)
    ax.set_ylabel("Cumulative Probability", fontsize=12, labelpad=8)
    ax.set_xlim(0.0, 25.5)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, facecolor="white", edgecolor="#D0D0D0", fontsize=10)
    fig.tight_layout()

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_traces(
    title: str,
    param_names: Sequence[str],
    full_draws: np.ndarray,
    n_warmup: int,
    path: Union[str, Path],
) -> None:
    """Plot MCMC trace plots across all iterations (including warmup), matching paper Figure 4b.

    full_draws shape: (chains, n_iter, n_params)
    """
    c, n_iter, d = full_draws.shape
    fig, axes = plt.subplots(d, 1, figsize=(10, 2.5 * d), squeeze=False)

    for j, p in enumerate(param_names):
        ax = axes[j, 0]
        # Shaded warmup region
        ax.axvspan(0, n_warmup, color="#EAEAEA", alpha=0.8, label="Warmup (5000)" if j == 0 else None)
        for ch in range(c):
            ax.plot(
                full_draws[ch, :, j],
                lw=0.5,
                color=CHAIN_COLORS[ch % len(CHAIN_COLORS)],
                label=f"Chain {ch+1}" if j == 0 else None,
                alpha=0.85,
            )
        ax.set_ylabel(f"{p}", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.set_xlim(0, n_iter)
        if j == d - 1:
            ax.set_xlabel("Iteration", fontsize=12)

    axes[0, 0].set_title(f"{title} — Trace Plots", fontsize=13, fontweight="bold", pad=10)
    if d > 0:
        axes[0, 0].legend(loc="upper right", fontsize=9, frameon=True)
    fig.tight_layout()

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_posteriors(
    title: str,
    param_names: Sequence[str],
    post_draws: np.ndarray,
    path: Union[str, Path],
) -> None:
    """Plot posterior histograms and credible intervals, matching paper Figure 4a.

    post_draws shape: (chains, n_post, n_params)
    """
    c, n, d = post_draws.shape
    fig, axes = plt.subplots(1, d, figsize=(4.5 * d, 4.0), squeeze=False)
    flat = post_draws.reshape(-1, d)

    for j, p in enumerate(param_names):
        ax = axes[0, j]
        col = flat[:, j]
        mean_val = float(np.mean(col))
        q025, q975 = np.percentile(col, [2.5, 97.5])

        ax.hist(col, bins=40, density=True, color="#4C72B0", edgecolor="white", alpha=0.7)
        ax.axvline(mean_val, color="#D62728", lw=1.8, label=f"Mean: {mean_val:.3f}")
        ax.axvline(q025, color="#555555", lw=1.2, ls="--", label=f"95% CI: [{q025:.3f}, {q975:.3f}]")
        ax.axvline(q975, color="#555555", lw=1.2, ls="--")

        ax.set_title(f"Parameter: {p}", fontsize=12, fontweight="bold")
        ax.set_xlabel(f"{p}", fontsize=11)
        ax.set_ylabel("Posterior Density", fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="upper right", fontsize=8, frameon=True)

    fig.suptitle(f"{title} — Parameter Posteriors", fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_mcmc_vs_mle_comparison(
    title: str,
    counts: np.ndarray,
    mcmc_params: Dict[str, Sequence[float]],
    mle_params: Dict[str, Sequence[float]],
    path: Union[str, Path],
) -> None:
    """Plot comparison of fitted distribution curves between MCMC and MLE."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    n = float(counts.sum())

    for ax, param_dict, subtitle in [(ax1, mcmc_params, "MCMC (Bayesian Posterior Mean)"),
                                     (ax2, mle_params, "MLE (Maximum Likelihood)")]:
        ax.bar(CENTERS, counts, width=0.48, color="#E0E0E0", edgecolor="#999999", alpha=0.7, label="Observed Data")
        for dname, theta in param_dict.items():
            if dname not in DISTS:
                continue
            color = PALETTE.get(dname, "#333333")
            is_proposed = (dname == "Proposed")
            lw = 2.4 if is_proposed else 1.4
            ls = "-" if is_proposed else "--"
            expected_counts = n * DISTS[dname].bin_probs(theta)
            ax.plot(CENTERS, expected_counts, label=dname, color=color, lw=lw, linestyle=ls)

        ax.set_title(f"{title}: {subtitle}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Headway [sec]", fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.set_xlim(0.0, 25.5)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(frameon=True, fontsize=9)

    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_runtime_comparison(
    mcmc_runtimes: Dict[str, float],
    mle_runtimes: Dict[str, float],
    path: Union[str, Path],
) -> None:
    """Bar chart comparing execution runtime between MCMC and MLE."""
    dists = [d for d in DISTS.keys() if d in mcmc_runtimes and d in mle_runtimes]
    mcmc_vals = [mcmc_runtimes[d] for d in dists]
    mle_vals = [mle_runtimes[d] for d in dists]

    x = np.arange(len(dists))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - width / 2, mcmc_vals, width, label="MCMC (2 chains x 10,000)", color="#4C72B0")
    ax.bar(x + width / 2, mle_vals, width, label="MLE (Numerical Optimization)", color="#55A868")

    ax.set_ylabel("Runtime [seconds] (log scale)", fontsize=11)
    ax.set_yscale("log")
    ax.set_title("Runtime Comparison: MCMC vs. Maximum Likelihood Estimation", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(dists, rotation=25, ha="right", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.5, axis="y")
    ax.legend(frameon=True, fontsize=10)
    fig.tight_layout()

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_goodness_of_fit_metrics(
    mcmc_metrics: Dict[str, Dict[str, float]],
    mle_metrics: Optional[Dict[str, Dict[str, float]]],
    path: Union[str, Path],
) -> None:
    """Grouped bar chart comparing KL divergence and Wasserstein distance."""
    dists = list(mcmc_metrics.keys())
    mcmc_kl = [mcmc_metrics[d]["kl"] for d in dists]
    mcmc_w = [mcmc_metrics[d]["wasserstein"] for d in dists]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(dists))
    width = 0.35 if mle_metrics else 0.5

    # KL Divergence
    ax1.bar(x - width/2 if mle_metrics else x, mcmc_kl, width, label="MCMC", color="#4C72B0")
    if mle_metrics:
        mle_kl = [mle_metrics[d]["kl"] for d in dists]
        ax1.bar(x + width/2, mle_kl, width, label="MLE", color="#55A868")
    ax1.set_title("Kullback-Leibler Divergence (Lower is Better)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("D_KL", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(dists, rotation=30, ha="right", fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(frameon=True)

    # Wasserstein Distance
    ax2.bar(x - width/2 if mle_metrics else x, mcmc_w, width, label="MCMC", color="#4C72B0")
    if mle_metrics:
        mle_w = [mle_metrics[d]["wasserstein"] for d in dists]
        ax2.bar(x + width/2, mle_w, width, label="MLE", color="#55A868")
    ax2.set_title("Distributional Wasserstein Distance [s] (Lower is Better)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("W_1 [seconds]", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(dists, rotation=30, ha="right", fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(frameon=True)

    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_simulation_validation(
    empirical: np.ndarray,
    simulated: np.ndarray,
    dist_name: str,
    path: Union[str, Path],
) -> None:
    """Plot empirical vs simulated headway histograms and CDF comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Histogram overlay
    ax1.hist(empirical, bins=EDGES, density=True, alpha=0.5, color="#1F77B4", label="Empirical Observations", edgecolor="white")
    ax1.hist(simulated, bins=EDGES, density=True, alpha=0.5, color="#D62728", label=f"Simulated ({dist_name})", edgecolor="white")
    ax1.set_title(f"{dist_name}: Empirical vs Simulated Density", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Headway [sec]", fontsize=11)
    ax1.set_ylabel("Probability Density", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(frameon=True)

    # CDF overlay
    s_emp = np.sort(empirical)
    s_sim = np.sort(simulated)
    ax2.plot(s_emp, np.arange(1, len(s_emp) + 1) / len(s_emp), color="#1F77B4", lw=1.8, label="Empirical CDF")
    ax2.plot(s_sim, np.arange(1, len(s_sim) + 1) / len(s_sim), color="#D62728", lw=1.8, linestyle="--", label=f"Simulated CDF ({dist_name})")
    ax2.set_title(f"{dist_name}: Empirical vs Simulated CDF", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Headway [sec]", fontsize=11)
    ax2.set_ylabel("Cumulative Probability", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(frameon=True)

    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_arrival_process_sample_paths(
    t_grid: np.ndarray,
    path_counts: np.ndarray,
    expected_counts: np.ndarray,
    title: str,
    path: Union[str, Path],
) -> None:
    """Plot multiple stochastic sample paths of vehicle arrivals N(t) over time (ASM Course Page 4)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    n_paths = path_counts.shape[0]

    # Plot individual stochastic sample paths
    for i in range(n_paths):
        ax.step(
            t_grid,
            path_counts[i],
            where="post",
            lw=1.2,
            alpha=0.65,
            label=f"Sample Path {i+1}",
        )

    # Plot theoretical renewal expectation m(t)
    ax.plot(
        t_grid,
        expected_counts,
        color="#000000",
        lw=2.5,
        linestyle="--",
        label=r"Theoretical Renewal Mean $\mathbb{E}[N(t)]$",
    )

    ax.set_title(f"{title} — Vehicle Arrival Renewal Process $N(t)$", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Elapsed Time $t$ [seconds]", fontsize=11)
    ax.set_ylabel("Cumulative Vehicle Arrivals $N(t)$", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", frameon=True, fontsize=9)
    fig.tight_layout()

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_markov_transition_heatmap(
    transition_matrix: np.ndarray,
    regime_names: Sequence[str],
    title: str,
    path: Union[str, Path],
) -> None:
    """Plot 2D heatmap of the car-following regime transition probability matrix (ASM Course Page 4)."""
    fig, ax = plt.subplots(figsize=(7, 6))
    P = np.asarray(transition_matrix, dtype=float)

    im = ax.imshow(P, cmap="Blues", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Transition Probability $p_{ij}$", fontsize=10)

    # Text annotations in each cell
    n = len(regime_names)
    for i in range(n):
        for j in range(n):
            val = P[i, j]
            text_color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=text_color, fontweight="bold", fontsize=11)

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(regime_names, fontsize=10)
    ax.set_yticklabels(regime_names, fontsize=10)
    ax.set_xlabel("Next Regime State $X_{n+1}$", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Current Regime State $X_n$", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title(fr"{title} — Regime Transition Matrix $\mathbf{{P}}$", fontsize=12, fontweight="bold", pad=12)
    fig.tight_layout()

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_markov_state_probabilities_vs_time(
    trajectory: np.ndarray,
    stationary_dist: np.ndarray,
    regime_names: Sequence[str],
    title: str,
    path: Union[str, Path],
) -> None:
    """Plot transient state probabilities pi_i(n) converging to the stationary distribution (ASM Course Page 4)."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    steps = trajectory.shape[0]
    x_steps = np.arange(steps)

    colors = ["#D62728", "#1F77B4", "#2CA02C"]
    for i, name in enumerate(regime_names):
        ax.plot(
            x_steps,
            trajectory[:, i],
            color=colors[i],
            lw=2.0,
            marker="o",
            markersize=4,
            label=fr"$\pi_{{{name}}}(n)$",
        )
        # Stationary asymptote
        ax.axhline(
            stationary_dist[i],
            color=colors[i],
            linestyle="--",
            lw=1.3,
            alpha=0.7,
            label=fr"Stationary $\pi_{{{name}}} = {stationary_dist[i]:.3f}$",
        )

    ax.set_title(fr"{title} — Markov Transient State Probabilities vs. Steps", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Transition Step $n$ (Successive Vehicles)", fontsize=11)
    ax.set_ylabel(r"State Probability $\mathbb{P}(X_n = i)$", fontsize=11)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="right", frameon=True, fontsize=9)
    fig.tight_layout()

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

