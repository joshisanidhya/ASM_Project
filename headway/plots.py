"""Frequency histogram with fitted curves (paper Figure 5)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .distributions import CENTERS, DISTS


def plot_posterior(title, param_names, draws, path):
    """Posterior histogram + trace per parameter, chains overlaid (paper Figure 4).

    draws: (chains, n, params), post-warmup.
    """
    fig, axes = plt.subplots(len(param_names), 2, figsize=(10, 2.6 * len(param_names)), squeeze=False)
    for j, p in enumerate(param_names):
        for c in range(draws.shape[0]):
            axes[j, 0].hist(draws[c, :, j], bins=40, alpha=0.5, density=True)
            axes[j, 1].plot(draws[c, :, j], lw=0.4)
        axes[j, 0].set(title=f"{p} posterior")
        axes[j, 1].set(title=f"{p} trace", xlabel="iteration (post-warmup)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_fits(name, counts, params, path):
    n = counts.sum()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(CENTERS, counts, width=0.5, color="0.75", label="Observation")
    for dname, theta in params.items():
        ax.plot(CENTERS, n * DISTS[dname].bin_probs(theta), label=dname, lw=1.5)
    ax.set(title=name, xlabel="Headway [sec]", ylabel="Frequency")
    ax.legend()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
