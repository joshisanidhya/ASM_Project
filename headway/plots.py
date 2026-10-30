"""Frequency histogram with fitted curves (paper Figure 5)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .distributions import CENTERS, DISTS


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
