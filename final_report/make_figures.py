"""Two-panel (highD | NGSIM) versions of the paper's figures for the condensed report.

Nothing is re-estimated: parameters, transition matrices and posterior draws are read from
outputs/truncated; the observed histogram, the renewal sample paths and the simulated count distribution
are re-created from the (seeded, hence identical) headway samples, and the simulated moments are checked
against the saved validation_renewal.csv before any figure is written.

Run from the project root:  python final_report/make_figures.py --workers 6
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm, poisson

from headway.datasets import LOADERS
from headway.distributions import ALPHA, CENTERS, DISTS, EDGES
from headway.plots import PALETTE
from headway.stochastic_process import TrafficRegimeMarkovChain, analyze_stochastic_processes
from headway.validation import renewal_validation

plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.grid": True, "grid.linestyle": ":",
                     "grid.alpha": 0.5, "figure.dpi": 100})
OUT = ROOT / "final_report" / "figures"
RES = ROOT / "outputs" / "truncated"
NAMES = {"highd": "highD", "ngsim": "NGSIM"}
REG = ["Close", "Platoon", "Free"]
REGCOL = ["#D62728", "#1F77B4", "#2CA02C"]
W = 7.0  # figure width in inches (scaled to the text width in the paper)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def diagram(ax, P, title):
    pos = np.array([[-1.0, -0.55], [1.0, -0.55], [0.0, 0.95]])
    ax.set_xlim(-1.75, 1.75); ax.set_ylim(-1.2, 1.65); ax.set_aspect("equal"); ax.axis("off")
    for i, nm in enumerate(REG):
        ax.add_patch(plt.Circle(pos[i], 0.40, color=REGCOL[i], alpha=0.25, ec=REGCOL[i], lw=1.8))
        ax.text(pos[i, 0], pos[i, 1] + 0.08, nm, ha="center", va="center", fontsize=7.5, fontweight="bold")
        ax.text(pos[i, 0], pos[i, 1] - 0.13, f"stay {P[i, i]:.3f}", ha="center", va="center", fontsize=7)
    rad = 0.28
    for i in range(3):
        for j in range(3):
            if i == j or P[i, j] < 0.005:
                continue
            ax.annotate("", xy=pos[j], xytext=pos[i], arrowprops=dict(
                arrowstyle="-|>", lw=0.7 + 6 * P[i, j], color="0.35", shrinkA=30, shrinkB=30,
                connectionstyle=f"arc3,rad={rad}"))
            d = pos[j] - pos[i]
            bend = np.array([d[1], -d[0]])
            lab = (pos[i] + pos[j]) / 2 + 0.5 * rad * bend + 0.11 * bend / np.linalg.norm(bend)
            ax.text(lab[0], lab[1], f"{P[i, j]:.3f}", ha="center", va="center", fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
    ax.set_title(title, fontsize=9, fontweight="bold")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    args = ap.parse_args()

    D = {}
    for k in ("highd", "ngsim"):
        if k == "highd":
            data, groups = LOADERS[k](Path(args.data_dir) / "highd", workers=args.workers, return_groups=True)
        else:
            f = sorted((Path(args.data_dir) / "ngsim").glob("*.csv"), key=lambda p: -p.stat().st_size)[0]
            data, groups = LOADERS[k](f, workers=args.workers, return_groups=True, location="i-80", lanes=None, period=1)
        counts = np.histogram(data, bins=EDGES)[0].astype(float)
        pr = pd.read_csv(RES / k / "params.csv").set_index("distribution")
        theta = {d: tuple(float(pr.loc[d, p]) for p in DISTS[d].params) for d in DISTS}
        mc_res, rp = analyze_stochastic_processes(data, n_sample_paths=5, t_simulation=300.0, seed=42, groups=groups)
        times = (30, 60, 120, 300)
        rdf, simc = renewal_validation(data, times=times, seed=42)
        saved = pd.read_csv(RES / k / "validation_renewal.csv")
        err = float(np.abs(rdf[["mean_sim", "var_sim"]].values - saved[["mean_sim", "var_sim"]].values).max())
        print(f"[{k}] n={len(data):,} max |sim - saved| over mean_sim/var_sim = {err:.2e}")
        assert err < 1e-8, "re-created simulation does not match the saved outputs"
        assert len(data) == int(pd.read_csv(RES / k / "dataset_info.csv").n_headways[0])
        mk = pd.read_csv(RES / k / "stochastic_markov_chain.csv").iloc[0]
        P = np.array([[mk.p_close_close, mk.p_close_platoon, mk.p_close_free],
                      [mk.p_platoon_close, mk.p_platoon_platoon, mk.p_platoon_free],
                      [mk.p_free_close, mk.p_free_platoon, mk.p_free_free]])
        pi = np.array([mk.pi_close, mk.pi_platoon, mk.pi_free])
        traj = TrafficRegimeMarkovChain(P).transient_distribution([1.0, 0.0, 0.0], steps=60)
        post = np.load(RES / k / "draws" / "proposed.npz")["draws"].reshape(-1, 2)
        D[k] = dict(counts=counts, theta=theta, rp=rp, rdf=rdf, simc=simc, P=P, pi=pi, traj=traj, post=post, times=times)

    # ---- Fig: fitted distributions -------------------------------------------------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.9))
    for ax, k in zip(axs, D):
        c, n = D[k]["counts"], D[k]["counts"].sum()
        ax.bar(CENTERS, c / 1e3, width=0.48, color="#D3D3D3", edgecolor="#808080", alpha=0.85, label="Observed")
        for dn, th in D[k]["theta"].items():
            ax.plot(CENTERS, n * DISTS[dn].bin_probs(th) / 1e3, color=PALETTE[dn], lw=2.0 if dn == "Proposed" else 1.1,
                    ls="-" if dn == "Proposed" else "--", label=dn)
        ax.set_xlim(0, 15); ax.set_xlabel("Headway (s)"); ax.set_ylabel("Frequency (thousands per 0.5 s bin)")
        ax.set_title(NAMES[k], fontweight="bold")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    save(fig, "fig_fits.png")

    # ---- Fig: renewal sample paths -------------------------------------------------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.5))
    for ax, k in zip(axs, D):
        rp = D[k]["rp"]
        for i in range(rp.sample_paths_counts.shape[0]):
            ax.step(rp.sample_paths_time, rp.sample_paths_counts[i], where="post", lw=0.9, alpha=0.7,
                    label="Simulated paths (5)" if i == 0 else None)
        ax.plot(rp.sample_paths_time, rp.expected_counts, "k--", lw=2.0, label=r"Renewal function $m(t)$")
        ax.set_xlabel(r"Elapsed time $t$ (s)"); ax.set_ylabel(r"Arrivals $N(t)$"); ax.set_title(NAMES[k], fontweight="bold")
        ax.legend(loc="upper left")
    fig.tight_layout()
    save(fig, "fig_paths.png")

    # ---- Fig: count distribution at 60 s --------------------------------------------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.5))
    for ax, k in zip(axs, D):
        rdf, simc, times = D[k]["rdf"], D[k]["simc"], D[k]["times"]
        j = times.index(60)
        cnt = simc[:, j]; r60 = rdf.iloc[j]; rate = D[k]["rp"].asymptotic_rate
        kk = np.arange(cnt.min() - 3, cnt.max() + 4); sd = np.sqrt(r60.var_theory)
        ax.bar(kk, np.bincount(cnt - kk[0], minlength=len(kk))[:len(kk)] / len(cnt), width=0.85, color="#BBBBBB",
               edgecolor="#777777", label=f"Simulated ({len(cnt):,} paths)")
        ax.plot(kk, norm.cdf((kk + .5 - r60.mean_theory) / sd) - norm.cdf((kk - .5 - r60.mean_theory) / sd),
                color="#D62728", lw=1.8, marker="o", ms=3, label="Renewal theory (normal)")
        ax.plot(kk, poisson.pmf(kk, rate * 60), color="#1F77B4", lw=1.5, ls="--", marker="s", ms=2.5,
                label=f"Poisson, same rate")
        ax.set_xlabel(r"Vehicles arriving in 60 s, $N(60)$"); ax.set_ylabel("Probability"); ax.set_title(NAMES[k], fontweight="bold")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.5)
        ax.legend(loc="upper left")
    fig.tight_layout()
    save(fig, "fig_counts.png")

    # ---- Fig: state-transition diagrams ---------------------------------------------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.8))
    for ax, k in zip(axs, D):
        diagram(ax, D[k]["P"], NAMES[k])
    fig.tight_layout()
    save(fig, "fig_diagram.png")

    # ---- Fig: transient state probabilities -----------------------------------------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.6), sharey=True)
    for ax, k in zip(axs, D):
        for i, nm in enumerate(["Close following", "Stable platoon", "Free flow"]):
            ax.plot(np.arange(D[k]["traj"].shape[0]), D[k]["traj"][:, i], color=REGCOL[i], lw=1.8, label=nm)
            ax.axhline(D[k]["pi"][i], color=REGCOL[i], ls="--", lw=1.1, alpha=0.8)
        ax.set_ylim(-0.02, 1.02); ax.set_xlabel(r"Step $n$ (s along one trajectory)"); ax.set_title(NAMES[k], fontweight="bold")
    axs[0].set_ylabel(r"$P(X_n=i)$ from close following")
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=REGCOL[i], lw=1.8) for i in range(3)] + [Line2D([], [], color="0.4", ls="--", lw=1.1)]
    fig.legend(hs, ["Close following", "Stable platoon", "Free flow", "Stationary $\pi_i$ (dashed)"], loc="lower center",
               ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    save(fig, "fig_stateprob.png")

    # ---- Fig: posteriors of the proposed distribution -------------------------------------------------------
    fig, axs = plt.subplots(1, 4, figsize=(W, 2.0))
    for c, (k, j, pn) in enumerate([("highd", 0, "a"), ("highd", 1, "b"), ("ngsim", 0, "a"), ("ngsim", 1, "b")]):
        ax = axs[c]; col = D[k]["post"][:, j]
        ax.hist(col, bins=40, density=True, color="#4C72B0", edgecolor="white", alpha=0.8)
        ax.axvline(col.mean(), color="#D62728", lw=1.4)
        lo, hi = np.percentile(col, [2.5, 97.5]); ax.axvline(lo, color="0.35", ls="--", lw=0.9); ax.axvline(hi, color="0.35", ls="--", lw=0.9)
        ax.set_xlabel(f"${pn}$" + (" (s)" if pn == "a" else "")); ax.set_title(f"{NAMES[k]}: ${pn}$", fontsize=8.5, fontweight="bold")
        ax.xaxis.set_major_locator(plt.MaxNLocator(3)); ax.tick_params(axis="x", labelsize=7)
        if c == 0:
            ax.set_ylabel("Posterior density")
    fig.tight_layout()
    save(fig, "fig_posterior.png")


if __name__ == "__main__":
    main()
