"""Offline animated walkthrough of saved pipeline results, without reading raw data."""
import base64
import io
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .distributions import ALPHA, CENTERS, DISTS, EDGES, T_MAX
from .simulation import sample_distribution


STAGES = [
    "What is a headway?", "Select and filter observations", "Build the histogram",
    "Define the probability model", "Sample the posterior with MCMC",
    "Summarize uncertainty", "Compare all seven distributions", "Measure fit quality",
    "Generate new headways", "Validate the generated sample", "Simulate vehicle arrivals",
    "Follow traffic regimes", "Check modelling assumptions", "Read the final outputs",
]
BLUE, ORANGE, GREEN = "#2563eb", "#e97928", "#139580"
COLORS = [BLUE, ORANGE, GREEN, "#9854bd", "#d34562", "#8c6a35", "#64748b"]


class Walkthrough:
    def __init__(self, results, dataset, model="Proposed"):
        self.directory = Path(results) / dataset
        self.dataset, self.model = dataset, model
        self.info = pd.read_csv(self.directory / "dataset_info.csv").iloc[0]
        self.counts = pd.read_csv(self.directory / "headway_histogram.csv")["count"].to_numpy(float)
        if (self.counts.shape != (49,) or not np.isfinite(self.counts).all()
                or (self.counts < 0).any() or self.counts.sum() != self.info.n_headways):
            raise ValueError("Histogram must contain 49 nonnegative bins matching dataset_info.csv")
        self.params = pd.read_csv(self.directory / "params.csv", index_col="distribution")
        self.metrics = pd.read_csv(self.directory / "metrics.csv", index_col="distribution")
        self.diagnostics = pd.read_csv(self.directory / "diagnostics.csv", index_col="distribution")
        self.posterior = pd.read_csv(self.directory / "posterior.csv")
        self.tests = pd.read_csv(self.directory / "tests.csv", index_col="distribution")
        self.sim_summary = pd.read_csv(self.directory / "simulation_summary.csv", index_col="distribution")
        self.renewal = pd.read_csv(self.directory / "stochastic_renewal_process.csv").iloc[0]
        self.markov = pd.read_csv(self.directory / "stochastic_markov_chain.csv").iloc[0]
        self.renewal_validation = pd.read_csv(self.directory / "validation_renewal.csv")
        self.independence = pd.read_csv(self.directory / "validation_independence.csv")
        self.markov_validation = pd.read_csv(self.directory / "validation_markov.csv").iloc[0]
        slug = model.lower().replace(" ", "_")
        with np.load(self.directory / "draws" / f"{slug}.npz", allow_pickle=False) as saved:
            self.full_draws = saved["full_draws"]
            self.draws = saved["draws"]
        if self.full_draws.ndim != 3 or self.draws.shape[2] != len(DISTS[model].params):
            raise ValueError("Invalid saved MCMC trace dimensions")
        self.warmup = self.full_draws.shape[1] - self.draws.shape[1]
        self.replay = {}
        replay_path = self.directory / "animation_replay.npz"
        if replay_path.exists():
            with np.load(replay_path, allow_pickle=False) as saved:
                self.replay = {key: saved[key] for key in saved.files}
        self.seed = int(self.info.get("seed", 42))
        self.sim = self.replay.get("simulated_headways")
        if self.sim is None:
            self.sim = sample_distribution(DISTS["Proposed"], self.theta("Proposed"),
                                          min(int(self.info.n_headways), 20000), self.seed)
        self.x = np.linspace(ALPHA, T_MAX, 600)
        self.probabilities = self.counts / self.counts.sum()
        self.P = np.array([[self.markov[f"p_{i}_{j}"] for j in ("close", "platoon", "free")]
                           for i in ("close", "platoon", "free")])
        self.pi = self.markov[["pi_close", "pi_platoon", "pi_free"]].to_numpy(float)
        self.fig = plt.figure(figsize=(12, 7), facecolor="#f8fafc")

    def theta(self, name):
        return self.params.loc[name, list(DISTS[name].params)].to_numpy(float)

    def empirical(self, ax):
        ax.bar(CENTERS, self.probabilities / 0.5, width=0.47, color="#cbd5e1", label="All filtered data")
        ax.set(xlim=(ALPHA, T_MAX), xlabel="Headway (seconds)", ylabel="Density (1 / second)")

    def frame(self, stage, progress):
        """Draw one deterministic frame; progress controls reveal, never the estimates."""
        self.fig.clear()
        fig = self.fig
        ax = fig.add_axes((0.08, 0.40, 0.57, 0.41))
        lower = fig.add_axes((0.08, 0.12, 0.57, 0.18))
        for plot in (ax, lower):
            plot.spines[["top", "right"]].set_visible(False)
            plot.tick_params(labelsize=9)
            plot.grid(alpha=0.12)
        source = "RAW-DATA RESULTS" if self.info.source == "raw" else "SYNTHETIC DEMO - NOT PAPER RESULTS"
        fig.text(0.05, 0.96, f"{self.dataset.upper()}  /  {source}", color=BLUE, fontsize=10, weight="bold")
        fig.text(0.05, 0.90, f"{stage + 1:02d}   {STAGES[stage]}", fontsize=21, weight="bold", color="#0f172a")
        fig.text(0.05, 0.035, "DATA  >  MODEL  >  INFERENCE  >  COMPARISON  >  SIMULATION  >  VALIDATION",
                 fontsize=9, color="#64748b")
        notes = []
        fraction = float(np.clip(progress, 0, 1))

        if stage == 0:
            # A detector illustration, not a reconstruction of vehicle positions.
            h = float(self.info.mean_s)
            clock = fraction * (h + 3)
            ax.axhspan(-0.28, 0.28, color="#e2e8f0")
            for arrival, label, color in [(1, "Leader", BLUE), (1 + h, "Follower", ORANGE)]:
                x = clock - arrival
                ax.scatter(x, 0, s=1000, marker="s", color=color)
                ax.text(x, 0.30, label, ha="center", fontsize=10)
            ax.axvline(0, color="#0f172a", ls="--")
            ax.set(xlim=(-h - 1, h + 2), ylim=(-0.5, 0.7), yticks=[], xlabel="Illustrative travel past a fixed detector")
            ax.set_title(f"Time gap between crossings: h = {h:.2f} s (dataset mean)", fontsize=11)
            hz = 25 if self.dataset == "highd" else 10
            frames = np.arange(2 * hz + 1)
            lower.scatter(frames / hz, np.zeros(len(frames)), s=20, color="#cbd5e1")
            kept = frames % hz == 0
            lower.scatter(frames[kept] / hz, np.zeros(kept.sum()), color=GREEN, s=60)
            lower.set(yticks=[], xlabel="Illustration: one follower's recording time (s)", title=f"{hz} Hz source -> 1 Hz extraction")
            notes = ["A time headway describes the temporal spacing between a follower and its leader.",
                     "The loaders read supplied thw / Time_Headway values. They do not calculate them from this road animation.",
                     "Vehicle positions and frame dots here are illustrative; no raw trajectories are replayed."]
        elif stage == 1:
            example = np.array([0.2, 0.5, 1.2, 3, 25, 29])
            keep = (example >= ALPHA) & (example <= T_MAX)
            ax.axvspan(ALPHA, T_MAX, alpha=0.12, color=GREEN)
            n = max(1, int(np.ceil(fraction * len(example))))
            ax.scatter(example[:n], np.zeros(n), s=110,
                       c=np.where(keep[:n], GREEN, ORANGE))
            for value in example[:n]:
                ax.annotate(f"{value:g}s", (value, 0), xytext=(0, 14), textcoords="offset points", ha="center")
            ax.set(xlim=(-1, 31), ylim=(-1, 1), yticks=[], xlabel="Illustrative filter examples (seconds)")
            lower.axis("off")
            selection = (f"NGSIM: {self.info.ngsim_location}; period {self.info.ngsim_period}; lanes {self.info.ngsim_lanes}"
                         if self.dataset == "ngsim" else "highD: recordings supplied to the loader")
            lower.text(0, 0.85, "1 Hz frames -> leader present -> finite 0.5 <= h <= 25", fontsize=11)
            lower.text(0, 0.48, selection, fontsize=10)
            lower.text(0, 0.1, f"Saved retained sample: {int(self.info.n_headways):,} headways", fontsize=12, weight="bold")
            notes = ["Green examples survive the headway window; orange examples are excluded, not clipped to the endpoints.",
                     "Raw rejection counts were not recorded. This is a rule illustration, not an audit of removed rows.",
                     "For synthetic runs, the generator starts inside the window; raw frame/leader filtering is not executed."]
        elif stage == 2:
            n = max(1, int(np.ceil(fraction * 49)))
            ax.bar(CENTERS[:n], self.counts[:n], width=0.47, color=BLUE)
            ax.set(xlim=(ALPHA, T_MAX), ylim=(0, max(self.counts) * 1.1), xlabel="Headway (seconds)", ylabel="Observed count")
            preview = self.replay.get("headway_preview")
            if preview is not None:
                lower.plot(np.arange(1, len(preview) + 1), preview, lw=0.8, color=GREEN)
                lower.set(xlabel="First saved filtered rows (loader order)", ylabel="Headway (s)")
            else:
                lower.axis("off")
                lower.text(0, 0.5, "Older run: no row preview saved. Histogram above uses all observations.")
            notes = ["49 bins, each 0.5 seconds wide. The animation reveals saved bin counts from left to right; this is not ingestion order.",
                     f"N = {int(self.info.n_headways):,}\nMean = {self.info.mean_s:.3f} s\nMedian = {self.info.median_s:.3f} s",
                     "Histogram probabilities are used for evaluation. Fitting uses the individual observations, not these bins."]
        elif stage == 3:
            self.empirical(ax)
            n = max(2, int(fraction * len(self.x)))
            ax.plot(self.x[:n], DISTS[self.model].pdf(self.x[:n], *self.theta(self.model)), color=BLUE, lw=2)
            ax.set_title(f"{self.model}: natural-support PDF at the final estimate", fontsize=11)
            lower.axis("off")
            lower.text(0, 0.75, "log posterior = log prior + sum log f(h_i | parameters)", fontsize=11)
            lower.text(0, 0.33, "Sample log / logit coordinates + change-of-variables Jacobian", fontsize=10)
            notes = ["Table 1 defines the model PDFs and parameter priors. This curve uses the final estimate to illustrate that model.",
                     "All filtered observations contribute. Repeated values are counted exactly; the plot preview is not the fitting sample.",
                     "The likelihood uses natural-support PDFs. Evaluation and generated samples are conditioned on [0.5, 25] seconds."]
        elif stage == 4:
            total = self.full_draws.shape[1]
            end = max(2, int(fraction * total))
            indexes = np.unique(np.r_[np.arange(0, end, max(1, total // 1200)), end - 1])
            names = DISTS[self.model].params
            for chain, values in enumerate(self.full_draws):
                ax.plot(indexes, values[indexes, 0], lw=0.7, color=COLORS[chain % 7], label=f"Chain {chain + 1}")
            ax.axvspan(0, self.warmup, color=ORANGE, alpha=0.15)
            ax.set(xlim=(0, total), xlabel="MCMC iteration", ylabel=names[0])
            ax.legend(fontsize=8, loc="upper right")
            remaining = self.full_draws[:, :, 1:]
            center, scale = remaining.mean(axis=(0, 1)), remaining.std(axis=(0, 1))
            for j, name in enumerate(names[1:]):
                for chain in range(len(remaining)):
                    lower.plot(indexes, (remaining[chain, indexes, j] - center[j]) / max(scale[j], 1e-15),
                               color=COLORS[j], ls="-" if chain == 0 else "--", lw=0.7,
                               label=name if chain == 0 else None)
            lower.axvspan(0, self.warmup, color=ORANGE, alpha=0.15)
            lower.set(xlim=(0, total), xlabel="MCMC iteration", ylabel="Other parameters\n(standardized)")
            lower.legend(fontsize=8, loc="upper right")
            d = self.diagnostics.loc[self.model]
            notes = [f"Actual saved chains: {self.model}. The view reveals {end:,} of {total:,} iterations.",
                     f"Orange: {self.warmup:,} warmup iterations. Proposal adaptation stops after warmup; retained draws determine the estimate.",
                     f"Final max R-hat: {d.max_rhat:.4f}\nAcceptance: {d.acceptance_rate:.1%}\nRepeated trace values can mean rejected proposals. Rejected candidates were not saved."]
        elif stage == 5:
            n = max(4, int(fraction * self.draws.shape[1]))
            ax.hist(self.draws[:, :n, 0].ravel(), bins=35, density=True, color=BLUE, alpha=0.7)
            ax.set(xlabel=DISTS[self.model].params[0], ylabel="Posterior density", title=f"{self.model}: retained draws only")
            lower.axis("off")
            rows = self.posterior[self.posterior.distribution == self.model]
            lines = ["Parameter      Mean         95% credible interval       ESS / R-hat"]
            for row in rows.itertuples():
                lines.append(f"{row.param:<10} {row.mean:>9.4f}   [{row.q025:.4f}, {row.q975:.4f}]   {row.ess:.0f} / {row.rhat:.3f}")
            lower.text(0, 0.95, "\n".join(lines), va="top", family="monospace", fontsize=9)
            notes = ["The histogram grows using retained draws from every chain. The table shows final saved summaries.",
                     "Posterior mean = the point estimate. The 2.5% and 97.5% quantiles describe posterior uncertainty.",
                     "R-hat checks agreement between chains; ESS estimates how much independent information the correlated draws contain. Low R-hat alone does not validate the model."]
        elif stage == 6:
            self.empirical(ax)
            count = max(1, int(np.ceil(fraction * len(DISTS))))
            for color, name in zip(COLORS[:count], list(DISTS)[:count]):
                ax.plot(self.x, DISTS[name].trunc_pdf(self.x, self.theta(name)), lw=1.4, color=color, label=name)
            ax.legend(fontsize=7)
            lower.step(np.r_[ALPHA, EDGES[1:]], np.r_[0, np.cumsum(self.probabilities)],
                       where="post", color="#64748b", label="Observed cumulative bin masses")
            lower.plot(self.x, DISTS[self.model].trunc_cdf(self.x, self.theta(self.model)), color=BLUE, label=self.model)
            lower.set(xlim=(ALPHA, T_MAX), xlabel="Headway (seconds)", ylabel="Cumulative mass")
            lower.legend(fontsize=7)
            notes = ["The same data are fitted independently with the proposed distribution and six baselines.",
                     "These evaluation curves are normalized to the retained window. Grey bars contain all filtered data.",
                     "The observed cumulative curve is binned for display. The pipeline's KS statistic uses individual observations."]
        elif stage == 7:
            labels = ["Proposed", "Shift log-N", "Weibull", "Log-logistic", "Gamma", "Burr", "Shift exp"]
            metrics = self.metrics.loc[list(DISTS)]
            count = max(1, int(np.ceil(fraction * 7)))
            ax.barh(np.arange(count), metrics.kl.iloc[:count], color=COLORS[:count])
            ax.set(yticks=np.arange(7), yticklabels=labels, xlabel="KL divergence (natural log); lower is better")
            ax.invert_yaxis()
            lower.bar(np.arange(count), metrics.wasserstein_seconds.iloc[:count], color=COLORS[:count])
            lower.set(xticks=np.arange(7), xticklabels=labels, ylabel="W1 (seconds)")
            lower.tick_params(axis="x", labelsize=8)
            tests = self.tests.loc[self.model]
            notes = [f"{self.model}\nKS D = {tests.ks_D:.4f}\nNominal KS p = {tests.ks_p:.3g}\nChi-square p = {tests.chi2_p:.3g}",
                     "KL compares bin probabilities. W1 moves probability mass across headway times, measured in seconds.",
                     f"Legacy W_bin = {metrics.loc[self.model, 'wasserstein']:.5f}. This sorted-probability proxy is separate from W1; the paper's exact inputs remain unspecified."]
        elif stage == 8:
            cdf = DISTS["Proposed"].trunc_cdf(self.x, self.theta("Proposed"))
            ax.plot(self.x, cdf, lw=2, color=BLUE)
            count = min(16, len(self.sim))
            i = min(int(fraction * count), count - 1)
            h = float(self.sim[i])
            u = float(DISTS["Proposed"].trunc_cdf(np.array([h]), self.theta("Proposed"))[0])
            ax.plot([ALPHA, h, h], [u, u, 0], ls="--", color=ORANGE)
            ax.scatter([h], [u], s=70, color=ORANGE)
            ax.set(xlim=(ALPHA, T_MAX), ylim=(0, 1), xlabel="Generated headway h = F inverse(u), seconds", ylabel="Conditional CDF F(h)")
            lower.bar(np.arange(i + 1), self.sim[:i + 1], color=GREEN)
            lower.set(xlabel="Generated sample index", ylabel="Headway (s)")
            notes = ["Draw u uniformly in [0, 1]. Move horizontally to the fitted CDF, then down to obtain a headway.",
                     f"Example: u approximately {u:.4f} produces h = {h:.3f} s. The project implements the inverse using a fine lookup grid.",
                     "Generation always uses the fitted Proposed model. " + ("Samples are replayed from this run." if self.replay else "Older run: samples are regenerated with its saved parameters and seed.")]
        elif stage == 9:
            self.empirical(ax)
            n = max(2, int(fraction * len(self.sim)))
            probs = np.histogram(self.sim[:n], EDGES)[0] / n
            ax.step(CENTERS, probs / 0.5, where="mid", color=GREEN, label=f"Generated: {n:,} samples")
            ax.plot(self.x, DISTS["Proposed"].trunc_pdf(self.x, self.theta("Proposed")), color=BLUE, label="Fitted Proposed")
            ax.legend(fontsize=8)
            lower.axis("off")
            validation = self.sim_summary.loc["Validation"]
            comparison = self.sim_summary.loc["Comparison"]
            lower.text(0, 0.8, f"Saved full-sample checks: theory KS p = {validation.theory_ks_p:.4g}", fontsize=11)
            lower.text(0, 0.35, f"Empirical versus simulated: KS D = {comparison.ks_statistic:.4f}", fontsize=11)
            notes = ["The growing green histogram shows generated headways. Grey bars show the complete input histogram.",
                     "Theory versus simulation checks the random generator. Empirical versus simulation checks how well the fitted model describes your data.",
                     "A high theory-validation p-value does not establish a good fit to real data. These are different comparisons."]
        elif stage == 10:
            time = self.replay.get("renewal_time")
            if time is not None:
                end = max(2, int(fraction * len(time)))
                for i, counts in enumerate(self.replay["renewal_counts"]):
                    ax.step(time[:end], counts[:end], where="post", lw=1, color=COLORS[i % 7], alpha=0.75)
                ax.plot(time[:end], self.replay["renewal_expected"][:end], color="#0f172a", ls="--", label="Asymptotic renewal mean")
                ax.set(xlim=(0, time[-1]), ylim=(0, self.replay["renewal_counts"].max() * 1.1 + 1))
                ax.legend(fontsize=8)
            else:
                ax.text(0.05, 0.5, "No arrival paths saved in this older run.\nRerun run.py to record them.", transform=ax.transAxes)
            ax.set(xlabel="Simulated time (seconds)", ylabel="Cumulative arrivals N(t)")
            check = self.renewal_validation
            lower.plot(check.t_s, check.mean_sim, "o-", color=GREEN, label="Simulated mean")
            lower.plot(check.t_s, check.mean_theory, "--", color=BLUE, label="Renewal approximation")
            lower.set(xlabel="Validation horizon (seconds)", ylabel="Mean N(t)")
            lower.legend(fontsize=8)
            notes = ["ASM extension: independently resample empirical headways. Add them to obtain arrival times; count arrivals up to each time t.",
                     f"Rate = 1 / mean(h)\n= {self.renewal.arrival_rate_veh_s:.4f} vehicles/s\n= {self.renewal.capacity_flow_veh_h:.0f} vehicles/hour (model).",
                     "These are simulated detector arrivals, not observed traffic trajectories. This branch resamples empirical data, not the Proposed draws in the preceding stage."]
        elif stage == 11:
            ax.imshow(self.P, cmap="Blues", vmin=0, vmax=1, aspect="auto")
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, f"{self.P[i, j]:.3f}", ha="center", va="center", color="white" if self.P[i, j] > 0.5 else "black")
            ax.set(xticks=range(3), yticks=range(3), xticklabels=["Close", "Platoon", "Free"],
                   yticklabels=["Close", "Platoon", "Free"], xlabel="Next regime", ylabel="Current regime")
            ax.grid(False)
            paths = self.replay.get("markov_paths")
            if paths is not None:
                path = paths[0]
                step = max(1, min(int(fraction * (len(path) - 1)), len(path) - 1))
                ax.scatter(path[step], path[step - 1], s=1800, facecolors="none", edgecolors=ORANGE, linewidths=3)
                ax.set_title(f"Saved simulated regime path: transition {step} of {len(path) - 1}", fontsize=10)
            transient = self.replay.get("markov_transient")
            if transient is None:
                transient = np.array([np.array([1., 0., 0.]) @ np.linalg.matrix_power(self.P, n) for n in range(21)])
            end = max(2, int(fraction * len(transient)))
            for i, label in enumerate(["Close", "Platoon", "Free"]):
                lower.plot(np.arange(end), transient[:end, i], color=COLORS[i], label=label)
                lower.axhline(self.pi[i], color=COLORS[i], ls="--", alpha=0.5)
            lower.set(xlim=(0, len(transient) - 1), ylim=(0, 1), xlabel="Markov step (not seconds)", ylabel="Probability")
            lower.legend(fontsize=7)
            notes = ["ASM extension: classify each headway.\nClose: h < 1.5 s\nPlatoon: 1.5 <= h < 4 s\nFree: h >= 4 s.",
                     "Estimate P from consecutive rows. " + ("Orange ring: one saved simulated transition. " if paths is not None else "No sampled state path saved in this older run. ") + "Below: start in Close, then update p(next) = p(now) P.",
                     "Dashed lines: stationary probabilities. Flattened file order crosses vehicle boundaries; this is not a verified physical traffic-state sequence."]
        elif stage == 12:
            check = self.independence
            ax.bar(check.lag, check.autocorr, color=BLUE)
            ax.axhline(check.band95.iloc[0], color=ORANGE, ls="--")
            ax.axhline(-check.band95.iloc[0], color=ORANGE, ls="--")
            ax.set(xlabel="Lag in filtered row order", ylabel="Headway autocorrelation", title="Independence check: nominal white-noise band")
            r = self.renewal_validation
            n = max(1, int(np.ceil(fraction * len(r))))
            lower.plot(r.t_s[:n], r.var_sim[:n], "o-", color=GREEN, label="Simulated variance")
            lower.plot(r.t_s, r.var_theory, "--", color=BLUE, label="Renewal theory")
            lower.plot(r.t_s, r.var_poisson, ":", color=ORANGE, label="Poisson alternative")
            lower.set(xlabel="Validation horizon (seconds)", ylabel="Var N(t)")
            lower.legend(fontsize=7)
            m = self.markov_validation
            notes = ["Independently resampling headways assumes independence. Strong autocorrelation in the input challenges that assumption.",
                     f"Markov two-step error = {m.chapman_kolmogorov_2_max_err:.4f}\nSecond-order test p = {m.second_order_p:.3g}\nSecond-order effect TV = {m.second_order_effect_tv:.4f}",
                     "Compare simulated moments with theory and check model assumptions. A simulation can run correctly even when its assumptions fit the real data poorly."]
        else:
            ax.axis("off")
            lower.axis("off")
            lines = ["DATA       dataset_info.csv / headway_histogram.csv",
                     "INFERENCE  draws/ / params.csv / posterior.csv",
                     "QUALITY    diagnostics.csv / tests.csv / metrics.csv",
                     "PAPER      paper_comparison.csv",
                     "SIMULATE   simulation_summary.csv / stochastic_*.csv",
                     "VALIDATE   validation_*.csv",
                     "REPLAY     animation_replay.npz / this HTML"]
            n = max(1, int(np.ceil(fraction * len(lines))))
            ax.text(0, 0.95, "\n\n".join(lines[:n]), va="top", family="monospace", fontsize=10)
            failed = self.diagnostics.index[self.diagnostics.max_rhat.ge(1.05) | ~np.isfinite(self.diagnostics.max_rhat)].tolist()
            lower.text(0, 0.8, "Convergence review: " + (", ".join(failed) if failed else "all saved max R-hat values < 1.05"), fontsize=10, wrap=True)
            best = self.metrics.kl.idxmin()
            lower.text(0, 0.35, f"Smallest saved KL: {best} ({self.metrics.loc[best, 'kl']:.5f})", fontsize=11)
            notes = ["Each animation stage reads saved results or clearly identifies an illustrative calculation. It does not refit parameters or change the paper's comparison targets.",
                     "The detailed trace/posterior view uses " + self.model + ". Use --model to inspect any of the seven distributions.",
                     "Review diagnostics before ranking fits. Synthetic demo rankings are not evidence of performance on highD or NGSIM."]

        wrapped = "\n\n".join("\n".join(textwrap.fill(line, 37) for line in note.splitlines()) for note in notes)
        self.description = "\n\n".join(notes)
        fig.text(0.70, 0.79, wrapped, va="top", fontsize=10.5, linespacing=1.45, color="#334155")
        return fig


def export_html(walkthrough, destination, frames_per_stage=16):
    """Render with Matplotlib and embed a small, dependency-free offline player."""
    if not 2 <= frames_per_stage <= 60:
        raise ValueError("frames_per_stage must be between 2 and 60")
    frames, descriptions = [], []
    with matplotlib.rc_context({"font.family": "DejaVu Sans", "font.size": 10}):
        for stage, title in enumerate(STAGES):
            print(f"Rendering {stage + 1}/{len(STAGES)}: {title}", flush=True)
            for progress in np.linspace(0, 1, frames_per_stage):
                fig = walkthrough.frame(stage, progress)
                buffer = io.BytesIO()
                fig.savefig(buffer, format="png", dpi=90, facecolor=fig.get_facecolor())
                frames.append(base64.b64encode(buffer.getvalue()).decode("ascii"))
                descriptions.append(walkthrough.description)
    template = Path(__file__).with_name("animation_player.html").read_text(encoding="utf-8")
    payload = json.dumps({"frames": frames, "descriptions": descriptions, "stages": STAGES, "framesPerStage": frames_per_stage,
                          "dataset": walkthrough.dataset, "source": str(walkthrough.info.source)})
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(template.replace("__ANIMATION_DATA__", payload.replace("<", "\\u003c")), encoding="utf-8")
    plt.close(walkthrough.fig)
    return destination
