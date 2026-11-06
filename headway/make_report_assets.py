"""Turn the pipeline outputs into the numbers, tables and figures used by report/main.tex.

    python -m headway.make_report_assets [--truncated outputs/truncated] [--paper outputs/paper]
                                         [--detector outputs/detector] [--out report]

Every scalar in the paper text is a LaTeX macro written to report/results.tex, and every table body
is written to report/tables/*.tex, both straight from the CSV files that run.py and
headway.detector produced, so no number is typed by hand. Figures are copied from the same
output folders into report/figures/.
"""
import argparse
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .paper_reference import PAPER_TABLE3_PARAMS, PAPER_TABLE5_KL, PAPER_TABLE5_W

ORDER = ["Proposed", "Shifted Log-normal", "Weibull", "Log-logistic", "Gamma", "Burr", "Shifted Exponential"]
DS = {"highd": "hd", "ngsim": "ng"}
DS_NAME = {"highd": "highD", "ngsim": "NGSIM"}
PARAM_TEX = {"a": "a", "b": "b", "mu": r"\mu", "sigma": r"\sigma", "gamma": r"\gamma", "alpha": r"\alpha",
             "beta": r"\beta", "lambda": r"\lambda"}
PARAMS = {"Proposed": ["a", "b"], "Shifted Log-normal": ["mu", "sigma", "gamma"], "Weibull": ["alpha", "beta"],
          "Log-logistic": ["alpha", "beta"], "Gamma": ["alpha", "beta"], "Burr": ["alpha", "beta", "lambda"],
          "Shifted Exponential": ["lambda", "gamma"]}
REGIMES = ["Close following", "Stable platoon", "Free flow"]


def f(x, d=3):
    s = f"{x:.{d}f}"
    if float(s) == 0.0:
        s = s.lstrip("-")
    return s.replace("-", "\\textminus{}")


def thousands(n):
    return f"{int(round(n)):,}".replace(",", "{,}")


def pval(p):
    if not np.isfinite(p):
        return "n/a"
    if p >= 1e-3:
        return f"{p:.3f}"
    if p < 1e-300:
        return r"$<\!10^{-300}$"
    m, e = f"{p:.1e}".split("e")
    return rf"${m}\!\times\!10^{{{int(e)}}}$"


def bold(s, on):
    return rf"\textbf{{{s}}}" if on else s


class Macros:
    def __init__(self):
        self.lines = []

    def add(self, name, value, comment=""):
        assert name.isalpha(), name
        self.lines.append(rf"\newcommand{{\{name}}}{{{value}}}" + (f"  % {comment}" if comment else ""))

    def write(self, path, header):
        Path(path).write_text(header + "\n" + "\n".join(self.lines) + "\n", encoding="utf-8", newline="\n")


def load(root: Path, d: str):
    r = root / d
    rd = lambda n: pd.read_csv(r / n)
    return {
        "info": rd("dataset_info.csv").iloc[0], "params": rd("params.csv").set_index("distribution"),
        "metrics": rd("metrics.csv").set_index("distribution"), "tests": rd("tests.csv").set_index("distribution"),
        "diag": rd("diagnostics.csv").set_index("distribution"), "post": rd("posterior.csv"),
        "markov": rd("stochastic_markov_chain.csv").iloc[0], "renewal": rd("stochastic_renewal_process.csv").iloc[0],
        "rv": rd("validation_renewal.csv"), "ind": rd("validation_independence.csv"),
        "mv": rd("validation_markov.csv").iloc[0], "sim": rd("simulation_summary.csv"),
    }


def build(trunc_dir, paper_dir, det_dir, out):
    trunc_dir, paper_dir, det_dir, out = map(Path, (trunc_dir, paper_dir, det_dir, out))
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    T = {d: load(trunc_dir, d) for d in DS}
    P = {d: load(paper_dir, d) for d in DS}
    M = Macros()
    def w(name, text):
        # the last row is left unterminated: main.tex writes `\input{tables/x}\\` (a file that ends in a row
        # terminator right before \bottomrule triggers "Misplaced \noalign")
        body = re.sub(r"\\\\(\[[^\]]*\])?\s*$", "", text.rstrip())
        (out / "tables" / f"{name}.tex").write_text(body + "\n", encoding="utf-8", newline="\n")

    # ---------------------------------------------------------------- scalars ------------------------------
    for d, k in DS.items():
        t, info = T[d], T[d]["info"]
        assert info.source == "raw" and info.likelihood == "truncated" and P[d]["info"].likelihood == "paper"
        assert int(info.n_headways) == int(P[d]["info"].n_headways)
        M.add(k + "N", thousands(info.n_headways)); M.add(k + "Runs", thousands(info.n_runs))
        M.add(k + "Trans", thousands(info.n_transitions))
        M.add(k + "Mean", f(info.mean_s, 2)); M.add(k + "Median", f(info.median_s, 2)); M.add(k + "Std", f(info.std_s, 2))
        M.add(k + "Min", f(info.min_s, 2)); M.add(k + "Max", f(info.max_s, 2))
        pa = t["params"].loc["Proposed"]
        M.add(k + "A", f(pa.a)); M.add(k + "B", f(pa.b)); M.add(k + "Kappa", f(-np.log(pa.b)))
        po = t["post"][t["post"].distribution == "Proposed"].set_index("param")
        M.add(k + "ALo", f(po.loc["a", "q025"])); M.add(k + "AHi", f(po.loc["a", "q975"]))
        M.add(k + "BLo", f(po.loc["b", "q025"])); M.add(k + "BHi", f(po.loc["b", "q975"]))
        kl, klp = t["metrics"]["kl"], P[d]["metrics"]["kl"]
        M.add(k + "KLProp", f(kl["Proposed"], 4)); M.add(k + "KLPropPaper", f(klp["Proposed"], 4))
        for nm, key in [("SLN", "Shifted Log-normal"), ("Burr", "Burr"), ("LLog", "Log-logistic")]:
            M.add(k + "KL" + nm, f(kl[key], 4))
        M.add(k + "KLBest", kl.idxmin()); M.add(k + "KLBestPaper", klp.idxmin())
        M.add(k + "KLRankProp", str(int(kl.rank()["Proposed"])))
        M.add(k + "KLRankPropPaper", str(int(klp.rank()["Proposed"])))
        pub = pd.Series(PAPER_TABLE5_KL[d]); pubw = pd.Series(PAPER_TABLE5_W[d])
        M.add(k + "KLMaxDiffPaper", f((klp - pub).abs().max(), 4)); M.add(k + "KLMaxDiffPaperName", (klp - pub).abs().idxmax())
        M.add(k + "KLMaxDiffTrunc", f((kl - pub).abs().max(), 4)); M.add(k + "KLMaxDiffTruncName", (kl - pub).abs().idxmax())
        wv, wvp = t["metrics"]["wasserstein"], P[d]["metrics"]["wasserstein"]
        M.add(k + "WvecBest", wv.idxmin()); M.add(k + "WvecBestPaper", wvp.idxmin())
        M.add(k + "WvecMaxDiffPaper", f((wvp - pubw).abs().max(), 4)); M.add(k + "WvecMaxDiffPaperName", (wvp - pubw).abs().idxmax())
        wt = t["metrics"]["wasserstein_time"]
        M.add(k + "WtimeBest", wt.idxmin()); M.add(k + "WtimeBestVal", f(wt.min(), 3))
        M.add(k + "WtimeProp", f(wt["Proposed"], 3))
        pk = (P[d]["post"].groupby("distribution")["rhat"].max())
        M.add(k + "RhatMaxTrunc", f(t["diag"]["max_rhat"].max())); M.add(k + "RhatMaxPaper", f(pk.max()))
        M.add(k + "RhatMaxPaperName", pk.idxmax())
        M.add(k + "AccMin", f(100 * t["diag"]["acceptance_rate"].min(), 1)); M.add(k + "AccMax", f(100 * t["diag"]["acceptance_rate"].max(), 1))
        M.add(k + "EssMin", thousands(t["post"]["ess"].min()))
        M.add(k + "McmcTime", f(t["diag"]["mcmc_runtime_s"].sum(), 1))
        assert bool(t["diag"]["converged"].all()) and bool(P[d]["diag"]["converged"].all()), "a chain did not converge"
        r = t["renewal"]
        M.add(k + "Rate", f(r.arrival_rate_veh_s, 4)); M.add(k + "Flow", thousands(r.capacity_flow_veh_h))
        M.add(k + "CV", f(r.fano_factor, 2)); M.add(k + "CVInv", f(1 / r.fano_factor, 1))
        rv = t["rv"]
        M.add(k + "RenMeanMax", f(100 * rv.mean_rel_err.max(), 1)); M.add(k + "RenVarMax", f(100 * rv.var_rel_err.max(), 1))
        M.add(k + "RenVarMaxT", str(int(rv.t_s[rv.var_rel_err.idxmax()])))
        M.add(k + "KsNormal", f(rv.ks_normal[rv.t_s == 60].iloc[0], 3)); M.add(k + "KsPoisson", f(rv.ks_poisson[rv.t_s == 60].iloc[0], 3))
        mk = t["markov"]
        for s, nm in zip(["close", "platoon", "free"], ["C", "P", "F"]):
            M.add(k + "Pi" + nm, f(mk["pi_" + s])); M.add(k + "Rec" + nm, f(mk["recurrence_" + s], 2))
        for s_, nm in zip(["close", "platoon", "free"], ["C", "P", "F"]):
            M.add(k + "Soj" + nm, f(1.0 / (1.0 - mk[f"p_{s_}_{s_}"]), 1))
        Pm_ = np.array([[mk[f"p_{a}_{b}"] for b in ("close", "platoon", "free")] for a in ("close", "platoon", "free")])
        pi_ = np.array([mk.pi_close, mk.pi_platoon, mk.pi_free])
        for n_ in (20, 60):
            M.add(k + "MixTV" + {20: "Twenty", 60: "Sixty"}[n_], f(0.5 * np.abs(np.eye(3)[0] @ np.linalg.matrix_power(Pm_, n_) - pi_).sum(), 3))
        M.add(k + "RunLen", f(info.n_headways / info.n_runs, 1))
        ind = t["ind"]
        M.add(k + "LagOne", f(ind.autocorr[0])); M.add(k + "LagFive", f(ind.autocorr[4])); M.add(k + "Band", f(ind.band95[0], 4))
        mv = t["mv"]
        M.add(k + "CKtwo", f(mv.chapman_kolmogorov_2_max_err, 3)); M.add(k + "CKfive", f(mv.chapman_kolmogorov_5_max_err, 3))
        M.add(k + "StatTV", f(mv.stationary_tv_distance, 3)); M.add(k + "SecondTV", f(mv.second_order_effect_tv, 4))
        M.add(k + "SecondGsq", f(mv.second_order_g2, 1))
        se = t["params"].loc["Shifted Exponential"]
        M.add(k + "ShExpLambda", f(se["lambda"]))
        M.add(k + "ShExpGammaPaper", f(P[d]["params"].loc["Shifted Exponential", "gamma"]))
        M.add(k + "ShExpLambdaPaper", f(P[d]["params"].loc["Shifted Exponential", "lambda"]))
        M.add(k + "ShExpKLTrunc", f(kl["Shifted Exponential"], 4)); M.add(k + "ShExpKLPaper", f(klp["Shifted Exponential"], 4))
        M.add(k + "ProposedMean", f(T[d]["sim"].iloc[1]["mean"], 2)); M.add(k + "ProposedStd", f(T[d]["sim"].iloc[1]["std"], 2))
        M.add(k + "BurrAlpha", f(t["params"].loc["Burr", "alpha"]))
    pr = pd.read_csv(trunc_dir.parent / "verification" / "ngsim_periods.csv")
    M.add("ngPeriodTwoN", thousands(pr.n_headways[1])); M.add("ngPeriodTwoMean", f(pr.mean_s[1], 2))
    M.add("ngBothN", thousands(pr.n_headways[2])); M.add("ngBothMean", f(pr.mean_s[2], 2))
    im = pd.read_csv(trunc_dir.parent / "verification" / "published_implied_means.csv")
    im = im[im.dataset == "ngsim"].iloc[:, 2]
    M.add("ngImpliedLo", f(im.min(), 2)); M.add("ngImpliedHi", f(im.max(), 2)); M.add("ngImpliedN", str(len(im)))
    M.add("hdSLNGamma", f(T["highd"]["params"].loc["Shifted Log-normal", "gamma"]))
    M.add("ngSLNGamma", f(T["ngsim"]["params"].loc["Shifted Log-normal", "gamma"]))

    # ---------------------------------------------------------------- detector scalars ------------------------
    have_det = (det_dir / "ngsim_gap_summary.csv").exists()
    D = {}
    if have_det:
        for d, k in DS.items():
            s = pd.read_csv(det_dir / f"{d}_gap_summary.csv").iloc[0]
            ac = pd.read_csv(det_dir / f"{d}_gap_autocorr.csv")
            wc = pd.read_csv(det_dir / f"{d}_window_counts.csv")
            ex = pd.read_csv(det_dir / f"{d}_gap_vs_exponential.csv").iloc[0]
            D[d] = {"s": s, "ac": ac, "wc": wc, "ex": ex}
            w60 = wc[wc.window_s == 60.0].iloc[0]
            M.add(k + "DetN", thousands(s.n_gaps)); M.add(k + "DetUnits", thousands(s.n_units))
            M.add(k + "DetMean", f(s.mean_s, 2)); M.add(k + "DetMedian", f(s.median_s, 2)); M.add(k + "DetCV", f(s.cv2, 2))
            M.add(k + "DetFlow", thousands(s.flow_veh_h_lane)); M.add(k + "DetFracBelow", f(100 * s.frac_below_lo, 1))
            M.add(k + "DetFracAbove", f(100 * s.frac_above_hi, 1)); M.add(k + "DetLagOne", f(ac.autocorr[0], 3))
            M.add(k + "DetLagTwo", f(ac.autocorr[1], 3)); M.add(k + "DetBand", f(ac.band95[0], 4))
            M.add(k + "DetKsD", f(ex.ks_D_vs_exponential, 3))
            M.add(k + "DetFanoObs", f(w60.fano_observed, 2)); M.add(k + "DetFanoRen", f(w60.fano_renewal_theory, 2))
            M.add(k + "DetWindows", thousands(w60.n_windows)); M.add(k + "DetSeries", thousands(w60.n_series))
            M.add(k + "DetMeanCount", f(w60.mean_count, 1))
        rt = (det_dir / "tests_count.txt")
    tests_file = out.parent / "logs" / "pytest_count.txt"
    M.add("nTests", tests_file.read_text().strip() if tests_file.exists() else "0")
    M.write(out / "results.tex", "% GENERATED by `python -m headway.make_report_assets` from outputs/*/*.csv -- do not edit by hand.")

    # ---------------------------------------------------------------- tables -------------------------------
    # parameter tables (one per dataset): corrected | paper likelihood | published
    for d, k in DS.items():
        rows = []
        for n in ORDER:
            for j, p in enumerate(PARAMS[n]):
                tr, pa, pu = T[d]["params"].loc[n, p], P[d]["params"].loc[n, p], PAPER_TABLE3_PARAMS[d][n][j]
                tr_s = f(tr)
                if n == "Shifted Exponential" and p == "gamma":
                    tr_s = r"\textit{not identified}"
                name = n if j == 0 else ""
                rows.append(f"{name} & ${PARAM_TEX[p]}$ & {tr_s} & {f(pa)} & {f(pu)} \\\\")
            rows[-1] += r"[0.3ex]"
        w(f"params_{k}", "\n".join(rows) + "\n")

    # KL / W-vec: corrected, paper likelihood, published, per dataset
    for name, col, pubd in [("kl", "kl", PAPER_TABLE5_KL), ("wvec", "wasserstein", PAPER_TABLE5_W)]:
        rows = []
        best = {(d, v): min(ORDER, key=lambda n: (T[d]["metrics"][col] if v == "t" else P[d]["metrics"][col])[n])
                for d in DS for v in "tp"}
        for n in ORDER:
            cells = []
            for d in DS:
                cells += [bold(f(T[d]["metrics"].loc[n, col], 4), best[(d, "t")] == n),
                          bold(f(P[d]["metrics"].loc[n, col], 4), best[(d, "p")] == n),
                          f(pubd[d][n], 4)]
            rows.append(f"{n} & " + " & ".join(cells) + r" \\")
        w(name, "\n".join(rows) + "\n")
    rows = []
    for n in ORDER:
        cells = []
        for d in DS:
            for src in (T, P):
                col = src[d]["metrics"]["wasserstein_time"]
                cells.append(bold(f(col[n], 3), col.idxmin() == n))
        rows.append(f"{n} & " + " & ".join(cells) + r" \\")
    w("wtime", "\n".join(rows) + "\n")

    # goodness-of-fit tests on the corrected fits
    rows = []
    for n in ORDER:
        cells = []
        for d in DS:
            r = T[d]["tests"].loc[n]
            cells += [f(r.ks_D, 3), f(r.ks_2samp_D, 3), pval(r.ks_2samp_p), pval(r.chi2_p)]
        rows.append(f"{n} & " + " & ".join(cells) + r" \\")
    w("tests", "\n".join(rows) + "\n")

    # convergence: max split R-hat (corrected and paper likelihood), min ESS and acceptance of the corrected run
    rows = []
    for n in ORDER:
        cells = []
        for d in DS:
            rt = T[d]["post"][T[d]["post"].distribution == n]
            rp = P[d]["post"][P[d]["post"].distribution == n]
            cells += [f(rt.rhat.max(), 3), f(rp.rhat.max(), 3), thousands(rt.ess.min()), f(100 * T[d]["diag"].loc[n, "acceptance_rate"], 1) + r"\%"]
        rows.append(f"{n} & " + " & ".join(cells) + r" \\")
    w("diag", "\n".join(rows) + "\n")

    # posterior summary of the proposed distribution
    rows = []
    for d in DS:
        po = T[d]["post"][T[d]["post"].distribution == "Proposed"].set_index("param")
        for j, p in enumerate(["a", "b"]):
            r = po.loc[p]
            rows.append(f"{DS_NAME[d] if j == 0 else ''} & ${p}$ & {f(r['mean'], 4)} & {f(r.sd, 4)} & [{f(r.q025, 4)}, {f(r.q975, 4)}] & {f(r.rhat, 3)} & {thousands(r.ess)} \\\\")
    w("posterior", "\n".join(rows) + "\n")

    # renewal simulation vs theory
    rows = []
    for d in DS:
        rv = T[d]["rv"]
        for j, r in rv.iterrows():
            rows.append(f"{DS_NAME[d] if j == 0 else ''} & {int(r.t_s)} & {f(r.mean_sim, 2)} & {f(r.mean_theory, 2)} & {f(r.var_sim, 2)} & "
                        f"{f(r.var_theory, 2)} & {f(r.var_poisson, 2)} & {f(100 * r.mean_rel_err, 2)}\\% & {f(100 * r.var_rel_err, 1)}\\% \\\\")
        rows[-1] += r"[0.3ex]"
    w("renewal", "\n".join(rows) + "\n")

    # Markov: transition matrix, stationary distribution, mean recurrence times
    rows = []
    for d in DS:
        mk = T[d]["markov"]
        for i, (s, nm) in enumerate(zip(["close", "platoon", "free"], REGIMES)):
            ps = [mk[f"p_{s}_{t}"] for t in ["close", "platoon", "free"]]
            rows.append(f"{DS_NAME[d] if i == 0 else ''} & {nm} & " + " & ".join(f(p, 3) for p in ps) +
                        f" & {f(mk['pi_' + s], 3)} & {f(mk['recurrence_' + s], 2)} \\\\")
        rows[-1] += r"[0.3ex]"
    w("markov", "\n".join(rows) + "\n")

    # Markov / independence validation
    rows = []
    items = [("Pairs / transitions used", lambda mv, ind: thousands(mv.n_transitions)),
             ("Chapman--Kolmogorov max error, $n=2$", lambda mv, ind: f(mv.chapman_kolmogorov_2_max_err, 3)),
             ("Chapman--Kolmogorov max error, $n=3$", lambda mv, ind: f(mv.chapman_kolmogorov_3_max_err, 3)),
             ("Chapman--Kolmogorov max error, $n=5$", lambda mv, ind: f(mv.chapman_kolmogorov_5_max_err, 3)),
             (r"Stationary $\pi$ vs observed regime fractions (TV)", lambda mv, ind: f(mv.stationary_tv_distance, 4)),
             ("Simulated chain: occupation vs $\\pi$ (TV)", lambda mv, ind: f(mv.sim_occupation_tv_distance, 4)),
             ("Simulated chain: max error of re-estimated $P$", lambda mv, ind: f(mv.sim_P_reestimate_max_err, 4)),
             (r"Second-order test $G^2$ (12 d.f.)", lambda mv, ind: f(mv.second_order_g2, 1)),
             (r"Second-order test $p$-value", lambda mv, ind: pval(mv.second_order_p)),
             ("Second-order effect size (mean TV)", lambda mv, ind: f(mv.second_order_effect_tv, 4))]
    for lab, fn in items:
        rows.append(f"{lab} & " + " & ".join(fn(T[d]["mv"], T[d]["ind"]) for d in DS) + r" \\")
    w("markov_val", "\n".join(rows) + "\n")

    # autocorrelation of the one-second headways within a trajectory run (+ detector gaps)
    rows = []
    for lag in range(1, 6):
        cells = [f(T[d]["ind"].autocorr[lag - 1], 3) for d in DS]
        if have_det:
            cells += [f(D[d]["ac"].autocorr[lag - 1], 3) for d in DS]
        rows.append(f"{lag} & " + " & ".join(cells) + r" \\")
    rows.append(r"\midrule")
    rows.append("95\\% band $\\pm$ & " + " & ".join(f(T[d]["ind"].band95[0], 4) for d in DS) +
                (" & " + " & ".join(f(D[d]["ac"].band95[0], 4) for d in DS) if have_det else "") + r" \\")
    w("autocorr", "\n".join(rows) + "\n")

    if have_det:
        rows = []
        for d in DS:
            s = D[d]["s"]
            rows.append(f"{DS_NAME[d]} & {thousands(s.n_gaps)} & {f(s.mean_s, 2)} & {f(s.median_s, 2)} & {f(s.std_s, 2)} & {f(s.cv2, 2)} & "
                        f"{thousands(s.flow_veh_h_lane)} & {f(100 * s.frac_below_lo, 1)}\\% & {f(100 * s.frac_above_hi, 1)}\\% \\\\")
        w("detector_gaps", "\n".join(rows) + "\n")
        rows = []
        for d in DS:
            for j, r in D[d]["wc"].iterrows():
                rows.append(f"{DS_NAME[d] if j == 0 else ''} & {int(r.window_s)} & {thousands(r.n_windows)} & {f(r.mean_count, 1)} & "
                            f"{f(r.fano_observed, 2)} & {f(r.fano_renewal_theory, 2)} & 1 \\\\")
            rows[-1] += r"[0.3ex]"
        w("detector_counts", "\n".join(rows) + "\n")

    # ---------------------------------------------------------------- figures -------------------------------
    wanted = {"sample_paths": "arrival_process_sample_paths", "event_count": "event_count_distribution",
              "fitted": "fitted_distributions", "posterior": "posterior_proposed", "traces": "trace_proposed",
              "heatmap": "markov_transition_heatmap", "diagram": "markov_state_diagram",
              "stateprob": "markov_state_probabilities", "cdf": "empirical_cdf", "sim": "simulation"}
    for d, k in DS.items():
        for short, src in wanted.items():
            shutil.copy(trunc_dir / d / "figures" / f"{src}.png", out / "figures" / f"{k}_{short}.png")
    from .plots import plot_markov_state_probabilities_vs_time
    from .stochastic_process import REGIME_NAMES, TrafficRegimeMarkovChain
    for d, k in DS.items():  # transient evolution from close following, 60 one-second steps, from the saved P
        mk = T[d]["markov"]
        Pm = np.array([[mk[f"p_{a}_{b}"] for b in ("close", "platoon", "free")] for a in ("close", "platoon", "free")])
        mc = TrafficRegimeMarkovChain(Pm)
        plot_markov_state_probabilities_vs_time(mc.transient_distribution([1.0, 0.0, 0.0], steps=60), mc.stationary_distribution(),
                                                REGIME_NAMES, DS_NAME[d], out / "figures" / f"{k}_stateprob.png")
    if have_det:
        from .detector import plot_gaps
        for d, k in DS.items():
            g = pd.read_csv(det_dir / f"{d}_gaps.csv", usecols=["gap"])
            plot_gaps(DS_NAME[d] + ": detector inter-arrival times", g, D[d]["ac"], out / "figures" / f"{k}_detector.png",
                      T[d]["ind"])
    print(f"wrote {out/'results.tex'}, {len(list((out/'tables').glob('*.tex')))} tables and figures to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--truncated", default="outputs/truncated")
    ap.add_argument("--paper", default="outputs/paper")
    ap.add_argument("--detector", default="outputs/detector")
    ap.add_argument("--out", default="report")
    a = ap.parse_args()
    build(a.truncated, a.paper, a.detector, a.out)
