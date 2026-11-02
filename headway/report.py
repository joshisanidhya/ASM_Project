"""Build the project report (Markdown and Word) from the outputs of a real run.

    uv run --group report python -m headway.report [--outputs outputs] [--out report]

The narrative and derivations live in headway/report_template.md. Every number, table and figure
is read from `outputs/<dataset>/`, so the report cannot drift from the results: a placeholder
@@key@@ with no value stops the build, and so does data from a synthetic run.
"""
import argparse
import inspect
import re
import shutil
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import distributions, mcmc, validation
from .distributions import DISTS
from .paper_reference import PAPER_TABLE3_PARAMS, PAPER_TABLE5_KL, PAPER_TABLE5_W

DATASETS = {"highd": "highD", "ngsim": "NGSIM"}
FIGURES = {  # name used in the template -> file under outputs/<dataset>/figures/
    "fitted": "fitted_distributions.png", "sample_paths": "arrival_process_sample_paths.png",
    "event_count": "event_count_distribution.png", "state_diagram": "markov_state_diagram.png",
    "state_probabilities": "markov_state_probabilities.png", "posterior": "posterior.png",
    "traces": "traces.png",
}
SYMBOL = {"mu": "μ", "sigma": "σ", "gamma": "γ", "alpha": "α", "beta": "β", "lambda": "λ"}
REGIMES = ["Close following", "Stable platoon", "Free flow"]


# --- small formatting helpers -----------------------------------------------------------------
def slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def num(x, d=3):
    return f"{x:,.{d}f}"


def pval(p):
    return f"{p:.1e}" if p < 1e-3 else f"{p:.3f}"


def md_table(header, rows, caption):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    return "\n".join(lines) + f"\n\nTable: {caption}"


def pstring(name, theta):
    return ", ".join(f"{SYMBOL.get(p, p)}={v:.3f}" for p, v in zip(DISTS[name].params, theta))


# --- per-dataset context ----------------------------------------------------------------------
def load(outputs, ds):
    d = Path(outputs) / ds
    need = ["dataset_info", "params", "metrics", "tests", "diagnostics", "stochastic_markov_chain",
            "stochastic_renewal_process", "validation_renewal", "validation_markov", "validation_independence"]
    missing = [n for n in need if not (d / f"{n}.csv").exists()]
    if missing:
        raise SystemExit(f"{d}: missing {missing}. Run `uv run python run.py --period 1 --real-only` first.")
    r = lambda n, **k: pd.read_csv(d / f"{n}.csv", **k)
    return dict(info=r("dataset_info").iloc[0], params=r("params", index_col=0), metrics=r("metrics", index_col=0),
                tests=r("tests", index_col=0), diag=r("diagnostics", index_col=0),
                mk=r("stochastic_markov_chain").iloc[0], rp=r("stochastic_renewal_process").iloc[0],
                vr=r("validation_renewal"), vm=r("validation_markov").iloc[0], vi=r("validation_independence"))


def disp(f):
    kind = "sub-Poisson" if f < 0.95 else "super-Poisson" if f > 1.05 else "close to Poisson"
    return f"{kind} (variance-to-mean ratio {f:.2f})"


def indep(lag1, band):
    if abs(lag1) > max(3 * band, 0.1):
        return "strong serial dependence between successive headways (the i.i.d. assumption of the renewal model does not hold at the one-second sampling scale)"
    return "weak but statistically significant serial dependence" if abs(lag1) > band else "no significant serial dependence"


def markov_text(v):
    p, tv = v.second_order_p, v.second_order_effect_tv
    verdict = ("not rejected" if p >= 0.05 else
               "rejected statistically, but the extra dependence on the second-previous regime is small" if tv < 0.05 else
               "clearly violated: the previous two regimes carry information beyond the last one")
    return (f"the largest gap between the empirical two-step transition matrix and $P^2$ is {v.chapman_kolmogorov_2_max_err:.3f} "
            f"({v.chapman_kolmogorov_5_max_err:.3f} at five steps); the stationary distribution differs from the observed regime "
            f"fractions by a total-variation distance of {v.stationary_tv_distance:.4f}; a simulated long path reproduces $\\pi$ "
            f"(distance {v.sim_occupation_tv_distance:.4f}). The likelihood-ratio test of first against second order gives "
            f"p = {pval(p)} with an average effect size (total variation between $P(\\cdot\\mid i,j)$ and $P(\\cdot\\mid j)$) of "
            f"{tv:.3f}, so the first-order assumption is {verdict}.")


def dataset_context(D, ds, label):
    c, info, kl, w = {}, D["info"], D["metrics"].kl, D["metrics"].wasserstein
    P, rp, mk, vr, vm, vi = D["params"], D["rp"], D["mk"], D["vr"], D["vm"], D["vi"]
    for k, v in dict(n=f"{int(info.n_headways):,}", mean=num(info.mean_s, 2), median=num(info.median_s, 2),
                     std=num(info.std_s, 2)).items():
        c[f"{ds}.{k}"] = v
    c[f"{ds}.rate"], c[f"{ds}.flow"] = num(rp.arrival_rate_veh_s, 3), f"{rp.capacity_flow_veh_h:,.0f}"
    c[f"{ds}.cv2"], c[f"{ds}.disp"] = num(rp.fano_factor, 2), disp(rp.fano_factor)
    c[f"{ds}.mcmc_time"] = num(D["diag"].mcmc_runtime_s.sum(), 1)

    best = kl.idxmin()
    rank = int(kl.rank()["Proposed"])
    c[f"{ds}.kl.best"], c[f"{ds}.kl.best_value"] = best, num(kl[best], 4)
    c[f"{ds}.kl.proposed"], c[f"{ds}.kl.proposed_rank"] = num(kl["Proposed"], 4), str(rank)
    c[f"{ds}.kl.rank_phrase"] = "lowest" if rank == 1 else f"{rank}{'nd' if rank == 2 else 'rd' if rank == 3 else 'th'}-lowest"
    c[f"{ds}.w.proposed"] = num(w["Proposed"], 4)
    for name in DISTS:
        c[f"{ds}.kl.{slug(name)}"] = num(kl[name], 4)
    dkl = pd.Series({n: abs(kl[n] - PAPER_TABLE5_KL[ds][n]) for n in DISTS})
    dw = pd.Series({n: abs(w[n] - PAPER_TABLE5_W[ds][n]) for n in DISTS})
    c[f"{ds}.kl.max_abs_diff"] = f"{num(dkl.max(), 4)} ({dkl.idxmax()})"
    c[f"{ds}.w.max_abs_diff"] = f"{num(dw.max(), 4)} ({dw.idxmax()})"

    a, b = P.loc["Proposed", "a"], P.loc["Proposed", "b"]
    c[f"{ds}.a"], c[f"{ds}.b"], c[f"{ds}.kappa"] = num(a), num(b), num(-np.log(b))
    c[f"{ds}.paper_a"], c[f"{ds}.paper_b"] = (num(x) for x in PAPER_TABLE3_PARAMS[ds]["Proposed"])

    pi = np.array([mk.pi_close, mk.pi_platoon, mk.pi_free])
    c[f"{ds}.pi_close"], c[f"{ds}.pi_platoon"], c[f"{ds}.pi_free"] = (num(x, 3) for x in pi)
    c[f"{ds}.dom"] = f"{REGIMES[int(pi.argmax())].lower()} ({pi.max():.0%})"

    for t in (60, 300):
        row = vr[vr.t_s == t].iloc[0]
        c[f"{ds}.mean_sim_{t}"], c[f"{ds}.mean_th_{t}"] = num(row.mean_sim, 1), num(row.mean_theory, 1)
        c[f"{ds}.var_sim_{t}"], c[f"{ds}.var_th_{t}"] = num(row.var_sim, 1), num(row.var_theory, 1)
        c[f"{ds}.var_poisson_{t}"] = num(row.var_poisson, 1)
    r60 = vr[vr.t_s == 60].iloc[0]
    c[f"{ds}.ks_normal_60"], c[f"{ds}.ks_poisson_60"] = num(r60.ks_normal, 3), num(r60.ks_poisson, 3)

    lag1, band = vi.autocorr.iloc[0], vi.band95.iloc[0]
    c[f"{ds}.lag1"], c[f"{ds}.band"] = num(lag1, 3), num(band, 4)
    c[f"{ds}.indep"], c[f"{ds}.markovval"] = indep(lag1, band), markov_text(vm)
    c[f"{ds}.rhat_max"] = num(D["diag"].max_rhat.max(), 3)

    opts = [("ngsim_location", info.ngsim_location), ("ngsim_period", info.ngsim_period)]
    if ds == "ngsim":
        period, loc = str(info.ngsim_period), str(info.ngsim_location).upper()
        lanes = "" if str(info.ngsim_lanes) == "all" else f", lanes {info.ngsim_lanes}"
        c["ngsim.selection"] = (f"the {loc} data, recording period {period} (the first contiguous recording block){lanes}"
                                if period != "all" else f"the {loc} data, all recording periods{lanes}")
        c["ngsim.period_note"] = (
            "The paper does not say which NGSIM recording period it used. Of the two periods in the export, we chose the one "
            "whose mean headway and sample size are consistent with the values implied by the paper's fitted parameters "
            "(a mean of about 2.9 s); this is an inference, not something the paper states."
            if period == "1" else "We use every recording period in the export.")
    return c


# --- tables -------------------------------------------------------------------------------------
def tables(data, ctx):
    T, H, N = {}, data["highd"], data["ngsim"]
    T["table.dataset"] = md_table(
        ["Dataset", "Selection", "Samples", "Mean (s)", "Median (s)", "Std (s)"],
        [["highD", "all 60 recordings", ctx["highd.n"], ctx["highd.mean"], ctx["highd.median"], ctx["highd.std"]],
         ["NGSIM", ctx["ngsim.selection"], ctx["ngsim.n"], ctx["ngsim.mean"], ctx["ngsim.median"], ctx["ngsim.std"]]],
        "**Table 1.** Headway samples used (one per second per follower, 0.5 to 25 s).")
    T["table.model"] = md_table(
        ["Component", "Random variable", "State space", "Time parameter", "Model"],
        [["Headway", "$H_i$", "$[0.5,\\infty)$ s", "vehicle index $i$", "i.i.d., density $b^{\\lvert t-a\\rvert}/Z$"],
         ["Arrival count", "$N(t)$", "$\\{0,1,2,\\dots\\}$", "continuous, $t\\ge0$", "renewal process with inter-arrival times $H_i$"],
         ["Car-following regime", "$X_n$", "{close, platoon, free}", "discrete, 1 s steps", "first-order Markov chain, matrix $P$"]],
        "**Table 2.** Random variables, state spaces and time parameters.")

    rows = []
    for t in (60, 300):
        for ds, lab in DATASETS.items():
            r = data[ds]["vr"]
            r = r[r.t_s == t].iloc[0]
            rows.append([lab, t, num(r.mean_sim, 2), num(r.mean_theory, 2), num(r.var_sim, 2), num(r.var_theory, 2),
                         num(r.var_poisson, 2), f"{100 * r.mean_rel_err:.2f}%", f"{100 * r.var_rel_err:.1f}%"])
    T["table.renewal"] = md_table(
        ["Dataset", "$t$ (s)", "E[N] sim", "E[N] theory", "Var[N] sim", "Var[N] theory", "Var if Poisson", "mean err", "var err"],
        rows, "**Table 3.** Simulated renewal counts (2,000 paths) against renewal theory.")

    rows = []
    for ds, lab in DATASETS.items():
        mk = data[ds]["mk"]
        P = np.array([[mk.p_close_close, mk.p_close_platoon, mk.p_close_free],
                      [mk.p_platoon_close, mk.p_platoon_platoon, mk.p_platoon_free],
                      [mk.p_free_close, mk.p_free_platoon, mk.p_free_free]])
        pi = [mk.pi_close, mk.pi_platoon, mk.pi_free]
        rec = [mk.recurrence_close, mk.recurrence_platoon, mk.recurrence_free]
        for i in range(3):
            rows.append([lab if i == 0 else "", REGIMES[i], *(num(x) for x in P[i]), num(pi[i]), num(rec[i], 2)])
    T["table.markov"] = md_table(["Dataset", "From", "to close", "to platoon", "to free", "$\\pi_i$", "recurrence $1/\\pi_i$ (steps)"],
                                 rows, "**Table 4.** Estimated transition matrices, stationary distributions and mean recurrence times.")

    def compare(key, paper, cap, d):
        rows = []
        for n in DISTS:
            row = [n]
            for ds in DATASETS:
                row += [num(data[ds]["metrics"][key][n], d), num(paper[ds][n], d)]
            rows.append(row)
        return md_table(["Distribution", "highD ours", "highD paper", "NGSIM ours", "NGSIM paper"], rows, cap)

    rows = []
    for n in DISTS:
        row = [n]
        for ds in DATASETS:
            row += [pstring(n, data[ds]["params"].loc[n, list(DISTS[n].params)].to_numpy(float)),
                    pstring(n, PAPER_TABLE3_PARAMS[ds][n])]
        rows.append(row)
    T["table.params"] = md_table(["Distribution", "highD ours", "highD paper", "NGSIM ours", "NGSIM paper"], rows,
                                 "**Table 5.** Posterior-mean parameters against the paper's Table 3.")
    T["table.kl"] = compare("kl", PAPER_TABLE5_KL, "**Table 6.** KL divergence (lower is better) against the paper's Table 5.", 4)
    T["table.w"] = compare("wasserstein", PAPER_TABLE5_W, "**Table 7.** Wasserstein distance (Eq. 17, lower is better) against the paper's Table 5.", 4)

    def vcol(ds):
        v, vi = data[ds]["vm"], data[ds]["vi"]
        return [f"{vi.autocorr.iloc[0]:.3f} (±{vi.band95.iloc[0]:.4f})", num(v.chapman_kolmogorov_2_max_err),
                num(v.chapman_kolmogorov_5_max_err), num(v.stationary_tv_distance, 4), num(v.sim_occupation_tv_distance, 4),
                pval(v.second_order_p), num(v.second_order_effect_tv)]
    labels = ["Lag-1 autocorrelation of headways (95% band)", "Chapman-Kolmogorov, 2-step, max error",
              "Chapman-Kolmogorov, 5-step, max error", "Stationary vs observed regime fractions (TV)",
              "Simulated path vs $\\pi$ (TV)", "First- vs second-order test, p-value", "Second-order effect size (TV)"]
    T["table.valid"] = md_table(["Check", "highD", "NGSIM"], [[l, a, b] for l, a, b in zip(labels, vcol("highd"), vcol("ngsim"))],
                                "**Table 8.** Validation of the independence and Markov assumptions.")

    rows = []
    for n in DISTS:
        row = [n]
        for ds in DATASETS:
            t = data[ds]["tests"].loc[n]
            row += [num(t.ks_2samp_D), pval(t.ks_2samp_p), pval(t.chi2_p)]
        rows.append(row)
    T["table.tests"] = md_table(["Distribution", "highD KS D", "KS p", "χ² p", "NGSIM KS D", "KS p", "χ² p"], rows,
                                "**Table A1.** Two-sample KS and chi-square tests.")
    rows = []
    for n in DISTS:
        row = [n]
        for ds in DATASETS:
            g = data[ds]["diag"].loc[n]
            row += [num(g.max_rhat), f"{g.acceptance_rate:.1%}"]
        rows.append(row)
    T["table.diag"] = md_table(["Distribution", "highD max R-hat", "acceptance", "NGSIM max R-hat", "acceptance"], rows,
                               "**Table A2.** MCMC convergence (split R-hat; 2 chains, 5,000 post-warmup draws each).")
    rows = [[k + 1, num(data["highd"]["vi"].autocorr.iloc[k], 4), num(data["ngsim"]["vi"].autocorr.iloc[k], 4)] for k in range(5)]
    T["table.indep"] = md_table(["Lag (s)", "highD", "NGSIM"], rows, "**Table A3.** Autocorrelation of the headway sequence.")
    return T


def combined_context(data, ctx):
    H, N = data["highd"], data["ngsim"]
    e = lambda col, ds: data[ds]["vr"][col].max()
    ctx["both.mean_err_max"] = f"{100 * max(e('mean_rel_err', d) for d in data):.1f}"
    ctx["both.var_err_max"] = f"{100 * max(e('var_rel_err', d) for d in data):.1f}"
    f = [data[d]["rp"].fano_factor for d in data]
    ctx["both.poisson_phrase"] = (
        f"the arrival process is clearly not Poisson (variance-to-mean ratios of {f[0]:.2f} on highD and {f[1]:.2f} on NGSIM instead of 1)"
        if max(f) < 0.95 else
        f"the arrival process has variance-to-mean ratios of {f[0]:.2f} on highD and {f[1]:.2f} on NGSIM, against 1 for a Poisson process")
    ctx["both.disp_meaning"] = (
        "Both streams are therefore more regular than Poisson, as expected from car following, which suppresses both very short and very long gaps."
        if max(f) < 0.95 else "The two streams are not both more regular than a Poisson stream.")
    bh, bn = ctx["highd.kl.best"], ctx["ngsim.kl.best"]
    ctx["both.paper_agreement"] = (
        "This reproduces the qualitative findings of the paper (proposed distribution best on highD, Burr best on NGSIM)."
        if (bh, bn) == ("Proposed", "Burr") else
        f"This differs from the paper, where the proposed distribution is best on highD and Burr is best on NGSIM (ours: {bh} and {bn}).")
    ctx["both.model_comparison"] = (
        f"On highD the proposed distribution ranks {ctx['highd.kl.proposed_rank']} of 7 by KL divergence (best: {bh}, {ctx['highd.kl.best_value']}); "
        f"on NGSIM it ranks {ctx['ngsim.kl.proposed_rank']} of 7 (best: {bn}, {ctx['ngsim.kl.best_value']}).")

    p = pd.concat([data[d]["tests"][["ks_2samp_p", "chi2_p"]] for d in data]).to_numpy()
    ctx["both.tests_phrase"] = (
        ("The KS and chi-square tests reject every fitted distribution at the 5% level. " if (p < 0.05).all() else
         "Not every distribution is rejected by the KS and chi-square tests at the 5% level (Appendix, Table A1). ")
        + "With samples of this size even small deviations are statistically significant, so KL divergence and "
          "Wasserstein distance are the more informative measures (Appendix, Table A1).")
    bad = [f"{DATASETS[d]}: " + ", ".join(f"{n} (R-hat {r.max_rhat:.2f})" for n, r in data[d]["diag"].iterrows() if r.max_rhat >= 1.05)
           for d in data if (data[d]["diag"].max_rhat >= 1.05).any()]
    ctx["report.unconverged"] = (
        f"Chains did not converge (split R-hat of at least 1.05) for {'; '.join(bad)}. We did not investigate this further, so the point estimates for these fits are less reliable."
        if bad else "All chains converged (split R-hat below 1.05 for every parameter of every distribution).")
    acc = pd.concat([data[d]["diag"].acceptance_rate for d in data])
    ctx["both.acc_range"] = f"{acc.min():.0%} and {acc.max():.0%}"


# --- build --------------------------------------------------------------------------------------
def build(outputs="outputs", out="report", allow_synthetic=False):
    out = Path(out)
    data = {ds: load(outputs, ds) for ds in DATASETS}
    if not allow_synthetic and any(d["info"].source != "raw" for d in data.values()):
        raise SystemExit("These outputs come from synthetic data. Run `uv run python run.py --period 1 --real-only`, "
                         "or pass --allow-synthetic to test the report build.")

    ctx = {"date": date.today().strftime("%d %B %Y")}
    for ds, lab in DATASETS.items():
        ctx.update(dataset_context(data[ds], ds, lab))
    ctx.update(tables(data, ctx))
    combined_context(data, ctx)
    for key, fn in {"proposed": distributions._proposed_cdf, "metropolis": mcmc.metropolis,
                    "renewal": validation.renewal_validation}.items():
        ctx[f"code.{key}"] = inspect.getsource(fn).rstrip()

    text = (Path(__file__).parent / "report_template.md").read_text(encoding="utf-8")
    missing = sorted(set(re.findall(r"@@([\w.]+)@@", text)) - set(ctx))
    if missing:
        raise SystemExit(f"template placeholders without a value: {missing}")
    text = re.sub(r"@@([\w.]+)@@", lambda m: ctx[m.group(1)], text)

    (out / "figures").mkdir(parents=True, exist_ok=True)
    for ds in DATASETS:
        for name, fname in FIGURES.items():
            src = Path(outputs) / ds / "figures" / fname
            if not src.exists():
                raise SystemExit(f"missing figure {src}; rerun run.py")
            shutil.copyfile(src, out / "figures" / f"{ds}_{name}.png")
    (out / "report.md").write_text(text, encoding="utf-8")

    import pypandoc
    pypandoc.convert_file(str(out / "report.md"), "docx", format="markdown", outputfile=str(out / "report.docx"),
                          extra_args=["--number-sections", f"--resource-path={out}"])
    main = text.split("# Appendix")[0]  # the page limit applies to the report body
    prose = len(re.sub(r"\$\$.*?\$\$|!\[.*?\]\(.*?\)(\{.*?\})?|^\|.*$", "", main, flags=re.S | re.M).split())
    figs, tabs = len(re.findall(r"!\[", main)), len(re.findall(r"^Table:", main, flags=re.M))
    print(f"wrote {out / 'report.docx'} and report.md: {prose:,} words of prose, {tabs} tables, {figs} figures in the body")
    print(f"rough body length: {prose / 525 + 0.3 * figs + 0.22 * tabs:.0f} pages "
          f"(an estimate: open the .docx and check it against the 10-12 page target)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--out", default="report")
    ap.add_argument("--allow-synthetic", action="store_true", help="only to test the build")
    a = ap.parse_args()
    build(a.outputs, a.out, a.allow_synthetic)
