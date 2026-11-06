"""Unified execution pipeline for headway distribution fitting (MCMC), simulation, and diagnostics.

Supported Datasets (2 exclusively):
1. highD: German highway drone dataset (25 Hz -> 1 Hz, uninterrupted highway flow)
2. NGSIM: FHWA I-80 trajectory dataset (10 Hz -> 1 Hz, interrupted/uninterrupted highway flow)

Features:
- Reproduces paper MCMC procedure (2 chains x 10,000 iterations, 5,000 warmup)
- Stochastic simulation module for synthetic headway generation & validation
- Goodness-of-fit evaluation as in the paper: KS, Chi-Square, KL divergence, Wasserstein (Eq. 17)
- Publication-quality visualizations and standardized CSV exports under `outputs/`
"""
import argparse
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from headway.datasets import LOADERS
from headway.distributions import ALPHA, DISTS, EDGES, T_MAX
from headway.fit import fit_all
from headway.simulation import (
    compare_empirical_and_simulated,
    compute_summary_statistics,
    sample_distribution,
    validate_simulated_against_theoretical,
)

from headway.paper_reference import PAPER_TABLE3_PARAMS  # paper Table 3, used for the synthetic fallback


def find_dataset_path(dataset: str, root: Union[str, Path]) -> Optional[Path]:
    """Locate a dataset's raw files under root/<dataset>; None if absent."""
    d = Path(root) / dataset
    if not d.exists():
        return None
    if dataset == "ngsim":
        hits = sorted(d.glob("*.csv"), key=lambda p: -p.stat().st_size)
        return hits[0] if hits else None
    if dataset == "highd":
        return d if any(d.glob("*_tracks.csv")) else None
    return None


def generate_synthetic_headways(dataset: str, size: int = 15000, seed: int = 42) -> np.ndarray:
    """Generate realistic synthetic headway observations using paper's fitted parameters."""
    params_dict = PAPER_TABLE3_PARAMS.get(dataset, PAPER_TABLE3_PARAMS["highd"])
    prop_params = params_dict["Proposed"]
    # Sample from proposed distribution as primary generator
    return sample_distribution(DISTS["Proposed"], prop_params, size, seed=seed)


def run_dataset_pipeline(name: str, path: Optional[Path], args: argparse.Namespace) -> Dict[str, Any]:
    """Execute end-to-end MCMC, simulation, metrics, and visualization for one dataset."""
    from headway.diagnostics import split_rhat, summarize_posterior
    from headway.metrics import evaluate
    from headway.plots import (
        plot_empirical_cdf,
        plot_fitted_distributions,
        plot_goodness_of_fit_metrics,
        plot_posteriors,
        plot_simulation_validation,
        plot_traces,
    )

    out_dir = Path(args.out) / name
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    if path is not None:
        kw = {}
        if name == "ngsim":
            kw = {"location": args.location, "lanes": args.lanes, "period": args.period}
        data, groups = LOADERS[name](path, workers=args.workers, return_groups=True, **kw)
        source_desc = f"raw file ({path})"
    else:
        groups = None  # synthetic i.i.d. draws: one sequence
        if args.real_only:
            raise SystemExit(f"[{name}] no raw data under {args.data_dir}/{name}/ and --real-only is set.")
        print(f"[{name}] Raw dataset not found on disk. Generating synthetic benchmark sample...")
        data = generate_synthetic_headways(name, size=args.synthetic_size, seed=args.seed)
        source_desc = "synthetic benchmark generator"

    if len(data) == 0:
        raise ValueError(f"[{name}] No valid car-following headways extracted from {source_desc}.")

    counts = np.histogram(data, bins=EDGES)[0].astype(float)
    tmin = float(data.min())
    print(f"\n=======================================================")
    print(f" Dataset: {name.upper()} | Samples: {len(data):,} | Mean: {data.mean():.2f}s | Source: {source_desc}")
    print(f"=======================================================")
    # The report generator refuses synthetic data, so record where this run's data came from.
    pd.DataFrame([{
        "dataset": name, "source": "raw" if path is not None else "synthetic",
        "n_headways": len(data), "mean_s": data.mean(), "median_s": np.median(data),
        "std_s": data.std(ddof=1), "min_s": data.min(), "max_s": data.max(),
        "ngsim_location": args.location if name == "ngsim" else "",
        "ngsim_period": (args.period or "all") if name == "ngsim" else "",
        "ngsim_lanes": ("all" if not args.lanes else ",".join(map(str, args.lanes))) if name == "ngsim" else "",
        "likelihood": args.likelihood,
        # trajectory runs: stretches of consecutive seconds of one follower (what Markov transitions use)
        "n_runs": int(groups.max()) + 1 if groups is not None and len(groups) else 1,
        "n_transitions": int((groups[:-1] == groups[1:]).sum()) if groups is not None else len(data) - 1,
    }]).to_csv(out_dir / "dataset_info.csv", index=False)

    # 2. Parameter Estimation: MCMC
    print(f"[{name}] Running MCMC (2 chains x {args.iters:,} iters, {args.warmup:,} warmup)...")
    t0_mcmc = time.perf_counter()
    mcmc_runtimes = {}
    fits = {}
    for dname in DISTS.keys():
        t0_d = time.perf_counter()
        fit_res = fit_all(
            counts,
            tmin,
            n_iter=args.iters,
            n_warmup=args.warmup,
            chains=args.chains,
            seed=args.seed,
            names=[dname],
            workers=args.workers,
            likelihood=args.likelihood,
        )[dname]
        mcmc_runtimes[dname] = time.perf_counter() - t0_d
        fits[dname] = fit_res

    total_mcmc_time = time.perf_counter() - t0_mcmc
    print(f"[{name}] MCMC complete in {total_mcmc_time:.2f}s.")

    # 3. Evaluate MCMC Metrics and Posteriors
    mcmc_params = {}
    metric_rows = []
    posterior_rows = []
    diagnostics_rows = []

    for dname, fit_res in fits.items():
        theta = fit_res.theta
        draws = fit_res.draws
        full_draws = fit_res.full_draws
        acc_rates = fit_res.acceptance_rates
        dist = DISTS[dname]
        pnames = dist.params
        slug = dname.lower().replace(" ", "_")
        mcmc_params[dname] = theta

        # Metrics
        m = evaluate(dist, theta, data, counts, seed=args.seed)
        row = {"distribution": dname, **dict(zip(pnames, theta)), **m}
        metric_rows.append(row)

        # Posterior diagnostics
        avg_acc = float(np.mean(acc_rates)) if acc_rates else np.nan
        post_summary = summarize_posterior(draws, pnames, acceptance_rate=avg_acc)
        for ps in post_summary:
            posterior_rows.append({"distribution": dname, **ps})

        rhat = split_rhat(draws)
        is_converged = bool((rhat < 1.05).all())
        diagnostics_rows.append({
            "distribution": dname,
            "max_rhat": float(np.nanmax(rhat)),
            "converged": is_converged,
            "acceptance_rate": avg_acc,
            "mcmc_runtime_s": mcmc_runtimes[dname],
        })

        # Save draws compressed
        draws_dir = out_dir / "draws"
        draws_dir.mkdir(exist_ok=True)
        np.savez_compressed(draws_dir / f"{slug}.npz", draws=draws, full_draws=full_draws, params=np.array(pnames))

        # Plot trace & posterior for this distribution
        plot_traces(f"{name.upper()} — {dname}", pnames, full_draws, args.warmup, fig_dir / f"trace_{slug}.png")
        plot_posteriors(f"{name.upper()} — {dname}", pnames, draws, fig_dir / f"posterior_{slug}.png")

        flag = " [OK]" if is_converged else " [NOT CONVERGED]"
        print(f"  {dname:22s} | KL={m['kl']:.4f} | W={m['wasserstein']:.4f} | W_time={m['wasserstein_time']:.4f}s | max R-hat={np.nanmax(rhat):.3f} | acc={avg_acc:.2%}{flag}")

    # Plot proposed trace and posterior under default names
    if "Proposed" in fits:
        plot_traces(f"{name.upper()} — Proposed", DISTS["Proposed"].params, fits["Proposed"].full_draws, args.warmup, fig_dir / "traces.png")
        plot_posteriors(f"{name.upper()} — Proposed", DISTS["Proposed"].params, fits["Proposed"].draws, fig_dir / "posterior.png")

    mcmc_metrics_dict = {r["distribution"]: r for r in metric_rows}

    # 5. Stochastic Simulation Module
    print(f"\n[{name}] Running Stochastic Simulation module (Proposed distribution)...")
    prop_theta = mcmc_params["Proposed"]
    sim_samples = sample_distribution(DISTS["Proposed"], prop_theta, size=min(len(data), 20000), seed=args.seed)
    sim_stats = compute_summary_statistics(sim_samples, name="Simulated_Proposed")
    emp_stats = compute_summary_statistics(data, name="Empirical_Data")
    sim_validation = validate_simulated_against_theoretical(DISTS["Proposed"], prop_theta, sim_samples)
    sim_comparison = compare_empirical_and_simulated(data, sim_samples)

    sim_summary_rows = [
        emp_stats.to_dict(),
        sim_stats.to_dict(),
        {"distribution": "Validation", "sample_size": len(sim_samples), **sim_validation},
        {"distribution": "Comparison", "sample_size": len(sim_samples), **sim_comparison},
    ]
    pd.DataFrame(sim_summary_rows).to_csv(out_dir / "simulation_summary.csv", index=False)
    print(f"[{name}] Simulation validation: KS p-value={sim_validation['theory_ks_p']:.4f}, Chi2 p-value={sim_validation['theory_chi2_p']:.4f}")

    # 6. Stochastic Process Analysis (ASM Course Page 2-4: Renewal Process & Markov Chain)
    print(f"\n[{name}] Running Stochastic Process Analysis (Arrival Renewal & Regime Markov Chain)...")
    from headway.stochastic_process import (
        REGIME_NAMES,
        TrafficRegimeMarkovChain,
        analyze_stochastic_processes,
    )
    from headway.plots import (
        plot_arrival_process_sample_paths,
        plot_markov_state_probabilities_vs_time,
        plot_markov_transition_heatmap,
    )

    mc_results, rp_results = analyze_stochastic_processes(
        data,
        n_sample_paths=args.sample_paths,
        t_simulation=args.sim_time,
        seed=args.seed,
        groups=groups,
    )

    # Compute transient evolution from close-following state [1.0, 0.0, 0.0]
    mc = TrafficRegimeMarkovChain(mc_results.transition_matrix)
    transient_traj = mc.transient_distribution([1.0, 0.0, 0.0], steps=20)

    # Save stochastic process figures
    plot_arrival_process_sample_paths(
        rp_results.sample_paths_time,
        rp_results.sample_paths_counts,
        rp_results.expected_counts,
        name.upper(),
        fig_dir / "arrival_process_sample_paths.png",
    )
    plot_markov_transition_heatmap(
        mc_results.transition_matrix,
        REGIME_NAMES,
        name.upper(),
        fig_dir / "markov_transition_heatmap.png",
    )
    plot_markov_state_probabilities_vs_time(
        transient_traj,
        mc_results.stationary_distribution,
        REGIME_NAMES,
        name.upper(),
        fig_dir / "markov_state_probabilities.png",
    )

    # Save stochastic process CSV summaries
    pd.DataFrame([mc_results.to_dict()]).to_csv(out_dir / "stochastic_markov_chain.csv", index=False)
    pd.DataFrame([rp_results.to_dict()]).to_csv(out_dir / "stochastic_renewal_process.csv", index=False)

    print(f"[{name}] Markov stationary distribution: Close={mc_results.stationary_distribution[0]:.3f}, Platoon={mc_results.stationary_distribution[1]:.3f}, Free={mc_results.stationary_distribution[2]:.3f}")
    print(f"[{name}] Arrival renewal rate: {rp_results.asymptotic_rate:.4f} veh/s ({rp_results.hourly_flow_rate:.0f} veh/h/lane)")

    # 6b. Quantitative validation of the stochastic models (course brief, section 6)
    from headway.plots import plot_event_count_distribution, plot_markov_state_diagram
    from headway.validation import independence_check, markov_validation, renewal_validation

    times = (30, 60, 120, 300)
    renewal_df, sim_counts = renewal_validation(data, times=times, seed=args.seed)
    renewal_df.to_csv(out_dir / "validation_renewal.csv", index=False)
    independence_check(data, groups=groups).to_csv(out_dir / "validation_independence.csv", index=False)
    mv = markov_validation(data, seed=args.seed, groups=groups)
    pd.DataFrame([mv]).to_csv(out_dir / "validation_markov.csv", index=False)

    j = times.index(60)
    r60 = renewal_df.iloc[j]
    plot_event_count_distribution(sim_counts[:, j], 60, r60.mean_theory, r60.var_theory,
                                  rp_results.asymptotic_rate, name.upper(), fig_dir / "event_count_distribution.png")
    plot_markov_state_diagram(mc_results.transition_matrix, REGIME_NAMES, name.upper(),
                              fig_dir / "markov_state_diagram.png")
    print(f"[{name}] Renewal check at t=300s: E[N] sim/theory = {renewal_df.mean_sim.iloc[-1]:.2f}/"
          f"{renewal_df.mean_theory.iloc[-1]:.2f}, Var[N] sim/theory = {renewal_df.var_sim.iloc[-1]:.2f}/"
          f"{renewal_df.var_theory.iloc[-1]:.2f}; Poisson would give Var = {renewal_df.var_poisson.iloc[-1]:.2f}")
    print(f"[{name}] Markov check: Chapman-Kolmogorov(2-step) max err = {mv['chapman_kolmogorov_2_max_err']:.4f}, "
          f"2nd-order test p = {mv['second_order_p']:.3g} (effect TV = {mv['second_order_effect_tv']:.3f})")

    # 7. Save Publication Figures
    plot_fitted_distributions(f"{name.upper()} — Fitted Headway Distributions", counts, mcmc_params, fig_dir / "fitted_distributions.png")
    plot_empirical_cdf(f"{name.upper()} — Cumulative Distribution Functions", data, mcmc_params, fig_dir / "empirical_cdf.png")
    plot_goodness_of_fit_metrics(mcmc_metrics_dict, fig_dir / "goodness_of_fit.png")
    plot_simulation_validation(data, sim_samples, "Proposed Distribution", fig_dir / "simulation.png")

    # 8. Save Standardized Output CSVs
    res_df = pd.DataFrame(metric_rows).set_index("distribution")
    stat_cols = ["ks_D", "ks_p", "ks_2samp_D", "ks_2samp_p", "chi2", "chi2_dof", "chi2_p", "kl", "wasserstein", "wasserstein_time"]
    param_cols = [c for c in res_df.columns if c not in stat_cols]

    res_df[param_cols].to_csv(out_dir / "params.csv")
    res_df[["ks_D", "ks_p", "ks_2samp_D", "ks_2samp_p", "chi2", "chi2_dof", "chi2_p"]].to_csv(out_dir / "tests.csv")
    res_df[["kl", "wasserstein", "wasserstein_time"]].to_csv(out_dir / "metrics.csv")
    pd.DataFrame(posterior_rows).to_csv(out_dir / "posterior.csv", index=False)
    pd.DataFrame(diagnostics_rows).to_csv(out_dir / "diagnostics.csv", index=False)

    print(f"[{name}] Successfully generated all CSV tables and figures under {out_dir}/")
    return {
        "name": name,
        "n_headways": len(data),
        "mean": float(data.mean()),
        "res_df": res_df,
    }


def main():
    parser = argparse.ArgumentParser(description="Headway distribution modeling via MCMC.")
    parser.add_argument("--dataset", choices=["highd", "ngsim", "all"], default="all",
                        help="Target dataset: 'highd', 'ngsim', or 'all'")
    parser.add_argument("--data-dir", default="data", help="Root directory holding dataset folders")
    parser.add_argument("--path", help="Explicit path overriding data-dir for a single dataset")
    parser.add_argument("--location", default="i-80", help="Location filter for NGSIM (default 'i-80')")
    parser.add_argument("--likelihood", choices=["paper", "truncated"], default="paper",
                        help="'paper': the paper's un-truncated density (default, reproduces its tables); "
                             "'truncated': conditional on the observed [0.5, 25] s window")
    parser.add_argument("--real-only", action="store_true",
                        help="fail if a dataset's raw files are missing instead of using synthetic data")
    parser.add_argument("--lanes", type=lambda s: [int(x) for x in s.split(",")],
                        help="NGSIM only: keep these Lane_IDs, e.g. 1,2,3,4,5,6 (paper states no lane filter)")
    parser.add_argument("--period", type=int,
                        help="NGSIM only: keep one recording period (1 = first, 2 = second); "
                             "see `python -m headway.ngsim_breakdown`")
    parser.add_argument("--iters", type=int, default=10000, help="MCMC total iterations per chain (default 10,000)")
    parser.add_argument("--warmup", type=int, default=5000, help="MCMC warmup iterations per chain (default 5,000)")
    parser.add_argument("--chains", type=int, default=2, help="MCMC number of independent chains (default 2)")
    parser.add_argument("--synthetic-size", type=int, default=15000, help="Size of synthetic sample if raw dataset absent")
    parser.add_argument("--sample-paths", type=int, default=5, help="Number of realization sample paths to simulate (default 5)")
    parser.add_argument("--sim-time", type=float, default=300.0, help="Arrival process simulation horizon in seconds (default 300.0s)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--workers", type=int, default=None, help="Parallel worker processes (default: CPU count)")
    parser.add_argument("--out", default="outputs", help="Output directory")
    args = parser.parse_args()

    target_datasets = ["highd", "ngsim"] if args.dataset == "all" else [args.dataset]
    if args.path and len(target_datasets) != 1:
        parser.error("--path can only be specified when a single --dataset is selected.")

    all_dataset_results = []
    comparisons_dir = Path(args.out) / "comparisons"
    comparisons_dir.mkdir(parents=True, exist_ok=True)

    for dname in target_datasets:
        path = Path(args.path) if args.path else find_dataset_path(dname, args.data_dir)
        res = run_dataset_pipeline(dname, path, args)
        all_dataset_results.append(res)

    # Cross-dataset aggregated summaries
    cross_rows = []
    for item in all_dataset_results:
        for dist_name, row in item["res_df"].iterrows():
            cross_rows.append({
                "dataset": item["name"],
                "distribution": dist_name,
                "n_headways": item["n_headways"],
                **row.to_dict(),
            })

    if cross_rows:
        cross_df = pd.DataFrame(cross_rows)
        cross_df.to_csv(comparisons_dir / "cross_dataset_comparison.csv", index=False)
        cross_df.to_csv(Path(args.out) / "all_results.csv", index=False)

    print(f"\n=======================================================")
    print(f" PIPELINE COMPLETE: Wrote all outputs to {args.out}/")
    print(f"=======================================================\n")


if __name__ == "__main__":
    main()
