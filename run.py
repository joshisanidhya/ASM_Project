"""Fit the proposed + baseline headway distributions (Leungbootnak et al.).

    uv run python run.py                       # ngsim + highd, whichever is found under data/
    uv run python run.py --dataset highd       # one dataset from data/highd
    uv run python run.py --dataset lyft        # opt-in; needs `uv sync --extra lyft`

Expected layout (any dataset that is missing is skipped):
    data/ngsim/<export>.csv     NGSIM Data.gov export
    data/highd/NN_tracks.csv    highD recordings (+ NN_recordingMeta.csv)
    data/lyft/<name>.zarr       Lyft Level 5 Prediction zarr folder (not in the default run)

Writes, per dataset, outputs/<dataset>/{params,tests,metrics}.csv + fit.png
(paper Tables 3-5, Figure 5) and, across datasets, outputs/all_results.csv.
Uses every core by default (--workers 1 to run serially).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from headway.datasets import LOADERS
from headway.distributions import DISTS, EDGES
from headway.fit import fit_all


def find_path(dataset, root):
    """Locate a dataset's files under root/<dataset>; None if absent."""
    d = Path(root) / dataset
    if dataset == "ngsim":
        hits = sorted(d.glob("*.csv"), key=lambda p: -p.stat().st_size)  # biggest csv = the export
        return hits[0] if hits else None
    if dataset == "lyft":
        return next(iter(sorted(d.glob("*.zarr"))), None)
    return d if any(d.glob("*_tracks.csv")) else None


def run_dataset(name, path, a):
    # Imported here, not at the top: spawned workers re-import this file, and
    # matplotlib / scipy.stats would add seconds to every worker's start-up.
    from headway.diagnostics import split_rhat
    from headway.metrics import evaluate
    from headway.plots import plot_fits, plot_posterior

    kw = {"location": a.location} if name == "ngsim" and a.location else {}
    data = LOADERS[name](path, workers=a.workers, **kw)
    if len(data) == 0:
        raise SystemExit(f"{name}: no headways found in {path} (wrong --location?)")
    counts = np.histogram(data, bins=EDGES)[0].astype(float)
    print(f"\n{name}: {len(data):,} headways, mean {data.mean():.2f}s  ({path})")

    fits = fit_all(counts, data.min(), n_iter=a.iters, n_warmup=a.iters // 2, workers=a.workers)
    out = Path(a.out) / name
    (out / "draws").mkdir(parents=True, exist_ok=True)
    (out / "traces").mkdir(exist_ok=True)

    params, rows, post = {}, [], []
    for dname, (theta, draws) in fits.items():
        pnames, slug = DISTS[dname].params, dname.lower().replace(" ", "_")
        params[dname] = theta
        rows.append({"distribution": dname, **dict(zip(pnames, theta)),
                     **evaluate(DISTS[dname], theta, data, counts)})
        rhat = split_rhat(draws)
        flat = draws.reshape(-1, draws.shape[-1])
        post += [{"distribution": dname, "param": p, "mean": m, "sd": s, "rhat": r}
                 for p, m, s, r in zip(pnames, flat.mean(0), flat.std(0), rhat)]
        np.savez_compressed(out / "draws" / f"{slug}.npz", draws=draws, params=np.array(pnames))
        plot_posterior(f"{name}: {dname}", pnames, draws, out / "traces" / f"{slug}.png")
        flag = "" if (rhat < 1.05).all() else "  <-- not converged"
        print(f"  {dname:22s} KL={rows[-1]['kl']:.4f}  W={rows[-1]['wasserstein']:.4f}  "
              f"max R-hat={np.nanmax(rhat):.3f}{flag}")

    res = pd.DataFrame(rows).set_index("distribution")
    stats = ["ks_D", "ks_p", "chi2", "chi2_dof", "chi2_p", "kl", "wasserstein"]
    pd.DataFrame(post).to_csv(out / "posterior.csv", index=False)
    res.drop(columns=stats).to_csv(out / "params.csv")
    res[["ks_D", "ks_p", "chi2", "chi2_dof", "chi2_p"]].to_csv(out / "tests.csv")
    res[["kl", "wasserstein"]].to_csv(out / "metrics.csv")
    plot_fits(name, counts, params, out / "fit.png")
    print(f"  wrote {out}/")
    return res.assign(dataset=name, n_headways=len(data)).reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=[*LOADERS, "all"], default="all",
                    help="'all' = ngsim + highd (Lyft is opt-in)")
    ap.add_argument("--data-dir", default="data", help="root holding ngsim/, highd/, lyft/")
    ap.add_argument("--path", help="explicit path, overrides --data-dir (single dataset only)")
    ap.add_argument("--location", default="i-80", help="NGSIM only: Location to keep (paper: i-80)")
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=None, help="default: all cores")
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()

    names = ["ngsim", "highd"] if a.dataset == "all" else [a.dataset]
    if a.path and len(names) != 1:
        ap.error("--path needs a single --dataset")

    results = []
    for n in names:
        path = a.path or find_path(n, a.data_dir)
        if path is None:
            print(f"\n{n}: nothing under {Path(a.data_dir) / n}, skipped")
            continue
        results.append(run_dataset(n, path, a))
    if not results:
        raise SystemExit(f"no datasets found under {a.data_dir}/")

    allres = pd.concat(results, ignore_index=True)
    first = ["dataset", "distribution", "n_headways"]
    Path(a.out).mkdir(exist_ok=True)
    allres[first + [c for c in allres if c not in first]].to_csv(Path(a.out) / "all_results.csv", index=False)
    print(f"\nwrote {Path(a.out) / 'all_results.csv'}")


if __name__ == "__main__":
    main()
