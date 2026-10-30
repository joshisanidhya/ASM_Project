"""Fit the proposed + baseline headway distributions on one dataset.

    uv run python run.py --dataset ngsim --csv data/raw/ngsim.csv --location i-80

Writes outputs/<dataset>/{params,tests,metrics}.csv (paper Tables 3-5) and fit.png (Figure 5).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from headway.datasets import LOADERS
from headway.distributions import DISTS, EDGES
from headway.fit import fit
from headway.metrics import evaluate
from headway.plots import plot_fits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=LOADERS, default="ngsim")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--location", help="NGSIM Location filter, e.g. i-80")
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()

    kw = {"location": a.location} if a.location else {}
    data = LOADERS[a.dataset](a.csv, **kw)
    counts = np.histogram(data, bins=EDGES)[0].astype(float)
    print(f"{a.dataset}: {len(data):,} headways, mean {data.mean():.2f}s")

    params, rows = {}, []
    for name, dist in DISTS.items():
        theta, _ = fit(dist, counts, data.min(), n_iter=a.iters, n_warmup=a.iters // 2)
        params[name] = theta
        rows.append({"distribution": name, **dict(zip(dist.params, theta)),
                     **evaluate(dist, theta, data, counts)})
        print(f"  {name:22s} KL={rows[-1]['kl']:.4f}  W={rows[-1]['wasserstein']:.4f}")

    res = pd.DataFrame(rows).set_index("distribution")
    out = Path(a.out) / a.dataset
    out.mkdir(parents=True, exist_ok=True)
    res[[c for c in res if c not in {"ks_D", "ks_p", "chi2", "chi2_dof", "chi2_p", "kl", "wasserstein"}]].to_csv(out / "params.csv")
    res[["ks_D", "ks_p", "chi2", "chi2_dof", "chi2_p"]].to_csv(out / "tests.csv")
    res[["kl", "wasserstein"]].to_csv(out / "metrics.csv")
    plot_fits(a.dataset, counts, params, out / "fit.png")
    print(f"wrote {out}/")


if __name__ == "__main__":
    main()
