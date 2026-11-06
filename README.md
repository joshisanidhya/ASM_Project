# Stochastic Modeling and Simulation of Vehicle Headways on Highways

Bayesian headway distribution (Leungbootnak et al., "Modeling Headway in Heterogeneous and Mixed Traffic Flow: A Statistical Distribution
Based on a General Exponential Function"), a renewal arrival process and a regime Markov chain, on the **highD** and **NGSIM (I-80)**
trajectory datasets. The research paper is in [`report/`](report/) (`main.tex`, compiled `main.pdf`); see
[`report/REVISION_NOTES.md`](report/REVISION_NOTES.md) for the corrections made and the open limitations.

## What is implemented
* `headway/distributions.py`: the proposed density f(t|a,b) ∝ b^|t-a| (normalised on [0.5, ∞)) and six baselines (shifted log-normal, Weibull,
  log-logistic, gamma, Burr XII, shifted exponential), window-conditioned PDFs/CDFs on [0.5, 25] s.
* `headway/mcmc.py`, `headway/fit.py`: adaptive random-walk Metropolis-Hastings (2 chains x 10,000, 5,000 warmup), binned multinomial posterior with two likelihoods:
  `paper` (the published, un-truncated one) and `truncated` (conditioned on the observation window; the correct one), log-gap reparametrisation of the shifted-exponential threshold.
* `headway/metrics.py`, `headway/diagnostics.py`: KS (one- and two-sample), chi-square with bin merging, KL divergence, two Wasserstein variants (sorted bin-probability vectors, and the time-axis distance in seconds), split R-hat, ESS.
* `headway/datasets/`: highD and NGSIM loaders (1 Hz sampling, leader present, headway in [0.5, 25] s) that also label *trajectory runs*, so Markov transitions and autocorrelations never cross a vehicle or a time gap.
* `headway/stochastic_process.py`, `headway/validation.py`: renewal counting process, 3-state regime Markov chain, renewal/Poisson comparison, Chapman-Kolmogorov and first-vs-second-order checks.
* `headway/detector.py`: virtual-detector inter-arrival times reconstructed from the trajectories, their autocorrelation and the count dispersion in windows.
* `headway/make_report_assets.py`, `headway/make_code_appendix.py`: write every number, table, figure and listing used by the paper from the output files.

## Data
Place the raw files under `data/` (not tracked):
```
data/highd/*_tracks.csv, *_recordingMeta.csv      (highD, 60 recordings)
data/ngsim/<NGSIM Data.gov export>.csv            (largest *.csv in the folder is used)
```
If a dataset is missing, `run.py` falls back to synthetic samples unless `--real-only` is given (never used for the paper).

## Run
```bash
pip install numpy pandas scipy matplotlib pytest          # or: uv sync
python run.py --dataset all --likelihood truncated --period 1 --workers 6 --out outputs/truncated
python run.py --dataset all --likelihood paper     --period 1 --workers 6 --out outputs/paper
python -m headway.detector --highd data/highd --ngsim data/ngsim/<file>.csv --out outputs/detector
python -m pytest -q tests                                  # 52 tests
python -m headway.make_report_assets && python -m headway.make_code_appendix --out report
```
`--period 1` selects the first NGSIM recording period (see the paper, Section 3.4). Use a moderate `--workers` value: every worker holds its own copy of the data it loads.

Outputs per dataset (`outputs/<run>/<dataset>/`): `params.csv`, `metrics.csv`, `tests.csv`, `posterior.csv`, `diagnostics.csv`, `simulation_summary.csv`,
`stochastic_markov_chain.csv`, `stochastic_renewal_process.csv`, `validation_*.csv`, `dataset_info.csv`, `draws/`, `figures/`.

`docs/` contains earlier working notes from before the paper was written; where they disagree with the paper, the paper and `report/REVISION_NOTES.md` apply
(for example, Lyft Level 5 and a maximum-likelihood extension mentioned there are not part of the current code base).
