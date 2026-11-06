# Vehicle headway modelling — highD and NGSIM

Implementation of *Modeling Headway in Heterogeneous and Mixed Traffic Flow:
A Statistical Distribution Based on a General Exponential Function*, with
renewal-process and traffic-regime Markov-chain extensions for the ASM course.

The executable pipeline supports **highD and NGSIM only**. It fits the proposed
distribution and six baselines using Metropolis-Hastings. There is no separate
MLE estimation pipeline in this version.

## Setup and run

Python 3.11 or later; dependencies and the lockfile are in this repository.

```powershell
uv sync
uv run python run.py --period 1 --real-only
```

Place highD recordings under `data/highd/` and the NGSIM CSV under `data/ngsim/`.
The highD loader uses `*_tracks.csv` plus matching recording metadata. The NGSIM
loader uses `Frame_ID`, `Preceding`, `Time_Headway`, `Location`, and
`Global_Time` when selecting a period. It defaults to I-80. Period 1 means the
first recording block; omit `--period` to use all blocks.

Both loaders retain leader-present observations at 1 Hz and headways in
**[0.5, 25] seconds**. `--real-only` prevents synthetic fallback when data is
missing. `--path` overrides discovery for a single `--dataset highd` or
`--dataset ngsim`. The code does not download data.

## Fitting and comparison with the paper

- Seven distributions: Proposed, Shifted Log-normal, Weibull, Log-logistic,
  Gamma, Burr XII and Shifted Exponential.
- Table 1 PDF likelihoods on individual observations and Table 1 priors.
  Exact repeated observations are compressed without rounding; 49 half-second
  bins are used for evaluation.
- Two chains × 10,000 iterations, with 5,000 warmup and 5,000 retained draws
  per chain. Point estimates are the pooled posterior means.
- Log/logit transformations with Jacobians and covariance-scaled proposals
  allow sampling close to parameter boundaries.
- Both shift priors end at the **observed minimum headway**, as stated in Table 1.
- Split R-hat, ESS, credible intervals and acceptance rates are exported.

[Paper reproduction notes](docs/PAPER_REPRODUCTION.md) explain the corrections,
likelihood interpretation, data requirements and unresolved details in the paper.

The console distinguishes `W_bin`, the legacy sorted-probability proxy, from
`W1` in seconds, the distance between weighted headway bin centers. The paper
does not specify the inputs to Equation 17, so the proxy is not a confirmed
reproduction of its Table 5 metric. See the mathematical bound in the notes.

## Outputs

Each dataset writes to `outputs/<dataset>/`:

| File | Contents |
| --- | --- |
| `dataset_info.csv` | Data source, selection, summaries and fit settings |
| `headway_histogram.csv` | The 49 observed bin counts |
| `params.csv`, `posterior.csv` | Estimates, uncertainty, R-hat and ESS |
| `diagnostics.csv` | Convergence flags, acceptance rates and runtime |
| `tests.csv`, `metrics.csv` | KS, chi-square, KL, probability proxy and W1 in seconds |
| `paper_comparison.csv` | Fitted results, published values, and published parameters evaluated on the current data |
| `simulation_summary.csv` | Synthetic sampling checks |
| `stochastic_*.csv`, `validation_*.csv` | Renewal and Markov analyses and assumption checks |
| `draws/`, `figures/` | Full/retained chains and plots |

Cross-dataset results are written to `outputs/comparisons/` and
`outputs/all_results.csv`. Convergence failures are flagged, not repaired by
substituting published parameter values.

## Stochastic extensions and reports

The renewal extension resamples empirical headways independently and compares
arrival-count moments with renewal theory. The three-state Markov chain uses
thresholds 1.5 s and 4 s, estimates transitions and stationary probabilities,
and checks independence and first-order assumptions. These are modelling
extensions: flattened trajectory rows do not preserve vehicle boundaries and
are not a detector arrival sequence.

```powershell
uv run --group report python -m headway.report
uv run --group report pytest -q
```

The report builder uses current CSVs and figures and rejects synthetic results
unless explicitly enabled for testing. Raw datasets are required to validate
agreement with published empirical results. A synthetic run only checks the
pipeline.

The documents in `docs/` other than `PAPER_REPRODUCTION.md` include historical
reports referring to Lyft, MLE and earlier numerical results. They are not
evidence of the current implementation or a current real-data run.
