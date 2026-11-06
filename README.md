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
| `animation_replay.npz` | Small filtered-row preview, generated headways, renewal paths and Markov evolution |

Cross-dataset results are written to `outputs/comparisons/` and
`outputs/all_results.csv`. Convergence failures are flagged, not repaired by
substituting published parameter values.

## Animated project walkthrough

Generate an offline, browser-playable presentation with the existing Python
dependencies (Matplotlib, NumPy and pandas). No video encoder or web server is needed.

```powershell
# Use saved results from your real-data pipeline run:
uv run python animate.py --dataset highd --open
uv run python animate.py --dataset ngsim --open

# Without datasets: run an explicitly synthetic demo, then animate its results:
uv run python animate.py --demo --dataset highd --open
```

The demo uses 2,000 synthetic observations and the normal 10,000/5,000 MCMC
settings. Its files stay under `outputs/animation_demo/`; it never reads `data/`
or replaces the real results under `outputs/highd/` and `outputs/ngsim/`.

Open `outputs/animations/highd.html` or `ngsim.html` to present or share the
real-result animation. Demo HTML is under `outputs/animation_demo/animations/`.
The HTML embeds its pictures and controls and works offline. Use Play/Pause,
Previous/Next, the stage menu, the timeline and speed selector. Space toggles
playback; arrow keys jump stages. A text explanation is available below the player.

Fourteen stages explain headway measurement, filtering, histogram construction,
PDF likelihoods and priors, actual MCMC chains, posterior uncertainty, all seven
fitted distributions, goodness-of-fit metrics, inverse-CDF sampling, simulation
validation, renewal arrivals, Markov regimes, assumption checks and output files.
The last three modelling stages are the ASM extensions, separate from the paper.

Use `--model "Burr"` (or another full distribution name) to change the detailed
trace/posterior view; the comparison still includes all seven models. Use
`--results outputs/verification` for a different saved run, `--out talk.html` for
a custom destination, or `--frames-per-stage 8` for a smaller file. Each stage
lasts four seconds at normal speed, independently of its frame count.

New pipeline runs save a compact `animation_replay.npz`. Older outputs can still
be animated if they contain the histogram and current metric CSVs: missing row
previews and renewal paths are explicitly marked, and generated headways are
regenerated from saved parameters/seed. Rerun the pipeline for a complete replay.
The animation never estimates raw rejection counts or reconstructs vehicle
positions from a histogram. Preprocessing examples are labelled illustrations.

### Download and export

The player has **Download interactive HTML** and **Download current frame (PNG)**
buttons. Add `--gif` to generate a looping GIF for slides and embed a third
**Download animation (GIF)** button. The interactive HTML remains self-contained,
including its GIF download; the GIF is also saved beside it.

```powershell
uv run python animate.py --dataset highd --gif --open

# Export an existing animation without rerunning fitting or rendering:
uv run python animate.py --export-existing outputs/animation_demo/animations/highd.html
uv run python animate.py --export-existing outputs/animation_demo/animations/ngsim.html
```

GIF preserves the 14-stage sequence (about 56 seconds, looping) but has no player
controls or audio. HTML retains playback controls and text explanations. GIF uses
Pillow, already installed with Matplotlib; no FFmpeg installation is required.

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
