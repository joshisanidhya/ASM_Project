# Reproducing the paper on highD and NGSIM

This note describes the current executable implementation. Older reports in `docs/`
describe a previous three-dataset/MLE version and contain results that have not been
verified against this checkout. Current scope is **highD and NGSIM only**.

## What was corrected

- **Observation likelihood.** The sampler now uses `sum(log f(headway | parameters))`
  from the PDFs in paper Table 1, rather than a 49-bin likelihood. Equal observations
  are compressed using their exact multiplicities, with no rounding or subsampling.
  The proposed model, Gamma and shifted exponential use exact sufficient statistics;
  the other likelihoods evaluate the distinct observations.
- **Parameter support.** Both shifts have the Table 1 prior
  `Uniform(-10, minimum_observed_headway)`. The previous 0.4999 s ceiling excluded
  the published NGSIM log-normal shift 0.528 and exponential shift 0.558. Arbitrary
  finite bounds on positive parameters and normal-prior parameters were also removed.
- **Boundary sampling.** Metropolis-Hastings operates on log/logit coordinates with
  the change-of-variables Jacobian. Local posterior curvature initializes proposal
  covariance; adaptation remains limited to warmup. Estimates are posterior means,
  never optimizer results. The defaults remain 2 chains × 10,000 iterations,
  5,000 warmup and 5,000 retained draws per chain.
- **Diagnostics.** Constant chains fail convergence checks. Very narrow posteriors
  are no longer mistaken for constant chains. ESS uses paired positive monotone
  autocorrelations with between-chain variance.

The data loaders and the 1 Hz, leader-present, 0.5–25 s filters are unchanged.
The supplied run command already selects NGSIM period 1. No subset was selected
or parameter tuned to make Table 5 numbers match.

## What the paper leaves unspecified

Sections 3.2–3.3 describe MCMC with Table 1 PDFs and priors, but give no explicit
likelihood implementation. The observation-PDF likelihood is the direct
interpretation used here. Table 1 gives unconditioned baseline PDFs and a proposed
PDF normalized on `[0.5, infinity)`, so fitting does not add a `[0.5, 25]`
conditioning term. Histogram comparisons and simulation do condition on that
window so their probability masses sum to one. This distinction is explicit;
changing to a fully conditional likelihood would be a different modelling choice.

The normal prior `N(0, 10)` does not identify whether 10 is variance or standard
deviation. This implementation retains its existing standard-deviation-10
interpretation. The standard log-logistic PDF is used: its denominator exponent
is the shape parameter alpha; the beta exponent printed in Table 1 is inconsistent
with that density.

The paper cites reconstructed NGSIM I-80 trajectories, but does not give an exact
recording/lane selection. The loader reads the supplied CSV's `Time_Headway` field;
it cannot establish whether those trajectories have been reconstructed. Agreement
requires the same source data and subset, as well as the same estimation method.

## The large NGSIM Gamma Wasserstein discrepancy

The old `W` is a distance between sorted histogram **probabilities**, not headway
times. It is retained as `W_bin` in the console and `wasserstein` in CSV files for
comparison with earlier runs. A separate `wasserstein_seconds` column computes
Wasserstein-1 between weighted headway bin centers.

Equation 17 does not say which vectors the authors used. Similar numerical scale
does not establish that their metric is the sorted-bin proxy. In fact, for 49 bins,
natural-log KL and normalized probabilities p and q:

`W_bin = ||sort(p) - sort(q)||_1 / 49 <= ||p - q||_1 / 49 <= sqrt(2 KL(p||q)) / 49`.

For the paper's NGSIM Gamma KL of 0.1182, this upper bound is about **0.0099**,
less than its reported W of **0.0173**. Thus those two published values cannot both
come from this proxy on the same probability vectors. This is a limit on the
proxy interpretation, not proof of which metric the authors actually computed.
Matching that number by altering the fitted model would not resolve the ambiguity.

KS p-values remain nominal because parameters are fitted on the same observations
and headways may be serially dependent. Chi-square pooling retains the paper's
observed-count rule; it does not guarantee large expected counts. Interpret the
distance measures and convergence diagnostics alongside the p-values.

## Run and inspect

Run the same command on the system holding the raw datasets:

```powershell
uv run python run.py --period 1 --real-only
```

Each dataset now additionally exports:

- `headway_histogram.csv`: all 49 counts, useful for comparisons without raw data.
- `paper_comparison.csv`: fitted parameters and metrics, published Tables 3/5,
  metrics obtained with the **fixed published parameters on your data**, model
  mass within the retained window, and the proxy-bound compatibility check.
- `dataset_info.csv`: sample summaries, selection, likelihood, seed and chain settings.

Read `diagnostics.csv` and `posterior.csv` first. A nonconverged estimate is not
validated by a high acceptance rate. In `paper_comparison.csv`, if fixed Table 3
parameters still give different metrics, fitting alone cannot explain the gap;
data and metric conventions also matter. Lower KL than the paper is not itself
an error. Published parameter and metric values are used only for comparison and
the explicitly labelled synthetic fallback, never to fit real observations.

## Verification limits

Regression tests compare all seven likelihoods with independent SciPy PDFs,
check the transformation Jacobian, enforce the actual shift support, and compare
a million-observation shifted-exponential fit with its analytically tractable
posterior. Full-pipeline tests use explicitly synthetic data. The real highD and
NGSIM data are on another system, so empirical improvement must be checked there.

Validation in this checkout: all 64 tests passed. A separate synthetic run with
15,000 observations per dataset configuration, two chains, 10,000 iterations and
5,000 warmup completed for all 14 fits, with maximum R-hat below 1.005. These
synthetic results check the implementation, not reproduction of Table 5.
