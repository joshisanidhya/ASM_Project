# Technical Audit and Discrepancy Analysis

**Project**: Applied Stochastic Modeling (ASM) Research Project — Vehicle Headway Distribution  
**Reference Paper**: *Modeling Headway in Heterogeneous and Mixed Traffic Flow: A Statistical Distribution Based on a General Exponential Function* (Natchaphon Leungbootnak, Texas A&M University)  
**Repository**: `joshisanidhya/ASM_Project`  
**Auditor**: Senior Research Engineer and Scientific Computing Specialist  
**Date**: October 2026  

---

## 1. Executive Summary & Scope Constraint

This technical audit provides a line-by-line inspection of the existing codebase, an exact extraction and mathematical verification of the reference paper (`Paper.pdf`), an audit of dataset schemas (`All_5_Datasets_Schemas.txt`), and a prioritized roadmap for bringing this repository to university-grade, scientifically defensible standards.

### Dataset Scope Directive
Per explicit project instructions:
* **Included Datasets (3 Supported)**:
  1. **highD** (German Highway drone dataset, 25 Hz, uninterrupted flow, HDVs)
  2. **NGSIM** (FHWA I-80 freeway trajectory dataset, 10 Hz, interrupted/uninterrupted highway flow, HDVs)
  3. **Lyft Level 5** (AV prediction dataset in Zarr format, 10 Hz, urban interrupted mixed traffic)
* **Excluded Datasets**:
  * **exiD** and **Waymo Open Motion Dataset** are **strictly excluded** from all loaders, execution pipelines, configurations, documentation, and comparative analyses.

---

## 2. Existing Repository Architecture & File Audit

| File | Size | Role in Existing Repo | Technical Status & Audit Findings |
| :--- | :--- | :--- | :--- |
| `headway/distributions.py` | 4,425 B | Proposed & baseline distribution CDFs, priors, parameter definitions | **Partially sound, but has boundary issues**. Proposed CDF antiderivative is clever but returns negative values for $t < \alpha$ without clipping. `raw_bin_probs` computes unnormalized bin differences rather than conditioned on $[0.5, 25]$ s. Log-logistic prior uses rate/shape $\Gamma(0.5, 0.5)$. |
| `headway/mcmc.py` | 1,486 B | Custom NumPy Metropolis-Hastings random-walk sampler | **Incomplete & unmonitored**. Warmup draws are discarded inside the function, preventing full-trajectory trace plots (as shown in paper Figure 4b). No acceptance-rate tracking or logging. Proposal tuning uses heuristic updates. Differences from paper's PyMC3/Theano pipeline are unvalidated. |
| `headway/fit.py` | 2,696 B | Log-posterior construction and parallel MCMC fitting orchestration | **Functional but limited**. Both chains start from identical Nelder-Mead MAP estimates rather than dispersed starting points. Binned multinomial likelihood is used instead of raw point likelihood; this approximation is efficient but needs documented justification. |
| `headway/metrics.py` | 2,419 B | Goodness-of-fit: KS, Chi-Square, KL, Wasserstein | **CRITICAL FLAW IN WASSERSTEIN**. Line 56 computes `np.mean(np.abs(np.sort(p_obs) - np.sort(p_model)))`. This sorts the 49 bin probability masses and averages their differences, discarding temporal coordinates entirely. KS test uses random 2-sample simulation rather than exact continuous CDF testing. |
| `headway/diagnostics.py` | 613 B | Gelman-Rubin Split $\hat{R}$ convergence check | **Correct formula**, but lacks ESS (effective sample size), chain comparison diagnostics, parameter uncertainty summaries (credible intervals), and automated convergence warning thresholds. |
| `headway/parallel.py` | 899 B | Shared `ProcessPoolExecutor` wrapper | **Works**, handles Windows multiprocessing peculiarities by avoiding per-call pool instantiation. Needs fallback for single-core execution and graceful error recovery. |
| `headway/plots.py` | 1,399 B | Visualizations for trace plots and fitted histograms | **Minimal**. Only plots post-warmup traces; does not plot empirical vs. theoretical CDFs, MCMC vs. MLE comparisons, or publication-quality multi-panel diagnostics. |
| `headway/datasets/common.py` | 251 B | Headway clipping to $[\alpha, T_{max}]$ ($[0.5, 25]$ s) | **Correct**, matches paper section 4. |
| `headway/datasets/highd.py` | 1,012 B | highD CSV loader (1 Hz subsampling, `thw`) | **Sound basic logic**. Reads `thw` where `precedingId > 0` and `frame % 25 == 0`. Lacks schema validation, unit checks, and standalone synthetic fixture testing. |
| `headway/datasets/ngsim.py` | 821 B | NGSIM CSV loader (1 Hz subsampling, `Time_Headway`) | **Sound basic logic**. Filters `Preceding > 0`, `Frame_ID % 10 == 0`. Location filtering handles string case. Lacks synthetic fixture test and handling of alternative column casing or space vs. time headway verification. |
| `headway/datasets/lyft.py` | 3,495 B | Lyft Level 5 Zarr loader | **Approximation requires explicit documentation**. Headway is derived via bumper-to-bumper gap divided by speed along an ego-heading corridor ($\pm 1.8$ m). Does not use HD map lane polylines. Requires documentation of speed thresholding and limitations. |
| `run.py` | 5,604 B | CLI entrypoint | **Functional**, but lacks MLE extension, simulation commands, structured CSV exports according to user specifications, and includes references to incomplete workflows. |
| `tests/` | 3 files | Basic unit tests | Pass, but coverage is narrow; lacks tests for metric mathematical correctness, MLE, simulation, normalization bounds, and synthetic dataset loading. |

---

## 3. Mathematical Extraction & Verification from Paper

### 3.1 Proposed Headway Distribution
The paper proposes an exponential function with variable base $b$ and modal location $a$:
$$g(t) = b^{|t - a|}, \quad 0 < b < 1, \quad a \in \mathbb{R}$$
where:
* $a$: Central tendency parameter representing the most frequent headway value (mode).
* $b$: Dispersion parameter ($0 < b < 1$). Lower $b$ indicates tighter concentration around $a$; as $b \to 1$, the function flattens toward uniform.
* $\alpha$: Minimum physical headway threshold, set to $\alpha = 0.5$ s (human reaction time floor).

#### Normalization Constant
The normalization integral over $[\alpha, \infty)$ is derived as:

$$
\int_\alpha^\infty b^{|t-a|}\,dt = \begin{cases}
\dfrac{b^{a-\alpha} - 2}{\ln b}, & a > \alpha \\
\dfrac{-b^{\alpha-a}}{\ln b}, & a \le \alpha
\end{cases}
$$

Since $0 < b < 1$, $\ln b < 0$. Furthermore, when $a > \alpha$, $0 < b^{a-\alpha} < 1$, so $b^{a-\alpha} - 2 < 0$. Thus, the integral is strictly positive.

The normalized PDF on $[\alpha, \infty)$ is:

$$
f(t \mid a, b) = \frac{b^{|t-a|}}{\int_\alpha^\infty b^{|s-a|}\,ds} = \begin{cases}
\dfrac{\ln b \cdot b^{|t-a|}}{b^{a-\alpha} - 2}, & a > \alpha \\
-\ln b \cdot b^{t-\alpha}, & a \le \alpha
\end{cases} \quad (t \ge \alpha)
$$

#### Cumulative Distribution Function (CDF)
For $t \ge \alpha$:
- **Case 1: $a > \alpha$**:

$$
F(t) = \begin{cases}
\dfrac{b^{a-\alpha} - b^{a-t}}{b^{a-\alpha} - 2}, & \alpha \le t \le a \\
\dfrac{b^{a-\alpha} + b^{t-a} - 2}{b^{a-\alpha} - 2}, & t > a
\end{cases}
$$

- **Case 2: $a \le \alpha$**:

$$
F(t) = 1 - b^{t-\alpha}, \quad t \ge \alpha
$$

For $t < \alpha$, $F(t) = 0$.

#### Interval Probability Formula (Paper Equation 2)
For $\alpha \le t_1 < t_2$:

$$
P(t_1 \le h \le t_2) = \begin{cases}
\dfrac{b^{a-t_1} + b^{t_2-a} - 2}{b^{a-\alpha} - 2}, & \alpha \le t_1 < a < t_2 \\
\dfrac{b^{a-t_1} - b^{a-t_2}}{b^{a-\alpha} - 2}, & \alpha \le t_1 < t_2 \le a \\
\dfrac{b^{t_2-a} - b^{t_1-a}}{b^{a-\alpha} - 2}, & \alpha \le a \le t_1 < t_2 \\
\dfrac{b^{t_2-a} - b^{t_1-a}}{-b^{\alpha-a}}, & a < \alpha \le t_1 < t_2
\end{cases}
$$

### 3.2 Six Baseline Distributions (Paper Table 1)
All candidate distributions are defined with support $t > 0$ or $t > \gamma$:

1. **Shifted Log-normal**:
   $$f(t \mid \mu, \sigma, \gamma) = \frac{1}{(t-\gamma)\sigma\sqrt{2\pi}} \exp\left(-\frac{(\ln(t-\gamma)-\mu)^2}{2\sigma^2}\right), \quad t > \gamma, \ t > 0$$
   $$F(t) = \Phi\left(\frac{\ln(t-\gamma)-\mu}{\sigma}\right)$$
   Priors: $\mu \sim \mathcal{N}(0, 10)$, $\sigma \sim \text{Gamma}(0.5, 0.5)$, $\gamma \sim \mathcal{U}(-10, \min(t))$.

2. **Weibull**:
   $$f(t \mid \alpha, \beta) = \frac{\alpha}{\beta} \left(\frac{t}{\beta}\right)^{\alpha-1} \exp\left(-\left(\frac{t}{\beta}\right)^\alpha\right), \quad t \ge 0$$
   $$F(t) = 1 - \exp\left(-\left(\frac{t}{\beta}\right)^\alpha\right)$$
   Priors: $\alpha, \beta \sim \text{Gamma}(0.5, 0.5)$.

3. **Log-logistic**:
   $$f(t \mid \alpha, \beta) = \frac{\frac{\alpha}{\beta}(t/\beta)^{\alpha-1}}{\left[1 + (t/\beta)^\alpha\right]^2}, \quad t > 0$$
   $$F(t) = \frac{1}{1 + (t/\beta)^{-\alpha}}$$
   Priors: $\alpha, \beta \sim \text{Gamma}(0.5, 0.5)$.
   *(Note: The paper text had a typographical exponent error $\beta$ instead of $\alpha$ in the denominator; the standard Fisk form with exponent $\alpha$ is mathematically consistent and used in our implementation).*

4. **Gamma**:
   $$f(t \mid \alpha, \beta) = \frac{\beta^\alpha}{\Gamma(\alpha)} t^{\alpha-1} e^{-\beta t}, \quad t > 0$$
   $$F(t) = \frac{\gamma(\alpha, \beta t)}{\Gamma(\alpha)} = P(\alpha, \beta t)$$
   Priors: $\alpha, \beta \sim \text{Gamma}(0.5, 0.5)$ ($\beta$ is rate parameter).

5. **Burr (Type XII)**:
   $$f(t \mid \alpha, \beta, \lambda) = \frac{\alpha\beta}{\lambda} \left(\frac{t}{\lambda}\right)^{\alpha-1} \left[1 + \left(\frac{t}{\lambda}\right)^\alpha\right]^{-\beta-1}, \quad t > 0$$
   $$F(t) = 1 - \left[1 + \left(\frac{t}{\lambda}\right)^\alpha\right]^{-\beta}$$
   Priors: $\alpha, \beta, \lambda \sim \text{Gamma}(0.5, 0.5)$.

6. **Shifted Exponential**:
   $$f(t \mid \lambda, \gamma) = \lambda e^{-\lambda(t-\gamma)}, \quad t \ge \gamma, \ t \ge 0$$
   $$F(t) = 1 - e^{-\lambda(t-\gamma)}$$
   Priors: $\lambda \sim \text{Gamma}(0.5, 0.5)$, $\gamma \sim \mathcal{U}(-10, \min(t))$.

---

## 4. Parameter Estimation Methodology Audit

### 4.1 Paper MCMC Specification
* **Software**: PyMC3 with Theano backend.
* **Sampler**: Random-Walk Metropolis-Hastings.
* **Chains**: 2 independent chains.
* **Iterations**: 10,000 per chain (5,000 warmup/burn-in, 5,000 retained posterior draws).
* **Point Estimates**: Posterior mean of the 5,000 retained samples per chain ($10,000$ total pooled samples).

### 4.2 Audit of Repository's Custom NumPy Metropolis Implementation
Because PyMC3/Theano is deprecated and lacks Python 3.13 compatibility, the repository implements a custom NumPy Metropolis algorithm. The audit reveals several differences and areas for enhancement:

1. **Warmup Discard**: `metropolis()` only returns `draws[n_warmup:]`. This prevents generating full trace plots from iteration 0 to 10,000 (as in paper Figure 4b).
2. **Missing Acceptance Rate**: The sampler does not record or return the post-warmup acceptance rate.
3. **Identical Initialization**: Both chains start at the identical Nelder-Mead MAP estimate `x0`. Proper MCMC diagnostic practice requires overdispersed starting states to verify that chains converge to the same stationary distribution.
4. **Binned Likelihood Approximation**: The custom implementation uses a binned multinomial log-likelihood evaluated over the 49 bins:
   $$\ln \mathcal{L}(\theta) = \sum_{k=1}^{49} O_k \ln p_k(\theta)$$
   This reduces evaluation cost from $\mathcal{O}(N)$ to $\mathcal{O}(49)$ per step. While fast and statistically consistent with the 0.5 s bin resolution of the paper, this differs from continuous point likelihood $\sum_{i=1}^N \ln f(t_i \mid \theta)$. Both should be supported and compared.

---

## 5. Statistical Validation & Goodness-of-Fit Audit

### 5.1 The Wasserstein Distance Issue
**Paper Equation (17)**:
$$W_p(P, Q) = \left(\frac{1}{n}\sum_{i=1}^n \|X_{(i)} - Y_{(i)}\|^p\right)^{1/p}$$
citing Vallender (1974) (*Calculation of the Wasserstein Distance between Probability Distributions on the Line*).

**Existing Code (`headway/metrics.py:56`)**:
```python
w = float(np.mean(np.abs(np.sort(p_obs) - np.sort(p_model))))
```

#### Why the Existing Code is Defective:
* The existing code sorts the **49 bin probability masses** $p_{obs}$ and $p_{model}$ and computes the mean absolute difference between sorted probabilities.
* This completely ignores the **horizontal headway axis** (seconds). Two distributions with identical shapes located at 1 second vs. 20 seconds would yield $W = 0$ under the existing code!
* By Vallender's theorem on $\mathbb{R}$, the true 1D Wasserstein-1 metric between probability distributions $P$ and $Q$ is:
  $$W_1(P, Q) = \int_{-\infty}^\infty |F_P(t) - F_Q(t)| dt$$
  which has units of **seconds** (representing expected transportation cost along the headway domain).
* On the 49 discretized intervals, this corresponds to:
  $$W_1(P, Q) = \sum_{k=1}^{49} |F_{obs}(t_k) - F_{model}(t_k)| \Delta t$$
  or computed via `scipy.stats.wasserstein_distance`.
* In our updated implementation, we will compute both the rigorous distributional Wasserstein distance $W_1(P, Q) = \int |F_{obs}(t) - F_{model}(t)| dt$ and provide an explanation of how the paper's reported values arose.

### 5.2 Kolmogorov-Smirnov (KS) Test
* **Paper Equation (14)**:
  $$D = \sup_x |F_{obs}(x) - F_{model}(x)|$$
* In `headway/metrics.py`, KS is evaluated by generating a synthetic pseudo-sample of size $N$ and running `scipy.stats.ks_2samp(data, sim)`.
* This introduces Monte Carlo sampling noise into the statistic and p-value.
* **Correction**: Implement the exact 1-sample KS test against the analytic/numerical CDF `scipy.stats.kstest(data, lambda t: dist.trunc_cdf(t, theta))`, while also retaining the 2-sample method if sample comparison is desired.

### 5.3 Chi-Square Goodness-of-Fit Test
* **Paper Equation (15)**:
  $$\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}$$
* Grouping rule: Bins with $O_i < 5$ must be pooled with adjacent bins until $O_i \ge 5$.
* Degrees of freedom: $df = K_{\text{pooled}} - 1 - p$ (where $p$ is the number of estimated parameters).
* Expected frequencies: $E_i = N \cdot p_i$, where $\sum p_i = 1$ under the truncated distribution.

### 5.4 Kullback-Leibler (KL) Divergence
* **Paper Equation (16)**:
  $$D_{KL}(P \parallel Q) = \sum_{x \in X} P(x) \ln\left(\frac{P(x)}{Q(x)}\right)$$
* Requires proper handling of $P(x) = 0$ ($0 \ln 0 = 0$) and regularization for $Q(x) \to 0$ to prevent numerical overflow.

---

## 6. Dataset Preprocessing & Loader Audit

| Dataset | Paper Specs | Existing Loader Audit | Actions Required |
| :--- | :--- | :--- | :--- |
| **highD** | 25 Hz drone recordings, highway, HDVs. Subsample to 1 Hz, filter $[0.5, 25]$ s. | Implemented in `headway/datasets/highd.py`. Reads `thw` where `precedingId > 0` and `frame % 25 == 0`. | Validate schema, preserve working logic, add synthetic test fixture and standardized return format. |
| **NGSIM** | 10 Hz video tracking, I-80 freeway, HDVs. Subsample to 1 Hz, filter $[0.5, 25]$ s. | Implemented in `headway/datasets/ngsim.py`. Reads `Time_Headway` where `Preceding > 0` and `Frame_ID % 10 == 0`. | Validate units (seconds vs feet), confirm location filtering, add synthetic test fixture and standardized return format. |
| **Lyft Level 5** | 10 Hz Zarr prediction dataset, urban, mixed HDV/AV. Subsample to 1 Hz, filter $[0.5, 25]$ s. | Implemented in `headway/datasets/lyft.py`. Derives headway via leader gap / ego speed along lateral corridor. | Audit Zarr schema, document headway proxy limitations (`gap / speed`), add synthetic Zarr fixture and standardized return format. |

---

## 7. Research Extension: Maximum Likelihood Estimation (MLE)

As an applied research extension beyond the paper's MCMC approach, we design a dedicated MLE estimation pipeline:
* **Formulation**:
  $$\hat{\theta}_{\text{MLE}} = \arg\max_\theta \sum_{k=1}^{49} O_k \ln p_k(\theta) \quad \text{subject to parameter bounds}$$
* **Optimization**: SciPy `minimize(..., method='L-BFGS-B' / 'Nelder-Mead')` with explicit parameter bounds:
  * Proposed: $a \in [\alpha, 15.0]$, $b \in (10^{-5}, 0.9999)$
  * Shifted Log-normal: $\mu \in \mathbb{R}, \sigma > 0, \gamma < \min(t)$
  * Weibull: $\alpha > 0, \beta > 0$
  * Log-logistic: $\alpha > 0, \beta > 0$
  * Gamma: $\alpha > 0, \beta > 0$
  * Burr: $\alpha > 0, \beta > 0, \lambda > 0$
  * Shifted Exponential: $\lambda > 0, \gamma < \min(t)$
* **Comparison Dimensions**:
  * Estimated parameters
  * Execution runtime (seconds)
  * Convergence status
  * Log-likelihood
  * Goodness-of-fit metrics (KS, $\chi^2$, KL, Wasserstein)
  * Parameter comparison plots and summary tables.

---

## 8. Stochastic Simulation Module

A dedicated simulation module (`headway/simulation.py`) will be implemented to:
1. Sample synthetic headway observations from all seven fitted distributions using inverse-CDF transform or rejection sampling on $[\alpha, T_{max}]$.
2. Support configurable sample sizes and reproducible random seeds.
3. Compute empirical vs. simulated summary statistics (mean, standard deviation, median, IQR, percentiles).
4. Perform two-sample KS and Wasserstein validation between simulated and observed data.
5. Generate comparison plots (overlaid histograms and empirical CDFs).

---

## 9. Prioritized Implementation Roadmap

1. **Phase 1: Core Mathematical Foundation & Distribution Validation**
   * Refine `headway/distributions.py` for exact analytical CDF, boundary clipping, and numerical stability.
   * Add analytical vs. numerical integration tests.
2. **Phase 2: Correct Goodness-of-Fit Metrics**
   * Fix Wasserstein distance to true 1D distributional transport metric ($W_1(P, Q) = \int |F_P - F_Q| dt$).
   * Refine 1-sample KS test and bin-merged Chi-square calculations.
3. **Phase 3: Parameter Estimation Pipeline (MCMC + MLE)**
   * Enhance `headway/mcmc.py` to record full traces (including warmup), track acceptance rates, and support diagnostic metrics.
   * Implement `headway/mle.py` for Maximum Likelihood Estimation across all 7 distributions.
   * Implement split $\hat{R}$, ESS, and parameter uncertainty summaries in `headway/diagnostics.py`.
4. **Phase 4: Dataset Loaders Standardization (highD, NGSIM, Lyft Level 5)**
   * Standardize loader interface returning clean NumPy arrays with metadata.
   * Ensure exiD and Waymo are completely omitted.
   * Create realistic synthetic fixtures for testing all three loaders.
5. **Phase 5: Stochastic Simulation Module**
   * Build `headway/simulation.py` with reproducible sampling, summary statistics, and validation tests.
6. **Phase 6: Visualization and Automated Reporting**
   * Enhance `headway/plots.py` for publication-quality figures: fitted curves, traces, posterior distributions, MCMC vs. MLE comparisons, and simulation CDFs.
   * Update CLI `run.py` to support MCMC, MLE, simulation, and output generation (`outputs/` directory structure).
7. **Phase 7: Comprehensive Test Suite & Documentation**
   * Write comprehensive unit and integration tests across distributions, metrics, MCMC, MLE, simulation, and datasets.
   * Create complete documentation: `README.md`, `docs/MATHEMATICAL_MODEL.md`, `docs/DATASET_SCHEMA.md`, `docs/METHODOLOGY.md`, `docs/EXPERIMENTS.md`.
