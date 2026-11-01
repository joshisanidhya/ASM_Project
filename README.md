# Modeling Headway in Heterogeneous and Mixed Traffic Flow: A Statistical Distribution Based on a General Exponential Function

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests: Pytest](https://img.shields.io/badge/tests-35%20passed-brightgreen.svg)](tests/)

A reproducible, university-grade scientific implementation and research extension of the traffic headway modeling paper:

> **Reference**: Leungbootnak, Natchaphon. *Modeling Headway in Heterogeneous and Mixed Traffic Flow: A Statistical Distribution Based on a General Exponential Function.* Texas A&M University (2021).

---

## 1. Research Overview

Time headway—the elapsed time between successive vehicles in a traffic lane passing a longitudinal cross-section—is a foundational microscopic traffic flow variable governing highway capacity, safety, congestion, and autonomous driving algorithms. 

In mixed traffic environments where human-driven vehicles (HDVs) interact with autonomous vehicles (AVs), conventional parametric headway distributions (e.g., Log-normal, Weibull, Shifted Exponential) often struggle to capture both the sharp central modal clustering of tight highway car-following and the elongated, dispersed tails of urban stop-and-go maneuvers.

This project implements:
1. **The Proposed General Exponential Distribution**: $g(t) = b^{|t - a|}$, normalized on $[\alpha, \infty)$ with $\alpha = 0.5\text{ s}$.
2. **Six Classical Baseline Distributions**: Shifted Log-normal, Weibull, Log-logistic, Gamma, Burr Type XII, Shifted Exponential.
3. **MCMC Parameter Estimation**: 2 independent chains $\times$ 10,000 iterations (5,000 warmup, 5,000 retained posterior draws).
4. **Three Empirical Car-Following Datasets**:
   * **highD**: Uninterrupted highway flow, HDVs (25 Hz $\to$ 1 Hz).
   * **NGSIM**: Interrupted freeway flow on I-80, HDVs (10 Hz $\to$ 1 Hz).
   * **Lyft Level 5**: Urban mixed traffic, AV/HDV interaction (10 Hz $\to$ 1 Hz).
5. **Research Extension — Maximum Likelihood Estimation (MLE)**: Fast frequentist estimation pipeline with runtime, convergence, and parameter comparisons against MCMC ($70\times$ speedup).
6. **Stochastic Simulation Module**: Synthetic headway generation via inverse-CDF transform with empirical and theoretical validation.
7. **Rigorous Goodness-of-Fit Metrics**: 1-sample KS test, Chi-Square test with pooling ($O_i \ge 5$), KL divergence, and true 1D Distributional Wasserstein-1 distance (Vallender 1974).
8. **Stochastic Process Analysis (ASM Course Requirement)**:
   * **Vehicle Arrival Renewal Process**: Inter-arrival counting process $N(t)$, renewal function $m(t) \approx \lambda t$, capacity flow $q = 3600\lambda$, and multiple sample path realizations.
   * **3-State Car-Following Markov Chain**: Regime transitions (Close-Following, Platoon, Free-Flow), transition matrix $\mathbf{P}$, stationary distribution $\boldsymbol{\pi}$, mean recurrence times $\mu_i = 1/\pi_i$, and transient trajectories $\boldsymbol{\pi}(n)$.

---

## 2. Mathematical Model

### The Proposed General Exponential Distribution

The probability density function (PDF) based on the general exponential kernel $g(t) = b^{|t - a|}$ is defined on $[\alpha, \infty)$ with physical threshold $\alpha = 0.5\text{ s}$:

$$
f(t \mid a, b) = \frac{b^{|t - a|}}{\int_\alpha^\infty b^{|s - a|}\,ds}, \quad 0 < b < 1, \quad a \ge \alpha, \quad t \ge \alpha
$$

#### Model Parameters
- **Modal Location $a$**: Central tendency parameter representing the peak / most frequent time headway ($a \ge 0.5\text{ s}$).
- **Dispersion Parameter $b$**: Concentration parameter controlling the rate of decay away from the mode ($0 < b < 1$).
- **Physical Threshold $\alpha$**: Minimum physical following threshold ($\alpha = 0.5\text{ s}$).

#### Normalization Constant

The closed-form normalization constant $I = \int_\alpha^\infty b^{|t - a|}\,dt$ guarantees that the total area integrates to 1:

$$
I = \int_\alpha^\infty b^{|t - a|}\,dt = \begin{cases}
\dfrac{b^{a - \alpha} - 2}{\ln b}, & a > \alpha \\
\dfrac{-b^{\alpha - a}}{\ln b}, & a \le \alpha
\end{cases}
$$

*(Note: Because $0 < b < 1$, we have $\ln b < 0$ and $b^{a - \alpha} - 2 < 0$, ensuring $I > 0$ strictly.)*

#### Analytical Cumulative Distribution Function (CDF)

For the realistic traffic regime where $a > \alpha$:

$$
F(t) = \begin{cases}
0, & t < \alpha \\
\dfrac{b^{a - \alpha} - b^{a - t}}{b^{a - \alpha} - 2}, & \alpha \le t \le a \\
\dfrac{b^{a - \alpha} + b^{t - a} - 2}{b^{a - \alpha} - 2}, & t > a
\end{cases}
$$

For complete mathematical derivations, baseline equations, priors, and proofs, see [docs/MATHEMATICAL_MODEL.md](docs/MATHEMATICAL_MODEL.md).

---

## 3. Dataset Scope & Requirements

Per project specifications, this repository supports **three datasets exclusively**:

| Dataset | Regime | Sensors | AV Involved | Original Hz | Subsampled Hz | Primary Source |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **highD** | Uninterrupted Highway | Drone Video | No | 25 Hz | 1 Hz | [highD Dataset](https://www.highd-dataset.com/) |
| **NGSIM** | Interrupted Freeway (I-80) | Fixed Camera | No | 10 Hz | 1 Hz | [FHWA Data.gov](https://data.gov/) |
| **Lyft Level 5** | Interrupted Urban Road | LiDAR / Radar | Yes | 10 Hz | 1 Hz | [Level 5 Prediction](https://level-5.global/) |

*(Note: exiD and Waymo Open Motion Dataset are strictly excluded from all pipelines and documentation).*

### Data Placement Layout
Place downloaded raw files in the `data/` directory:
```
data/
├── highd/
│   ├── 01_tracks.csv
│   ├── 01_recordingMeta.csv
│   └── ...
├── ngsim/
│   └── Next_Generation_Simulation__NGSIM__Vehicle_Trajectories_and_Supporting_Data.csv
└── lyft/
    └── sample.zarr/ (or train.zarr/)
```
> **Synthetic Benchmark Fallback**: If raw datasets are not downloaded, the pipeline automatically generates synthetic benchmark samples calibrated to the paper's reported parameters, allowing full end-to-end execution and testing without gigabytes of data.

---

## 4. Environment Setup & Installation

### Requirements
* Python 3.11+ (Python 3.13 tested)
* NumPy, SciPy, Pandas, Matplotlib, Pytest
* Zarr `< 3` (for Lyft Level 5 Zarr stores)

### Quick Setup
```bash
# Clone the repository
git clone https://github.com/joshisanidhya/ASM_Project.git
cd ASM_Project

# Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install required dependencies
pip install numpy pandas scipy matplotlib pytest "zarr<3"
```

---

## 5. Execution Commands

### Run Full Pipeline Across All 3 Supported Datasets
```bash
# Runs MCMC + MLE + Simulation + Visualizations for highD, NGSIM, and Lyft
python run.py --dataset all --iters 10000 --warmup 5000
```

### Run on a Single Dataset
```bash
# Run on highD exclusively
python run.py --dataset highd --iters 10000 --warmup 5000

# Run on NGSIM (with I-80 location filter)
python run.py --dataset ngsim --location i-80 --iters 10000 --warmup 5000

# Run on Lyft Level 5 AV dataset
python run.py --dataset lyft --iters 10000 --warmup 5000
```

### Fast Demonstration / Synthetic Mode
```bash
# Quick demonstration run (1,000 iters, 500 warmup, synthetic benchmark data)
python run.py --dataset all --iters 1000 --warmup 500 --synthetic-size 5000 --workers 2
```

---

## 6. Output Files & Results Structure

All results are saved systematically under `outputs/`:
```
outputs/
├── highd/
│   ├── params.csv               # MCMC posterior mean parameters
│   ├── posterior.csv            # Posterior mean, sd, 95% CI, R-hat, ESS
│   ├── tests.csv                # KS and Chi-Square statistics and p-values
│   ├── metrics.csv              # KL divergence and Wasserstein distances
│   ├── diagnostics.csv          # Convergence status, R-hat, acceptance rates
│   ├── simulation_summary.csv   # Summary statistics & simulation validation
│   ├── mle_results.csv          # MLE parameters, runtime, log-likelihood, metrics
│   ├── stochastic_markov_chain.csv   # Markov transition matrix P, stationary vector pi, recurrence mu
│   ├── stochastic_renewal_process.csv # Renewal rate, highway capacity flow, Fano factor
│   ├── draws/                   # Compressed MCMC sample chains (.npz)
│   └── figures/
│       ├── fitted_distributions.png  # Paper Figure 5 histogram overlay
│       ├── empirical_cdf.png         # Empirical vs model CDFs
│       ├── traces.png                # Paper Figure 4b MCMC trace plot
│       ├── posterior.png             # Paper Figure 4a posterior densities
│       ├── mcmc_vs_mle.png           # Side-by-side MCMC vs MLE fit comparison
│       ├── runtime_comparison.png    # Runtime speedup bar chart
│       ├── goodness_of_fit.png       # KL and Wasserstein comparison bars
│       ├── simulation.png            # Empirical vs simulated validation
│       ├── arrival_process_sample_paths.png # Renewal process N(t) multiple sample paths
│       ├── markov_transition_heatmap.png    # 3-state transition probability heatmap
│       └── markov_state_probabilities.png   # Transient state convergence vs time
├── ngsim/ ...
├── lyft/ ...
└── comparisons/
    ├── cross_dataset_comparison.csv  # Combined results across all 3 datasets
    ├── mcmc_vs_mle.csv              # Detailed MCMC vs MLE comparative table
    └── runtime_comparison.csv       # Runtime benchmarks per distribution
```

---

## 7. Testing & Quality Assurance

A comprehensive test suite of 35 unit and integration tests verifies mathematical correctness, metrics, MCMC, MLE, simulation, stochastic processes, and dataset loaders:

```bash
# Run full pytest suite
python -m pytest -v
```

### Test Suite Structure
* `tests/test_headway.py`: Proposed CDF closed-form vs numerical integration, paper Eq. (2) literal agreement, all 7 distributions normalization, CDF derivative matching PDF.
* `tests/test_metrics.py`: KS test (1-sample & 2-sample), Chi-square pooling ($O_i \ge 5$), KL divergence properties, 1D distributional Wasserstein distance vs sorted-bin proxy.
* `tests/test_mcmc.py`: Metropolis-Hastings full history preservation, acceptance rate monitoring, Split $\hat{R}$, ESS, credible intervals.
* `tests/test_mle.py`: MLE parameter recovery on known synthetic distributions, runtime benchmarking, bound constraints.
* `tests/test_simulation.py`: Sampling bounds $[0.5, 25.0]$ s, reproducibility via seeds, statistical validation against theoretical CDFs.
* `tests/test_stochastic_process.py`: Markov transition matrix row-stochasticity, $n$-step powers, stationary invariance $\boldsymbol{\pi}\mathbf{P} = \boldsymbol{\pi}$, mean recurrence times $\mu_i = 1/\pi_i$, renewal counting process $N(t)$, and multiple sample path monotonicity.
* `tests/test_datasets.py`: 1 Hz subsampling, filtering, leader gap computation, synthetic highD, NGSIM, and Lyft Zarr fixtures.

---

## 8. Research Documentation & Deliverables

Detailed scientific documentation is provided in `docs/`:
* [docs/ADDITIONAL_INSTRUCTIONS_IMPLEMENTATION_CHECKLIST.md](docs/ADDITIONAL_INSTRUCTIONS_IMPLEMENTATION_CHECKLIST.md): Requirements matrix mapping every instruction from `ASM Project 2026M_instructions.pdf`.
* [docs/ASM_PROJECT_REPORT.md](docs/ASM_PROJECT_REPORT.md): Complete 10–12 page academic report draft covering all 18 required report sections.
* [docs/PRESENTATION_SLIDES_GUIDE.md](docs/PRESENTATION_SLIDES_GUIDE.md): 15-minute presentation guide with slide-by-slide script, timing, and terminal demonstration commands.
* [docs/RESEARCH_IMPLEMENTATION_AUDIT.md](docs/RESEARCH_IMPLEMENTATION_AUDIT.md): Technical audit, line-by-line inspection, and discrepancies analysis.
* [docs/MATHEMATICAL_MODEL.md](docs/MATHEMATICAL_MODEL.md): Rigorous mathematical derivations, baseline PDFs/CDFs, and metric formulas.
* [docs/DATASET_SCHEMA.md](docs/DATASET_SCHEMA.md): HighD, NGSIM, and Lyft Level 5 schemas, field definitions, and preprocessing rules.
* [docs/METHODOLOGY.md](docs/METHODOLOGY.md): MCMC algorithm, adaptive proposal mechanism, MLE extension, stochastic processes, and simulation pipeline.
* [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md): Paper-reported baseline results, experimental findings, MCMC vs MLE benchmarks, and simulation validation.

---

## 9. Limitations & Future Work

1. **Lyft Heading Corridor**: In the absence of bundled HD vector map lane polylines, leader vehicles in Lyft are identified via a lateral corridor ($\pm 1.8$ m) along the ego vehicle's heading vector. Future work can integrate HD map graph networks for curved intersection maneuvers.
2. **Speed Thresholding**: Headway proxy $h = \text{gap} / v$ becomes ill-posed at zero speed ($v = 0$). Vehicles traveling below $0.5$ m/s are excluded, matching paper recommendations.
3. **MCMC vs MLE Tradeoff**: MCMC provides full posterior parameter uncertainty at higher computational cost (~1 s), while MLE delivers near-instantaneous estimates (~0.02 s) suited for real-time vehicular control.
