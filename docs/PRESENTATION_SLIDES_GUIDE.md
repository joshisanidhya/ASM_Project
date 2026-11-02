# Applied Stochastic Modeling (ASM) 15-Minute Presentation Guide

## Project Title
**Modeling Headway in Heterogeneous and Mixed Traffic Flow: A Statistical Distribution Based on a General Exponential Function, Maximum Likelihood Estimation, and Stochastic Process Analysis**

**Course**: Applied Stochastic Modeling (ASM 2026M)  
**Presenter**: Sanidhya Joshi  
**Duration**: 15 Minutes (12 min presentation + 3 min Q&A)  
**Artifacts & Slides**: Includes slide outline, visual cues, speaking script, mathematical formulas, and terminal demonstration commands.

---

## Presentation Structure Overview

| Slide | Topic | Suggested Time | Core Takeaway |
| :--- | :--- | :--- | :--- |
| **Slide 1** | Title & Introduction | 0:00 – 0:45 (45s) | Research context, university ASM project, paper reference. |
| **Slide 2** | Motivation & Background | 0:45 – 1:45 (60s) | Why headway matters for highway capacity, safety, and mixed CAV traffic. |
| **Slide 3** | Problem Statement & Limitations | 1:45 – 2:45 (60s) | Multimodality, asymmetric skewness, and existing open-source metric flaws. |
| **Slide 4** | Datasets & Preprocessing | 2:45 – 3:45 (60s) | highD (Autobahn), NGSIM (US freeway), Lyft (urban AV). 1 Hz sampling, $h \in [0.5, 25]$s. |
| **Slide 5** | Proposed General Exponential Distribution | 3:45 – 5:00 (75s) | Kernel $g(t) = b^{|t-a|}$, normalization constant $I(a,b)$, and closed-form piecewise CDF. |
| **Slide 6** | Baseline Distributions | 5:00 – 5:45 (45s) | 6 benchmark models: Shifted Log-normal, Weibull, Log-logistic, Gamma, Burr, Shifted Exp. |
| **Slide 7** | Parameter Estimation: MCMC vs. MLE | 5:45 – 7:00 (75s) | Adaptive Metropolis-Hastings (2 chains $\times$ 10k iters) vs fast MLE (Nelder-Mead/L-BFGS-B). |
| **Slide 8** | Rigorous Evaluation Metrics | 7:00 – 8:00 (60s) | 1-sample KS, Chi-Square ($O_i \ge 5$ pooling), KL divergence, true 1D Wasserstein vs. sorted proxy. |
| **Slide 9** | Empirical Results & Comparative Fit | 8:00 – 9:30 (90s) | Proposed model dominates across all 3 datasets; modal shifts $a = 1.35\,\text{s} \to 1.54\,\text{s} \to 1.82\,\text{s}$. |
| **Slide 10** | Engineering Extension: MLE vs MCMC | 9:30 – 10:15 (45s) | $70\times$ speedup ($1.8\,\text{s} \to 0.02\,\text{s}$) with parameter delta $< 0.003$ for real-time ITS. |
| **Slide 11** | Stochastic Process: Vehicle Arrival Renewal Process | 10:15 – 11:15 (60s) | Counting process $N(t)$, asymptotic rate $\lambda = 1/\mathbb{E}[H]$, renewal function $m(t)$, sample paths. |
| **Slide 12** | Stochastic Process: Car-Following Markov Chain | 11:15 – 12:15 (60s) | 3 states: Close-Following, Platoon, Free-Flow. Transition matrix $\mathbf{P}$, $\boldsymbol{\pi}$, recurrence $\mu_i$. |
| **Slide 13** | Simulation Module & Synthetic Validation | 12:15 – 13:00 (45s) | Inverse-CDF transform sampling, synthetic validation tests (KS $p > 0.40$, Chi2 $p > 0.70$). |
| **Slide 14** | Discussion & Engineering Implications | 13:00 – 14:00 (60s) | Why proposed kernel wins, Fano factor $< 1$ disproves Poisson, Markov regime persistence. |
| **Slide 15** | Limitations, Conclusion & Q&A | 14:00 – 15:00 (60s) | Non-stationarity, vehicle pairing extensions, concluding remarks, opening Q&A. |

---

## Detailed Slide-by-Slide Content & Script

### Slide 1: Title & Introduction
- **Visuals**: Title banner, university logo, repository badge `github.com/joshisanidhya/ASM_Project`, paper citation: *Leungbootnak (2021), Texas A&M University*.
- **Presenter Script**:
  > "Good morning, Professor and colleagues. Today, I am presenting our Applied Stochastic Modeling research project: *Modeling Headway in Heterogeneous and Mixed Traffic Flow: A Statistical Distribution Based on a General Exponential Function, Maximum Likelihood Estimation, and Stochastic Process Analysis*.
  > Time headway is the foundational microscopic metric of traffic engineering. In this presentation, I will walk you through our full-spectrum reproduction of Leungbootnak's general exponential headway distribution, our technical resolution of critical metric flaws in prior implementations, our high-speed Maximum Likelihood Estimation extension, and our formulation of two core stochastic processes—Vehicle Arrival Renewal Processes and Car-Following Markov Chains."

---

### Slide 2: Motivation & Background
- **Visuals**: Split graphic: human driver in Autobahn platoon vs. Autonomous Vehicle (AV) sensing leader in urban traffic. Diagram of time headway definition: $h = t_{\text{rear, leader}} - t_{\text{rear, follower}} = \Delta y / v_{\text{follower}}$.
- **Key Points**:
  - Time headway governs roadway capacity: $q = 3600 / \mathbb{E}[H]$.
  - Microscopic safety: human reaction time bounds ($0.7 - 1.5\,\text{s}$) vs. AV reaction times ($0.1 - 0.3\,\text{s}$).
  - Mixed traffic flows create severe heterogeneity: sharp peaks from platoons combined with long exponential tails from free-flowing traffic.
- **Presenter Script**:
  > "Why does time headway matter? At a macroscopic level, mean headway determines highway capacity. At a microscopic level, headway dictates collision risk and safe following distances. As autonomous vehicles integrate with human drivers, traffic distributions can no longer be modeled by simple textbook exponentials. We observe sharp close-following peaks bounded by human physical limits alongside heavy tails from unconstrained drivers."

---

### Slide 3: Problem Statement & Existing Flaws
- **Visuals**: Side-by-side plot of empirical histogram vs. traditional Exponential distribution (showing severe mismatch). Callout box highlighting the mathematical flaws in previous codebases.
- **Key Points**:
  - Empirical headways are strictly positive and truncated at $\alpha = 0.5\,\text{s}$.
  - Conventional models (Gamma, Weibull) fail to decouple the peak mode from tail decay.
  - **Audit Finding 1**: Prior open-source code computed Wasserstein-1 distance by sorting probability bin heights ($\frac{1}{K}\sum |p_{(k)} - q_{(k)}|$), which completely erased the time axis!
  - **Audit Finding 2**: Chi-Square tests did not pool tail bins with expected counts $< 5$, violating Cochran's rule.
- **Presenter Script**:
  > "Prior studies and open-source attempts faced severe modeling and statistical challenges. Standard distributions lack the flexibility to capture the asymmetric cliff around typical following distances. Furthermore, during our technical audit, we identified two severe flaws in existing codebases: Chi-Square testing without low-frequency bin pooling, and an invalid Wasserstein metric that sorted probability amplitudes without considering the time axis. Our implementation fixes both issues rigorously."

---

### Slide 4: Datasets & Preprocessing
- **Visuals**: Three dataset cards with photos/diagrams:
  1. **highD**: German Autobahn aerial drone video, 6 lanes, high-speed free-flow.
  2. **NGSIM**: US-101 / I-80 California freeways, congested stop-and-go waves.
  3. **Lyft Level 5**: Palo Alto urban AV fleet, intersections, turns, high variance.
- **Key Points**:
  - Resampled to $1\,\text{Hz}$ to eliminate auto-correlation across video frames.
  - Domain bounds: $\alpha = 0.5\,\text{s}$ (reaction time lower bound), $\beta = 25.0\,\text{s}$ (upper traffic bound).
  - Preserved strict research scope: 3 datasets only (no exiD, no Waymo).
- **Presenter Script**:
  > "To ensure rigorous scientific validity, we evaluate three real-world datasets spanning different driving regimes: highD for disciplined German highway flow, NGSIM for congested California freeways, and Lyft Level 5 for complex urban autonomous driving. All trajectories were cleaned, harmonized to 1 Hz, and bounded between 0.5 and 25 seconds."

---

### Slide 5: Proposed General Exponential Distribution
- **Visuals**: Formula cards with highlighted components:
  - Kernel: $g(t) = b^{|t - a|}, \quad 0 < b < 1, \; a \ge \alpha$
  - Normalization: $I(a, b) = \frac{2 - b^{a - \alpha}}{-\ln b}$
  - PDF: $f(t) = \frac{-\ln b}{2 - b^{a - \alpha}} b^{|t - a|}$
  - Piecewise CDF curves showing left-branch exponential growth and right-branch exponential decay.
- **Presenter Script**:
  > "The centerpiece of the paper is the General Exponential Distribution. The kernel is $g(t) = b^{|t-a|}$, where parameter $a$ represents the modal headway where density peaks, and $b \in (0, 1)$ governs the exponential decay rate. By integrating across the left branch $[\alpha, a]$ and right branch $[a, \infty)$, we derive the exact closed-form normalization constant $I(a, b) = (2 - b^{a-\alpha}) / (-\ln b)$ and its piecewise CDF."

---

### Slide 6: Six Baseline Distributions
- **Visuals**: Table of 6 comparative benchmark distributions:
  1. Shifted Log-normal $(\mu, \sigma)$
  2. Weibull $(k, \lambda)$
  3. Log-logistic / Fisk $(\alpha_0, \beta_0)$
  4. Gamma $(k, \theta)$
  5. Burr Type XII $(c, k, \gamma)$
  6. Shifted Exponential $(\lambda)$
- **Key Points**:
  - All distributions left-truncated at $\alpha = 0.5\,\text{s}$ and normalized over $[\alpha, \beta]$.
- **Presenter Script**:
  > "To evaluate whether the proposed distribution truly offers superior performance, we benchmark it against six classical traffic flow models: Shifted Log-normal, Weibull, Log-logistic, Gamma, the three-parameter Burr distribution, and the memoryless Shifted Exponential. All baselines are strictly truncated and re-normalized over the exact same interval."

---

### Slide 7: Parameter Estimation: MCMC vs. MLE
- **Visuals**: Diagram comparing the two estimation pipelines:
  - **MCMC**: Random walk Metropolis-Hastings, 2 chains $\times$ 10,000 iterations, 5,000 warmup, target acceptance $\approx 23.4\%$, Gelman-Rubin $\hat{R} < 1.05$.
  - **MLE Extension**: Direct negative log-likelihood minimization via Nelder-Mead simplex and L-BFGS-B gradient polishing.
- **Presenter Script**:
  > "For parameter estimation, we implemented two distinct paradigms. First, the paper's Bayesian MCMC framework using adaptive Random Walk Metropolis-Hastings with two independent chains of 10,000 iterations and Gelman-Rubin convergence diagnostics. Second, as an engineering extension, we built a high-speed Maximum Likelihood Estimation pipeline using bounded Nelder-Mead and L-BFGS-B optimization."

---

### Slide 8: Evaluation Metrics & Flaw Resolution
- **Visuals**: Math formulas for the 4 evaluation metrics:
  - Continuous 1-sample KS test: $D_{\text{KS}} = \sup |F_{\text{emp}}(t) - F_{\text{model}}(t)|$
  - Chi-Square with adaptive pooling: $O_k, E_k \ge 5$
  - Kullback-Leibler Divergence: $D_{\text{KL}}(P \parallel Q)$
  - 1D Distributional Wasserstein-1 Metric: $W_1 = \int |F_P(t) - F_Q(t)|\,dt$ in seconds.
- **Presenter Script**:
  > "We evaluate every distribution using four rigorous metrics: the continuous one-sample Kolmogorov-Smirnov distance, the Chi-Square test with Cochran-compliant bin pooling, Kullback-Leibler divergence, and the true one-dimensional Wasserstein-1 distance in seconds. Our code also computes the paper's original sorted proxy so we can reproduce their exact reported numbers."

---

### Slide 9: Experimental Results & Comparative Fit
- **Visuals**: Screenshot of `fitted_distributions.png` and summary table:
  - **highD**: Proposed ($D_{\text{KS}} = 0.0212$, $W_1 = 0.048\,\text{s}$), Burr (0.0298), Log-normal (0.0384), Weibull (0.0451), Exp (0.1250).
  - Modal parameter shift: highD ($a = 1.35\,\text{s}$) $\to$ NGSIM ($a = 1.54\,\text{s}$) $\to$ Lyft ($a = 1.82\,\text{s}$).
- **Presenter Script**:
  > "Here are our empirical results. Across all three datasets, the Proposed General Exponential distribution decisively outperforms every baseline model, achieving the lowest KS distance of 0.0212 on highD and the lowest Wasserstein distance. Notice the fascinating physical trend in the modal parameter $a$: it shifts from 1.35 seconds on the disciplined German Autobahn to 1.54 seconds on congested California freeways, and out to 1.82 seconds for cautious urban autonomous vehicles."

---

### Slide 10: Engineering Extension: MLE vs. MCMC
- **Visuals**: Bar chart from `runtime_comparison.png` and overlay plot from `mcmc_vs_mle.png`.
  - MCMC runtime: $1.4 - 1.8\,\text{s}$ per dataset.
  - MLE runtime: $0.015 - 0.025\,\text{s}$ per dataset.
  - Speedup: $\approx 70\times$ faster!
  - Parameter agreement: $|a_{\text{MCMC}} - a_{\text{MLE}}| < 0.003$.
- **Presenter Script**:
  > "While MCMC provides full posterior credible intervals, its computational overhead prevents real-time deployment. Our MLE extension achieves identical parameter estimates—differing by less than 0.003—while executing 70 times faster, in just 20 milliseconds. This makes the model feasible for edge deployment on roadside sensors and adaptive traffic signals."

---

### Slide 11: Stochastic Process: Vehicle Arrival Renewal Process
- **Visuals**: Plot of `arrival_process_sample_paths.png` showing 5 realization paths of $N(t)$ alongside the theoretical renewal function $m(t) \approx \lambda t$.
- **Key Points**:
  - Inter-arrival times $H_i$ form a renewal process.
  - Asymptotic arrival rate $\lambda = 1 / \mathbb{E}[H]$.
  - Practical capacity flow: highD = 1,638 veh/h/lane; NGSIM = 1,304 veh/h/lane; Lyft = 677 veh/h/lane.
  - Fano factor $D = \sigma^2 / \mu^2 = 0.56 < 1$ (underdispersed, non-Poisson!).
- **Presenter Script**:
  > "Transitioning into our stochastic process requirements: vehicle arrivals at a detector naturally form a Renewal Counting Process $N(t)$. By the Elementary Renewal Theorem, the arrival rate converges to $\lambda = 1/\mathbb{E}[H]$, yielding highway capacity of 1,638 vehicles per hour per lane for highD. Crucially, the Fano factor is 0.56—well below 1—proving that vehicle arrivals are regularized rather than Poisson."

---

### Slide 12: Stochastic Process: Car-Following Markov Chain
- **Visuals**: Heatmap from `markov_transition_heatmap.png` and transient convergence curves from `markov_state_probabilities.png`.
  - 3 States: Close-Following ($<1.5\,\text{s}$), Platoon ($1.5-4\,\text{s}$), Free-Flow ($>4\,\text{s}$).
  - Stationary distribution $\boldsymbol{\pi} = (0.432, 0.449, 0.119)$.
  - Mean recurrence times: $\mu_{\text{Close}} = 2.31$ vehicles, $\mu_{\text{Platoon}} = 2.23$ vehicles.
- **Presenter Script**:
  > "We also formulated a 3-state Car-Following Markov Chain. Using the transition matrix $\mathbf{P}$, we compute the stationary distribution $\boldsymbol{\pi}$, which shows that 88% of Autobahn vehicles operate in close-following or platoon regimes. The mean recurrence time reveals that a close-following state recurs every 2.3 vehicles. The transient state probabilities show rapid convergence to stationarity within 5 to 7 vehicle transitions."

---

### Slide 13: Simulation Module & Synthetic Validation
- **Visuals**: Histogram and QQ overlay from `simulation.png`. Validation metrics table:
  - Inverse-CDF transform sampling algorithm.
  - 1-sample KS test: $p = 0.407$ (null hypothesis retained).
  - Chi-Square test: $p = 0.878$.
- **Presenter Script**:
  > "To demonstrate generative modeling, we built a simulation engine using inverse-CDF transform sampling. Generating 10,000 synthetic headways from the calibrated distribution, we executed formal statistical hypothesis tests: the KS test yields a p-value of 0.41 and Chi-Square yields 0.88, confirming flawless statistical fidelity."

---

### Slide 14: Discussion & Key Insights
- **Key Takeaways**:
  1. The General Exponential distribution succeeds because it naturally decouples the mode $a$ from tail decay $b$.
  2. MCMC is the gold standard for scientific uncertainty quantification; MLE is optimal for real-time edge computing.
  3. Traffic streams exhibit strong Markovian memory ($P_{ii} \approx 0.45 - 0.55$).
  4. Poisson arrivals fail in dense traffic; Renewal Theory is necessary.
- **Presenter Script**:
  > "To synthesize our findings: Leungbootnak's distribution succeeds because it decouples reaction-time thresholds from tail dispersion. In terms of engineering deployment, MCMC and MLE form a complementary pair—MCMC for safety-critical offline audits, and MLE for 20-millisecond real-time signal control. Finally, our stochastic process models confirm that traffic possesses significant Markovian memory, rendering traditional memoryless queueing models obsolete in congested highway settings."

---

### Slide 15: Limitations, Conclusion & Q&A
- **Visuals**: Summary takeaways box, GitHub QR code/link, acknowledgment of course instructor.
- **Key Points**:
  - Limitations: Assumes stationary traffic; future work should model time-varying flow $\lambda(t)$ via Markov-Modulated Poisson Processes (MMPP).
  - All code, 35 unit tests, CSVs, and figures are 100% reproducible via `python run.py --dataset all`.
- **Presenter Script**:
  > "In conclusion, this project provides a rigorous, verified, and complete reproduction and extension of the General Exponential headway model, bridging mathematical theory with real-world stochastic processes. All 35 tests pass, and all figures and tables are completely reproducible. Thank you, and I look forward to your questions."

---

## Live Demo Cheat-Sheet (During Q&A)

If asked by the instructor to demonstrate execution in the terminal:
```bash
# 1. Run all 35 unit tests in ~3 seconds
python -m pytest -v

# 2. Run the complete pipeline on highD with MCMC, MLE, Simulation, and Stochastic Processes
python run.py --dataset highd --iters 1000 --warmup 500 --synthetic-size 5000

# 3. View the generated stochastic process figures in outputs/highd/figures/
# - arrival_process_sample_paths.png
# - markov_transition_heatmap.png
# - markov_state_probabilities.png
```
