# Research Methodology: Estimation, Evaluation, and Simulation

**Project**: Applied Stochastic Modeling (ASM) Research Project — Vehicle Headway Distribution  
**Reference Paper**: *Modeling Headway in Heterogeneous and Mixed Traffic Flow: A Statistical Distribution Based on a General Exponential Function* (Natchaphon Leungbootnak, Texas A&M University)  

---

## 1. Overview of the Research Pipeline

```
Raw Car-Following Trajectories (highD, NGSIM, Lyft Level 5)
                     │
                     ▼
       Standardized Preprocessing (1 Hz subsampling, [0.5, 25.0]s filtering)
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
  [Paper Core]             [Research Extension]
  Bayesian MCMC            Maximum Likelihood Estimation
  (Metropolis-Hastings)    (Numerical Optimization)
  2 chains x 10,000 draws  L-BFGS-B / Nelder-Mead
         │                       │
         └───────────┬───────────┘
                     ▼
         Statistical Validation & Metrics
         - 1-Sample & 2-Sample KS Test
         - Chi-Square Goodness-of-Fit (Oi >= 5 pooling)
         - Kullback-Leibler Divergence
         - True 1D Distributional Wasserstein-1 Distance
                     │
                     ▼
         Stochastic Simulation Module
         (Inverse-CDF Sampling, Empirical Validation)
                     │
                     ▼
         Outputs: Machine-Readable CSVs & Publication Visualizations
```

---

## 2. Parameter Estimation: Bayesian MCMC (Paper Reproduction)

### 2.1 The MCMC Sampling Scheme
The paper estimates parameters using Markov Chain Monte Carlo (MCMC) with the random-walk Metropolis-Hastings algorithm:
* **Number of Chains**: $C = 2$ independent Markov chains.
* **Total Iterations**: $N_{\text{iter}} = 10,000$ iterations per chain.
* **Warmup / Burn-in**: $N_{\text{warmup}} = 5,000$ iterations (discarded for posterior estimation).
* **Retained Posterior Draws**: $N_{\text{post}} = 5,000$ draws per chain ($10,000$ total pooled samples).
* **Point Estimates**: Calculated as the posterior mean of the pooled $10,000$ post-warmup draws:
  $$\hat{\theta} = \frac{1}{C \cdot N_{\text{post}}} \sum_{c=1}^C \sum_{i=1}^{N_{\text{post}}} \theta^{(c, i)}$$

### 2.2 Likelihood Formulation
For $N$ observations binned into the 49 half-second intervals $[t_{k-1}, t_k]$, the observed frequency counts are $\mathbf{O} = [O_1, \dots, O_{49}]$. Under the candidate distribution conditional on $[0.5, 25.0]$ s, the model probability mass per bin is:
$$p_k(\theta) = F_{\text{trunc}}(t_k \mid \theta) - F_{\text{trunc}}(t_{k-1} \mid \theta)$$
The binned multinomial log-likelihood is:
$$\ln \mathcal{L}(\theta \mid \mathbf{O}) = \sum_{k=1}^{49} O_k \ln\left(\max(p_k(\theta), 10^{-300})\right)$$
Combined with the prior log-density $\ln p(\theta)$ (from paper Table 1), the log-posterior is:
$$\ln p(\theta \mid \mathbf{O}) = \ln p(\theta) + \ln \mathcal{L}(\theta \mid \mathbf{O}) + \text{constant}$$

### 2.3 Adaptive Proposal Mechanism & Ergodicity
* **Initial Proposal Covariance**:
  $$\Sigma_0 = \text{diag}\left((0.1 \cdot \max(|x_0|, 0.5))^2\right), \quad L_0 = \text{cholesky}(\Sigma_0)$$
* **Proposal Step**:
  $$\theta^* = \theta^{(t)} + s \cdot L \cdot \mathbf{z}, \quad \mathbf{z} \sim \mathcal{N}(\mathbf{0}, I_d)$$
  where $s = 2.38 / \sqrt{d}$ is the optimal Gelman-Roberts-Gilks scaling factor for dimension $d$.
* **Acceptance Probability**:
  $$\alpha = \min\left(1, \exp\left(\ln p(\theta^* \mid \mathbf{O}) - \ln p(\theta^{(t)} \mid \mathbf{O})\right)\right)$$
* **Warmup Adaptation**:
  * Step scale $s$ is tuned adaptively every 50–100 iterations during warmup to steer acceptance toward the theoretical optimum $\sim 23.4\%$.
  * Proposal covariance $\Sigma$ is re-estimated from the empirical sample history during warmup with Tikhonov regularization ($\Sigma + 10^{-8} I$).
  * Adaptation strictly ceases after warmup ($i \ge 5000$) to preserve Markov chain ergodicity and detailed balance.

### 2.4 Convergence Diagnostics
1. **Gelman-Rubin Split $\hat{R}$**: Each chain is split in half (forming $2C = 4$ subchains of length $h = 2500$). Between-chain variance $B$ and within-chain variance $W$ are computed:
   $$\text{var}^+(\theta) = \frac{h-1}{h} W + \frac{B}{h}, \quad \hat{R} = \sqrt{\frac{\text{var}^+(\theta)}{W}}$$
   $\hat{R} < 1.05$ indicates acceptable convergence; $\hat{R} \ge 1.05$ flags non-convergence.
2. **Effective Sample Size (ESS)**: Computed via initial positive sequence autocorrelation over the chains.
3. **95% Bayesian Credible Intervals**: Equal-tailed $2.5\%$ and $97.5\%$ quantiles of the posterior draws.

---

## 3. Maximum Likelihood Estimation (Research Extension)

To investigate computational efficiency and compare Bayesian vs. frequentist performance, a dedicated MLE pipeline is introduced.

### 3.1 Mathematical Formulation
$$\hat{\theta}_{\text{MLE}} = \arg\max_{\theta \in \Theta} \sum_{k=1}^{49} O_k \ln p_k(\theta) \quad \text{subject to } \theta \in [\mathbf{l}, \mathbf{u}]$$
where $[\mathbf{l}, \mathbf{u}]$ enforces parameter bounds:
* Proposed: $a \in [-5.0, 25.0]$, $b \in [10^{-6}, 0.9999]$
* Shifted Log-normal: $\mu \in [-5.0, 5.0]$, $\sigma \in [10^{-4}, 10.0]$, $\gamma \in [-10.0, 0.4999]$
* Weibull: $\alpha \in [10^{-4}, 20.0]$, $\beta \in [10^{-4}, 50.0]$
* Log-logistic: $\alpha \in [10^{-4}, 20.0]$, $\beta \in [10^{-4}, 50.0]$
* Gamma: $\alpha \in [10^{-4}, 50.0]$, $\beta \in [10^{-4}, 50.0]$
* Burr: $\alpha \in [10^{-4}, 30.0]$, $\beta \in [10^{-4}, 20.0]$, $\lambda \in [10^{-4}, 50.0]$
* Shifted Exponential: $\lambda \in [10^{-4}, 20.0]$, $\gamma \in [-10.0, 0.4999]$

### 3.2 Numerical Optimization
Optimization employs Nelder-Mead simplex search with bound enforcement, followed by L-BFGS-B refinement. Runtime, log-likelihood, and convergence status are systematically recorded.

---

## 4. Stochastic Simulation Module

The simulation module generates synthetic headway observations for synthetic traffic generation, safety simulations, and theoretical validation:
1. **Sampling Algorithm**: High-precision monotonic inverse-CDF transform:
   $$U \sim \mathcal{U}(0, 1), \quad X = F_{\text{trunc}}^{-1}(U \mid \hat{\theta})$$
   using a 50,001-point monotonic grid on $[0.5, 25.0]$ s.
2. **Validation**: Two-stage validation:
   * **Theoretical Fidelity**: Evaluates simulated samples against continuous model CDF using 1-sample KS test and Chi-Square goodness-of-fit.
   * **Empirical Fidelity**: Evaluates simulated samples against empirical field observations using 2-sample KS test and 1D Wasserstein distance.

---

## 5. Goodness-of-Fit Evaluation Framework

* **1-Sample KS Test**: Exact supremum distance against continuous model CDF: $D = \sup_t |F_n(t) - F_{\text{trunc}}(t)|$.
* **Chi-Square Test**: $\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}$ with pooling of bins $O_i < 5$ into adjacent neighbors.
* **Kullback-Leibler Divergence**: $D_{KL}(P \parallel Q) = \sum P_i \ln(P_i / Q_i)$.
* **1D Distributional Wasserstein-1 Distance**: $W_1 = \int_\alpha^{T_{\text{max}}} |F_P(t) - F_Q(t)| dt$ [seconds].
* **Paper Sorted Proxy**: $\frac{1}{49}\sum |\text{sort}(p_{\text{obs}}) - \text{sort}(p_{\text{model}})|$ (retains exact comparability with paper Table 5 values).
