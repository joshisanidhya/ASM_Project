---
title: "Stochastic Modeling and Simulation of Vehicle Headways on Highways"
subtitle: "A Bayesian headway distribution, a renewal arrival process and a regime Markov chain, fitted to the highD and NGSIM trajectory datasets"
author: "[Group member names and roll numbers: to be filled in]"
date: "@@date@@"
---

# Abstract {-}

The time gap between successive vehicles in a lane, the *headway*, is a random quantity that governs highway capacity and safety. We study it as a stochastic system on two public trajectory datasets: highD (@@highd.n@@ one-second headway samples from 60 drone recordings of German highways) and NGSIM (@@ngsim.n@@ samples from @@ngsim.selection@@). Three models are built and tested against data. (i) A recently proposed headway density $f(t)\propto b^{|t-a|}$ and six standard distributions are fitted by Bayesian MCMC. (ii) Vehicle arrivals are modelled as a renewal counting process $N(t)$. (iii) Car-following regimes are modelled as a three-state Markov chain. Simulation is compared with theory quantitatively. @@both.model_comparison@@ Simulated renewal counts agree with theory to within @@both.mean_err_max@@% in the mean and @@both.var_err_max@@% in the variance, and @@both.poisson_phrase@@. How well the independent-headway assumption and the first-order Markov assumption hold is quantified in Section 11.

# Introduction

Headway is the key microscopic variable of traffic flow theory. Its reciprocal is the flow of vehicles past a point, short headways raise capacity but reduce safety margins, and its distribution is an input to signal timing, capacity analysis and microscopic simulation. Headways are random: they depend on driver behaviour, vehicle type, speed, traffic volume and road geometry, so no single deterministic value describes a lane. This project follows the pipeline required by the course:

*real-world problem → data → probability model → stochastic process → mathematical analysis → simulation → validation → interpretation.*

The probability model is the headway distribution of Leungbootnak et al. [1]. The stochastic processes built on it are a renewal counting process for vehicle arrivals and a Markov chain for car-following regimes. We implement everything in Python, estimate parameters from real data, simulate multiple sample paths, and compare simulation, theory and data with quantitative measures.

# Problem Statement

**The real-world problem.** Describe the randomness of the gaps between consecutive vehicles on a highway lane, and the random number of vehicles that pass a point in a given time.

**Why randomness is involved.** Each driver chooses a gap depending on perception, reaction time, speed and the vehicle ahead; these choices are heterogeneous and unobserved, so the headway $H$ is a random variable and the arrival count $N(t)$ is a random process.

**Questions we answer.**

1. Which probability distribution describes headways, and does the proposed density $b^{|t-a|}$ outperform six standard alternatives (shifted log-normal, Weibull, log-logistic, gamma, Burr, shifted exponential)?
2. Is vehicle arrival well described by a Poisson process, or by a more general renewal process? How well does simulation agree with renewal theory?
3. Do car-following regimes (close following, platoon, free flow) evolve as a Markov chain, and what is its long-run behaviour?
4. How closely do we reproduce the results published for the proposed distribution [1]?

# Literature Review

Headway modelling began with Adams, who treated road traffic as a random series and used the negative exponential distribution, which corresponds to Poisson arrivals [2]. Because the exponential law allows arbitrarily short headways, shifted and composite forms were introduced later, including the shifted exponential, log-normal, hyper-exponential and Pearson type III distributions. Cowan [3] separated a *tracking* component (vehicles following closely) from a *free* component, giving the M1 to M4 family of composite models. Later studies showed that headway depends on speed [4] and on vehicle type [5] and that no single standard distribution fits all conditions: log-normal and log-logistic fits are favoured in different flow states, and gamma for some vehicle pairs, as reviewed in [1]. Mixture and Bayesian model-averaging approaches [6] fit well but are expensive to estimate. Leungbootnak et al. [1] instead modify the exponential function to $b^{|t-a|}$ with a free base, giving a two-parameter unimodal density with interpretable parameters, estimated by MCMC. The stochastic-process tools we use are classical: renewal theory [7, 8] for the arrival count, and finite Markov chains [9] for the regime sequence.

# Data Description

Both datasets are public trajectory collections. **highD** [10] contains drone recordings of German highways at 25 Hz, in 60 recordings; for each vehicle it reports the preceding vehicle (`precedingId`) and the time headway to it (`thw`). **NGSIM** [11] is the FHWA Next Generation Simulation program; we use the I-80 data (Emeryville, California), recorded at 10 Hz by cameras, with the preceding vehicle and `Time_Headway` per record. We use the original Data.gov export, not the reconstructed trajectories that the paper cites [1].

Following the paper, we keep one sample per second per following vehicle, keep only vehicles that have a leader, and discard headways below 0.5 s (below human reaction time) and above 25 s (not car following). For NGSIM we use @@ngsim.selection@@. @@ngsim.period_note@@

@@table.dataset@@

# Mathematical Background

**Renewal theory.** Let $H_1,H_2,\dots$ be i.i.d. positive random variables with mean $\mu$ and variance $\sigma^2$, and $S_n=H_1+\dots+H_n$. The counting process $N(t)=\max\{n:S_n\le t\}$ is a *renewal process*; $\{N(t)\ge n\}=\{S_n\le t\}$. The renewal function $m(t)=\mathbb E[N(t)]$ satisfies $m(t)/t\to 1/\mu$ (elementary renewal theorem), and for a non-lattice distribution with finite variance

$$m(t)=\frac{t}{\mu}+\frac{\sigma^{2}-\mu^{2}}{2\mu^{2}}+o(1),\qquad \operatorname{Var}N(t)\sim\frac{\sigma^{2}t}{\mu^{3}},\qquad N(t)\ \dot\sim\ \mathcal N\!\Big(\frac{t}{\mu},\frac{\sigma^{2}t}{\mu^{3}}\Big).$$

If $H$ is exponential, $N(t)$ is a Poisson process: $\operatorname{Var}N(t)=\mathbb E N(t)=t/\mu$. In general $\operatorname{Var}N(t)/\mathbb E N(t)\to\sigma^2/\mu^2$, the squared coefficient of variation of the headway. Values below 1 mean arrivals are more regular than Poisson.

**Markov chains.** A discrete-time Markov chain on a finite state space with transition matrix $P=(p_{ij})$ satisfies the Chapman-Kolmogorov equation $P^{(n)}=P^{n}$. If the chain is irreducible and aperiodic, $P^{n}\to \mathbf 1\pi$ where the stationary distribution solves $\pi P=\pi,\ \sum_i\pi_i=1$, and the mean recurrence time of state $i$ is $1/\pi_i$. The maximum-likelihood estimate from an observed path is $\hat p_{ij}=n_{ij}/n_{i\cdot}$.

**Bayesian estimation.** For parameters $\theta$ with prior $p(\theta)$, the posterior is $p(\theta\mid x)\propto p(x\mid\theta)p(\theta)$. The Metropolis-Hastings algorithm draws a candidate $\theta'=\theta+\varepsilon$, $\varepsilon\sim\mathcal N(0,s^2\Sigma)$, and accepts it with probability $\min\{1,p(\theta'\mid x)/p(\theta\mid x)\}$. Convergence is checked with the split $\hat R$ statistic across chains [12].

**Goodness-of-fit measures** (paper, Sections 3.4-3.5): the Kolmogorov-Smirnov statistic, the chi-square statistic with bins merged until the observed count is at least 5, the Kullback-Leibler divergence $D_{KL}(P\|Q)=\sum_k p_k\log(p_k/q_k)$ between the observed and model bin probabilities, and a Wasserstein distance (Eq. 17 of [1]).

# Stochastic Model

@@table.model@@

## The headway distribution

The proposed density is $f(t\mid a,b)=b^{|t-a|}/Z$ on $t\ge\alpha=0.5$ s, with mode $a$ and dispersion $0<b<1$. Writing $\kappa=-\ln b>0$, it is $f\propto e^{-\kappa|t-a|}$: a Laplace density truncated to $t\ge\alpha$, with $b$ controlling the decay. For $a\ge\alpha$ the normalising constant, obtained by integrating separately on $[\alpha,a]$ and $[a,\infty)$, is $Z=(2-b^{a-\alpha})/\kappa$ and the distribution function is

$$F(t)=\begin{cases}\dfrac{b^{a-t}-b^{a-\alpha}}{2-b^{a-\alpha}}, & \alpha\le t\le a,\\[2ex]\dfrac{2-b^{a-\alpha}-b^{t-a}}{2-b^{a-\alpha}}, & t\ge a.\end{cases}$$

This agrees with the closed form in [1] and is verified against numerical integration in our test suite. It also gives an inverse-transform sampler: with $c=b^{a-\alpha}$ and $u\sim U(0,1)$, $t=a-\ln(u(2-c)+c)/\ln b$ if $u\le F(a)$, otherwise $t=a+\ln((1-u)(2-c))/\ln b$.

## The arrival process and the regime chain

*Arrivals.* Successive headways $H_i$ are modelled as i.i.d. draws from the headway distribution; the number of vehicles that have passed a detector by time $t$ is the renewal process $N(t)$ of Section 5, with state space $\{0,1,2,\dots\}$ and continuous time. The assumptions are: one lane, stationary traffic, and independent headways. *Regimes.* Each one-second sample of a following vehicle is assigned a regime $X_n\in\{\text{Close following}\ (0.5\le h<1.5\text{ s}),\ \text{Stable platoon}\ (1.5\le h<4\text{ s}),\ \text{Free flow}\ (h\ge4\text{ s})\}$. The sequence $(X_n)$ is modelled as a time-homogeneous, first-order Markov chain with discrete time (one step is one second of one vehicle's trajectory). The regime boundaries are modelling choices, not estimated quantities.

# Methodology

1. **Data.** Load the datasets, sample at 1 Hz, keep vehicles with a leader, and keep headways in $[0.5,25]$ s.
2. **Estimation.** For each of the seven distributions, run two Metropolis-Hastings chains of 10,000 iterations (5,000 warmup) with the priors of [1]; the estimate is the posterior mean. The likelihood uses the counts in 49 bins of width 0.5 s, $\log L=\sum_k n_k\log p_k(\theta)$, which makes each step cost independent of the sample size.
3. **Goodness of fit.** KS, chi-square, KL divergence and Wasserstein distance between the observed and fitted bin probabilities.
4. **Stochastic processes.** Estimate $P$ by transition counts; compute $P^n$, $\pi$ and recurrence times; simulate renewal paths and Markov paths.
5. **Validation.** (a) Simulated $N(t)$ against renewal theory (mean, variance and distribution at $t=30,60,120,300$ s from 2,000 simulated paths); (b) autocorrelation of the headway sequence, to test the independence assumption; (c) Chapman-Kolmogorov check of $P^n$ against empirical $n$-step frequencies, stationary distribution against observed regime fractions, and a likelihood-ratio test of first against second order; (d) our results against those published in [1].

# Computational Implementation

All code is our own Python (NumPy, SciPy, pandas, Matplotlib); the Metropolis-Hastings sampler is written from scratch. The package `headway/` is organised by role: `distributions.py` (the seven distributions, their CDFs and priors), `mcmc.py` and `fit.py` (sampler and posterior), `metrics.py` (goodness of fit), `datasets/` (highD and NGSIM loaders), `stochastic_process.py` (renewal process and Markov chain), `validation.py` (the quantitative checks above), `simulation.py` and `plots.py`. `run.py` executes the full pipeline and writes CSV tables and figures; a pytest suite covers the distributions, sampler, loaders, metrics and stochastic processes. Independent chains and independent recordings run in parallel on all CPU cores. The complete benchmark is reproduced by `uv run python run.py --period 1 --real-only`, with seed 42; MCMC for all seven distributions took @@highd.mcmc_time@@ s on highD and @@ngsim.mcmc_time@@ s on NGSIM.

# Simulation Results

## Renewal arrival process

Figure 1 shows five simulated sample paths of $N(t)$ with the renewal function $m(t)$ for highD. Headways were resampled i.i.d. from each dataset, so these are realisations of the model of Section 6. Table 3 compares 2,000 simulated paths with renewal theory.

![**Figure 1.** Five simulated sample paths of the arrival count $N(t)$ against the renewal function (highD; the NGSIM figure is in the project outputs).](figures/highd_sample_paths.png){width=65%}

@@table.renewal@@

On highD the simulated mean and variance of $N(t)$ at $t=300$ s are @@highd.mean_sim_300@@ and @@highd.var_sim_300@@, against theoretical values of @@highd.mean_th_300@@ and @@highd.var_th_300@@. A Poisson process with the same rate would have a variance of @@highd.var_poisson_300@@. On NGSIM the corresponding values are @@ngsim.mean_sim_300@@ and @@ngsim.var_sim_300@@ against @@ngsim.mean_th_300@@ and @@ngsim.var_th_300@@ (Poisson: @@ngsim.var_poisson_300@@). The arrival process is therefore @@highd.disp@@ on highD and @@ngsim.disp@@ on NGSIM. Figure 2 shows the full distribution of $N(60)$. At $t=60$ s the largest gap between the simulated and the theoretical count distribution is @@highd.ks_normal_60@@ for the renewal-theory normal and @@highd.ks_poisson_60@@ for a Poisson process on highD, and @@ngsim.ks_normal_60@@ and @@ngsim.ks_poisson_60@@ on NGSIM.

![**Figure 2.** Distribution of $N(60\text{ s})$: simulation, renewal theory and a Poisson process (highD).](figures/highd_event_count.png){width=60%}

## Regime Markov chain

Figure 3 shows the estimated state-transition diagrams and Table 4 the transition matrices, stationary distributions and mean recurrence times. Figure 4 shows $\pi(n)=\pi(0)P^n$ starting from close following, converging to the stationary distribution.

![**Figure 3a.** State-transition diagram of the regime chain (highD).](figures/highd_state_diagram.png){width=48%}

![**Figure 3b.** State-transition diagram of the regime chain (NGSIM).](figures/ngsim_state_diagram.png){width=48%}

@@table.markov@@

![**Figure 4.** State probabilities $\pi(n)$ against the step $n$, converging to the long-run distribution (highD).](figures/highd_state_probabilities.png){width=60%}

# Real Data Results

Table 5 lists the posterior-mean parameters next to the published values [1]; Tables 6 and 7 give the KL divergence and Wasserstein distance. Figure 5 overlays the fitted densities on the observed histograms.

@@table.params@@

@@table.kl@@

@@table.w@@

![**Figure 5a.** Observed headway histogram with the seven fitted distributions (highD).](figures/highd_fitted.png){width=62%}

![**Figure 5b.** Observed headway histogram with the seven fitted distributions (NGSIM).](figures/ngsim_fitted.png){width=62%}

On highD, the proposed distribution has a mode $a=$ @@highd.a@@ s (paper: @@highd.paper_a@@) and $b=$ @@highd.b@@ (paper: @@highd.paper_b@@), i.e. a decay rate $\kappa=$ @@highd.kappa@@ per second. On NGSIM, $a=$ @@ngsim.a@@ s (paper: @@ngsim.paper_a@@), $b=$ @@ngsim.b@@ (paper: @@ngsim.paper_b@@), $\kappa=$ @@ngsim.kappa@@. @@both.tests_phrase@@ The posterior and trace plots for the proposed distribution are in Appendix, Figure A1.

# Model Validation

**Against the published results.** Our KL divergences differ from Table 5 of the paper by at most @@highd.kl.max_abs_diff@@ on highD and @@ngsim.kl.max_abs_diff@@ on NGSIM (Table 6). The best model by KL is @@highd.kl.best@@ on highD in our run (paper: Proposed) and @@ngsim.kl.best@@ on NGSIM (paper: Burr). The Wasserstein values (Table 7) follow our reading of Eq. 17 (Section 13) and are on the paper's scale; the largest difference is @@highd.w.max_abs_diff@@ on highD and @@ngsim.w.max_abs_diff@@ on NGSIM. The KL of badly fitting baselines is sensitive to the exact sample, which differs from the paper's NGSIM sample (Section 13). The paper's Wasserstein value for Gamma on NGSIM (0.0173) is also inconsistent with its own KL ranking, so we do not expect to reproduce it.

**MCMC convergence.** The largest split $\hat R$ is @@highd.rhat_max@@ (highD) and @@ngsim.rhat_max@@ (NGSIM), with acceptance rates between @@both.acc_range@@ (the optimum for random-walk Metropolis is about 23%; Appendix, Table A2). @@report.unconverged@@

**Renewal process against theory.** Simulated means and variances of $N(t)$ agree with renewal theory within @@both.mean_err_max@@% and @@both.var_err_max@@% (Table 3), so the simulation reproduces the theory to that accuracy. Whether the *model* matches the data is a separate question: the headway sequence has lag-1 autocorrelation @@highd.lag1@@ on highD and @@ngsim.lag1@@ on NGSIM, against a 95% white-noise band of $\pm$@@highd.band@@ and $\pm$@@ngsim.band@@, so the data show @@highd.indep@@ on highD and @@ngsim.indep@@ on NGSIM (Appendix, Table A3).

**Markov chain.** Table 8 summarises the checks. On highD, @@highd.markovval@@ On NGSIM, @@ngsim.markovval@@

@@table.valid@@

# Discussion

*Traffic interpretation.* The renewal rate $1/\mathbb E[H]$ is the lane flow: @@highd.flow@@ veh/h/lane on highD (mean headway @@highd.mean@@ s) against @@ngsim.flow@@ veh/h/lane on NGSIM (mean headway @@ngsim.mean@@ s). The variance-to-mean ratio of the counts is @@highd.cv2@@ on highD and @@ngsim.cv2@@ on NGSIM (1 for Poisson). @@both.disp_meaning@@ The regime chain gives a long-run share of @@highd.pi_close@@ / @@highd.pi_platoon@@ / @@highd.pi_free@@ (close / platoon / free) on highD against @@ngsim.pi_close@@ / @@ngsim.pi_platoon@@ / @@ngsim.pi_free@@ on NGSIM, so the dominant regime is @@highd.dom@@ on highD and @@ngsim.dom@@ on NGSIM.

*Model comparison.* @@both.model_comparison@@ The parameters of the proposed distribution are directly interpretable: $a$ is the most frequent headway and $\kappa=-\ln b$ is the exponential decay rate on either side of it.

*Estimation.* Because the likelihood is evaluated on 49 bins, the cost of MCMC does not grow with the sample size; fitting all seven distributions took @@highd.mcmc_time@@ s on highD (@@highd.n@@ samples) and @@ngsim.mcmc_time@@ s on NGSIM (@@ngsim.n@@ samples).

# Limitations

1. **NGSIM sample.** The paper uses the reconstructed I-80 data; we use the original export and our own choice of recording period (Section 4). Some baseline KL values therefore differ from the paper's.
2. **Likelihood.** MCMC uses binned counts rather than every observation, which is exact only to the 0.5 s bin resolution.
3. **Wasserstein distance.** The paper does not define the two vectors in Eq. 17. We compare the sorted observed and model bin probabilities, the only reading that reproduces its magnitudes; this measure ignores where each bin lies on the time axis.
4. **Dependence.** Consecutive one-second samples of one follower are serially dependent, so the i.i.d. renewal assumption and the first-order Markov assumption are approximations, and the sequences contain a few vehicle-to-vehicle jumps (see Section 11).
5. **Chain definition.** One Markov step is one second of one vehicle's trajectory; the regime thresholds (1.5 s and 4 s) are our choice.
6. **Convergence.** @@report.unconverged@@
7. **Scope.** One lane at a time, no vehicle-type or speed dependence, and only two datasets; the Lyft autonomous-vehicle data used in the paper is not included.

# Conclusion

We modelled highway headways with a fitted probability distribution, a renewal arrival process and a regime Markov chain, estimated all parameters from real data, simulated sample paths, and tested simulation against theory and the models against data. By KL divergence the proposed distribution ranks @@highd.kl.proposed_rank@@ of 7 on highD and @@ngsim.kl.proposed_rank@@ of 7 on NGSIM. @@both.paper_agreement@@ In both datasets @@both.poisson_phrase@@. Simulation and renewal theory agree closely, and the validation quantifies how well the independence and Markov assumptions hold. Natural extensions are speed-dependent or semi-Markov regime models that capture the observed dependence, a vehicle-type-aware headway distribution, and the mixed human-autonomous traffic of the Lyft dataset.

# References {-}

1. N. Leungbootnak, Z. Li, Z. Wei, D. Lord, Y. Zhang. *Modeling Headway in Heterogeneous and Mixed Traffic Flow: A Statistical Distribution Based on a General Exponential Function.* Texas A&M University (manuscript).
2. W. F. Adams. Road traffic considered as a random series. *J. Institution of Civil Engineers* 4, 1936.
3. R. J. Cowan. Useful headway models. *Transportation Research* 9, 1975.
4. Chen et al.: headway and spacing distributions from NGSIM trajectory data depend on speed (work cited in [1]; see [1] for the full reference).
5. S. P. Hoogendoorn and P. H. L. Bovy: headway models calibrated per vehicle type (work cited in [1]; see [1] for the full reference).
6. Wu et al.: Bayesian model averaging of headway distributions (work cited in [1]; see [1] for the full reference).
7. D. R. Cox. *Renewal Theory.* Methuen, 1962.
8. S. M. Ross. *Stochastic Processes*, 2nd ed. Wiley, 1996.
9. J. R. Norris. *Markov Chains.* Cambridge University Press, 1997.
10. R. Krajewski, J. Bock, L. Kloeker, L. Eckstein. The highD dataset: a drone dataset of naturalistic vehicle trajectories on German highways for validation of highly automated driving systems. *IEEE ITSC*, 2018.
11. U.S. Department of Transportation, FHWA. Next Generation Simulation (NGSIM) vehicle trajectories and supporting data. catalog.data.gov/dataset/next-generation-simulation-ngsim-vehicle-trajectories-and-supporting-data
12. A. Vehtari, A. Gelman, D. Simpson, B. Carpenter, P.-C. Bürkner. Rank-normalization, folding, and localization: an improved $\hat R$ for assessing convergence of MCMC. *Bayesian Analysis* 16(2), 2021.

# Appendix: Code and Additional Tables {-}

## Reproducing the results {-}

```
uv sync
uv run python run.py --period 1 --real-only      # highD + NGSIM, all cores
uv run --group report python -m headway.report   # builds this report from outputs/
uv run pytest -q tests
```

## Key code {-}

The proposed distribution's distribution function (normalised on $[\alpha,\infty)$):

```python
@@code.proposed@@
```

The Metropolis-Hastings sampler:

```python
@@code.metropolis@@
```

The renewal-process validation:

```python
@@code.renewal@@
```

## Additional tables and figures {-}

@@table.tests@@

@@table.diag@@

@@table.indep@@

![**Figure A1a.** Posterior distributions of the proposed distribution's parameters (highD).](figures/highd_posterior.png){width=65%}

![**Figure A1b.** MCMC traces of the proposed distribution's parameters (highD).](figures/highd_traces.png){width=65%}
