# Mathematical Model & Theoretical Derivations

**Project**: Applied Stochastic Modeling (ASM) Research Project — Vehicle Headway Distribution  
**Reference Paper**: *Modeling Headway in Heterogeneous and Mixed Traffic Flow: A Statistical Distribution Based on a General Exponential Function* (Natchaphon Leungbootnak, Texas A&M University)  

---

## 1. The Proposed General Exponential Headway Distribution

### 1.1 Motivation & Functional Form
Vehicle time headway ($h$, in seconds) is the microscopic time interval between successive vehicles traveling in the same traffic lane past a common reference line. Traditional headway distributions (e.g., Exponential, Erlang, Log-normal) often fail to capture the multi-modal or sharp modal clustering present in mixed traffic where autonomous vehicles (AVs) and human-driven vehicles (HDVs) interact.

The paper proposes a general exponential function with variable base $b$ and central modal location $a$:
$$g(t) = b^{|t - a|}, \quad 0 < b < 1, \quad a \in \mathbb{R}$$
where:
* **$a$ (Modal Location / Central Tendency)**: Represents the most frequent headway value observed in the traffic stream. The function attains its peak value $g(a) = b^0 = 1$ at $t = a$.
* **$b$ (Decay / Dispersion Parameter)**: Governs the rate of decay as headways deviate from the modal point $a$. A smaller $b \to 0$ corresponds to a sharply concentrated distribution, whereas $b \to 1$ produces a broader, flatter dispersion.
* **$\alpha$ (Minimum Physical Headway Threshold)**: By definition, headways below human reaction time ($\alpha = 0.5$ s) are physically hazardous or invalid in car-following. Thus:
  $$P(h < \alpha) = 0$$

### 1.2 Analytical Normalization Constant
To convert $g(t)$ into a valid probability density function on $[\alpha, \infty)$, we compute the total area under the curve:
$$I = \int_\alpha^\infty b^{|t - a|} dt$$

Using base conversion $b^u = e^{u \ln b}$ (with $\ln b < 0$ since $0 < b < 1$):

#### Case 1: $a > \alpha$ (Mode exceeds physical threshold)
The domain partitions at $t = a$:
$$\int_\alpha^\infty b^{|t - a|} dt = \int_\alpha^a b^{a - t} dt + \int_a^\infty b^{t - a} dt$$
Evaluating the first integral:
$$\int_\alpha^a e^{-(t - a)\ln b} dt = \left[ -\frac{e^{-(t - a)\ln b}}{\ln b} \right]_\alpha^a = -\frac{1 - e^{-(\alpha - a)\ln b}}{\ln b} = \frac{b^{a - \alpha} - 1}{\ln b}$$
Evaluating the second integral (since $\lim_{t \to \infty} b^{t-a} = 0$):
$$\int_a^\infty e^{(t - a)\ln b} dt = \left[ \frac{e^{(t - a)\ln b}}{\ln b} \right]_a^\infty = 0 - \frac{1}{\ln b} = -\frac{1}{\ln b}$$
Summing both contributions (Paper Equation 12):
$$I = \frac{b^{a - \alpha} - 1}{\ln b} - \frac{1}{\ln b} = \frac{b^{a - \alpha} - 2}{\ln b}, \quad (a > \alpha)$$

#### Case 2: $a \le \alpha$ (Mode at or below physical threshold)
For all $t \ge \alpha \ge a$, $|t - a| = t - a$:
$$\int_\alpha^\infty b^{t - a} dt = \left[ \frac{b^{t - a}}{\ln b} \right]_\alpha^\infty = 0 - \frac{b^{\alpha - a}}{\ln b} = \frac{-b^{\alpha - a}}{\ln b}, \quad (a \le \alpha)$$

### 1.3 Normalized Probability Density Function (PDF)
Dividing $g(t)$ by $I$ yields the normalized PDF:
$$f(t \mid a, b) = \begin{cases}
\dfrac{\ln b \cdot b^{|t - a|}}{b^{a - \alpha} - 2}, & t \ge \alpha, \ a > \alpha \\[10pt]
-\ln b \cdot b^{t - \alpha}, & t \ge \alpha, \ a \le \alpha \\[10pt]
0, & t < \alpha
\end{cases}$$
Note that because $\ln b < 0$ and $b^{a - \alpha} - 2 < 0$, the ratio is strictly positive.

### 1.4 Analytical Cumulative Distribution Function (CDF)
The CDF $F(t) = \int_\alpha^t f(s \mid a, b) ds$ on $[\alpha, \infty)$ is piecewise:
* **For $a > \alpha$**:
  $$F(t) = \begin{cases}
  0, & t < \alpha \\[6pt]
  \dfrac{b^{a - \alpha} - b^{a - t}}{b^{a - \alpha} - 2}, & \alpha \le t \le a \\[10pt]
  \dfrac{b^{a - \alpha} + b^{t - a} - 2}{b^{a - \alpha} - 2}, & t > a
  \end{cases}$$
* **For $a \le \alpha$**:
  $$F(t) = \begin{cases}
  0, & t < \alpha \\[6pt]
  1 - b^{t - \alpha}, & t \ge \alpha
  \end{cases}$$

### 1.5 Closed-Form Interval Probability (Paper Equation 2)
For interval probabilities $P(t_1 \le h \le t_2) = F(t_2) - F(t_1)$ with $\alpha \le t_1 < t_2$:
$$P(t_1 \le h \le t_2) = \begin{cases}
\dfrac{b^{a - t_1} + b^{t_2 - a} - 2}{b^{a - \alpha} - 2}, & \alpha \le t_1 < a < t_2 \\[10pt]
\dfrac{b^{a - t_1} - b^{a - t_2}}{b^{a - \alpha} - 2}, & \alpha \le t_1 < t_2 \le a \\[10pt]
\dfrac{b^{t_2 - a} - b^{t_1 - a}}{b^{a - \alpha} - 2}, & \alpha \le a \le t_1 < t_2 \\[10pt]
\dfrac{b^{t_2 - a} - b^{t_1 - a}}{-b^{\alpha - a}}, & a < \alpha \le t_1 < t_2
\end{cases}$$

### 1.6 Conditioning on Observed Truncation Window $[\alpha, T_{max}]$
Since data filtering removes headways $> T_{max} = 25.0$ s, empirical distributions are supported on $[\alpha, T_{max}] = [0.5, 25.0]$ s. The truncated conditional CDF is:
$$F_{\text{trunc}}(t) = \frac{F(t) - F(\alpha)}{F(T_{max}) - F(\alpha)}, \quad t \in [\alpha, T_{max}]$$
The probability mass for bin $k = [t_{k-1}, t_k]$ is:
$$p_k = F_{\text{trunc}}(t_k) - F_{\text{trunc}}(t_{k-1})$$
satisfying $\sum_{k=1}^{49} p_k = 1.0$.

---

## 2. Six Baseline Headway Distributions (Paper Table 1)

All baseline distributions are evaluated under the same conditional truncation window $[\alpha, T_{max}]$:

### 2.1 Shifted Log-Normal
* **Support**: $t > \gamma, \ t > 0$
* **Parameters**: $\mu \in \mathbb{R}$ (log-mean), $\sigma > 0$ (log-std), $\gamma < \alpha$ (location shift)
* **PDF**:
  $$f(t \mid \mu, \sigma, \gamma) = \frac{1}{(t - \gamma)\sigma\sqrt{2\pi}} \exp\left(-\frac{(\ln(t - \gamma) - \mu)^2}{2\sigma^2}\right)$$
* **CDF**:
  $$F(t) = \Phi\left(\frac{\ln(t - \gamma) - \mu}{\sigma}\right)$$
* **Priors**: $\mu \sim \mathcal{N}(0, 10^2), \ \sigma \sim \text{Gamma}(0.5, 0.5), \ \gamma \sim \mathcal{U}(-10, \min(t))$

### 2.2 Weibull
* **Support**: $t \ge 0$
* **Parameters**: $\alpha > 0$ (shape), $\beta > 0$ (scale)
* **PDF**:
  $$f(t \mid \alpha, \beta) = \frac{\alpha}{\beta}\left(\frac{t}{\beta}\right)^{\alpha - 1} \exp\left(-\left(\frac{t}{\beta}\right)^\alpha\right)$$
* **CDF**:
  $$F(t) = 1 - \exp\left(-\left(\frac{t}{\beta}\right)^\alpha\right)$$
* **Priors**: $\alpha, \beta \sim \text{Gamma}(0.5, 0.5)$

### 2.3 Log-Logistic (Fisk)
* **Support**: $t > 0$
* **Parameters**: $\alpha > 0$ (shape), $\beta > 0$ (scale)
* **PDF**:
  $$f(t \mid \alpha, \beta) = \frac{\frac{\alpha}{\beta}(t / \beta)^{\alpha - 1}}{\left[1 + (t / \beta)^\alpha\right]^2}$$
* **CDF**:
  $$F(t) = \frac{1}{1 + (t / \beta)^{-\alpha}} = \frac{(t/\beta)^\alpha}{1 + (t/\beta)^\alpha}$$
* **Priors**: $\alpha, \beta \sim \text{Gamma}(0.5, 0.5)$

### 2.4 Gamma
* **Support**: $t > 0$
* **Parameters**: $\alpha > 0$ (shape), $\beta > 0$ (rate)
* **PDF**:
  $$f(t \mid \alpha, \beta) = \frac{\beta^\alpha}{\Gamma(\alpha)} t^{\alpha - 1} e^{-\beta t}$$
* **CDF**:
  $$F(t) = \frac{\gamma(\alpha, \beta t)}{\Gamma(\alpha)} = P(\alpha, \beta t)$$
* **Priors**: $\alpha, \beta \sim \text{Gamma}(0.5, 0.5)$

### 2.5 Burr Type XII
* **Support**: $t > 0$
* **Parameters**: $\alpha > 0$ (first shape), $\beta > 0$ (second shape), $\lambda > 0$ (scale)
* **PDF**:
  $$f(t \mid \alpha, \beta, \lambda) = \frac{\alpha\beta}{\lambda}\left(\frac{t}{\lambda}\right)^{\alpha - 1}\left[1 + \left(\frac{t}{\lambda}\right)^\alpha\right]^{-\beta - 1}$$
* **CDF**:
  $$F(t) = 1 - \left[1 + \left(\frac{t}{\lambda}\right)^\alpha\right]^{-\beta}$$
* **Priors**: $\alpha, \beta, \lambda \sim \text{Gamma}(0.5, 0.5)$

### 2.6 Shifted Exponential
* **Support**: $t \ge \gamma, \ t \ge 0$
* **Parameters**: $\lambda > 0$ (rate), $\gamma < \alpha$ (location shift)
* **PDF**:
  $$f(t \mid \lambda, \gamma) = \lambda e^{-\lambda(t - \gamma)}$$
* **CDF**:
  $$F(t) = 1 - e^{-\lambda(t - \gamma)}$$
* **Priors**: $\lambda \sim \text{Gamma}(0.5, 0.5), \ \gamma \sim \mathcal{U}(-10, \min(t))$

---

## 3. Goodness-of-Fit Evaluation Metrics

### 3.1 Kolmogorov-Smirnov (KS) Test (Paper Equation 14)
The Kolmogorov-Smirnov statistic evaluates the maximum deviation between the empirical CDF $F_n(t)$ and the fitted continuous theoretical CDF $F_{\text{trunc}}(t)$:
$$D = \sup_{t \in [\alpha, T_{max}]} |F_n(t) - F_{\text{trunc}}(t \mid \hat{\theta})|$$
* $H_0$: The field data sample is drawn from the continuous theoretical CDF $F_{\text{trunc}}(t)$.
* $p$-value: Derived from the Kolmogorov distribution. A $p$-value $> 0.05$ indicates failure to reject $H_0$ at the 95% confidence level.

### 3.2 Chi-Square Goodness-of-Fit Test (Paper Equation 15)
The Chi-Square statistic evaluates binned frequencies over the 49 intervals of width 0.5 s:
$$\chi^2 = \sum_{i=1}^{K_{\text{pooled}}} \frac{(O_i - E_i)^2}{E_i}$$
* **Bin Merging Rule (Paper Section 3.4)**: If observed count $O_i < 5$, the bin is merged with adjacent bins until $O_i \ge 5$.
* **Expected Frequency**: $E_i = N \cdot p_i$, where $p_i = F_{\text{trunc}}(t_i) - F_{\text{trunc}}(t_{i-1})$.
* **Degrees of Freedom**: $df = K_{\text{pooled}} - 1 - p$, where $p$ is the number of estimated parameters.

### 3.3 Kullback-Leibler (KL) Divergence (Paper Equation 16)
Quantifies the information lost when approximating the empirical discrete distribution $P$ by model $Q$:
$$D_{KL}(P \parallel Q) = \sum_{k=1}^{49} P(x_k) \ln\left(\frac{P(x_k)}{Q(x_k)}\right)$$
where $P(x_k) = O_k / N$ and $Q(x_k) = p_k(\hat{\theta})$. Bins with $P(x_k) = 0$ contribute $0$.

### 3.4 1D Distributional Wasserstein Distance (Paper Equation 17 & Vallender 1974)
By Vallender's theorem (paper reference 42), the 1D Wasserstein-1 (Earth Mover's Distance) between probability distributions $P$ and $Q$ on $\mathbb{R}$ is:
$$W_1(P, Q) = \int_{-\infty}^\infty |F_P(t) - F_Q(t)| dt = \int_\alpha^{T_{max}} |F_n(t) - F_{\text{trunc}}(t \mid \hat{\theta})| dt$$
* **Physical Dimension**: Measured in **seconds** (representing expected transportation cost along the headway time axis).
* **Binned Computation**:
  $$W_1(P, Q) = \sum_{k=1}^{48} |F_P(t_k) - F_Q(t_k)| \Delta t$$
* **Note on Paper Proxy**: The paper reported values around $\sim 10^{-3}$ resulting from the average difference of sorted bin probabilities $\frac{1}{49}\sum |\text{sort}(p_{\text{obs}}) - \text{sort}(p_{\text{model}})|$. Our implementation reports both the rigorous distributional Wasserstein distance $W_1$ [seconds] and the sorted-bin proxy for paper comparability.
