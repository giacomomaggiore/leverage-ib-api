# Technical Appendix

---

## 1. Portfolio Construction

### Returns

Let $P_{i,t}$ be the adjusted closing price of asset $i$ on day $t$. Log-returns are:

$$r_{i,t} = \ln\left(\frac{P_{i,t}}{P_{i,t-1}}\right)$$

Over a window of $T$ observations, the sample mean vector and sample covariance matrix are:

$$\hat{\mu} = \frac{1}{T}\sum_{t=1}^{T} r_t \qquad S = \frac{1}{T-1}\sum_{t=1}^{T}(r_t - \hat{\mu})(r_t - \hat{\mu})^\top$$

---

### Covariance Shrinkage (Ledoit–Wolf)

The sample covariance $S$ is noisy when $T$ is not large relative to $n$ (number of assets). Ledoit–Wolf shrinks $S$ toward a structured target $F$:

$$\hat{\Sigma} = (1 - \alpha)\, S + \alpha\, F$$

The target is a scaled identity matrix:

$$F = \frac{\operatorname{tr}(S)}{n}\, I_n$$

The shrinkage intensity $\alpha^* \in [0, 1]$ is chosen analytically to minimise the expected Frobenius loss $\mathbb{E}\|\hat{\Sigma} - \Sigma\|_F^2$. A higher $\alpha$ pulls the matrix toward equal variances and zero correlations; useful when $T/n$ is small.

---

### Min-Variance Portfolio

$$\min_{w}\; w^\top \hat{\Sigma}\, w \quad \text{s.t.} \quad \mathbf{1}^\top w = 1,\quad 0 \le w_i \le w_{\max}$$

The unconstrained closed-form solution is:

$$w^* = \frac{\hat{\Sigma}^{-1}\,\mathbf{1}}{\mathbf{1}^\top \hat{\Sigma}^{-1}\,\mathbf{1}}$$

With box constraints ($w_i \le w_{\max}$) the problem is a convex quadratic program (QP) with no closed form; solved numerically.

---

### Max-Sharpe Portfolio

$$\max_{w}\; \frac{\hat{\mu}^\top w - r_f}{\sqrt{w^\top \hat{\Sigma}\, w}} \quad \text{s.t.} \quad \mathbf{1}^\top w = 1,\quad 0 \le w_i \le w_{\max}$$

Via the homogenisation trick (let $y = w\,/\,(\hat{\mu}^\top w - r_f)$), this is equivalent to the QP:

$$\min_{y}\; y^\top \hat{\Sigma}\, y \quad \text{s.t.} \quad (\hat{\mu} - r_f\,\mathbf{1})^\top y = 1,\quad y \ge 0$$

then recover $w^* = y\,/\,(\mathbf{1}^\top y)$.

---

### Market-Cap Portfolio

$$w_i = \frac{MC_i}{\sum_j MC_j}$$

where $MC_i$ is the free-float market capitalisation of asset $i$. No optimisation step; weights are set directly from market data.

---

## 2. Leverage and Margin

### Definitions

Let $E_t$ denote equity (NAV) at time $t$ and $G_t$ total gross exposure (sum of position values):

$$L_t = \frac{G_t}{E_t}$$

At target leverage $L = 1.7$:

$$G_t = L \cdot E_t \qquad D_t = (L-1)\cdot E_t = 0.7\cdot E_t$$

where $D_t$ is the borrowed amount (margin debt).

---

### Daily NAV Evolution

Let $r_{p,t} = w^\top r_t$ be the portfolio return on day $t$ and $r_m$ the annual margin rate. Position value after market move:

$$G_t = G_{t-1}(1 + r_{p,t})$$

Debt is unchanged intraday. Updated equity:

$$E_t = G_t - D_{t-1} = L\cdot E_{t-1}(1 + r_{p,t}) - (L-1)\cdot E_{t-1}$$

$$\boxed{E_t = E_{t-1}\left(1 + L\, r_{p,t}\right)}$$

Subtracting daily margin interest:

$$E_t = E_{t-1}\left(1 + L\, r_{p,t} - (L-1)\,\frac{r_m}{252}\right)$$

---

### Realized Leverage After a Move

After a portfolio return $r_{p,t}$ without rebalancing:

$$L_t = \frac{G_t}{E_t} = \frac{L(1 + r_{p,t})}{1 + L\, r_{p,t}}$$

The leverage drift is:

$$\Delta L_t = L_t - L = \frac{L\,(1 - L)\,r_{p,t}}{1 + L\, r_{p,t}}$$

Since $L > 1$: a positive return decreases leverage; a negative return increases it. Leverage is asymmetrically dangerous on the downside.

---

## 3. Leverage Safety Interval

Define the rebalance trigger as a fractional band $\delta$ around the target leverage $L$:

$$\text{Rebalance if}\quad L_t \notin \left[L(1 - \delta),\; L(1 + \delta)\right]$$

At $L = 1.7$ and $\delta = 0.10$, the interval is $[1.53,\; 1.87]$.

The critical portfolio return that breaches the upper bound (leverage rises above $L(1+\delta)$) satisfies:

$$\frac{L(1+r)}{1 + L\,r} = L(1+\delta) \implies r^* = \frac{-L\delta}{L(1 + \delta) - 1} \cdot \frac{1}{L}$$

For $L = 1.7$, $\delta = 0.10$: a single-day portfolio return below approximately $-7\%$ will trigger the threshold.

---

### Maintenance Margin

Under IBKR Reg T, the minimum equity ratio for ETFs is 25%:

$$\frac{E_t}{G_t} = \frac{1}{L_t} \ge 0.25 \implies L_t \le 4$$

In practice, the safety interval $[L(1-\delta), L(1+\delta)]$ is set well inside this hard limit. At $L=1.7$ and $\delta=0.10$, the upper bound $1.87$ is far from the margin-call level of $4$.

---

## 4. Performance Metrics

### CAGR

$$\text{CAGR} = \left(\frac{E_T}{E_0}\right)^{252/T} - 1$$

where $T$ is the number of trading days in the backtest.

---

### Annualised Volatility

$$\sigma_p = \sqrt{252}\;\operatorname{std}(r_{p,t})$$

---

### Sharpe Ratio

$$\text{Sharpe} = \frac{\mu_p^{\text{ann}} - r_f}{\sigma_p}$$

where $\mu_p^{\text{ann}} = 252\cdot\mathbb{E}[r_{p,t}]$ is the annualised mean return and $r_f$ is the risk-free rate.

---

### Maximum Drawdown

$$\text{MDD} = \min_t\;\frac{E_t - \max_{s \le t} E_s}{\max_{s \le t} E_s}$$

---

### Sortino Ratio

$$\text{Sortino} = \frac{\mu_p^{\text{ann}} - r_f}{\sigma_d}$$

where the downside deviation is:

$$\sigma_d = \sqrt{252\cdot\mathbb{E}\!\left[\min(r_{p,t} - r_f/252,\;0)^2\right]}$$

---

### Calmar Ratio

$$\text{Calmar} = \frac{\text{CAGR}}{|\text{MDD}|}$$
