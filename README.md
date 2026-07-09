Leverage Portfolio
==================

Build and backtest a leveraged long-only portfolio of US ETFs, then run it live via IB Gateway.

---

Strategy
--------
Four portfolios are compared in backtest and Monte Carlo:
- Min-variance (Ledoit–Wolf covariance shrinkage)
- Max-Sharpe
- Market-cap (100% VT)
- Equal-weight (1/N)

Target leverage: 1.7x. The best performer is used live.

Rebalancing triggers:
- Monthly (calendar)
- Threshold: rebalance early if actual leverage drifts ±10% from target

---

Pipeline
--------
1. Download 5y daily OHLCV via yfinance → data/
2. Compute covariances from returns (PyPortfolioOpt risk models)
3. Optimize weights with PyPortfolioOpt (min-vol, max-Sharpe); expected returns via mean_historical_return on prices (simple returns)
4. Backtest: monthly rebalance loop, 2x leverage, margin interest deducted
5. Monte Carlo: block bootstrap, collect return distribution
6. Pick best strategy
7. Live: query IB positions, compute delta to target weights, place orders

Workflow (end-to-end)
---------------------
- Universe setup: optionally cluster near-duplicates and select representatives.
- Build strategies (rebalancing at `freq`, e.g., `BMS`):
    - `min_variance`, `max_sharpe` via optimizers
    - `market_cap` = 100% VT (explicit VT column is always included)
    - `equal_weight` = 1/N baseline
- Backtest: turn daily weights into a value path with `backtest_portfolio`.
- Monte Carlo: simulate daily returns (parametric MVN or bootstrap), rebuild weights on simulated prices using the same logic, and backtest per path.

---

Covariance Models (Theory, Intuition, When to Use)
--------------------------------------------------
Portfolio optimizers depend critically on the return covariance matrix. This repo supports multiple estimators via the `cov_method` parameter in `min_variance` and `max_sharpe` (backed by PyPortfolioOpt):

- `shrunk` (default): Ledoit–Wolf shrinkage toward a scaled identity
- `empirical`: plain sample covariance
- `oas`: Oracle Approximating Shrinkage (sklearn)
- `ewma`: exponentially weighted covariance
- `factor`: simple PCA factor model

Each method trades bias vs variance differently. Below is a practical guide.

Empirical (sample) covariance — `cov_method='empirical'`
 - Definition:

    $$S = \frac{1}{T-1}\sum_t (r_t - \mu)(r_t - \mu)^\top$$
- Intuition: Uses only in-sample co-movements; no regularization.
- Pros: Preserves true cross-asset structure; differentiates assets strongly.
- Cons: Noisy/ill-conditioned when lookback T is short relative to number of assets N; can produce unstable/extreme weights.
- Use when: You have ample history vs asset count, accept more variability for higher fidelity to observed correlations.

Ledoit–Wolf shrinkage — `cov_method='shrunk'`
 - Definition:

    $$\hat{\Sigma} = (1-\alpha) S + \alpha \mu I \quad (\alpha\ \text{auto-chosen};\ \mu = \mathrm{tr}(S)/N)$$
- Intuition: Pulls estimates toward a spherical target to reduce sampling noise and improve conditioning.
- Pros: Stabler optimizer, fewer pathologies from noisy S; no hyperparameters.
- Cons: Can “wash out” correlation structure; when off-diagonals are small and variances similar, min-variance tends toward equal-weight under long-only constraints.
- Use when: Moderate history, many assets, need stability with minimal tuning.

OAS (Oracle Approximating Shrinkage) — `cov_method='oas'`
- Definition: Like LW but with an analytically derived shrinkage intensity optimal for Gaussian data.
- Intuition: Often shrinks slightly differently than LW; can be milder in practice.
- Pros: Strong conditioning with competitive bias/variance trade-off; no knobs to tune.
- Cons: Same structural limitation as LW if assets look similar.
- Use when: You want a robust, parameter-free shrinker; try this if LW produces overly uniform weights.

EWMA (Exponentially Weighted) — `cov_method='ewma'` with `cov_params={'alpha': a}` or `{'span': s}`
 - Definition (conceptually): recent returns get more weight; effective covariance:

    $$\mathrm{Cov} = \sum_t w_t (r_t - \mu)(r_t - \mu)^\top$$

    with decaying weights.
- Intuition: Emphasizes the recent regime; adapts to changing vol/correlation faster than static windows.
- Pros: Captures time-variation; can reduce stale-risk during regime shifts.
- Cons: More parameter sensitivity (α or span); can underuse older data and be noisier if span is too short.
- Use when: You want responsiveness to recent behavior; start with `span=60`–`120` trading days.

Factor (PCA) model — `cov_method='factor'` with `cov_params={'n_factors': k}`
- Construction (outline):
    - Demean returns and take SVD/PCA → top k factors (scores) and loadings.
        - Rebuild

            $$\hat{\Sigma} = B \Sigma_F B^\top + D$$

            where $B$ are loadings (matrix of factor loadings: each row is an asset, each column a factor, each entry in the matrix is asset's sensitivity to a factor), $\Sigma_F$ factor covariance, $D$ diagonal of specific variances.
- Intuition: Most co-movement is driven by a few latent factors; model common risk parsimoniously, reduce noise elsewhere.
- Pros: Good stability with retained structure; tunable complexity via `n_factors`.
- Cons: Choice of k matters; purely statistical (no economic labeling by default).
- Use when: N is large vs T and you want structure without heavy shrinkage; try `n_factors=2–4`.

Choosing a method (practical tips)
- Seeing equal weights with shrinkage? Try `empirical`, `oas`, or `ewma` to re-introduce structure.
- If weights look jumpy, prefer `oas`, `shrunk`, or `factor` (small `n_factors`).
- Increase lookback or rebalance on business-month-start (`BMS`) to improve estimation and reduce NaNs.
- Check constraints: tight per-asset caps push solutions toward uniform weights when assets look similar.
 - Prefer well-conditioned Σ: set `cov_method='oas'` or `'shrunk'`; or use `'factor'` with `cov_params={'n_factors': 2–4}` to retain structure while reducing noise.
 - Optional ridge (jitter): improve conditioning by adding a tiny diagonal term via `cov_params={'jitter': 1e-8}`.
 - Remove near-duplicates: avoid including aggregate ETFs (e.g., `AGG`) together with their sleeve constituents (e.g., `VGSH`, `IEF`, `VGLT`, `TLH`).

Code usage
```
# Min-variance with different covariances
min_variance(tickers, as_of=dt, timeframe_years=3, cov_method='empirical')
min_variance(tickers, as_of=dt, cov_method='oas')
min_variance(tickers, as_of=dt, cov_method='ewma',   cov_params={'span': 60})
min_variance(tickers, as_of=dt, cov_method='factor', cov_params={'n_factors': 3})

# Max-Sharpe mirrors the same interface
max_sharpe(tickers, as_of=dt, cov_method='shrunk')
```

PyPortfolioOpt specifics
- Expected returns: `pypfopt.expected_returns.mean_historical_return(prices, frequency=252, log_returns=False)` (simple returns on adjusted-close prices, annualized).
- Covariance (returns): `pypfopt.risk_models.sample_cov`, `CovarianceShrinkage(...).ledoit_wolf()`, `CovarianceShrinkage(...).oracle_approximating()`, or `exp_cov(..., span=s)` depending on `cov_method`.
- Optimization: `EfficientFrontier.min_volatility()` and `EfficientFrontier.max_sharpe(risk_free_rate=rf)` under long-only, fully-invested constraints.
 - Extras: You can pass `cov_params` to tweak estimators; this repo supports `{'span': s}` (EWMA), `{'n_factors': k}` (factor), and `{'jitter': 1e-8}` to add a small diagonal ridge for numerical stability.

Troubleshooting equal-weight outcomes
 - Cause: With strong shrinkage and similar variances, $\hat{\Sigma} \approx cI$; under long-only + sum-to-1, min-variance → equal-weight.
 - Remedies: Switch to `empirical`/`oas`/`ewma`, increase lookback, relax caps, or use `factor` with small `n_factors`.

---

Clustering Methodology
----------------------
Before optimization, the ETF universe can be reduced by clustering highly similar ETFs and selecting one representative per cluster.

Inputs:
- Universe CSV: first column is the ticker, second column is AUM in millions of USD.
- Price files: one CSV per ticker in `data/`, named `{ticker}.csv`, with columns `date` and `adj close`.

Method:
1. Load the universe mapping from `data/etf_universe.csv`.
2. Load each ticker price history from `data/{ticker}.csv`.
3. Filter out ETFs with less than 6 years of history by default.
4. Align all remaining ETFs on common dates.
5. Compute daily simple returns.
6. Compute the return correlation matrix.
7. Convert correlation into distance using `1 - |corr|` by default.
8. Run hierarchical clustering with average linkage.
9. Select the largest-AUM ETF inside each cluster.

Rationale:
- Reduces duplicate exposures such as overlapping equity or bond ETFs.
- Improves covariance conditioning by removing near-identical assets.
- Keeps the universe investable by preferring larger, more liquid ETFs.
- Preserves broad market coverage while reducing optimizer noise.

Usage:
```python
from helpers.clustering import cluster_select_representatives_from_csv

selected_tickers, clusters = cluster_select_representatives_from_csv(
    universe_csv_path="data/etf_universe.csv",
    data_dir="data",
    n_clusters=20,
    min_history_years=6,
)

selected_tickers

---

Recent Work (what I changed)
----------------------------
- `helpers/fetch.py`
    - Improved robustness when downloading with `yfinance`: catches download errors, returns empty DataFrames on failure, and avoids overwriting existing CSVs with empty results.
    - Suppresses noisy `yfinance` stdout/stderr messages during download to keep notebook output clean.
    - Filters empty fetch chunks before concatenation when updating local CSVs.

- `helpers/clustering.py`
    - Added `visualize_clusters(clusters, corr=None, returns=None, ...)` which computes a 2D embedding of tickers and plots them colored by cluster. Uses MDS (via `sklearn.manifold.MDS`) on a distance matrix derived from correlations (default `1 - |corr|`). Returns `(fig, ax)` for further customization or saving.

- `main.ipynb`
    - Inserted an example notebook cell that runs `cluster_select_representatives_from_csv(...)`, builds aligned returns for the clustered tickers and calls `visualize_clusters(...)` to render the plot.

- `helpers/backtest.py`
    - Added `backtest_portfolio(weights, start_value=10000.0, returns=None)` to turn any daily weights DataFrame into a portfolio value series. If `returns` is not provided, it loads adjusted-close prices via `load_data` and builds historical daily simple returns.

- `helpers/portfolio.py`
    - Added `build_portfolios_from_prices(prices, start, end, ...)` to compute dynamic rebalanced weights (min-variance, max-Sharpe, market-cap=100% VT, equal-weight) directly from a provided prices matrix. Mirrors the historical pipeline but without fetching.
    - `build_portfolios(...)` now honors the `freq` argument and always includes a `VT` column; adds `equal_weight` (1/N). `market_cap` is set to 100% VT.

- `helpers/montecarlo.py`
    - `simulate_parametric(...)`: multivariate-normal daily return paths estimated from historical mean/cov.
    - `simulate_bootstrap(...)`: IID or moving-block bootstrap on historical daily returns.
    - `apply_backtest_to_simulations(...)`: backtest constant weights on each simulated path.
    - `rebalance_on_simulation(...)`: turn simulated returns → prices, recompute dynamic weights with `build_portfolios_from_prices`, and backtest.
    - `apply_rebalanced_backtest_to_simulations(...)`: batch version returning `(weights, values)` per simulation.

Why these changes?
- Make data fetching more robust and idempotent so repeated notebook runs don't clobber existing data or print confusing warnings from upstream libraries.
- Provide a lightweight visual tool to inspect cluster structure before/after representative selection so you can study which ETFs were grouped together and why.

How to reproduce the visualization quickly
---------------------------------------
In a notebook cell (this is the example I added to `main.ipynb`):

```python
from helpers.clustering import cluster_select_representatives_from_csv, visualize_clusters
from helpers.fetch import load_data
import pandas as pd

ETF_UNIVERSE_CSV_PATH = "data/etf_universe.csv"
selected, clusters = cluster_select_representatives_from_csv(ETF_UNIVERSE_CSV_PATH)

# build aligned price matrix for tickers in the clusters
tickers = sorted([t for members in clusters.values() for t in members])
start_date = '2012-01-01'
end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
prices = pd.concat([load_data(t, start_date, end_date)['adj close'].rename(t) for t in tickers], axis=1, join='inner')
returns = prices.pct_change().dropna()

# show the cluster plot
fig, ax = visualize_clusters(clusters, returns=returns)
fig
```

Backtesting and Monte Carlo (examples)
--------------------------------------
Historical backtest from daily weights:

```python
from helpers.backtest import backtest_portfolio

values = backtest_portfolio(portfolios["min_variance"], start_value=10_000)
values.tail()
```

Parametric Monte Carlo with dynamic rebalancing on simulated paths:

```python
from helpers.montecarlo import simulate_parametric, rebalance_on_simulation

tickers = list(portfolios["min_variance"].columns)
sims = simulate_parametric(tickers, n_days=252, n_sims=10, lookback_years=3, seed=42)

# Rebuild weights and backtest on the first simulation using the same logic as historical
W, V = rebalance_on_simulation(sims[0], lookback_years=4, freq='BMS', which='min_variance', start_value=10_000)
V.tail()
```

Bootstrap Monte Carlo with batch rebalanced backtests:

```python
from helpers.montecarlo import simulate_bootstrap, apply_rebalanced_backtest_to_simulations

sims = simulate_bootstrap(tickers, n_days=252, n_sims=50, lookback_years=3, block_size=10, seed=1)
results = apply_rebalanced_backtest_to_simulations(sims, lookback_years=4, freq='BMS', which='max_sharpe', start_value=10_000)

# results is a list of (weights_df, values_series)
len(results), results[0][1].tail()
```


Technical: Multi-Dimensional Scaling (MDS)
---------------------------------------
Purpose
- Embed objects described by pairwise dissimilarities into a low-dimensional Euclidean space so that Euclidean distances approximate the original dissimilarities.

Classical MDS (Torgerson)
- Given the squared distance matrix $D^{(2)}$ with entries $d_{ij}^2$, let $J = I - \frac{1}{n}11^\top$ and
    $$B = -\tfrac{1}{2} J D^{(2)} J.$$
- Eigendecompose $B = V\Lambda V^\top$. Keep the top $k$ positive eigenvalues $\Lambda_k$ and corresponding eigenvectors $V_k$. The embedding is
    $$X = V_k \Lambda_k^{1/2},$$
    where row $i$ of $X$ gives the $k$-dimensional coordinates for object $i$.

Metric MDS (stress-minimizing)
- Find $X\in\mathbb{R}^{n\times k}$ that minimizes the stress
    $$\mathrm{Stress}(X) = \sqrt{\sum_{i<j} (d_{ij} - \|x_i - x_j\|)^2},$$
    solved iteratively (implemented by `sklearn.manifold.MDS` with `dissimilarity='precomputed'`).

Practical notes
- Convert correlation $\rho$ to distances via, e.g., $d_{ij}=1-|\rho_{ij}|$ (or $d_{ij}=1-\rho_{ij}$).
- Classical MDS requires a Euclidean distance matrix; if $B$ has negative eigenvalues, either drop negatives (use only positive spectrum) or use metric MDS.
- Complexity: forming the full distance matrix costs $O(n^2)$ memory; metric MDS is iterative and can be slow for large $n$ (practical limit depends on CPU and memory).
- Alternatives: PCA on returns (fast), t-SNE, UMAP (nonlinear, parameter-sensitive).

References
- Borg, I., & Groenen, P. J. F. (2005). Modern Multidimensional Scaling: Theory and Applications.
- sklearn.manifold.MDS documentation: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html

---
If you want, I can also:
- add PCA / t-SNE options to `visualize_clusters`,
- add an option to save the plot to file from the helper, or
- add an interactive Plotly-based visualization cell to the notebook.

---

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

