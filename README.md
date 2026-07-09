Leverage Portfolio
===================

Research pipeline that builds, backtests, and stress-tests a leveraged long-only ETF portfolio. Four allocation strategies are compared; the best performer is meant to be run live via IB Gateway (`ib_insync` is in `requirements.txt`, but the execution module is not implemented yet — everything below is backtest/research only).

---

Strategy
--------
Four candidate portfolios, rebalanced periodically over the same ETF universe:

- **Min-variance** — minimizes portfolio volatility (PyPortfolioOpt).
- **Max-Sharpe** — maximizes expected return per unit of risk.
- **Market-cap** — 100% VT (global market-cap-weighted equities), used as a passive benchmark.
- **Equal-weight (1/N)** — naive diversification benchmark.

Target leverage is **1.7x**, applied uniformly to whichever strategy is chosen. Rebalancing happens either on a fixed calendar schedule (e.g. monthly) or early if realized leverage drifts more than **±10%** from target — because leverage moves against you faster on drawdowns than it helps on rallies (a losing day shrinks equity while debt stays fixed, pushing leverage up).

---

Repository Structure
---------------------

### `helpers/fetch.py`
Downloads and caches adjusted-close prices.
- `load_data(ticker, start, end)` — reads `data/{ticker}.csv` if present and only fetches the missing date ranges from `yfinance` (before the cached start, after the cached end), so repeated runs don't re-download data you already have. Falls back to a full fetch if the file is missing or unreadable.
- `common_window(tickers)` — returns the widest `(start, end)` date range that is covered by *every* ticker's CSV, useful for picking a backtest window that all assets can support.
- `_fetch(ticker, start, end)` — internal wrapper around `yfinance.download` that silences its console output and normalizes the result to a single `adj close` column.

### `helpers/stats.py`
Small statistics building blocks used across the codebase.
- `log_returns(prices)` — log price differences, the return convention used for covariance estimation.
- `sharpe(prices, rf, periods)` — annualized Sharpe ratio of a single price series.
- `covariance(returns)` — plain sample covariance matrix.
- `covariance_shrunk(returns)` — Ledoit-Wolf shrinkage covariance (via scikit-learn), more stable than the sample covariance when history is short relative to the number of assets.

### `helpers/portfolio.py`
Turns price history into portfolio weights.
- `min_variance(tickers, as_of, timeframe_years, cov_method, ...)` — pulls a lookback window of returns, estimates a covariance matrix, and solves for the lowest-volatility long-only portfolio with PyPortfolioOpt.
- `max_sharpe(tickers, as_of, timeframe_years, cov_method, ...)` — same setup, but solves for the maximum Sharpe ratio. Tries several QP solvers in sequence (ECOS, CLARABEL, SCS) and falls back to quadratic-utility or min-volatility if all solvers fail to converge, so the pipeline never crashes on one bad optimization.
- `_compute_covariance(returns, method, params)` — shared covariance estimator behind both optimizers; supports `shrunk` (Ledoit-Wolf), `empirical` (sample), `oas` (Oracle Approximating Shrinkage), `ewma` (recency-weighted), and `factor` (PCA-based) methods, plus an optional diagonal jitter for numerical stability.
- `build_portfolios(tickers, start, end, lookback_years, freq, ...)` — end-to-end historical pipeline: fetches prices for all tickers, then delegates to `build_portfolios_from_prices`.
- `build_portfolios_from_prices(prices, start, end, lookback_years, freq, ...)` — the core rebalancing loop. At every rebalance date it takes a rolling lookback window, recomputes min-variance and max-Sharpe weights, sets `market_cap` to 100% VT, and sets `equal_weight` to 1/N, then forward-fills each schedule into a daily weights DataFrame. This is reused for both the historical backtest and the Monte Carlo rebalancing (same logic run on simulated prices).

### `helpers/backtest.py`
- `backtest_portfolio(weights, start_value, returns)` — converts a daily weights DataFrame into a portfolio value series by forward-filling weights onto return dates and compounding the weighted daily returns. If `returns` isn't supplied, it loads historical prices for the weight columns and derives returns itself.

### `helpers/montecarlo.py`
Stress-tests the strategies against simulated futures instead of the one realized history.
- `simulate_parametric(tickers, n_days, n_sims, lookback_years, seed)` — draws daily returns from a multivariate normal distribution fit to historical mean/covariance. Assumes returns are well-described by a Gaussian — fast, but understates fat tails and crash correlation.
- `simulate_bootstrap(tickers, n_days, n_sims, lookback_years, block_size, seed)` — resamples actual historical daily returns instead of assuming a distribution. With `block_size` set, it resamples contiguous blocks (moving-block bootstrap) to preserve autocorrelation/volatility clustering; without it, it's an IID resample.
- `apply_backtest_to_simulations(weights, simulations, start_value, hold)` — backtests a **fixed** weight vector (last or time-averaged from `weights`) across many simulated paths — use this to see how a static allocation performs under randomized markets.
- `rebalance_on_simulation(sim_returns, lookback_years, freq, which, ...)` — compounds one simulated return path into synthetic prices, then reruns the *same* dynamic rebalancing logic used historically (`build_portfolios_from_prices`) on that path. This is the realistic version: it tests whether the optimizer keeps working when the future doesn't look like the past.
- `apply_rebalanced_backtest_to_simulations(simulations, ...)` — batch version of the function above, one `(weights, values)` pair per simulation.
- `simulate_and_backtest_portfolios(tickers, n_days, n_sims, method, ...)` — simulates paths (parametric or bootstrap), rebuilds all four strategies on each path, and returns one combined DataFrame with a `(sim, strategy)` column MultiIndex.
- `save_simulations_parquet(values, path, engine)` — persists that MultiIndex DataFrame to Parquet under `output/`.

### `helpers/clustering.py`
Shrinks the ETF universe before optimization, so the optimizer isn't fed near-duplicate assets (e.g. an aggregate bond ETF alongside its own sleeve constituents), which would make the covariance matrix ill-conditioned.
- `cluster_select_representatives_from_csv(universe_csv_path, data_dir, n_clusters, ...)` — loads `data/etf_universe.csv` (ticker + AUM), filters out ETFs with too little price history, computes a correlation-based distance between the survivors, runs hierarchical clustering, and keeps the largest-AUM (most liquid) ETF from each cluster as its representative.
- `visualize_clusters(clusters, returns, ...)` — projects the correlation-distance matrix into 2D with MDS (multi-dimensional scaling) and scatter-plots tickers colored by cluster, so you can visually sanity-check which ETFs got grouped together.

### `main.ipynb`
The runnable, end-to-end walkthrough: cluster the universe → build all four portfolios over the historical window → backtest and compare them → run parametric and bootstrap Monte Carlo → export simulation results to `output/mc_values.parquet`.

### `data/`
- `etf_universe.csv` — the candidate ETF universe with AUM (in millions USD), used by the clustering step to break ties.
- `{ticker}.csv` — one file per ticker with `date` and `adj close` columns, incrementally maintained by `load_data`.

### `output/`
Generated artifacts (Parquet exports of Monte Carlo runs); not checked in as source data.

---

Covariance Estimators
----------------------
`cov_method` on `min_variance`/`max_sharpe`/`build_portfolios*` controls how the covariance matrix is estimated from returns:

| Method | Idea | Best for |
|---|---|---|
| `shrunk` (default) | Ledoit-Wolf: blend sample covariance with a scaled-identity target | General default; stable with minimal tuning |
| `empirical` | Plain sample covariance, no regularization | Long history relative to number of assets |
| `oas` | Oracle Approximating Shrinkage — an alternative shrinkage intensity to Ledoit-Wolf | Robust alternative when `shrunk` looks too uniform |
| `ewma` | Exponentially weighted, recent data counts more (`cov_params={'span': s}`) | Adapting quickly to a regime change |
| `factor` | PCA factor model (`cov_params={'n_factors': k}`) | Many assets, few independent risk drivers |

If min-variance keeps returning equal weights, the shrinkage is likely washing out real correlation structure — try `empirical`, `oas`, or `ewma`, or reduce clustering redundancy in the universe first.

---

Leverage Mechanics
-------------------
At target leverage `L`, gross exposure is `L × equity` and the rest is margin debt. A market move changes exposure before debt is repaid, so leverage drifts every day: a gain *reduces* leverage, a loss *increases* it (leverage compounds losses faster than gains). The ±10% rebalance band exists to catch that drift before it compounds — at `L=1.7`, a single-day portfolio loss of roughly 7% would already breach the band. IBKR's Reg T maintenance margin caps usable leverage around 4x for ETFs, so the 1.7x target with a 1.87x upper band leaves a wide safety margin before a margin call.

---

Setup
-----
```bash
pip install -r requirements.txt
jupyter lab main.ipynb
```

Quick usage:
```python
from helpers.portfolio import build_portfolios
from helpers.backtest import backtest_portfolio

portfolios = build_portfolios(["VTI", "IEF", "GLD", "VT"], "2015-01-01", "2024-01-01")
values = backtest_portfolio(portfolios["min_variance"], start_value=10_000)
```

```python
from helpers.montecarlo import simulate_bootstrap, rebalance_on_simulation

sims = simulate_bootstrap(["VTI", "IEF", "GLD", "VT"], n_days=252, n_sims=50, block_size=10, seed=1)
weights, values = rebalance_on_simulation(sims[0], which="min_variance")
```

---

Technical Appendix
===================

Only `sharpe()` (in `helpers/stats.py`) is currently implemented. CAGR, Sortino, and Calmar are documented here as the natural next metrics to add on top of the `values` series returned by `backtest_portfolio` — they all consume the same daily portfolio-value path.

### 1. Performance Metrics

Let $E_t$ be portfolio equity (value) at time $t$, and $r_{p,t} = E_t/E_{t-1} - 1$ the daily portfolio return.

**CAGR** — average annual growth rate, geometric rather than arithmetic because returns compound:

$$\text{CAGR} = \left(\frac{E_T}{E_0}\right)^{252/T} - 1$$

where $T$ is the number of trading days in the sample. The exponent $252/T$ annualizes whatever period the backtest actually covers.

**Annualized volatility** — standard deviation of daily returns scaled to a yearly horizon by $\sqrt{252}$ (variance scales linearly with time under the IID assumption, so standard deviation scales with its square root):

$$\sigma_p = \sqrt{252}\,\operatorname{std}(r_{p,t})$$

**Sharpe ratio** — excess return per unit of *total* risk:

$$\text{Sharpe} = \frac{\mu_p^{\text{ann}} - r_f}{\sigma_p}, \qquad \mu_p^{\text{ann}} = 252\cdot\mathbb{E}[r_{p,t}]$$

**Sortino ratio** — same idea, but only penalizes downside deviation, since an investor doesn't mind upside volatility:

$$\text{Sortino} = \frac{\mu_p^{\text{ann}} - r_f}{\sigma_d}, \qquad \sigma_d = \sqrt{252\cdot\mathbb{E}\!\left[\min(r_{p,t} - r_f/252,\,0)^2\right]}$$

**Maximum drawdown** — worst peak-to-trough decline, the metric leverage most directly threatens:

$$\text{MDD} = \min_t\ \frac{E_t - \max_{s \le t} E_s}{\max_{s \le t} E_s}$$

**Calmar ratio** — return earned per unit of worst-case pain, useful for comparing leveraged strategies where volatility alone understates tail risk:

$$\text{Calmar} = \frac{\text{CAGR}}{|\text{MDD}|}$$

---

### 2. Covariance Matrix Theory

The optimizers in `helpers/portfolio.py` need a covariance matrix $\Sigma$ of asset returns; how well $\Sigma$ is estimated determines how trustworthy the resulting weights are.

**Sample covariance** (`cov_method='empirical'`), with returns $r_t$ and mean $\hat\mu$ over $T$ observations:

$$S = \frac{1}{T-1}\sum_{t=1}^{T} (r_t - \hat\mu)(r_t - \hat\mu)^\top$$

This is unbiased but noisy whenever $T$ is not large relative to the number of assets $N$ — with $N$ assets there are $N(N+1)/2$ entries to estimate, and estimation error in $S$ gets amplified by the matrix inversion inside `min_volatility`/`max_sharpe`, producing extreme, unstable weights.

**Ledoit-Wolf shrinkage** (`cov_method='shrunk'`, the default) fixes this by blending $S$ with a low-variance target $F$ (a scaled identity matrix):

$$\hat\Sigma = (1-\alpha)\,S + \alpha\,F, \qquad F = \frac{\operatorname{tr}(S)}{N}I_N$$

The shrinkage intensity $\alpha^\* \in [0,1]$ is chosen analytically to minimize expected estimation error $\mathbb{E}\lVert\hat\Sigma - \Sigma\rVert_F^2$ — no manual tuning needed. Larger $\alpha$ pulls the matrix toward equal variances and zero correlation, which is why min-variance degenerates toward equal-weight when assets look statistically similar.

**OAS** (`cov_method='oas'`) is the same shrinkage-toward-identity idea, but with a shrinkage intensity derived under a Gaussian-data assumption instead of Ledoit-Wolf's distribution-free bound; in practice it often shrinks slightly less aggressively.

**EWMA** (`cov_method='ewma'`) replaces the uniform average in $S$ with exponentially decaying weights $w_t \propto (1-\lambda)\lambda^{T-t}$, so recent observations dominate:

$$\hat\Sigma_{\text{EWMA}} = \sum_t w_t\,(r_t-\hat\mu)(r_t-\hat\mu)^\top$$

This trades estimation stability for responsiveness — it adapts faster to a new volatility regime, at the cost of being noisier since it effectively uses fewer observations.

**Factor model** (`cov_method='factor'`) assumes returns are driven by a small number $k$ of latent factors plus idiosyncratic noise. PCA on demeaned returns $X_0$ gives factor scores $F$ (top-$k$ singular directions) and loadings $B$:

$$\hat\Sigma = B\,\Sigma_F\,B^\top + D$$

where $\Sigma_F$ is the ($k\times k$) factor covariance and $D$ is a diagonal of residual (specific) variances. This concentrates estimation effort on the few directions that actually explain co-movement, leaving less noise to estimate elsewhere — useful when $N$ is large relative to $T$.

---

### 3. Monte Carlo Theory

Monte Carlo (`helpers/montecarlo.py`) exists because a single historical backtest is one draw from one realized path — it can't tell you how a strategy behaves in futures that didn't happen to occur. Both simulators below produce daily-return paths that feed back into `build_portfolios_from_prices`, so the same rebalancing logic used historically is tested under alternative markets.

**Parametric (Gaussian) simulation** (`simulate_parametric`) assumes daily returns are drawn from a multivariate normal fit to historical mean $\hat\mu$ and covariance $\hat\Sigma$:

$$r_t \sim \mathcal{N}(\hat\mu, \hat\Sigma)$$

This is fast and lets you generate arbitrarily many paths, but real returns have fatter tails and time-varying correlation (correlations tend to spike in crashes) than a Gaussian captures — so parametric MC tends to understate tail risk.

**Bootstrap simulation** (`simulate_bootstrap`) instead resamples actual historical return rows, so the marginal distribution of returns (fat tails and all) is preserved exactly:

- *IID bootstrap* (`block_size=None`): sample $T$ rows independently with replacement. This breaks any autocorrelation or volatility clustering present in the original series (each day is treated as unrelated to its neighbors).
- *Moving-block bootstrap* (`block_size=b`): sample contiguous blocks of $b$ consecutive days instead of single days. Because each block preserves the correlation structure *within* it, this keeps volatility clustering and short-term momentum/mean-reversion patterns roughly intact — a closer proxy to how real markets actually move than the IID version.

**Compounding to prices**: once a simulated return path exists, `rebalance_on_simulation` compounds it into a synthetic price path via $\prod_t (1+r_t)$ starting from 1.0, then reruns the real optimizer/rebalancing pipeline on those synthetic prices — this is what makes the Monte Carlo test the *strategy*, not just the *asset returns*.

---

### 4. Multi-Dimensional Scaling (MDS) Theory

`visualize_clusters` (in `helpers/clustering.py`) needs to draw tickers as points in 2D even though the only information available is a pairwise distance (correlation-derived, not spatial). MDS finds a low-dimensional embedding whose Euclidean distances best approximate that dissimilarity matrix.

**Classical (Torgerson) MDS**: given squared distances $D^{(2)}$, double-center it,

$$B = -\tfrac{1}{2}J D^{(2)} J, \qquad J = I - \tfrac{1}{n}\mathbf{1}\mathbf{1}^\top$$

then eigendecompose $B = V\Lambda V^\top$ and keep the top-$k$ positive eigenpairs:

$$X = V_k \Lambda_k^{1/2}$$

Row $i$ of $X$ is the $k$-dimensional coordinate of ticker $i$. Double-centering removes the arbitrary choice of origin, leaving only relative distances — which is all correlation distance actually encodes.

**Metric MDS** (what `sklearn.manifold.MDS` uses here) instead solves an optimization problem directly, minimizing the stress between target and embedded distances:

$$\text{Stress}(X) = \sqrt{\sum_{i \lt j}\big(d_{ij} - \lVert x_i - x_j\rVert\big)^2}$$

This is more robust when the distance matrix isn't perfectly Euclidean (e.g. $1-|\rho_{ij}|$ distances don't always satisfy the triangle inequality exactly), which is the case here since correlation distance is only an approximate metric.

**Turning correlation into distance**: $d_{ij} = 1 - |\rho_{ij}|$ (used by `visualize_clusters` and the clustering step) treats strongly negatively correlated assets as "close" — appropriate here because the goal is deduplicating *redundant exposure*, and an asset highly anti-correlated with another is just as substitutable (e.g. via shorting or as a hedge pair) for diversification purposes as one highly positively correlated with it.
