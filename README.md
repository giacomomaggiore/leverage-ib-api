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
Reads cached adjusted-close prices.
- `load_data(ticker, start, end)` — reads and slices `data/{ticker}.csv`. Every requested CSV is expected to exist and contain valid `adj close` data.
- `common_window(tickers)` — returns the widest `(start, end)` date range that is covered by *every* ticker's CSV, useful for picking a backtest window that all assets can support.

### `helpers/stats.py`
Small statistics building blocks used across the codebase.
- `log_returns(prices)` — log price differences, the return convention used for covariance estimation.
- `sharpe(prices, rf, periods)` — annualized Sharpe ratio of a single price series.
- `covariance(returns)` — plain sample covariance matrix.
- `covariance_shrunk(returns)` — Ledoit-Wolf shrinkage covariance (via scikit-learn), more stable than the sample covariance when history is short relative to the number of assets.

### `helpers/portfolio.py`
Turns price history into portfolio weights.
- `min_variance(tickers, as_of, timeframe_years, cov_method, ...)` — pulls a lookback window of returns, estimates a covariance matrix, and solves for the lowest-volatility long-only portfolio with PyPortfolioOpt.
- `max_sharpe(tickers, as_of, timeframe_years, cov_method, ...)` — same setup, but solves the maximum-Sharpe objective directly. Solver failures surface as errors instead of silently changing the strategy objective.
- `_compute_covariance(returns, method, params)` — shared annualized covariance estimator behind both optimizers; supports `shrunk` (Ledoit-Wolf), `empirical` (sample), `oas` (Oracle Approximating Shrinkage), `ewma` (recency-weighted), and `factor` (PCA-based) methods, plus an optional diagonal jitter for numerical stability.
- `build_portfolios(tickers, start, end, lookback_years, freq, ...)` — end-to-end historical pipeline: fetches the preceding lookback history, then starts allocating only after that warm-up period.
- `build_portfolios_from_prices(prices, start, end, lookback_years, freq, ...)` — the core rebalancing loop. At every rebalance date it takes the available rolling lookback window, recomputes min-variance and max-Sharpe weights, sets `market_cap` to 100% VT, and sets `equal_weight` to 1/N, then forward-fills each schedule into a daily weights DataFrame. Monte Carlo calls use the same function but initially have only the simulated path's short, expanding history.

### `helpers/backtest.py`
- `portfolio_returns(weights, returns)` — forward-fills a daily weights DataFrame onto return dates and computes the weighted daily return series. If `returns` isn't supplied, it loads historical prices for the weight columns and derives returns itself. A weight decided using date $t$ is applied from the next return observation, preventing same-day look-ahead. Missing return columns are dropped and surviving weights are renormalized. Shared by `backtest_portfolio` and `leverage_backtest`.
- `backtest_portfolio(weights, start_value, returns)` — compounds `portfolio_returns` into a portfolio value series (unlevered, no margin loan).

### `helpers/leverage.py`
Simulates a margin-loan-financed leveraged portfolio, modeled on IBKR's Reg T margin mechanics.
- `load_margin_rate(index, spread)` — daily annualized margin borrow rate for each date in `index`: FRED EFFR (`data/FRED_EFFR.csv`) plus a `spread` approximating IBKR's real tiered markup over the benchmark rate (EFFR alone understates actual borrowing cost).
- `leverage_backtest(weights, leverage, start_value, returns, freq, band, hard_cap, spread)` — the core simulation. Each trading observation: gross exposure moves with the weighted return of the underlying assets (`portfolio_returns`), while margin debt accrues the annual rate over the elapsed calendar days using actual/360, and equity is the residual (`gross_exposure - debt`). Gross exposure is reset to `leverage × equity` (a rebalance) when:
  - `freq` is set to `'daily'`, `'weekly'`, or `'monthly'` — on the first trading day of each new period; or
  - `freq=None` — as soon as realized leverage drifts outside `[leverage×(1-band), leverage×(1+band)]` (default band ±10%, matching the target/band described below).
  - Independently of either mode, a `hard_cap` breach (default 4.0, IBKR's approximate Reg T ceiling) forces an immediate rebalance, modeling a margin call.
  - Raises if equity is wiped out (`equity <= 0`) rather than silently returning nonsense leverage.
  - Returns a DataFrame indexed by date with columns `portfolio_value, gross_exposure, debt, leverage, margin_rate, rebalanced`. On rebalancing days, `leverage` is measured before the reset while gross exposure and debt are reported after it.

### `helpers/montecarlo.py`
Stress-tests the strategies against simulated futures instead of the one realized history.
- `simulate_parametric(tickers, n_days, n_sims, lookback_years, seed)` — draws daily returns from a multivariate normal distribution fit to historical mean/covariance. Assumes returns are well-described by a Gaussian — fast, but understates fat tails and crash correlation.
- `simulate_bootstrap(tickers, n_days, n_sims, lookback_years, block_size, seed)` — resamples actual historical daily returns instead of assuming a distribution. With `block_size` set, it resamples contiguous blocks (moving-block bootstrap) to preserve autocorrelation/volatility clustering; without it, it's an IID resample.
- `apply_backtest_to_simulations(weights, simulations, start_value, hold)` — backtests a **fixed** weight vector (last or time-averaged from `weights`) across many simulated paths — use this to see how a static allocation performs under randomized markets.
- `rebalance_on_simulation(sim_returns, lookback_years, freq, which, ...)` — prepends a historical return window, compounds the combined history into synthetic prices, then reruns `build_portfolios_from_prices`. The first simulated allocation therefore has the requested full lookback. A `market_cap` simulation requires a `VT` return column.
- `apply_rebalanced_backtest_to_simulations(simulations, ...)` — batch version of the function above, one `(weights, values)` pair per simulation.
- `simulate_and_backtest_portfolios(tickers, n_days, n_sims, method, ...)` — simulates paths (parametric or bootstrap), rebuilds all four strategies on each path, and returns one combined DataFrame with a `(sim, strategy)` column MultiIndex.
- `leverage_backtest_on_simulation(sim_returns, ..., which, leverage, ...)` — builds dynamic weights on a single simulated path and runs the leveraged backtest (uses the simulated EFFR path if present). Useful when you want both the weight schedule and the full leverage state for diagnostics.
- `save_simulations_parquet(values, path, engine)` — persists a DataFrame with `(sim, strategy)` MultiIndex columns to Parquet. For compatibility the `sim` level is written as strings; when reading back, cast to integers if you want numeric indexing (see examples below).

### `helpers/clustering.py`
Shrinks the ETF universe before optimization, so the optimizer isn't fed near-duplicate assets (e.g. an aggregate bond ETF alongside its own sleeve constituents), which would make the covariance matrix ill-conditioned.
- `cluster_select_representatives_from_csv(universe_csv_path, data_dir, n_clusters, ...)` — loads `data/etf_universe.csv` (ticker + AUM), filters out ETFs with too little price history, computes a correlation-based distance between the survivors, runs hierarchical clustering, and keeps the largest-AUM (most liquid) ETF from each cluster as its representative. The current implementation loads through `helpers.fetch.DATA_DIR`; its `data_dir` argument is not yet applied.
- `visualize_clusters(clusters, returns, ...)` — projects the correlation-distance matrix into 2D with MDS (multi-dimensional scaling) and scatter-plots tickers colored by cluster, so you can visually sanity-check which ETFs got grouped together.

### `main.ipynb`
The runnable, end-to-end walkthrough: cluster the universe → build all four portfolios over the historical window → backtest and compare them → run bootstrap Monte Carlo → export unlevered simulation results to `output/mc_values.parquet` and leveraged Monte Carlo results by rebalance mode.

There are dedicated cells for both unlevered and leveraged exports. The leveraged cell simulates paths once, reuses those same paths across `daily`, `weekly`, `monthly`, and band-based rebalancing, and saves one Parquet file per mode so the results are directly comparable.

### `data/`
- `etf_universe.csv` — the candidate ETF universe with AUM (in millions USD), used by the clustering step to break ties.
- `{ticker}.csv` — one valid file per ticker with `date` and `adj close` columns.
- `FRED_EFFR.csv` — daily Effective Federal Funds Rate (percent) from FRED, used by `helpers/leverage.py` as the margin-loan benchmark rate.

### `output/`
Generated artifacts (Parquet exports of Monte Carlo runs); not checked in as source data.
- `mc_values.parquet` — unlevered Monte Carlo values for all four strategies.
- `mc_leverage_values_daily.parquet`, `mc_leverage_values_weekly.parquet`, `mc_leverage_values_monthly.parquet`, `mc_leverage_values_band.parquet` — leveraged Monte Carlo values for each leverage rebalancing mode.

---

Covariance Estimators
----------------------
`cov_method` on `min_variance`/`max_sharpe`/`build_portfolios*` controls how the covariance matrix is estimated from returns:

| Method | Idea | Best for |
|---|---|---|
| `shrunk` | Ledoit-Wolf: blend sample covariance with a scaled-identity target | Default for standalone `min_variance` and `max_sharpe`; stable with minimal tuning |
| `empirical` | Plain sample covariance, no regularization | Long history relative to number of assets |
| `oas` | Oracle Approximating Shrinkage — an alternative shrinkage intensity to Ledoit-Wolf | Default for `build_portfolios*`; robust when `shrunk` looks too uniform |
| `ewma` | Exponentially weighted, recent data counts more (`cov_params={'span': s}`) | Adapting quickly to a regime change |
| `factor` | PCA factor model (`cov_params={'n_factors': k}`) | Many assets, few independent risk drivers |

If min-variance keeps returning equal weights, the shrinkage is likely washing out real correlation structure — try `empirical`, `oas`, or `ewma`, or reduce clustering redundancy in the universe first.

### Why Max-Sharpe Can Resemble Min-Variance

Max-Sharpe is not a pure highest-return portfolio. It maximizes expected excess return per unit of volatility:

$$\max_w \frac{\mu^\top w - r_f}{\sqrt{w^\top \Sigma w}}$$

Min-variance ignores expected return and minimizes portfolio variance:

$$\min_w w^\top \Sigma w$$

In rolling historical windows, the expected-return estimate $\mu$ is often much noisier than the covariance estimate $\Sigma$. When the return signal is weak, unstable, or not large enough to compensate for risk, the Max-Sharpe optimizer is mostly governed by the same covariance matrix as min-variance. With long-only constraints and a shrinkage covariance estimator, both optimizers can therefore choose very similar defensive allocations.

The implementation solves the Max-Sharpe objective directly. If the solver cannot solve the problem, it raises rather than silently substituting a different portfolio. To test whether the similarity is structural, compare the weight distance between `portfolios["min_variance"]` and `portfolios["max_sharpe"]`, then vary `cov_method` (`empirical`, `oas`, `ewma`) or `lookback_years`.

### Current Modelling Limitations

The results are research diagnostics, not implementation-ready performance estimates. In particular:

- **Execution and costs**: weights are lagged by one return observation, but the model still excludes transaction costs, bid-ask spreads, taxes, and turnover constraints.
- **Rate paths and financing**: bootstrap asset returns retain sampled contemporaneous EFFR observations, but block boundaries can create rate-level jumps; parametric asset returns and EFFR changes are simulated separately. Interest accrues over elapsed calendar days using actual/360, but the rate spread remains a simplified approximation of broker pricing.
- **Clustering**: the default $1-|\rho|$ distance treats positive and negative correlations as equally redundant. For a long-only portfolio, a negatively correlated ETF can be valuable diversification; use `distance_metric="1-corr"` when preserving such hedges matters.

---

Leverage Mechanics
-------------------
At target leverage `L`, gross exposure is `L × equity` and the rest is margin debt. A market move changes exposure before debt is repaid, so leverage drifts every day: a gain *reduces* leverage, a loss *increases* it (leverage compounds losses faster than gains). The ±10% rebalance band exists to catch that drift before it compounds — at `L=1.7`, a single-day portfolio loss of roughly 7% would already breach the band. IBKR's Reg T maintenance margin caps usable leverage around 4x for ETFs, so the 1.7x target with a 1.87x upper band leaves a wide safety margin before a margin call.

This is implemented in `helpers/leverage.py` (`leverage_backtest`), which simulates the margin loan day by day rather than just applying a constant multiplier to returns:

- **Financing cost**: margin debt accrues daily interest at FRED EFFR plus a `spread` (default 1%, approximating IBKR's tiered markup over the benchmark rate), compounded under an actual/360 day-count — IBKR's own convention. This cost is what actually erodes equity between rebalances; a flat "leverage × return" backtest would miss it entirely.
- **Rebalance triggers**: either a fixed calendar schedule (`freq='daily'|'weekly'|'monthly'`) or, if no frequency is given, the ±10% drift band described above. A separate `hard_cap` (default 4.0) forces an emergency rebalance if leverage ever breaches it, independent of the normal schedule/band — a proxy for a margin call.
- Daily rebalancing pins leverage tightly to target (drift is bounded by one day's move); monthly rebalancing can let leverage drift substantially during sustained drawdowns (e.g. leverage rose to ~2.3x on a 1.7x target for VT during the 2022 selloff, under monthly rebalancing) before the next scheduled reset catches it — illustrating why the band exists as a faster backstop than a fixed calendar.

In Monte Carlo backtests, the simulated EFFR path (if present in the simulation as an `EFFR` column) is passed into `leverage_backtest` so financing costs evolve consistently with the simulated rate environment.

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

```python
from helpers.portfolio import build_portfolios
from helpers.leverage import leverage_backtest

portfolios = build_portfolios(["VTI", "IEF", "GLD", "VT"], "2015-01-01", "2024-01-01")

# Band-based rebalancing (default): resets to 1.7x whenever leverage drifts past +/-10%
margin = leverage_backtest(portfolios["min_variance"], leverage=1.7, start_value=10_000)

# Calendar-based rebalancing instead of the drift band
margin_monthly = leverage_backtest(portfolios["min_variance"], leverage=1.7, freq="monthly")
```

Unlevered Monte Carlo backtests (all four strategies) and Parquet export:
```python
from pathlib import Path
from helpers.montecarlo import simulate_and_backtest_portfolios, save_simulations_parquet

selected_tickers = ["VTI", "IEF", "GLD"]

values = simulate_and_backtest_portfolios(
    tickers=selected_tickers,
    n_days=252 * 5,
    n_sims=50,
    lookback_years=3,
    block_size=10,
    seed=1,
    method="bootstrap",
    start_value=10_000.0,
)

save_simulations_parquet(values, "output/mc_values.parquet", engine="pyarrow")
```

Leveraged Monte Carlo backtests (all four strategies) and per-frequency Parquet exports:
```python
from pathlib import Path
import pandas as pd
from helpers.montecarlo import simulate_bootstrap, save_simulations_parquet
from helpers.portfolio import build_portfolios_from_prices
from helpers.leverage import leverage_backtest

# Configuration
selected_tickers = ["VTI", "IEF", "GLD"]
n_days, n_sims, lookback_years = 252*5, 5, 3
leverage = 2.0
lev_rebalance_freq_list = ["daily", "weekly", "monthly", None]
band, hard_cap, spread = 0.05, 4.0, 0.01

# Ensure VT exists for the market-cap benchmark
base_tickers = list(dict.fromkeys(selected_tickers))
sim_tickers = base_tickers.copy()
if "VT" not in sim_tickers:
  sim_tickers.append("VT")

# 1) Simulate return paths (includes an EFFR column when available)
simulations = simulate_bootstrap(
  sim_tickers,
  n_days=n_days,
  n_sims=n_sims,
  lookback_years=lookback_years,
  block_size=10,
  seed=1,
)

# 2) Rebuild strategies on synthetic prices and run leveraged backtests
strategies = ["min_variance", "max_sharpe", "market_cap", "equal_weight"]
for lev_rebalance_freq in lev_rebalance_freq_list:
  all_values = []
  for i, sim in enumerate(simulations, start=1):
    idx = pd.bdate_range(start=pd.Timestamp("2000-01-03"), periods=sim.shape[0])
    effr = sim["EFFR"].copy() if "EFFR" in sim.columns else None
    if effr is not None:
      effr.index = idx
    rets = sim.drop(columns=["EFFR"], errors="ignore").copy()
    rets.index = idx

    # Optimized/equal-weight portfolios use selected tickers; VT remains available for market_cap returns.
    prices = (1.0 + rets[base_tickers]).cumprod()
    weights = build_portfolios_from_prices(prices, start=prices.index.min(), end=prices.index.max(), lookback_years=lookback_years, freq="BMS")
    for strategy in strategies:
      lev_df = leverage_backtest(
        weights[strategy],
        leverage=leverage,
        start_value=10_000.0,
        returns=rets,
        effr=effr,
        freq=lev_rebalance_freq,
        band=band,
        hard_cap=hard_cap,
        spread=spread,
      )
      all_values.append(lev_df["portfolio_value"].rename((i, strategy)))

  values_df = pd.concat(all_values, axis=1)
  values_df.columns = pd.MultiIndex.from_tuples(values_df.columns, names=["sim", "strategy"])
  values_df.index.name = "date"
  tag = lev_rebalance_freq if lev_rebalance_freq else "band"
  save_simulations_parquet(values_df, f"output/mc_leverage_values_{tag}.parquet", engine="pyarrow")
```

Reading and slicing the saved Parquet (note the `sim` level stored as strings):
```python
import pandas as pd
from helpers.stats import quantiles_df

df = pd.read_parquet("output/mc_leverage_values_monthly.parquet", engine="pyarrow")

# Cast the first (sim) level to int for numeric indexing like df[(1, "min_variance")]
if isinstance(df.columns, pd.MultiIndex):
  try:
    df.columns = pd.MultiIndex.from_tuples([(int(sim), strat) for sim, strat in df.columns], names=df.columns.names)
  except Exception:
    pass

series_one = df[(1, "min_variance")]
by_strategy = df.xs("min_variance", level="strategy", axis=1)
print(quantiles_df(by_strategy))
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

**Sharpe ratio** — excess return per unit of *total* risk. The following is the conventional simple-return definition:

$$\text{Sharpe} = \frac{\mu_p^{\text{ann}} - r_f}{\sigma_p}, \qquad \mu_p^{\text{ann}} = 252\cdot\mathbb{E}[r_{p,t}]$$

The current `sharpe()` helper instead uses log returns and subtracts the simple annual risk-free rate as $r_f/252$. That is an approximation to a log-return Sharpe and differs from the formula above, especially for volatile or leveraged paths.

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

The shrinkage intensity $\alpha^{*} \in [0,1]$ is chosen analytically to minimize expected estimation error
$\mathbb{E}\left\|\hat\Sigma - \Sigma\right\|_F^2$ — no manual tuning needed. Larger $\alpha$ pulls the matrix toward equal variances and zero correlation, which is why min-variance degenerates toward equal-weight when assets look statistically similar.

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

**Turning correlation into distance**: the default is $d_{ij} = 1 - |\rho_{ij}|$ (with `1-corr` also available). Absolute correlation treats strongly negative correlation as "close". This can be reasonable when the goal is purely exposure deduplication, but it is not generally appropriate for this long-only strategy: negatively correlated assets are diversification benefits, not substitutes. Use $d_{ij}=1-\rho_{ij}$ when that distinction matters.
