Leverage Portfolio
==================

Research pipeline for a leveraged, long-only ETF portfolio. It selects a small ETF universe, builds portfolios, backtests them, and stress-tests them with Monte Carlo simulation.

- Research and backtesting only.
- Live execution through IB Gateway is not implemented.

Strategy
--------

The same ETF universe is used for four periodically rebalanced portfolios:

- **Min-variance:** lowest estimated volatility.
- **Max-Sharpe:** highest estimated excess return per unit of risk.
- **Market-cap:** 100% VT, used as a passive benchmark.
- **Equal-weight:** equal allocation to each selected ETF.

The default research configuration uses **2.0x target leverage**. It can rebalance on a calendar schedule or when realized leverage moves outside a tolerance band. Losses raise leverage because debt remains while equity falls; gains reduce it.

Pipeline
--------

1. Select liquid, long-history ETF representatives from correlation clusters.
2. Estimate rolling portfolio weights and run an unlevered historical backtest.
3. Apply margin financing and leverage-rebalancing rules.
4. Generate bootstrap or parametric return paths.
5. Rebuild portfolios on each path and compare terminal values, risk, and drawdowns.


Project Layout
--------------

- `main.ipynb`: end-to-end workflow and result analysis.
- `helpers/fetch.py`: cached prices and synthetic history.
- `helpers/clustering.py`: correlation clustering and ETF selection.
- `helpers/portfolio.py`: rolling portfolio construction.
- `helpers/backtest.py`: unlevered backtest.
- `helpers/leverage.py`: margin and leverage simulation.
- `helpers/montecarlo.py`: parametric and bootstrap simulation.
- `helpers/stats.py`: return, covariance, and summary helpers.
- `data/`: cached prices, ETF universe, and EFFR data.
- `output/`: generated Parquet results.

The default run uses a three-year optimization window, a separate 20-year simulation history, 60-day moving blocks, and a five-year simulation horizon. Historical optimization uses date-aligned FRED EFFR as the risk-free rate. Bootstrap paths sample asset returns and EFFR jointly.

Optimized assets use configurable 10%-40% bounds by default. The same bounds apply to min-variance and max-Sharpe. `max_sharpe_sensitivity.csv` compares historical-mean, EMA, and equal expected-return assumptions; equal expected returns remove noisy cross-asset return ranking.

If no bounded portfolio has an expected return above the estimated risk-free rate, max-Sharpe is undefined. The implementation then uses the same capped min-variance solution instead of deleting the simulation path. This conservative fallback is economically interpretable: when every feasible risky mix has non-positive estimated excess return, the return forecast contains no usable Sharpe-ranking signal.

Synthetic Pre-Inception History
-------------------------------

Some ETFs do not have the 20-year history required for clustering. A longer-history proxy extends the missing period while preserving the ETF's observed prices after inception.

- Fund proxies already include their management fee, so no extra adjustment is applied.
- Raw indexes and futures do not include fund costs, so the ETF's daily TER drag is subtracted.
- The synthetic series is rebased to equal the ETF price at the splice date. Real observations are never changed.

For an ETF with first real observation $t_0$, proxy return $r_t^{proxy}$, and annual TER:

$$P_t^{synthetic}=P_{t_0}^{ETF}\prod_{s=t+1}^{t_0}\frac{1}{1+r_s^{proxy}-\text{TER}/252}, \qquad t<t_0$$

| ETF | Proxy | Extension | Limitation |
|---|---|---|---|
| `AGG`, `BND` | `VBMFX` | 1986 onward | Broad US aggregate-bond proxy. |
| `VOO` | `VFINX` | 1980 onward | Same S&P 500 benchmark. |
| `HYG` | `VWEHX` | 1980 onward | Similar, not identical, high-yield exposure. |
| `VNQ` | `VGSIX` | 1996 onward | US REIT proxy. |
| `EMLC` | `PREMX` | 1994 onward | USD EM debt differs from local-currency debt. |
| `GLD` | `GC=F` | 2000 onward | Gold futures; GLD TER is deducted. |
| `GSG` | `^SPGSCI` | 1984 onward | Excludes collateral yield. |
| `SGOV` | `BIL`, then `EFFR_CASH` | 2000 onward | Chained cash proxy. |
| `VNQI` | `CSRSX` | 1991 onward | International real-estate proxy. |
| `PFF` | `PREFX` | 2000 onward | Open-end preferred-stock proxy. |
| `IEMG` | `VEIEX` | 1994 onward | Emerging-markets equity proxy. |
| `VT` | `VTSMX` 55% + `VGTSX` 45% | 1996 onward | Static US/ex-US split is approximate. |

`DBMF` and `QAI` have no suitable long-history proxy and are excluded by the history threshold. `N_CLUSTERS=5` selects one large-AUM representative from each broad correlation cluster. Raising it makes the universe more granular; lowering it merges more exposures.

Covariance Estimators
---------------------

The optimizer estimates a covariance matrix $\Sigma$ from returns. Its quality materially affects portfolio weights.

| Method | Idea | Useful when |
|---|---|---|
| `shrunk` | Ledoit-Wolf shrinkage toward a scaled identity matrix | Stable general-purpose estimate. |
| `empirical` | Plain sample covariance | History is long relative to assets. |
| `oas` | Oracle Approximating Shrinkage | Less aggressive shrinkage. |
| `ewma` | Recent returns receive more weight | Volatility regimes may have changed. |
| `factor` | PCA factor covariance model | Many assets share few risk drivers. |

If min-variance repeatedly returns equal weights, shrinkage may be suppressing meaningful correlation differences. Try `empirical`, `oas`, or `ewma`, or reduce redundant assets before optimization.

### Why Max-Sharpe Can Resemble Min-Variance

Max-Sharpe maximizes expected excess return per unit of volatility:

$$\max_w\frac{\mu^\top w-r_f}{\sqrt{w^\top\Sigma w}}$$

Min-variance ignores expected returns and minimizes total variance:

$$\min_w w^\top\Sigma w$$

Expected returns $\mu$ are usually much noisier than covariance estimates $\Sigma$. If the estimated return advantage is weak or unstable, Max-Sharpe is driven mainly by the same covariance structure as min-variance. The model uses date-aligned EFFR for $r_f$, preventing cash-like ETFs from appearing to have excess return simply because the risk-free rate was set to zero.

Modelling Limits
---------------

- **Trading costs:** weights are lagged one return observation, but transaction costs, bid-ask spreads, taxes, and turnover limits are excluded.
- **Rates:** EFFR is used as both the optimizer's risk-free rate and the margin-rate benchmark. Bootstrap block boundaries can create rate jumps; parametric asset and rate paths remain simulated separately.
- **Failed paths:** optimizer failures are reported in `run_config.json`. Equity wipeouts are retained at zero and included in terminal and ruin statistics.
- **Clustering:** $1-|\rho|$ treats positive and negative correlation as equally redundant. Use $1-\rho$ when negatively correlated assets should remain distinct diversifiers.

Leverage Mechanics
------------------

At target leverage $L$, gross exposure is $L\times E$ and debt is $(L-1)\times E$, where $E$ is equity. Market moves affect gross exposure immediately while debt remains outstanding. Therefore a loss raises realized leverage and a gain lowers it.

The margin simulation:

- Accrues debt daily at EFFR plus a configurable broker spread.
- Uses actual/360 for elapsed calendar days.
- Resets on daily, weekly, or monthly schedules, or at a configured leverage band.
- Forces a reset at the hard cap, a simplified margin-call proxy.
- Liquidates a path at zero if equity becomes non-positive and keeps all later values at zero.

Daily resets keep leverage closest to target. Monthly resets allow more drift. A band balances the two by resetting only after a material move. Financing cost reduces equity independently of asset returns and is therefore essential to a leveraged backtest.

Setup
-----

```bash
pip install -r requirements.txt
jupyter lab main.ipynb
```

Run the notebook from top to bottom. It first clusters the 20-year ETF universe, selects the largest-AUM representative from each cluster, and passes those representatives to the explicit simulation configuration. It then writes results and diagnostics to `output/`.

For a reproducible non-interactive run:

```bash
python run_research.py
```

Use `python run_research.py --n-sims 10` for a quick smoke run. The default is 1,000 paths.

The command-line runner uses the explicit default universe `VT`, `BND`, `SGOV`, `GLD`, and `GSG`. The notebook recomputes that universe from clustering before each full run. In both cases, the final tickers are stored in `run_config.json`.

Generated artifacts:

- `mc_values_unlevered.parquet`
- `mc_values_leveraged_daily.parquet`
- `mc_values_leveraged_weekly.parquet`
- `mc_values_leveraged_monthly.parquet`
- `mc_values_leveraged_band.parquet`
- Matching `summary_*.csv` files
- `average_weights_by_simulation.csv` and `average_weights_summary.csv`
- `max_sharpe_sensitivity.csv`
- `run_config.json`

Every summary includes the horizon, trading-day count, starting value, leverage, rebalance mode, history windows, block size, and requested simulation count. CAGR and annualized volatility are conditional on survival because they are undefined after liquidation. Terminal statistics, maximum drawdown, and ruin rate include ruined paths.

Technical Appendix
==================

### 1. Performance Metrics

Let $E_t$ be equity at time $t$, and let $r_{p,t}=E_t/E_{t-1}-1$ be the daily portfolio return.

**CAGR** is the geometric annual growth rate:

$$\text{CAGR}=\left(\frac{E_T}{E_0}\right)^{252/T}-1$$

$T$ is the number of trading days. The exponent annualizes the full compounded change rather than averaging daily returns arithmetically.

**Annualized volatility** scales the standard deviation of daily returns by $\sqrt{252}$:

$$\sigma_p=\sqrt{252}\,\text{std}(r_{p,t})$$

This scaling assumes variance grows approximately linearly with time. It is a convention, not a guarantee, especially during clustered volatility.

**Sharpe ratio** measures excess return per unit of total volatility:

$$\text{Sharpe}=\frac{\mu_p^{ann}-r_f}{\sigma_p}, \qquad \mu_p^{ann}=252\,\mathbb{E}[r_{p,t}]$$

The current helper uses log returns and subtracts $r_f/252$. This approximates a log-return Sharpe and can differ from the simple-return definition, especially for volatile or leveraged portfolios.

**Sortino ratio** penalizes only downside deviation:

$$\text{Sortino}=\frac{\mu_p^{ann}-r_f}{\sigma_d}, \qquad \sigma_d=\sqrt{252\,\mathbb{E}[\min(r_{p,t}-r_f/252,0)^2]}$$

It distinguishes undesirable downside volatility from positive surprises.

**Maximum drawdown** is the worst peak-to-trough loss:

$$\text{MDD}=\min_t\frac{E_t-\max_{s\leq t}E_s}{\max_{s\leq t}E_s}$$

**Calmar ratio** compares long-run return with the worst observed drawdown:

$$\text{Calmar}=\frac{\text{CAGR}}{|\text{MDD}|}$$

### 2. Covariance Matrix Theory

With return vector $r_t$, mean $\hat\mu$, and $T$ observations, sample covariance is

$$S=\frac{1}{T-1}\sum_{t=1}^{T}(r_t-\hat\mu)(r_t-\hat\mu)^\top$$

It is unbiased, but noisy when the number of assets is large relative to the available history. Optimization can amplify that noise into unstable or concentrated weights.

**Ledoit-Wolf shrinkage** blends $S$ with a scaled identity target:

$$\hat\Sigma=(1-\alpha)S+\alpha F, \qquad F=\frac{\operatorname{tr}(S)}{N}I_N$$

The analytically selected $\alpha^*\in[0,1]$ minimizes expected squared estimation error $\mathbb{E}\lVert\hat\Sigma-\Sigma\rVert_F^2$. Larger $\alpha$ reduces estimation noise but pulls variances and correlations toward a common structure, sometimes producing equal-weight-like allocations.

**OAS** uses the same shrinkage target with an intensity derived under a Gaussian assumption. It often shrinks slightly less than Ledoit-Wolf.

**EWMA** emphasizes recent observations:

$$\hat\Sigma_{EWMA}=\sum_t w_t(r_t-\hat\mu)(r_t-\hat\mu)^\top, \qquad w_t\propto(1-\lambda)\lambda^{T-t}$$

It responds faster to regime changes but has higher sampling noise because its effective sample is smaller.

**Factor covariance** represents returns through $k$ common PCA factors plus residual risk:

$$\hat\Sigma=B\Sigma_F B^\top+D$$

$B$ contains factor loadings, $\Sigma_F$ is factor covariance, and $D$ is diagonal residual variance. This reduces the number of noisy relationships that must be estimated directly.

### 3. Monte Carlo Theory

A historical backtest is one realized market path. Monte Carlo generates alternative paths and reruns the allocation process, testing the strategy rather than only a fixed allocation.

**Parametric simulation** draws daily returns from a fitted multivariate normal distribution:

$$r_t\sim\mathcal{N}(\hat\mu,\hat\Sigma)$$

It is fast and can generate many paths, but Gaussian returns understate fat tails and the tendency of correlations to rise during market stress.

**Bootstrap simulation** resamples observed return rows and therefore retains their empirical marginal distribution.

- **IID bootstrap:** samples individual rows independently. It removes serial dependence and volatility clustering.
- **Moving-block bootstrap:** samples contiguous blocks of length $b$. It approximately preserves short-run autocorrelation and volatility clustering within each block.

Simulated returns are compounded into prices with $\prod_t(1+r_t)$, then the rolling portfolio construction is run again. This preserves the feedback between market path, estimated parameters, and rebalanced weights.

### 4. Multi-Dimensional Scaling Theory

Clustering uses correlation-derived distances, while the visualization needs two-dimensional coordinates. MDS finds coordinates with Euclidean distances close to the original distances.

**Classical MDS** double-centers squared distances $D^{(2)}$:

$$B=-\tfrac{1}{2}JD^{(2)}J, \qquad J=I-\tfrac{1}{n}\mathbf{1}\mathbf{1}^\top$$

After eigendecomposition $B=V\Lambda V^\top$, the two-dimensional representation is

$$X=V_k\Lambda_k^{1/2}$$

Double-centering removes the arbitrary origin. Only relative distances matter.

**Metric MDS** minimizes embedding stress directly:

$$\text{Stress}(X)=\sqrt{\sum_{i<j}\left(d_{ij}-\lVert x_i-x_j\rVert\right)^2}$$

This is useful when the distance matrix is not perfectly Euclidean. The default $d_{ij}=1-|\rho_{ij}|$ treats strong negative correlation as close. For long-only diversification, $d_{ij}=1-\rho_{ij}$ often better preserves negatively correlated hedges.