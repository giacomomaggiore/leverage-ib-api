Leverage Portfolio
==================

Research pipeline for a leveraged, long-only ETF portfolio. It selects a small ETF universe, builds rolling portfolios inside simulated market paths, and compares unlevered and margin-financed outcomes.

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
2. Bootstrap joint asset-return and EFFR paths with 60-day moving blocks.
3. Estimate monthly portfolio weights from a three-year warm-up and the simulated path.
4. Apply unlevered and margin-financed portfolio rules to the same returns.
5. Compare terminal values, CAGR, volatility, drawdown, and liquidation rates.


Code Map
--------

- `main.ipynb`: full research run and result charts.
- `run_research.py`: reproducible command-line run.
- `helpers/`: clustering, portfolio construction, leverage, simulation, and statistics.
- `data/`: committed CSV inputs. The research code does not download missing data.
- `output/`: generated paths, summaries, diagnostics, and charts.

The command-line defaults are a three-year optimization window, 20 years of simulation history, 60-day blocks, five simulated years, and 1,000 paths. The notebook deliberately uses a 10-year, 2,000-path configuration. Both use date-aligned FRED EFFR as the risk-free rate and sample asset returns with EFFR jointly.

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
- **Rates:** EFFR is used as both the optimizer's risk-free rate and the margin-rate benchmark. Bootstrap block boundaries can create rate jumps.
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

```bash
python run_research.py
```

The notebook reclusters the universe before every run; the command-line runner uses `VT`, `BND`, `SGOV`, `GLD`, and `GSG`. Results and the exact configuration are written to `output/`, including `summary_*.csv`, path-level Parquet files, weight diagnostics, and `run_config.json`.

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

**Maximum drawdown** is the worst peak-to-trough loss:

$$\text{MDD}=\min_t\frac{E_t-\max_{s\leq t}E_s}{\max_{s\leq t}E_s}$$

### 2. Covariance Matrix Theory

With return vector $r_t$, mean $\hat\mu$, and $T$ observations, sample covariance is

$$S=\frac{1}{T-1}\sum_{t=1}^{T}(r_t-\hat\mu)(r_t-\hat\mu)^\top$$

It is unbiased, but noisy when the number of assets is large relative to the available history. Optimization can amplify that noise into unstable or concentrated weights.

The implementation starts from daily simple returns and annualizes every covariance estimator:

$$
\hat\Sigma_{ann}=252\hat\Sigma_{daily}
$$

For portfolio weights $w$, the optimizer sees portfolio variance $w^\top\hat\Sigma_{ann}w$. Small errors in correlations can therefore change the allocation materially, especially when several assets have similar expected returns or volatility. The 10%-40% weight bounds limit this estimation-error amplification; they do not remove it.

**Ledoit-Wolf shrinkage** blends $S$ with a scaled identity target:

$$\hat\Sigma=(1-\alpha)S+\alpha F, \qquad F=\frac{\text{tr}(S)}{N}I_N$$

The analytically selected $\alpha^*\in[0,1]$ minimizes expected squared estimation error $\mathbb{E}\lVert\hat\Sigma-\Sigma\rVert_F^2$. Larger $\alpha$ reduces estimation noise but pulls variances and correlations toward a common structure, sometimes producing equal-weight-like allocations.

**OAS** uses the same target with an intensity derived under a Gaussian assumption. It often shrinks slightly less than Ledoit-Wolf. OAS is the current default in both the notebook and command-line configuration; an optional positive diagonal jitter is then added to make numerical optimization more stable.

**EWMA** emphasizes recent observations:

$$\hat\Sigma_{EWMA}=\sum_t w_t(r_t-\hat\mu)(r_t-\hat\mu)^\top, \qquad w_t\propto(1-\lambda)\lambda^{T-t}$$

It responds faster to regime changes but has higher sampling noise because its effective sample is smaller.

With decay parameter $\lambda$, a return observed $k$ days ago receives relative weight $\lambda^k$. A high $\lambda$ remembers more history and changes slowly; a low $\lambda$ adapts quickly but makes the estimate more sensitive to a small number of recent observations. The implementation accepts an EWMA span or alpha, using a 60-day span when neither is supplied.

**Factor covariance** represents returns through $k$ common PCA factors plus residual risk:

$$\hat\Sigma=B\Sigma_F B^\top+D$$

$B$ contains factor loadings, $\Sigma_F$ is factor covariance, and $D$ is diagonal residual variance. This reduces the number of noisy relationships that must be estimated directly.

The PCA implementation estimates $B$ from demeaned daily returns, retains the requested number of components, and sets $D$ from residual variances. It is a statistical compression, not an economic factor model: its factors are chosen to explain variance, not to represent named risks such as equities, duration, or inflation.

All covariance estimators should be compared out of sample. A lower in-sample $w^\top\hat\Sigma w$ is not evidence that the resulting portfolio will have lower realized volatility.

### 3. Monte Carlo Theory

A historical backtest is one realized market path. Monte Carlo generates alternative paths and reruns the allocation process, testing the strategy rather than only a fixed allocation. The research runner uses the moving-block bootstrap below.

**Bootstrap simulation** resamples observed return rows and therefore retains their empirical marginal distribution. For each historical date, the runner forms the joint vector

$$
z_t=(r_{VT,t},r_{BND,t},\ldots,r_{GSG,t},EFFR_t)
$$

after aligning all assets and forward-filling EFFR onto trading days. Sampling this vector jointly preserves the contemporaneous relationship between asset returns and financing rates.

- **IID bootstrap:** samples individual rows independently. It removes serial dependence and volatility clustering.
- **Moving-block bootstrap:** samples contiguous blocks of length $b$. It approximately preserves short-run autocorrelation and volatility clustering within each block.

For a moving-block bootstrap, independent start indices $s_1,s_2,\ldots$ are drawn uniformly from the feasible historical starts. A simulated path is the truncated concatenation

$$
z^{*}_{1:H}=(z_{s_1:s_1+b-1},z_{s_2:s_2+b-1},\ldots)_{1:H}
$$

where $H$ is the simulated horizon. The default $b=60$ preserves roughly three trading months at a time, but it breaks dependence at every block boundary. It cannot create shocks, regimes, or cross-asset relationships that are absent from the historical sample.

Asset returns are compounded into synthetic prices with $P_t=P_{t-1}(1+r_t)$. The first monthly allocation uses a separate three-year historical warm-up; later allocations use rolling three-year windows of the combined warm-up and synthetic price history. Each strategy then uses the same simulated portfolio returns across all leverage-reset modes. This pairing isolates the effect of the reset rule from differences in sampled markets.

Bootstrap percentiles describe variation conditional on the chosen history, block length, portfolio rules, and financing model. They are scenario statistics, not probabilities of future outcomes in a fully specified economic model.

### 4. Multi-Dimensional Scaling Theory

Clustering uses correlation-derived distances, while the visualization needs two-dimensional coordinates. MDS finds coordinates with Euclidean distances close to the original distances.

**Classical MDS** double-centers squared distances $D^{(2)}$:

$$B=-\tfrac{1}{2}JD^{(2)}J, \qquad J=I-\tfrac{1}{n}\mathbf{1}\mathbf{1}^\top$$

After eigendecomposition $B=V\Lambda V^\top$, the two-dimensional representation is

$$X=V_k\Lambda_k^{1/2}$$

Double-centering removes the arbitrary origin. Only relative distances matter.

**Metric MDS** minimizes embedding stress directly:

$$
	ext{Stress}(X) = \sqrt{\sum_{i<j}\left(d_{ij} - \lVert x_i - x_j \rVert\right)^2}
$$

This is useful when the distance matrix is not perfectly Euclidean. The default $d_{ij}=1-|\rho_{ij}|$ treats strong negative correlation as close. For long-only diversification, $d_{ij}=1-\rho_{ij}$ often better preserves negatively correlated hedges.