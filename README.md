Leverage Portfolio — IB API Research & Execution
===============================================

Project goal
------------
- Research and implement a long-only leveraged portfolio strategy using the Interactive Brokers (IB) API.
- Key flow: select n assets (5–10 US stocks / ETFs) → compute daily returns → estimate/shrink covariance → compute portfolio weights (tangent/max-Sharpe or min-variance with constraints) → apply margin loan to reach target leverage L → periodic rebalance and safety-band rebalances → backtest and Monte Carlo analysis.

Assumptions (from initial decisions)
-----------------------------------
- Asset universe: US ETFs and stocks, long-only.
- Number of assets: typically 5–10.
- Return frequency: daily returns for covariance estimation.
- Rebalance cadence: monthly or bi-monthly (configurable).
- Weighting: allow cash (weights need not sum to 1 before leverage scaling). Box constraints (max weight per asset) and minimum weight per asset will be enforced.
- Leverage: use IB margin loan to scale exposure to a target leverage ratio `L`. Safety interval: ±10% (configurable).
- Execution: automated via `ib_insync` against IB demo (paper) account; orders executed as fill-at-next-open in backtests.
- Backtest realism: include commissions, financing (margin) rates, and simulate margin-call behavior.
- Language & libs: Python with `pandas`, `numpy`, `scipy`, `cvxpy` (or `scipy.optimize`), `pyportfolioopt` (optional), and `ib_insync`.

High-level pipeline
-------------------
1. Data ingestion
	- Source historical prices (primary: IB historical data; fallback: external source for long histories). Clean for corporate actions, align timestamps, forward-fill or drop non-trading days.
2. Return and covariance
	- Compute daily log-returns (configurable). Build rolling-covariance with configurable lookback window.
	- Apply shrinkage (default: Ledoit–Wolf); provide options for constant-correlation, target diagonal, and factor-based shrinkage.
3. Portfolio construction
	- Two modes:
	  - Tangent portfolio (maximize expected Sharpe ratio with an estimated return vector, or using Risk Parity variations). Returns estimation can be simple historical mean or shrink/robust alternatives.
	  - Minimum-variance portfolio (no required sum(w)=1; allow cash allocation). Enforce box constraints: min weight per asset, max weight per asset, and optional turnover penalty.
	- Solve with `cvxpy` or `scipy.optimize` depending on constraints.
4. Leverage application
	- Given raw weights `w_raw` (sum may be <1 if cash exists), scale exposure using margin loan to reach target gross leverage `L`.
	- Define leverage metric: gross exposure / equity (recommended). Document exact formula in code.
5. Execution & rebalancing
	- Rebalance on scheduled cadence (monthly or bi-monthly) to target weights scaled by leverage.
	- Also monitor leverage continuously; if leverage breaches safety band (±10% around `L`), trigger an intraperiod rebalance back toward target.
6. Backtest & realism
	- Fill model: next-open fills for simplicity; optional VWAP slicing later.
	- Costs: commissions, spread/slippage model, and margin interest accrual per day.
	- Margin calls: if equity falls and maintenance margin is breached, simulate forced deleveraging (partial liquidation in order of lowest contribution to portfolio variance or pro-rata).
7. Monte Carlo
	- Framework hooks to support different scenario generators: parametric MVN, bootstrap resampling, factor + stochastic covariance, or Wishart dynamics. Collect distributional metrics across runs.
8. Reporting
	- Jupyter notebooks for exploratory analysis and plots: NAV, cumulative returns, leverage over time, drawdowns, heatmaps of weights and correlations, performance metrics (CAGR, Sharpe, max drawdown, VaR).

Design decisions & defaults (can be changed in config)
----------------------------------------------------
- Covariance shrinkage: Ledoit–Wolf (default) with switch to other methods for sensitivity tests.
- Returns: daily log-returns.
- Optimization solver: `cvxpy` with ECOS/OSQP for convex problems; fallback to `scipy.optimize` for simpler cases.
- Leverage metric: gross exposure / equity.
- Rebalance default: monthly; set `rebalance_period='1M'` or `'2M'`.
- Safety band default: 10% around target `L`.

IB integration notes
--------------------
- Library: `ib_insync` (recommended). Use IB Gateway or TWS in paper mode.
- Credentials: store API host/port and account details in a local config file (never commit credentials).
- Rate limits: throttle historical and live requests; respect IB pacing violations.
- Orders: use market-on-open / market with careful scheduling. For real trading, implement order sizing and routable order batching.

Backtest realism specifics
------------------------
- Next-open fill assumption for rebalances.
- Commission model: per-share or per-ticket configurable.
- Margin interest: treated as daily accrual on borrowed amount. (We'll ask you for the financing rate source when implementing.)
- Margin call behavior: simulate maintenance margin checks daily; if breached, deleverage according to configurable policy.

Monte Carlo placeholders
-----------------------
- Modular scenario generator interface to plug in:
  - `ParametricMVN` (mean vector, covariance),
  - `BlockBootstrap` (resample daily blocks),
  - `StochasticCovariance` (e.g., Wishart process),
  - `FactorModel` (simulate factors + idiosyncratic noise).
- Collect distributions: terminal NAV, max drawdown, frequency of leverage breaches, realized Sharpe, tail losses.

Project layout (recommended)
---------------------------
- `notebooks/` — Jupyter notebooks for exploration and reporting
- `src/data/` — ingestion and cleaning functions
- `src/estimation/` — returns, covariance, shrinkage modules
- `src/opt/` — portfolio optimization routines (tangent, minvar)
- `src/backtest/` — backtest engine, execution model, margin simulation
- `src/ib/` — IB execution & monitoring wrapper using `ib_insync`
- `src/sim/` — Monte Carlo scenario generators
- `requirements.txt` — Python dependencies
- `README.md` — this document

Next actions / questions for you
-------------------------------
- Please confirm the target leverage ratio `L` you want to use as default (e.g., 2.0x).  
- Specify a minimum weight per asset (e.g., 0.01 = 1%) and any maximum per-asset weight cap (e.g., 0.4 = 40%).  
- Pick default rebalance cadence: `monthly` or `bi-monthly` (I can implement both as options).  
- Do you want historical margin/financing rates from IB used, or a fixed assumed rate input at runtime? (You said you'll be asked when we implement — I'll prompt again before coding financing.)

How I'll proceed once you answer
--------------------------------
1. Create a minimal runnable backtest prototype in `src/backtest/` that: ingests CSV price history, computes daily returns and Ledoit–Wolf covariance, solves for min-variance and tangent portfolios with box constraints, applies leverage scaling, and simulates next-open fills + commissions + daily margin interest and margin-call logic.
2. Provide a `notebooks/` demo that runs a sample backtest on toy data and produces the requested plots/metrics.
3. Add an `src/ib/` module using `ib_insync` for paper-trading execution and a mapping layer from backtest orders to live orders.

If the above sounds good, tell me your default `L`, min/max weight choices, and preferred rebalance cadence and I'll start the prototype implementation.

Acknowledgements
----------------
This README is a living document — we'll refine defaults and add implementation notes as we build modules.

# leverage-ib-api
