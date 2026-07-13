from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from helpers.fetch import load_data
from helpers.backtest import backtest_portfolio
from helpers.portfolio import build_portfolios_from_prices
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data"


def _historical_returns(
	tickers: List[str],
	start: Optional[str] = None,
	end: Optional[str] = None,
	lookback_years: int = 3,
) -> pd.DataFrame:
	"""Build historical daily simple returns for given tickers and date range."""
	if end is None:
		end = pd.Timestamp.today().strftime("%Y-%m-%d")
	if start is None:
		start = (pd.Timestamp(end) - pd.DateOffset(years=lookback_years)).strftime("%Y-%m-%d")

	prices = pd.concat(
		[load_data(ticker, start, end)["adj close"].rename(ticker) for ticker in tickers],
		axis=1,
		join="inner",
	)
	return prices.pct_change().dropna(how="any")


def _historical_effr() -> pd.Series:
	"""Historical EFFR series in percentage points."""
	effr = pd.read_csv(DATA_DIR / "FRED_EFFR.csv", parse_dates=["date"], index_col="date")["EFFR"]
	return effr.sort_index().astype(float)


def _simulate_effr_path(n_days: int, method: str, rng: np.random.Generator) -> pd.Series:
	"""Simulate an EFFR scenario path in percentage points."""
	effr = _historical_effr()
	changes = effr.diff().dropna()

	last_level = float(effr.iloc[-1])
	if method == "bootstrap":
		draws = rng.choice(changes.to_numpy(), size=int(n_days), replace=True)
	else:
		mu = float(changes.mean())
		sigma = float(changes.std(ddof=1))
		draws = rng.normal(loc=mu, scale=sigma, size=int(n_days)) if sigma > 0 else np.full(int(n_days), mu)

	path = last_level + np.cumsum(draws)
	path = np.clip(path, a_min=0.0, a_max=None)
	return pd.Series(path, name="EFFR")


def simulate_parametric(
    tickers: List[str],
    n_days: int,
    n_sims: int = 1000,
    lookback_years: int = 3,
    seed: Optional[int] = None,
) -> List[pd.DataFrame]:
    """Parametric MC: draw MVN daily returns using μ,Σ from historical returns."""
    hist = _historical_returns(tickers, lookback_years=lookback_years)
    mu = hist.mean().to_numpy()
    cov = hist.cov().to_numpy()
    rng = np.random.default_rng(seed)

    sims: List[pd.DataFrame] = []
    for _ in range(int(n_sims)):
        draws = rng.multivariate_normal(mean=mu, cov=cov, size=int(n_days))
        df = pd.DataFrame(draws, columns=tickers, index=pd.RangeIndex(n_days))
        df["EFFR"] = _simulate_effr_path(n_days, method="parametric", rng=rng).to_numpy()
        sims.append(df)
    return sims


def simulate_bootstrap(
    tickers: List[str],
    n_days: int,
    n_sims: int = 1000,
    lookback_years: int = 3,
    block_size: Optional[int] = None,
    seed: Optional[int] = None,
) -> List[pd.DataFrame]:
	"""
	Bootstrap Monte Carlo on historical daily simple returns.
	- If block_size is None: IID bootstrap (sample rows with replacement).
	- Else: moving block bootstrap with block_size.

	Returns a list of length n_sims; each item is a DataFrame [n_days x tickers].
	"""
	hist = _historical_returns(tickers, lookback_years=lookback_years)
	effr = _historical_effr().reindex(hist.index, method="ffill")
	joint = hist.copy()
	joint["EFFR"] = effr

	rng = np.random.default_rng(seed)

	T = joint.shape[0]

	sims: List[pd.DataFrame] = []
	if not block_size or block_size <= 1:
		# IID bootstrap: sample rows with replacement
		for _ in range(int(n_sims)):
			idx = rng.integers(low=0, high=T, size=int(n_days))
			df = joint.iloc[idx].reset_index(drop=True)
			df.index = pd.RangeIndex(n_days)
			sims.append(df)
	else:
		b = int(block_size)
		for _ in range(int(n_sims)):
			starts = rng.integers(0, T - b + 1, size=int(np.ceil(n_days / b)))
			chunks = [joint.iloc[start : start + b] for start in starts]
			df = pd.concat(chunks, axis=0).iloc[:n_days].reset_index(drop=True)
			df.index = pd.RangeIndex(n_days)
			sims.append(df)
	return sims


def apply_backtest_to_simulations(
    weights: pd.DataFrame,
    simulations: List[pd.DataFrame],
    start_value: float = 10_000.0,
    hold: str = "last",
) -> List[pd.Series]:
	"""
	For each simulated return path, run the backtest and return the value series.

	- weights: daily weights DataFrame [dates x tickers]. Only the per-ticker allocations are used.
	- simulations: list of DataFrames [n_days x tickers] with daily simple returns.
	- hold: 'last' uses the last row of weights (constant through simulation);
			'mean' uses the time-average of weights (normalized).
	"""
	tickers = [ticker for ticker in simulations[0].columns if ticker != "EFFR"]

	if hold == "mean":
		base_w = weights[tickers].mean(axis=0)
	else:
		base_w = weights[tickers].iloc[-1]

	base_w = base_w / base_w.sum()

	results: List[pd.Series] = []
	for sim in simulations:
		# Build a constant-weights DataFrame over the simulation horizon
		W = pd.DataFrame(np.tile(base_w.to_numpy(), (sim.shape[0], 1)), index=sim.index, columns=tickers)
		values = backtest_portfolio(W, start_value=start_value, returns=sim[tickers])
		results.append(values)
	return results


def _simulation_weights(
	returns: pd.DataFrame,
	lookback_years: int,
	freq: str,
	cov_method: str | None,
	cov_params: dict | None,
	warmup_returns: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
	"""Build weights at the simulation start using a full historical lookback."""
	if warmup_returns is None:
		warmup_returns = _historical_returns(list(returns.columns), lookback_years=lookback_years)
	warmup = warmup_returns[returns.columns]

	warmup.index = pd.bdate_range(
		end=returns.index.min() - pd.offsets.BDay(),
		periods=warmup.shape[0],
	)
	prices = (1.0 + pd.concat([warmup, returns])).cumprod()
	return build_portfolios_from_prices(
		prices=prices,
		start=returns.index.min(),
		end=returns.index.max(),
		lookback_years=lookback_years,
		freq=freq,
		cov_method=cov_method,
		cov_params=cov_params,
	)


def rebalance_on_simulation(
    sim_returns: pd.DataFrame,
    lookback_years: int = 3,
    freq: str = "BMS",
    cov_method: str | None = None,
    cov_params: dict | None = None,
    which: str = "min_variance",
    start_value: float = 10_000.0,
) -> tuple[pd.DataFrame, pd.Series]:
	"""
	Given one simulated return path [n_days x tickers],
	- build synthetic prices by compounding from 1.0
	- compute a rebalanced weight schedule using the same logic as historical (lookback, optimization)
	- run the backtest on the simulated returns using the dynamic weights

	Returns (weights_df, values_series)
	"""
	n_days = sim_returns.shape[0]
	idx = pd.bdate_range(start=pd.Timestamp("2000-01-03"), periods=n_days)
	effr = sim_returns["EFFR"].copy() if "EFFR" in sim_returns.columns else None
	if effr is not None:
		effr.index = idx
	rets = sim_returns.drop(columns=["EFFR"], errors="ignore").copy()
	rets.index = idx

	# Start with a complete historical window before the first simulated return.
	weights_dict = _simulation_weights(
		returns=rets,
		lookback_years=lookback_years,
		freq=freq,
		cov_method=cov_method,
		cov_params=cov_params,
	)

	W = weights_dict[which]

	values = backtest_portfolio(W, start_value=start_value, returns=rets)
	return W, values


def leverage_backtest_on_simulation(
    sim_returns: pd.DataFrame,
    lookback_years: int = 3,
    freq: str = "BMS",
    cov_method: str | None = None,
    cov_params: dict | None = None,
    which: str = "min_variance",
    leverage: float = 2.0,
    start_value: float = 10_000.0,
    band: float = 0.05,
    hard_cap: float = 4.0,
    spread: float = 0.01,
) -> tuple[pd.DataFrame, pd.DataFrame]:
	"""Build dynamic weights on one simulation path and run the leveraged backtest."""
	n_days = sim_returns.shape[0]
	idx = pd.bdate_range(start=pd.Timestamp("2000-01-03"), periods=n_days)
	effr = sim_returns["EFFR"] if "EFFR" in sim_returns.columns else None
	rets = sim_returns.drop(columns=["EFFR"], errors="ignore").copy()
	rets.index = idx

	weights_dict = _simulation_weights(
		returns=rets,
		lookback_years=lookback_years,
		freq=freq,
		cov_method=cov_method,
		cov_params=cov_params,
	)

	W = weights_dict[which]

	from helpers.leverage import leverage_backtest

	values = leverage_backtest(
		W,
		leverage=leverage,
		start_value=start_value,
		returns=rets,
		effr=effr,
		freq=None,
		band=band,
		hard_cap=hard_cap,
		spread=spread,
	)
	return W, values


def apply_rebalanced_backtest_to_simulations(
    simulations: list[pd.DataFrame],
    lookback_years: int = 3,
    freq: str = "BMS",
    cov_method: str | None = None,
    cov_params: dict | None = None,
    which: str = "min_variance",
    start_value: float = 10_000.0,
) -> list[tuple[pd.DataFrame, pd.Series]]:
	"""
	For each simulation, compute dynamic weights (as per historical logic) and return (weights, values).
	"""
	return [
		rebalance_on_simulation(
			sim_returns=sim,
			lookback_years=lookback_years,
			freq=freq,
			cov_method=cov_method,
			cov_params=cov_params,
			which=which,
			start_value=start_value,
		)
		for sim in simulations
	]


def simulate_and_backtest_portfolios(
	tickers: List[str],
	n_days: int,
	n_sims: int,
	lookback_years: int = 3,
	block_size: int = 10,
	seed: int = 1,
	method: str = "parametric",
	start_value: float = 10_000.0,
) -> pd.DataFrame:
	"""
	Simulate returns, rebuild all portfolios on each path, and return value series.

	Output columns are: sim_1_min_variance, sim_1_max_sharpe, ...
	"""
	base_tickers = list(dict.fromkeys(tickers))
	sim_tickers = base_tickers.copy()
	if "VT" not in sim_tickers:
		sim_tickers.append("VT")

	# Run Monte Carlo return simulations.
	if method == "parametric":
		simulations = simulate_parametric(
			sim_tickers,
			n_days=n_days,
			n_sims=n_sims,
			lookback_years=lookback_years,
			seed=seed,
		)
	elif method == "bootstrap":
		simulations = simulate_bootstrap(
			sim_tickers,
			n_days=n_days,
			n_sims=n_sims,
			lookback_years=lookback_years,
			block_size=block_size,
			seed=seed,
		)
	else:
		raise ValueError("method must be 'parametric' or 'bootstrap'")

	strategies = ["min_variance", "max_sharpe", "equal_weight", "market_cap"]
	all_values: list[pd.Series] = []
	warmup_returns = _historical_returns(base_tickers, lookback_years=lookback_years)

	for i, sim in enumerate(simulations, start=1):
		# Use a business-day index so portfolio rebuilding can use calendar frequencies.
		idx = pd.bdate_range(start=pd.Timestamp("2000-01-03"), periods=sim.shape[0])
		rets = sim.copy()
		rets.index = idx

		# Build optimized/equal-weight portfolios from a full historical window.
		weights = _simulation_weights(
			returns=rets[base_tickers],
			lookback_years=lookback_years,
			freq="BMS",
			cov_method=None,
			cov_params=None,
			warmup_returns=warmup_returns,
		)

		# Backtest each strategy on the full simulated returns, including VT for market_cap.
		for strategy in strategies:
			values = backtest_portfolio(weights[strategy], start_value=start_value, returns=rets)
			values.name = (i, strategy)  # MultiIndex column (sim, strategy)
			all_values.append(values)

	df = pd.concat(all_values, axis=1)
	df.columns = pd.MultiIndex.from_tuples(df.columns, names=["sim", "strategy"])
	df.index.name = "date"
	return df


def save_simulations_parquet(values: pd.DataFrame, path: str, engine: str | None = None) -> None:
	"""Save simulation values with Parquet-compatible simulation labels."""
	df = values.copy()
	# fastparquet requires every MultiIndex level to be str/bytes (sim is currently int)
	df.columns = pd.MultiIndex.from_tuples(
		[(str(sim), strat) for sim, strat in df.columns], names=df.columns.names
	)
	if df.index.name is None:
		df.index.name = "date"
	# Ensure destination folder exists (relative paths preferred in notebooks)
	p = Path(path)
	if p.parent and str(p.parent) not in ("", "."):
		p.parent.mkdir(parents=True, exist_ok=True)
	try:
		df.to_parquet(path, compression="snappy", engine=engine)
	except ImportError as e:
		raise ImportError(
			"Parquet export requires 'pyarrow' or 'fastparquet'. Install with:\n"
			"  pip install pyarrow\n"
			"or:\n  pip install fastparquet"
		) from e

