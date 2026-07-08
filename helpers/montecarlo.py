from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from helpers.fetch import load_data
from helpers.backtest import backtest_portfolio
from helpers.portfolio import build_portfolios_from_prices


def _historical_returns(
	tickers: List[str],
	start: Optional[str] = None,
	end: Optional[str] = None,
	lookback_years: int = 3,
) -> pd.DataFrame:
    
    # Build historical returns DataFrame for the given tickers and date range.
    
	if end is None:
		end = pd.Timestamp.today().strftime("%Y-%m-%d")
	if start is None:
		start = (pd.Timestamp(end) - pd.DateOffset(years=lookback_years)).strftime("%Y-%m-%d")

	series: dict[str, pd.Series] = {}
	for t in tickers:
		try:
			df = load_data(t, start, end)
			s = df["adj close"].dropna().rename(t)
			if s.shape[0] >= 2:
				series[t] = s
		except Exception:
			continue

	if not series:
		return pd.DataFrame()

	prices = pd.concat(series.values(), axis=1, join="inner")
	prices.columns = list(series.keys())
	returns = prices.pct_change().dropna(how="any")
	return returns


def simulate_parametric(
	tickers: List[str],
	n_days: int,
	n_sims: int = 1000,
	lookback_years: int = 3,
	seed: Optional[int] = None,
) -> List[pd.DataFrame]:
	"""
	Parametric Monte Carlo using multivariate normal daily returns estimated
	from historical simple returns.

	Returns a list of length n_sims; each item is a DataFrame [n_days x tickers].
	"""
 
    # loads historical returns for the given tickers and lookback period
	hist = _historical_returns(tickers, lookback_years=lookback_years)
	if hist.empty:
		raise ValueError("Not enough historical data to estimate parameters")

    # compute mean vector and covariance matrix of historical returns
	mu = hist.mean().to_numpy()
	cov = hist.cov().to_numpy()
	rng = np.random.default_rng(seed)

	sims: List[pd.DataFrame] = []
	for _ in range(int(n_sims)):
     
        # generate multivariate normal draws for n_days using the estimated mean and covariance
		draws = rng.multivariate_normal(mean=mu, cov=cov, size=int(n_days))
		df = pd.DataFrame(draws, columns=tickers, index=pd.RangeIndex(n_days))
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
	if hist.empty:
		raise ValueError("Not enough historical data to bootstrap")

	rng = np.random.default_rng(seed)

	# number of historical observations available for bootstrapping
	T = hist.shape[0]

	sims: List[pd.DataFrame] = []
	if not block_size or block_size <= 1:
		# IID bootstrap: sample rows with replacement
		for _ in range(int(n_sims)):
			idx = rng.integers(low=0, high=T, size=int(n_days))
			df = hist.iloc[idx].reset_index(drop=True)
			df.index = pd.RangeIndex(n_days)
			sims.append(df)
	else:
		# Moving block bootstrap: sample contiguous blocks of length b
		b = int(block_size)
		for _ in range(int(n_sims)):
			chunks: list[pd.DataFrame] = []
			days_left = int(n_days)
			while days_left > 0:
				# number of valid start positions so a full block fits is (T - b + 1)
				if T - b + 1 > 0:
					start = int(rng.integers(low=0, high=T - b + 1))
					block = hist.iloc[start : start + b]
				else:
					# history shorter than block: take whole history as a block
					start = 0
					block = hist.copy()
				chunks.append(block)
				days_left -= block.shape[0]
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
	if not isinstance(weights, pd.DataFrame) or weights.empty:
		raise ValueError("'weights' must be a non-empty DataFrame")

	tickers = list(simulations[0].columns) if simulations else list(weights.columns)
	if not tickers:
		raise ValueError("No tickers available for simulations")

	if hold == "mean":
		base_w = weights[tickers].mean(axis=0)
	else:
		base_w = weights[tickers].iloc[-1]

	# Normalize to sum to 1
	s = float(base_w.sum())
	if s <= 0:
		raise ValueError("Weights must sum to a positive number")
	base_w = (base_w / s).fillna(0.0)

	results: List[pd.Series] = []
	for sim in simulations:
		# Build a constant-weights DataFrame over the simulation horizon
		W = pd.DataFrame(np.tile(base_w.to_numpy(), (sim.shape[0], 1)), index=sim.index, columns=sim.columns)
		values = backtest_portfolio(W, start_value=start_value, returns=sim)
		results.append(values)
	return results


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
	if not isinstance(sim_returns, pd.DataFrame) or sim_returns.empty:
		raise ValueError("sim_returns must be a non-empty DataFrame")

	# Synthesize business-day index for stability
	n_days = sim_returns.shape[0]
	idx = pd.bdate_range(start=pd.Timestamp("2000-01-03"), periods=n_days)
	rets = sim_returns.copy()
	rets.index = idx

	# Synthetic prices starting at 1.0
	prices = (1.0 + rets).cumprod()

	# Build dynamic portfolios on the simulated prices
	weights_dict = build_portfolios_from_prices(
		prices=prices,
		start=prices.index.min(),
		end=prices.index.max(),
		lookback_years=lookback_years,
		freq=freq,
		cov_method=cov_method,
		cov_params=cov_params,
	)

	if which not in weights_dict:
		raise ValueError(f"which must be one of {list(weights_dict.keys())}")
	W = weights_dict[which]

	values = backtest_portfolio(W, start_value=start_value, returns=rets)
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
	results: list[tuple[pd.DataFrame, pd.Series]] = []
	for sim in simulations:
		W, V = rebalance_on_simulation(
			sim_returns=sim,
			lookback_years=lookback_years,
			freq=freq,
			cov_method=cov_method,
			cov_params=cov_params,
			which=which,
			start_value=start_value,
		)
		results.append((W, V))
	return results

