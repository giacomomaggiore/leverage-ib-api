from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from helpers.fetch import load_data


def portfolio_returns(
	weights: pd.DataFrame,
	returns: Optional[pd.DataFrame] = None,
) -> pd.Series:
	"""
	Compute the day-by-day weighted return of an (unlevered) portfolio.

	Inputs
	- weights: DataFrame [dates x tickers], daily portfolio weights (sum ~ 1 per row).
			   If weights change over time, rebalancing is assumed at the start of each day.
	- returns: optional DataFrame [dates x tickers] of daily simple returns for the same tickers.
			   If None, historical returns are built from adjusted-close prices via `load_data`.

	Output
	- Series of daily weighted returns indexed by business days (intersection of weights/returns dates).
	"""
	if not isinstance(weights, pd.DataFrame) or weights.empty:
		raise ValueError("'weights' must be a non-empty DataFrame")

	# Ensure numeric and fill missing with 0
	w = weights.copy().astype(float).fillna(0.0)

	# If no returns provided, load historical prices and build returns for these tickers
	if returns is None:
		tickers = list(w.columns)
		if len(tickers) == 0:
			raise ValueError("Weights must have at least one ticker column")

		# Choose a date span covering the weights timeline
		start_date = pd.Timestamp(w.index.min()).strftime("%Y-%m-%d")
		end_date = pd.Timestamp(w.index.max()).strftime("%Y-%m-%d")

		series = {}
		for t in tickers:
			try:
				df = load_data(t, start_date, end_date)
				s = df["adj close"].dropna().rename(t)
				if s.shape[0] >= 2:
					series[t] = s
			except Exception:
				continue

		if not series:
			raise ValueError("Could not build historical prices for provided tickers")

		prices = pd.concat(series.values(), axis=1, join="inner")
		prices.columns = list(series.keys())
		rets = prices.pct_change().dropna(how="any")
	else:
		rets = returns.copy().astype(float).replace([np.inf, -np.inf], np.nan).dropna(how="all")

	if rets.empty:
		raise ValueError("No return observations available to run the backtest")

	# Align on common dates and tickers
	common_tickers = [c for c in w.columns if c in rets.columns]
	if not common_tickers:
		raise ValueError("No overlapping tickers between weights and returns")

	rets = rets[common_tickers]
	w = w[common_tickers]

	# Reindex weights to return dates using forward-fill to carry last allocation
	w_aligned = w.reindex(rets.index, method="ffill").fillna(0.0)

	# Normalize rows to sum to 1 when possible (avoid division by ~0)
	row_sums = w_aligned.sum(axis=1).replace(0.0, np.nan)
	w_norm = w_aligned.div(row_sums, axis=0).fillna(0.0)

	# Daily portfolio returns: weighted sum of constituent returns
	port_rets = (w_norm * rets).sum(axis=1)
	port_rets.name = "portfolio_return"
	return port_rets


def backtest_portfolio(
	weights: pd.DataFrame,
	start_value: float = 10_000.0,
	returns: Optional[pd.DataFrame] = None,
) -> pd.Series:
	"""
	Compute the day-by-day portfolio value time series given a weights DataFrame.

	See `portfolio_returns` for the meaning of `weights`/`returns`.
	"""
	port_rets = portfolio_returns(weights, returns)
	values = start_value * (1.0 + port_rets).cumprod()
	values.name = "portfolio_value"
	return values

