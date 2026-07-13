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
			   A weight decided at date t is applied from the next return observation.
	- returns: optional DataFrame [dates x tickers] of daily simple returns for the same tickers.
			   If None, historical returns are built from adjusted-close prices via `load_data`.

	Output
	- Series of daily weighted returns indexed by business days (intersection of weights/returns dates).
	"""
	w = weights.astype(float).fillna(0.0)

	# If no returns provided, load historical prices and build returns for these tickers
	if returns is None:
		start_date = pd.Timestamp(w.index.min()).strftime("%Y-%m-%d")
		end_date = pd.Timestamp(w.index.max()).strftime("%Y-%m-%d")
		prices = pd.concat(
			[load_data(ticker, start_date, end_date)["adj close"].rename(ticker) for ticker in w.columns],
			axis=1,
			join="inner",
		)
		rets = prices.pct_change().dropna(how="any")
	else:
		rets = returns.astype(float)

	rets = rets[w.columns]

	# Trade after the close: weights observed on t are applied to the return after t.
	w_aligned = w.reindex(rets.index, method="ffill").shift(1).fillna(0.0)

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

