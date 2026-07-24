from __future__ import annotations

from typing import Optional

import pandas as pd

from helpers.fetch import load_data


def portfolio_returns(
	weights: pd.DataFrame,
	returns: Optional[pd.DataFrame] = None,
) -> pd.Series:
	"""
	Compute daily returns while holdings drift between target-weight dates.

	Inputs
	- weights: DataFrame [rebalance dates x tickers] of target portfolio weights.
			   A target decided at date t is applied from the next return observation.
	- returns: optional DataFrame [dates x tickers] of daily simple returns for the same tickers.
			   If None, historical returns are built from adjusted-close prices via `load_data`.

	Output
	- Series of daily weighted returns indexed by business days (intersection of weights/returns dates).
	"""
	w = weights.sort_index().astype(float).fillna(0.0)

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

	available = [ticker for ticker in w.columns if ticker in rets.columns]
	if not available:
		raise ValueError("weights and returns have no ticker columns in common")
	w = w[available]
	rets = rets[available].sort_index()

	holdings = None
	target_position = 0
	target_dates = w.index
	daily_returns = []

	for return_date, asset_returns in rets.iterrows():
		new_target_date = None
		while target_position < len(target_dates) and target_dates[target_position] < return_date:
			new_target_date = target_dates[target_position]
			target_position += 1

		if new_target_date is not None:
			target = w.loc[new_target_date]
			target_sum = target.sum()
			if target_sum <= 0.0:
				raise ValueError(f"target weights must sum to a positive value on {new_target_date}")
			portfolio_value = 1.0 if holdings is None else holdings.sum()
			holdings = portfolio_value * target / target_sum

		if holdings is None:
			daily_returns.append(0.0)
			continue

		previous_value = holdings.sum()
		holdings = holdings * (1.0 + asset_returns)
		daily_returns.append(holdings.sum() / previous_value - 1.0)

	return pd.Series(daily_returns, index=rets.index, name="portfolio_return")


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

