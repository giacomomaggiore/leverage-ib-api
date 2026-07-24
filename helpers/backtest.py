import pandas as pd

def portfolio_returns(
	weights: pd.DataFrame,
	returns: pd.DataFrame,
) -> pd.Series:
	"""
	Compute daily returns while holdings drift between target-weight dates.

	A target decided at date t applies from the next return observation. Holdings then
	compound independently until the next target date, so portfolio weights drift.
	"""
	w = weights.sort_index().astype(float).fillna(0.0)
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

