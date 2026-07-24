from __future__ import annotations

from typing import List, Optional
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from helpers.fetch import load_data
from helpers.portfolio import build_portfolios_from_prices
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data"


def _historical_returns(
	tickers: List[str],
	start: Optional[str] = None,
	end: Optional[str] = None,
	lookback_years: int = 20,
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
	returns = prices.pct_change().dropna(how="any")
	actual_years = (returns.index.max() - returns.index.min()).days / 365.25
	if start is not None and actual_years + 0.1 < lookback_years:
		warnings.warn(
			f"Requested {lookback_years} years but the common return history contains {actual_years:.1f} years "
			f"({returns.index.min().date()} to {returns.index.max().date()}).",
			stacklevel=2,
		)
	return returns


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
	simulation_history_years: int = 20,
	seed: Optional[int] = None,
) -> List[pd.DataFrame]:
	"""Draw multivariate-normal daily returns from a historical calibration window."""
	hist = _historical_returns(tickers, lookback_years=simulation_history_years)
	mu = hist.mean().to_numpy()
	cov = hist.cov().to_numpy()
	rng = np.random.default_rng(seed)

	simulations: List[pd.DataFrame] = []
	for _ in range(int(n_sims)):
		draws = rng.multivariate_normal(mean=mu, cov=cov, size=int(n_days))
		simulation = pd.DataFrame(draws, columns=tickers, index=pd.RangeIndex(n_days))
		simulation["EFFR"] = _simulate_effr_path(n_days, method="parametric", rng=rng).to_numpy()
		simulations.append(simulation)
	return simulations


def simulate_bootstrap(
    tickers: List[str],
    n_days: int,
    n_sims: int = 1000,
	simulation_history_years: int = 20,
	block_size: Optional[int] = 60,
    seed: Optional[int] = None,
) -> List[pd.DataFrame]:
	"""
	Bootstrap Monte Carlo on historical daily simple returns.
	- If block_size is None: IID bootstrap (sample rows with replacement).
	- Else: moving block bootstrap with block_size.

	Returns a list of length n_sims; each item is a DataFrame [n_days x tickers].
	"""
	hist = _historical_returns(tickers, lookback_years=simulation_history_years)
	effr = _historical_effr().reindex(hist.index, method="ffill")
	joint = hist.copy()
	joint["EFFR"] = effr
	joint = joint.dropna(how="any")
	if joint.empty:
		raise ValueError("Asset returns and EFFR have no common simulation history")
	if block_size and block_size > len(joint):
		raise ValueError(f"block_size={block_size} exceeds {len(joint)} common history rows")

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


def simulation_weights(
	returns: pd.DataFrame,
	lookback_years: int,
	freq: str,
	cov_method: str | None = None,
	cov_params: dict | None = None,
	warmup_returns: pd.DataFrame | None = None,
	risk_free_rate: float | pd.Series = 0.0,
	min_weight: float = 0.10,
	max_weight: float = 0.40,
	expected_return_method: str = "mean",
) -> dict[str, pd.DataFrame]:
	"""Build weights at the simulation start using a full historical lookback."""
	if warmup_returns is None:
		warmup_returns = _historical_returns(list(returns.columns), lookback_years=lookback_years)
	warmup = warmup_returns[returns.columns].copy()

	warmup.index = pd.bdate_range(
		end=returns.index.min() - pd.offsets.BDay(),
		periods=warmup.shape[0],
	)
	initial_weight_date = returns.index.min() - pd.offsets.BDay()
	if isinstance(risk_free_rate, pd.Series) and initial_weight_date not in risk_free_rate.index:
		risk_free_rate = pd.concat(
			[pd.Series([risk_free_rate.iloc[0]], index=[initial_weight_date]), risk_free_rate]
		).sort_index()
	prices = (1.0 + pd.concat([warmup, returns])).cumprod()
	return build_portfolios_from_prices(
		prices=prices,
		start=initial_weight_date,
		end=returns.index.max(),
		lookback_years=lookback_years,
		freq=freq,
		cov_method=cov_method,
		cov_params=cov_params,
		risk_free_rate=risk_free_rate,
		min_weight=min_weight,
		max_weight=max_weight,
		expected_return_method=expected_return_method,
	)


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
		parquet_options = {"engine": engine} if engine is not None else {}
		df.to_parquet(path, compression="snappy", **parquet_options)
	except ImportError as e:
		raise ImportError(
			"Parquet export requires 'pyarrow' or 'fastparquet'. Install with:\n"
			"  pip install pyarrow\n"
			"or:\n  pip install fastparquet"
		) from e


def _terminal_quantile_paths(values: pd.DataFrame, quantile: float) -> pd.DataFrame:
	"""Select each strategy's complete path nearest its terminal-value quantile."""
	if not isinstance(values.columns, pd.MultiIndex) or "strategy" not in values.columns.names:
		raise ValueError("values must have MultiIndex columns including a 'strategy' level")
	if not 0.0 <= quantile <= 1.0:
		raise ValueError("quantile must be between 0 and 1")

	selected_paths = {}
	for strategy in values.columns.get_level_values("strategy").unique():
		strategy_paths = values.xs(strategy, level="strategy", axis=1)
		terminal_values = strategy_paths.iloc[-1].dropna()
		if terminal_values.empty:
			raise ValueError(f"strategy {strategy!r} has no terminal values")
		target = terminal_values.quantile(quantile)
		simulation = (terminal_values - target).abs().idxmin()
		selected_paths[strategy] = strategy_paths[simulation]

	return pd.DataFrame(selected_paths)


def plot_monthly_leverage_comparison(
	unlevered_values: pd.DataFrame,
	leveraged_monthly_values: pd.DataFrame,
	quantile: float = 0.50,
	title: str | None = None,
) -> plt.Axes:
	"""Compare VT unleveraged with monthly 2x equal-weight and max-Sharpe paths."""
	unlevered_paths = _terminal_quantile_paths(unlevered_values, quantile)
	leveraged_paths = _terminal_quantile_paths(leveraged_monthly_values, quantile)

	if "market_cap" not in unlevered_paths:
		raise ValueError("unlevered_values must include the market_cap strategy")
	missing = {"equal_weight", "max_sharpe"} - set(leveraged_paths.columns)
	if missing:
		raise ValueError(f"leveraged_monthly_values is missing strategies: {sorted(missing)}")

	comparison = pd.DataFrame(
		{
			"VT unleveraged": unlevered_paths["market_cap"],
			"Equal weight 2x monthly": leveraged_paths["equal_weight"],
			"Max Sharpe 2x monthly": leveraged_paths["max_sharpe"],
		}
	)
	comparison.index = pd.RangeIndex(1, len(comparison) + 1, name="Simulation day")
	quantile_label = "Median" if quantile == 0.50 else f"Q{quantile * 100:02.0f}"

	ax = comparison.plot(figsize=(12, 6))
	ax.set_xlim(1, len(comparison))
	ax.set_title(title or f"Terminal-{quantile_label} Simulation Comparison")
	ax.set_xlabel("Simulation day")
	ax.set_ylabel("Portfolio value")
	ax.grid(alpha=0.3)
	return ax


def plot_median_paths(values: pd.DataFrame, title: str | None = None) -> plt.Axes:
	"""Plot each strategy's complete path nearest the median terminal value."""
	median_paths = _terminal_quantile_paths(values, 0.50)

	median_paths.index = pd.RangeIndex(1, len(median_paths) + 1, name="Simulation day")
	ax = median_paths.plot(figsize=(12, 6))
	ax.set_xlim(1, len(median_paths))
	ax.set_title(title or "Monte Carlo Terminal-Median Simulation Paths")
	ax.set_xlabel("Simulation day")
	ax.set_ylabel("Portfolio value")
	ax.grid(alpha=0.3)
	return ax


def plot_q01_paths(values: pd.DataFrame, title: str | None = None) -> plt.Axes:
	"""Plot each strategy's complete path nearest the 1st-percentile terminal value."""
	q01_paths = _terminal_quantile_paths(values, 0.01)

	q01_paths.index = pd.RangeIndex(1, len(q01_paths) + 1, name="Simulation day")
	ax = q01_paths.plot(figsize=(12, 6))
	ax.set_xlim(1, len(q01_paths))
	ax.set_title(title or "Monte Carlo Terminal-Q01 Simulation Paths")
	ax.set_xlabel("Simulation day")
	ax.set_ylabel("Portfolio value")
	ax.grid(alpha=0.3)
	return ax

