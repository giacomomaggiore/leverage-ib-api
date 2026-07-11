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
	return prices.pct_change().dropna(how="any")


def _historical_effr() -> pd.Series:
	"""Historical EFFR series in percentage points."""
	effr = pd.read_csv(DATA_DIR / "FRED_EFFR.csv", parse_dates=["date"], index_col="date")["EFFR"]
	return effr.sort_index().astype(float)


def _simulate_effr_path(n_days: int, method: str, rng: np.random.Generator) -> pd.Series:
	"""Simulate an EFFR scenario path in percentage points."""
	effr = _historical_effr()
	if effr.empty:
		raise ValueError("Not enough historical EFFR data to simulate a rate path")

	changes = effr.diff().dropna()
	if changes.empty:
		raise ValueError("Not enough historical EFFR changes to simulate a rate path")

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
    if hist.empty:
        raise ValueError("Not enough historical data to estimate parameters")

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
	if hist.empty:
		raise ValueError("Not enough historical data to bootstrap")

	effr = _historical_effr().reindex(hist.index, method="ffill")
	joint = hist.copy()
	joint["EFFR"] = effr

	rng = np.random.default_rng(seed)

	# number of historical observations available for bootstrapping
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
		# Moving block bootstrap: sample contiguous blocks of length b
		b = int(block_size)
		for _ in range(int(n_sims)):
			chunks: list[pd.DataFrame] = []
			days_left = int(n_days)
			while days_left > 0:
				# number of valid start positions so a full block fits is (T - b + 1)
				if T - b + 1 > 0:
					start = int(rng.integers(low=0, high=T - b + 1))
					block = joint.iloc[start : start + b]
				else:
					# history shorter than block: take whole history as a block
					start = 0
					block = joint.copy()
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

	tickers = [c for c in (list(simulations[0].columns) if simulations else list(weights.columns)) if c != "EFFR"]
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
		W = pd.DataFrame(np.tile(base_w.to_numpy(), (sim.shape[0], 1)), index=sim.index, columns=tickers)
		values = backtest_portfolio(W, start_value=start_value, returns=sim[tickers])
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
	effr = sim_returns["EFFR"].copy() if "EFFR" in sim_returns.columns else None
	if effr is not None:
		effr.index = idx
	rets = sim_returns.drop(columns=["EFFR"], errors="ignore").copy()
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
	if not isinstance(sim_returns, pd.DataFrame) or sim_returns.empty:
		raise ValueError("sim_returns must be a non-empty DataFrame")

	n_days = sim_returns.shape[0]
	idx = pd.bdate_range(start=pd.Timestamp("2000-01-03"), periods=n_days)
	effr = sim_returns["EFFR"] if "EFFR" in sim_returns.columns else None
	rets = sim_returns.drop(columns=["EFFR"], errors="ignore").copy()
	rets.index = idx

	prices = (1.0 + rets).cumprod()
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

	for i, sim in enumerate(simulations, start=1):
		# Use a business-day index so portfolio rebuilding can use calendar frequencies.
		idx = pd.bdate_range(start=pd.Timestamp("2000-01-03"), periods=sim.shape[0])
		rets = sim.copy()
		rets.index = idx

		# Build optimized/equal-weight portfolios only on the selected tickers.
		prices = (1.0 + rets[base_tickers]).cumprod()
		weights = build_portfolios_from_prices(
			prices=prices,
			start=prices.index.min(),
			end=prices.index.max(),
			lookback_years=lookback_years,
			freq="BMS",
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
	"""
	Save simulation values DataFrame to Parquet. Ensures MultiIndex columns (sim, strategy).
	"""
	df = values.copy()
	if not isinstance(df.columns, pd.MultiIndex):
		# Try to parse flat names like 'sim_1_min_variance' → (1, 'min_variance')
		tuples = []
		for c in df.columns:
			sim = None
			strat = str(c)
			s = str(c)
			if s.startswith("sim_"):
				parts = s.split("_", 2)
				if len(parts) == 3 and parts[1].isdigit():
					sim = int(parts[1])
					strat = parts[2]
			tuples.append((sim if sim is not None else 0, strat))
		df.columns = pd.MultiIndex.from_tuples(tuples, names=["sim", "strategy"])
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

