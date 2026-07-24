"""Simulate a margin-loan-financed leveraged portfolio, IBKR-style.

Mechanics: gross exposure is invested at `leverage` x equity, financed by margin debt.
The debt accrues daily interest (EFFR + spread, actual/360 — IBKR's convention); the
underlying assets move gross exposure independently of the (fixed) debt, so realized
leverage drifts every day. Rebalancing resets gross exposure back to target.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from helpers.backtest import portfolio_returns

DATA_DIR = Path(__file__).parent.parent / "data"

# Maps the user-facing frequency name to the pandas period code used to detect
# "new period started" boundaries in the trading-day index.
_FREQ_TO_PERIOD = {"daily": "D", "weekly": "W", "monthly": "M"}


def load_margin_rate(index: pd.DatetimeIndex, spread: float = 0.01) -> pd.Series:
    """
    Annualized margin borrow rate for each date in `index`: FRED EFFR + IBKR-style spread.

    EFFR is only published on days the Fed is open, so it's forward-filled onto `index`.
    `spread` approximates IBKR's real tiered markup over the benchmark rate (EFFR alone
    understates actual borrowing cost).
    """
    effr = pd.read_csv(DATA_DIR / "FRED_EFFR.csv", index_col="date", parse_dates=True)["EFFR"]
    return (effr / 100.0 + spread).reindex(index, method="ffill")


def _rebalance_flags(index: pd.DatetimeIndex, freq: str) -> pd.Series:
    """True on the first trading day of each new period (day/week/month)."""
    periods = index.to_period(_FREQ_TO_PERIOD[freq])
    return pd.Series(periods, index=index).ne(pd.Series(periods, index=index).shift(1))


def leverage_backtest(
    weights: pd.DataFrame,
    leverage: float,
    start_value: float = 10_000.0,
    returns: pd.DataFrame | None = None,
    effr: pd.Series | None = None,
    freq: str | None = None,  # None | 'daily' | 'weekly' | 'monthly'
    band: float = 0.10,       # rebalance early if leverage drifts +/-10% from target (only used when freq is None)
    hard_cap: float = 4.0,    # IBKR Reg T ceiling; forces a rebalance regardless of freq/band
    spread: float = 0.01,     # markup over EFFR approximating IBKR's real borrow rate
    portfolio_return_series: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Simulate leverage `leverage` applied to the strategy defined by `weights`.

    Rebalancing (reset gross exposure to `leverage * equity`) happens when:
    - `freq` is given: on the first trading day of each period (day/week/month).
    - `freq` is None: whenever leverage drifts outside [leverage*(1-band), leverage*(1+band)].
    In both cases, a breach of `hard_cap` forces an immediate rebalance (margin call).

    A non-positive equity event liquidates the account. The event row and all later
    rows remain at zero so Monte Carlo aggregation includes ruined paths.
    """
    if leverage <= 1.0:
        raise ValueError("leverage must be > 1.0 to represent a margin loan")
    if freq is not None and freq not in _FREQ_TO_PERIOD:
        raise ValueError(f"freq must be one of {list(_FREQ_TO_PERIOD)} or None")

    if portfolio_return_series is None:
        port_rets = portfolio_returns(weights, returns)
    else:
        port_rets = portfolio_return_series.astype(float).sort_index()

    if effr is None:
        annual_rate = load_margin_rate(port_rets.index, spread=spread)
    else:
        annual_rate = effr.astype(float).reindex(port_rets.index).ffill() / 100.0 + spread
    if annual_rate.isna().any():
        raise ValueError("Missing EFFR for one or more leveraged return dates")
    day_counts = port_rets.index.to_series().diff().dt.days.fillna(1.0)
    accrual_rate = annual_rate * day_counts / 360.0

    due = _rebalance_flags(port_rets.index, freq) if freq is not None else None
    lower_band, upper_band = leverage * (1 - band), leverage * (1 + band)
    equity = start_value
    gross = leverage * start_value
    debt = gross - equity

    rows = []
    ruined = False
    for dt in port_rets.index:
        if ruined:
            rows.append((dt, 0.0, 0.0, 0.0, float("nan"), annual_rate.loc[dt], False, True))
            continue

        gross *= 1.0 + port_rets.loc[dt]
        debt *= 1.0 + accrual_rate.loc[dt]
        equity = gross - debt

        if equity <= 0:
            ruined = True
            equity = 0.0
            gross = 0.0
            debt = 0.0
            rows.append((dt, equity, gross, debt, float("nan"), annual_rate.loc[dt], False, True))
            continue

        current_leverage = gross / equity
        forced = current_leverage > hard_cap
        scheduled = due.loc[dt] if due is not None else False
        
        drifted = freq is None and not (lower_band <= current_leverage <= upper_band)

        rebalanced = bool(forced or scheduled or drifted)
        if rebalanced:
            gross = leverage * equity
            debt = gross - equity
            current_leverage = leverage

        rows.append((dt, equity, gross, debt, current_leverage, annual_rate.loc[dt], rebalanced, False))

    return pd.DataFrame(
        rows,
        columns=["date", "portfolio_value", "gross_exposure", "debt", "leverage", "margin_rate", "rebalanced", "ruined"],
    ).set_index("date")
