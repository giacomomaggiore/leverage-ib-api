"""Simulate a margin-loan-financed leveraged portfolio, IBKR-style.

Mechanics: gross exposure is invested at `leverage` x equity, financed by margin debt.
The debt accrues daily interest (EFFR + spread, actual/360 — IBKR's convention); the
underlying assets move gross exposure independently of the (fixed) debt, so realized
leverage drifts every day. Rebalancing resets gross exposure back to target.
"""

from __future__ import annotations

import pandas as pd

_FREQ_TO_PERIOD = {"daily": "D", "weekly": "W", "monthly": "M"}


def _rebalance_flags(index: pd.DatetimeIndex, freq: str) -> pd.Series:
    """True on the first trading day of each new period (day/week/month)."""
    periods = index.to_period(_FREQ_TO_PERIOD[freq])
    return pd.Series(periods, index=index).ne(pd.Series(periods, index=index).shift(1))


def leverage_backtest(
    portfolio_returns: pd.Series,
    effr: pd.Series,
    leverage: float,
    start_value: float = 10_000.0,
    freq: str | None = None,
    band: float = 0.10,
    hard_cap: float = 4.0,
    spread: float = 0.01,
) -> pd.DataFrame:
    """
    Simulate leverage applied to an unlevered portfolio return series.

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

    port_rets = portfolio_returns.astype(float).sort_index()
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
