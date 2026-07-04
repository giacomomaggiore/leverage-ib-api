import numpy as np
import pandas as pd
import cvxpy as cp
from scipy.optimize import minimize
from datetime import date
from dateutil.relativedelta import relativedelta

from helpers.download import load_data
from helpers.estimation import log_returns, covariance_shrunk


def _returns(tickers: list[str], as_of: date, timeframe_years: int) -> pd.DataFrame:
    start = (as_of - relativedelta(years=timeframe_years)).strftime("%Y-%m-%d")
    end = as_of.strftime("%Y-%m-%d")

    series: dict[str, pd.Series] = {}
    for t in tickers:
        s = load_data(t, start, end)["close"].dropna()
        # keep only tickers with at least 2 price points in the window
        if s.shape[0] >= 2:
            series[t] = s

    if not series:
        return pd.DataFrame()

    # Align on common business days across the kept tickers
    prices = pd.concat(series.values(), axis=1, join="inner")
    prices.columns = list(series.keys())

    if prices.shape[0] < 2 or prices.shape[1] == 0:
        return pd.DataFrame()

    return log_returns(prices)


def min_variance(
    tickers: list[str],
    as_of: date = None,
    timeframe_years: int = 3,
    max_weight: float = 0.3,
) -> pd.Series:
    if as_of is None:
        as_of = date.today()

    rets = _returns(tickers, as_of, timeframe_years)
    avail = list(rets.columns) if isinstance(rets, pd.DataFrame) else []
    if rets is None or rets.shape[0] == 0 or len(avail) == 0:
        raise ValueError(f"No usable return observations for optimisation window ending {as_of}.")

    sigma = covariance_shrunk(rets).values
    sigma = np.asarray(sigma, dtype=float)
    if sigma.ndim == 0:
        sigma = sigma.reshape(1, 1)
    elif sigma.ndim == 1:
        sigma = np.diag(sigma)

    # Ensure weight cap is feasible given number of assets
    cap = max(max_weight, 1.0 / len(avail))

    # Minimise variance via SLSQP with box constraints and full-investment
    def var_obj(w):
        return float(w @ sigma @ w)

    n = len(avail)
    x0 = np.ones(n) / n
    bounds = [(0.0, cap)] * n
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    res = minimize(var_obj, x0=x0, method="SLSQP", bounds=bounds, constraints=cons)

    out = pd.Series(0.0, index=tickers)
    out.loc[avail] = res.x
    return out


def max_sharpe(
    tickers: list[str],
    as_of: date = None,
    timeframe_years: int = 3,
    max_weight: float = 0.3,
    rf: float = 0.0,
) -> pd.Series:
    if as_of is None:
        as_of = date.today()

    rets = _returns(tickers, as_of, timeframe_years)
    avail = list(rets.columns) if isinstance(rets, pd.DataFrame) else []
    if rets is None or rets.shape[0] == 0 or len(avail) == 0:
        raise ValueError(f"No usable return observations for optimisation window ending {as_of}.")

    sigma = covariance_shrunk(rets).values
    sigma = np.asarray(sigma, dtype=float)
    if sigma.ndim == 0:
        sigma = sigma.reshape(1, 1)
    elif sigma.ndim == 1:
        sigma = np.diag(sigma)
    mu = rets.mean().values * 252  # annualised expected returns

    def neg_sharpe(w):
        # annualised Sharpe — sigma is daily so vol scales by sqrt(252)
        # mu @ w is the expected return of the portfolio
        # rf is the risk-free rate
        # w @ sigma @ w gives the portfolio variance. 
        # 
        # The negative sign is used because we want to maximize the Sharpe ratio, but the optimizer minimizes functions.
        return -(mu @ w - rf) / np.sqrt(w @ sigma @ w * 252)

    # minimize the negative Sharpe ratio 
    # subject to full investment and weight cap
    # Ensure weight cap is feasible given number of assets
    cap = max(max_weight, 1.0 / len(avail))

    result = minimize(
        neg_sharpe,
        x0=np.ones(len(avail)) / len(avail),
        method="SLSQP",
        bounds=[(0, cap)] * len(avail),
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
    )

    out = pd.Series(0.0, index=tickers)
    out.loc[avail] = result.x
    return out
