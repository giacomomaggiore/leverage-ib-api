import cvxpy as cp
import pandas as pd
from datetime import date
from dateutil.relativedelta import relativedelta

from helpers.download import download
from helpers.estimation import log_returns, covariance_shrunk


def min_variance(
    tickers: list[str],
    as_of: date = None,
    timeframe_years: int = 3,
    max_weight: float = 0.3,
) -> pd.Series:
    if as_of is None:
        as_of = date.today()

    start = (as_of - relativedelta(years=timeframe_years)).strftime("%Y-%m-%d")
    end = as_of.strftime("%Y-%m-%d")

    prices = pd.DataFrame({t: download(t, start, end)["close"] for t in tickers})
    returns = log_returns(prices)
    sigma = covariance_shrunk(returns).values

    w = cp.Variable(len(tickers))
    objective = cp.Minimize(cp.quad_form(w, sigma))
    constraints = [cp.sum(w) == 1, w >= 0, w <= max_weight]
    cp.Problem(objective, constraints).solve()

    return pd.Series(w.value, index=tickers)
