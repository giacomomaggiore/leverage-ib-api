import numpy as np
import pandas as pd
import cvxpy as cp
from scipy.optimize import minimize
from datetime import date
from dateutil.relativedelta import relativedelta
from sklearn.covariance import OAS

from helpers.fetch import load_data
from helpers.stats import log_returns, covariance_shrunk


def _returns(tickers: list[str], as_of: date, timeframe_years: int) -> pd.DataFrame:
    start = (as_of - relativedelta(years=timeframe_years)).strftime("%Y-%m-%d")
    end = as_of.strftime("%Y-%m-%d")

    series: dict[str, pd.Series] = {}
    for t in tickers:
        s = load_data(t, start, end)["adj close"].dropna()
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
    cov_method: str = "shrunk",  # 'shrunk' | 'empirical' | 'oas' | 'ewma' | 'factor'
    cov_params: dict | None = None,
) -> pd.Series:
    if as_of is None:
        as_of = date.today()

    rets = _returns(tickers, as_of, timeframe_years)
    avail = list(rets.columns) if isinstance(rets, pd.DataFrame) else []
    if rets is None or rets.shape[0] == 0 or len(avail) == 0:
        raise ValueError(f"No usable return observations for optimisation window ending {as_of}.")

    sigma = _compute_covariance(rets, method=cov_method, params=cov_params)
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
    cov_method: str = "shrunk",  # 'shrunk' | 'empirical' | 'oas' | 'ewma' | 'factor'
    cov_params: dict | None = None,
) -> pd.Series:
    if as_of is None:
        as_of = date.today()

    rets = _returns(tickers, as_of, timeframe_years)
    avail = list(rets.columns) if isinstance(rets, pd.DataFrame) else []
    if rets is None or rets.shape[0] == 0 or len(avail) == 0:
        raise ValueError(f"No usable return observations for optimisation window ending {as_of}.")

    sigma = _compute_covariance(rets, method=cov_method, params=cov_params)
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


def _compute_covariance(rets: pd.DataFrame, method: str, params: dict | None) -> np.ndarray:
    """Compute covariance matrix from returns with multiple methods.

    method:
      - 'shrunk'    : Ledoit–Wolf shrinkage (default)
      - 'empirical' : simple sample covariance
      - 'oas'       : Oracle Approximating Shrinkage
      - 'ewma'      : exponentially weighted covariance (params: span or alpha)
      - 'factor'    : simple PCA factor model (params: n_factors)
    """
    if params is None:
        params = {}

    if method == "shrunk":
        return covariance_shrunk(rets).values

    if method == "empirical":
        return rets.cov().values

    if method == "oas":
        cov = OAS().fit(rets.values).covariance_
        return cov

    if method == "ewma":
        alpha = params.get("alpha")
        span = params.get("span")
        if alpha is None:
            if span is None:
                span = 60  # ~3 months of daily data
            alpha = 2.0 / (span + 1.0)
        x = rets.values
        n, k = x.shape
        # weights: older get lower weight; sum to 1
        w = (1 - alpha) ** np.arange(n - 1, -1, -1, dtype=float)
        w /= w.sum()
        mu = (w[:, None] * x).sum(axis=0)
        xc = x - mu
        cov = np.einsum('ti,tj,t->ij', xc, xc, w)
        return cov

    if method == "factor":
        n_f = int(params.get("n_factors", 3))
        x = rets.values
        # PCA via SVD on demeaned returns
        x0 = x - x.mean(axis=0, keepdims=True)
        # economy SVD
        U, S, Vt = np.linalg.svd(x0, full_matrices=False)
        # factor scores (T x n_f): U[:, :n_f] * S[:n_f]
        F = U[:, :n_f] * S[:n_f]
        # loadings (N x n_f)
        B = Vt[:n_f, :].T
        # factor covariance
        Fcov = np.cov(F, rowvar=False)
        # specific variances from residuals
        X_hat = F @ B.T
        resid = x0 - X_hat
        spec_var = resid.var(axis=0, ddof=1)
        cov = B @ Fcov @ B.T + np.diag(spec_var)
        return cov

    raise ValueError("cov_method must be one of: 'shrunk', 'empirical', 'oas', 'ewma', 'factor'")
