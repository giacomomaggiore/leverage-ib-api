import numpy as np
import pandas as pd
from datetime import date
from dateutil.relativedelta import relativedelta
from pypfopt import risk_models, expected_returns
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import exceptions as ppo_exceptions

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


def _prices(tickers: list[str], as_of: date, timeframe_years: int) -> pd.DataFrame:
    start = (as_of - relativedelta(years=timeframe_years)).strftime("%Y-%m-%d")
    end = as_of.strftime("%Y-%m-%d")

    series: dict[str, pd.Series] = {}
    for t in tickers:
        s = load_data(t, start, end)["adj close"].dropna()
        if s.shape[0] >= 2:
            series[t] = s

    if not series:
        return pd.DataFrame()

    prices = pd.concat(series.values(), axis=1, join="inner")
    prices.columns = list(series.keys())
    if prices.shape[0] < 2 or prices.shape[1] == 0:
        return pd.DataFrame()
    return prices


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

    # Covariance matrix (daily) via PyPortfolioOpt risk models (using returns)
    sigma = _compute_covariance(rets, method=cov_method, params=cov_params)

    # Expected returns from prices using PyPortfolioOpt (simple returns, annualized)
    prices = _prices(tickers, as_of, timeframe_years)
    mu = expected_returns.mean_historical_return(prices, frequency=252, log_returns=False)

    # Preserve previous behavior: long-only with no per-asset cap by default
    ef = EfficientFrontier(expected_returns=mu, cov_matrix=sigma, weight_bounds=(0.0, 1.0))

    # Minimise portfolio volatility
    ef.min_volatility()
    w = ef.clean_weights()

    out = pd.Series(0.0, index=tickers)
    for t in avail:
        out.loc[t] = float(w.get(t, 0.0))
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

    # Covariance matrix (daily) via PyPortfolioOpt risk models
    sigma = _compute_covariance(rets, method=cov_method, params=cov_params)
    # Expected returns from prices using PyPortfolioOpt (simple returns, annualized)
    prices = _prices(tickers, as_of, timeframe_years)
    mu = expected_returns.mean_historical_return(prices, frequency=252, log_returns=False)

    # Some CVXPY solver defaults can hit a 'user_limit' status. Try a
    # robust sequence of solvers with generous iteration budgets.
    last_err: Exception | None = None
    for solver, kwargs in [
        ("ECOS", {"max_iters": 50000}),
        ("CLARABEL", {"max_iter": 80000, "time_limit": 120.0}),
        ("SCS", {"max_iters": 100000}),
        (None, {}),  # let CVXPY choose as a final attempt
    ]:
        try:
            # Fresh EF each attempt to avoid accumulating constraints internally
            ef = EfficientFrontier(expected_returns=mu, cov_matrix=sigma, weight_bounds=(0.0, 1.0))
            if solver is None:
                ef.max_sharpe(risk_free_rate=rf)
            else:
                ef.max_sharpe(risk_free_rate=rf, solver=solver, **kwargs)
            last_err = None
            break
        except (ppo_exceptions.OptimizationError, Exception) as e:  # fall back to next solver
            last_err = e

    if last_err is not None:
        # Fallback 1: maximize quadratic utility as a proxy for Sharpe
        try:
            
            ef = EfficientFrontier(expected_returns=mu, cov_matrix=sigma, weight_bounds=(0.0, 1.0))
            ef.max_quadratic_utility(risk_aversion=1.0)
            print("max_sharpe failed, falling back to max_quadratic_utility.")
            last_err = None
        except Exception as e1:
            last_err = e1

    if last_err is not None:
        # Fallback 2: resort to minimum volatility to keep pipeline running
        try:
            ef = EfficientFrontier(expected_returns=mu, cov_matrix=sigma, weight_bounds=(0.0, 1.0))
            ef.min_volatility()
            print("max_sharpe failed, falling back to min_volatility.")
            last_err = None
        except Exception as e2:
            last_err = e2

    if last_err is not None:
        raise last_err
    w = ef.clean_weights()

    out = pd.Series(0.0, index=tickers)
    for t in avail:
        out.loc[t] = float(w.get(t, 0.0))
    return out


def _compute_covariance(rets: pd.DataFrame, method: str, params: dict | None) -> pd.DataFrame:
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

    cov: pd.DataFrame | np.ndarray | None = None

    if method == "shrunk":
        # Ledoit–Wolf shrinkage via PyPortfolioOpt
        cs = risk_models.CovarianceShrinkage(rets, returns_data=True)
        cov = cs.ledoit_wolf()

    if method == "empirical":
        cov = risk_models.sample_cov(rets, returns_data=True)

    if method == "oas":
        cs = risk_models.CovarianceShrinkage(rets, returns_data=True)
        # Oracle Approximating Shrinkage
        cov = cs.oracle_approximating()

    if method == "ewma":
        alpha = params.get("alpha")
        span = params.get("span")
        if span is None and alpha is not None and alpha > 0:
            span = 2.0 / alpha - 1.0
        if span is None:
            span = 60  # ~3 months of daily data by default
        cov = risk_models.exp_cov(rets, span=int(span), returns_data=True)

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
        cov = pd.DataFrame(cov, index=rets.columns, columns=rets.columns)
    
    if cov is None:
        raise ValueError("cov_method must be one of: 'shrunk', 'empirical', 'oas', 'ewma', 'factor'")

    # Ensure DataFrame
    if not isinstance(cov, pd.DataFrame):
        cov = pd.DataFrame(cov, index=rets.columns, columns=rets.columns)

    # Optional diagonal jitter for conditioning
    try:
        jitter = float(params.get("jitter", 0.0))
    except Exception:
        jitter = 0.0
    if jitter and jitter > 0:
        arr = cov.to_numpy(copy=True)
        idx = np.arange(arr.shape[0])
        arr[idx, idx] = arr[idx, idx] + jitter
        cov = pd.DataFrame(arr, index=cov.index, columns=cov.columns)
    return cov

