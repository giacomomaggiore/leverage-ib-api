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

    elif method == "empirical":
        cov = risk_models.sample_cov(rets, returns_data=True)

    elif method == "oas":
        cs = risk_models.CovarianceShrinkage(rets, returns_data=True)
        # Oracle Approximating Shrinkage
        cov = cs.oracle_approximating()

    elif method == "ewma":
        alpha = params.get("alpha")
        span = params.get("span")
        if span is None and alpha is not None and alpha > 0:
            span = 2.0 / alpha - 1.0
        if span is None:
            span = 60  # ~3 months of daily data by default
        cov = risk_models.exp_cov(rets, span=int(span), returns_data=True)

    elif method == "factor":
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

    else:
        cov = cov  # no-op to keep structure

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



def build_portfolios(tickers, start, end, lookback_years=4, freq="MS", cov_method=None, cov_params=None):
    # Wrapper around build_portfolios_from_prices using a shared price matrix.
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    # Load enough history to cover the first lookback window
    load_start = start_ts
    # For rolling windows inside build_portfolios_from_prices, it will access (dt - lookback_years)
    # Ensure we have sufficient earlier data by extending the load start back by lookback_years
    load_start = start_ts - pd.DateOffset(years=lookback_years)

    # Assemble aligned adjusted-close prices for the entire period
    series: dict[str, pd.Series] = {}
    for t in tickers:
        s = load_data(t, load_start.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d"))["adj close"].dropna().rename(t)
        if s.shape[0] >= 2:
            series[t] = s
    if not series:
        raise ValueError("No price data available to build portfolios")
    px = pd.concat(series.values(), axis=1, join="inner").sort_index()

    # Rebalance window starts after enough history is accumulated
    rebalance_start = start_ts + pd.DateOffset(years=lookback_years)

    return build_portfolios_from_prices(
        prices=px,
        start=rebalance_start,
        end=end_ts,
        lookback_years=lookback_years,
        freq=freq,
        cov_method=cov_method,
        cov_params=cov_params,
    )


def build_portfolios_from_prices(
    prices: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    lookback_years: int = 4,
    freq: str = "BMS",
    cov_method: str | None = None,
    cov_params: dict | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Build dynamic portfolios using provided adjusted-close price history (no fetching).

    - prices: DataFrame [dates x tickers] of adjusted-close prices
    - start/end: rebalancing window (inclusive)
    - lookback_years: rolling window length used for estimation
    - freq: rebalancing frequency (e.g., 'BMS' business-month start)
    - cov_method/cov_params: covariance configuration passed to the optimizer

    Returns dict of daily weight DataFrames with keys: 'min_variance', 'max_sharpe', 'market_cap' (100% VT), 'equal_weight' (1/N).
    """
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise ValueError("prices must be a non-empty DataFrame of adjusted-close values")

    # Clean and ensure proper ordering
    px = prices.sort_index().astype(float).dropna(how="all", axis=1)
    px = px.dropna(how="any")  # require full alignment for simplicity
    if px.shape[0] < 2 or px.shape[1] == 0:
        raise ValueError("Insufficient price history")

    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    # Use business-month start (default) to mirror historical pipeline
    rebalance_dates = pd.date_range(start, end, freq=freq)

    # Default covariance settings consistent with build_portfolios
    COV_METHOD = "oas"
    COV_PARAMS = {"jitter": 1e-8}
    if cov_method is None:
        cov_method = COV_METHOD
    if cov_params is None:
        cov_params = COV_PARAMS

    tickers = list(px.columns)
    # Ensure VT column exists in weights output for market_cap
    cols = sorted(set(tickers) | {"VT"})
    mv, ms, mc, ew = {}, {}, {}, {}

    for dt in rebalance_dates:
        # Select rolling lookback window ending at dt (inclusive)
        window_start = dt - pd.DateOffset(years=lookback_years)
        win = px.loc[(px.index >= window_start) & (px.index <= dt)]
        if win.shape[0] < 2:
            # Not enough data; carry previous if available, else skip
            if mv:
                mv[dt] = mv[max(mv.keys())]
            if ms:
                ms[dt] = ms[max(ms.keys())]
            if mc:
                mc[dt] = mc[max(mc.keys())]
            continue

        # Compute daily returns for covariance
        rets = win.pct_change().dropna(how="any")
        if rets.shape[0] < 2:
            if mv:
                mv[dt] = mv[max(mv.keys())]
            if ms:
                ms[dt] = ms[max(ms.keys())]
            if mc:
                mc[dt] = mc[max(mc.keys())]
            continue

        # Expected returns from prices using PyPortfolioOpt (simple returns, annualized)
        mu = expected_returns.mean_historical_return(win, frequency=252, log_returns=False)
        # Covariance via shared helper
        sigma = _compute_covariance(rets, method=cov_method, params=cov_params)

        # Solve portfolios
        try:
            ef_mv = EfficientFrontier(expected_returns=mu, cov_matrix=sigma, weight_bounds=(0.0, 1.0))
            ef_mv.min_volatility()
            w_mv = ef_mv.clean_weights()
        except Exception:
            w_mv = {t: 0.0 for t in tickers}

        try:
            ef_ms = EfficientFrontier(expected_returns=mu, cov_matrix=sigma, weight_bounds=(0.0, 1.0))
            ef_ms.max_sharpe(risk_free_rate=0.0)
            w_ms = ef_ms.clean_weights()
        except Exception:
            w_ms = {t: 0.0 for t in tickers}

        # Market-cap proxy: force 100% VT (if VT not present, this will be all zeros)
        w_mc = {t: (1.0 if t == "VT" else 0.0) for t in tickers}
        # (equal-weight is constructed explicitly below)

        # Reindex MV/MS to include VT with 0 if not present in window
        mv[dt] = pd.Series({t: float(w_mv.get(t, 0.0)) for t in tickers}).reindex(cols, fill_value=0.0)
        ms[dt] = pd.Series({t: float(w_ms.get(t, 0.0)) for t in tickers}).reindex(cols, fill_value=0.0)
        # Market-cap: 100% VT across full cols
        mc[dt] = pd.Series(w_mc).reindex(cols, fill_value=0.0)
        # Equal-weight over available price tickers; 0 for VT if not present
        ew_row = {t: (1.0 / len(tickers)) for t in tickers}
        for t in set(cols) - set(tickers):
            ew_row[t] = 0.0
        ew[dt] = pd.Series(ew_row).reindex(cols, fill_value=0.0)

    def to_daily(d: dict[pd.Timestamp, pd.Series]) -> pd.DataFrame:
        if not d:
            # fallback to a single equal-weight row at start
            d = {pd.Timestamp(start): pd.Series({t: 1.0 / len(tickers) for t in tickers})}
        df = pd.DataFrame(d).T.sort_index()
        first = df.index.min()
        daily_idx = pd.bdate_range(first, end)
        return df.reindex(daily_idx, method="ffill").reindex(px.index, method="ffill").loc[(px.index >= first) & (px.index <= end)]

    return {
        "min_variance": to_daily(mv),
        "max_sharpe": to_daily(ms),
        "market_cap": to_daily(mc),
        "equal_weight": to_daily(ew),
    }
    