import numpy as np
import pandas as pd
from datetime import date
from dateutil.relativedelta import relativedelta
from pypfopt import risk_models, expected_returns
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.exceptions import OptimizationError

from helpers.fetch import load_data


def _validate_weight_bounds(n_assets: int, min_weight: float, max_weight: float) -> None:
    if min_weight < 0 or max_weight > 1 or min_weight > max_weight:
        raise ValueError("weight bounds must satisfy 0 <= min_weight <= max_weight <= 1")
    if n_assets * min_weight > 1 or n_assets * max_weight < 1:
        raise ValueError(f"weight bounds ({min_weight}, {max_weight}) are infeasible for {n_assets} assets")


def _solve_max_sharpe(
    mu: pd.Series,
    sigma: pd.DataFrame,
    rf: float = 0.0,
    weight_bounds: tuple[float, float] = (0.10, 0.40),
) -> dict[str, float]:
    min_weight, max_weight = weight_bounds
    ordered_returns = mu.sort_values(ascending=False)
    remaining = 1.0 - len(mu) * min_weight
    feasible_weights = pd.Series(min_weight, index=mu.index)
    for ticker in ordered_returns.index:
        addition = min(remaining, max_weight - min_weight)
        feasible_weights[ticker] += addition
        remaining -= addition
        if remaining <= 1e-12:
            break

    ef = EfficientFrontier(
        expected_returns=mu,
        cov_matrix=sigma,
        weight_bounds=weight_bounds,
        solver="CLARABEL",
        solver_options={"max_iter": 500},
    )
    if float(feasible_weights @ mu) <= rf:
        ef.min_volatility()
        return ef.clean_weights()
    try:
        ef.max_sharpe(risk_free_rate=rf)
    except OptimizationError:
        ef = EfficientFrontier(
            expected_returns=mu,
            cov_matrix=sigma,
            weight_bounds=weight_bounds,
            solver="CLARABEL",
            solver_options={"max_iter": 500},
        )
        ef.min_volatility()
    return ef.clean_weights()

def _prices(tickers: list[str], as_of: date, timeframe_years: int) -> pd.DataFrame:
    start = (as_of - relativedelta(years=timeframe_years)).strftime("%Y-%m-%d")
    end = as_of.strftime("%Y-%m-%d")
    return pd.concat(
        [load_data(ticker, start, end)["adj close"].rename(ticker) for ticker in tickers],
        axis=1,
        join="inner",
    )


def min_variance(
    tickers: list[str],
    as_of: date = None,
    timeframe_years: int = 3,
    cov_method: str = "shrunk",  # 'shrunk' | 'empirical' | 'oas' | 'ewma' | 'factor'
    cov_params: dict | None = None,
    min_weight: float = 0.10,
    max_weight: float = 0.40,
) -> pd.Series:
    if as_of is None:
        as_of = date.today()

    prices = _prices(tickers, as_of, timeframe_years)
    returns = prices.pct_change().dropna(how="any")
    sigma = _compute_covariance(returns, method=cov_method, params=cov_params)
    mu = expected_returns.mean_historical_return(prices, frequency=252, log_returns=False)
    _validate_weight_bounds(len(tickers), min_weight, max_weight)
    ef = EfficientFrontier(expected_returns=mu, cov_matrix=sigma, weight_bounds=(min_weight, max_weight))
    ef.min_volatility()
    return pd.Series(ef.clean_weights()).reindex(tickers, fill_value=0.0)


def max_sharpe(
    tickers: list[str],
    as_of: date = None,
    timeframe_years: int = 3,
    rf: float = 0.0,
    cov_method: str = "shrunk",  # 'shrunk' | 'empirical' | 'oas' | 'ewma' | 'factor'
    cov_params: dict | None = None,
    min_weight: float = 0.10,
    max_weight: float = 0.40,
    expected_return_method: str = "mean",
) -> pd.Series:
    if as_of is None:
        as_of = date.today()

    prices = _prices(tickers, as_of, timeframe_years)
    returns = prices.pct_change().dropna(how="any")
    sigma = _compute_covariance(returns, method=cov_method, params=cov_params)
    mu = _expected_returns(prices, method=expected_return_method)
    _validate_weight_bounds(len(tickers), min_weight, max_weight)
    return pd.Series(
        _solve_max_sharpe(mu, sigma, rf=rf, weight_bounds=(min_weight, max_weight))
    ).reindex(tickers, fill_value=0.0)


def _expected_returns(prices: pd.DataFrame, method: str = "mean") -> pd.Series:
    """Estimate annual returns or remove cross-asset return ranking for sensitivity tests."""
    if method == "mean":
        return expected_returns.mean_historical_return(prices, frequency=252, log_returns=False)
    if method == "ema":
        return expected_returns.ema_historical_return(prices, frequency=252, log_returns=False)
    if method == "equal":
        return pd.Series(1.0, index=prices.columns)
    raise ValueError("expected_return_method must be one of: 'mean', 'ema', 'equal'")


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

    if method == "shrunk":
        cs = risk_models.CovarianceShrinkage(rets, returns_data=True)
        cov = cs.ledoit_wolf()
    elif method == "empirical":
        cov = risk_models.sample_cov(rets, returns_data=True)
    elif method == "oas":
        cs = risk_models.CovarianceShrinkage(rets, returns_data=True)
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
        Fcov = np.atleast_2d(np.cov(F, rowvar=False))
        # specific variances from residuals
        X_hat = F @ B.T
        resid = x0 - X_hat
        spec_var = resid.var(axis=0, ddof=1)
        cov = 252.0 * (B @ Fcov @ B.T + np.diag(spec_var))
        cov = pd.DataFrame(cov, index=rets.columns, columns=rets.columns)

    else:
        raise ValueError("cov_method must be one of: 'shrunk', 'empirical', 'oas', 'ewma', 'factor'")

    jitter = float(params.get("jitter", 0.0))
    if jitter > 0:
        arr = cov.to_numpy(copy=True)
        idx = np.arange(arr.shape[0])
        arr[idx, idx] = arr[idx, idx] + jitter
        cov = pd.DataFrame(arr, index=cov.index, columns=cov.columns)
    return cov


def covariance_diagnostics(sigma: pd.DataFrame) -> pd.Series:
    """Return annualized asset volatilities implied by an annual covariance matrix."""
    return pd.Series(np.sqrt(np.diag(sigma)), index=sigma.index, name="annualized_volatility")



def build_portfolios(
    tickers,
    start,
    end,
    lookback_years=4,
    freq="MS",
    cov_method=None,
    cov_params=None,
    risk_free_rate: float | pd.Series = 0.0,
    min_weight: float = 0.10,
    max_weight: float = 0.40,
    expected_return_method: str = "mean",
):
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    load_start = start_ts - pd.DateOffset(years=lookback_years)
    px = pd.concat(
        [load_data(ticker, load_start.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d"))["adj close"].rename(ticker) for ticker in tickers],
        axis=1,
        join="inner",
    ).sort_index()
    return build_portfolios_from_prices(
        prices=px,
        start=start_ts,
        end=end_ts,
        lookback_years=lookback_years,
        freq=freq,
        cov_method=cov_method,
        cov_params=cov_params,
        risk_free_rate=risk_free_rate,
        min_weight=min_weight,
        max_weight=max_weight,
        expected_return_method=expected_return_method,
    )


def build_portfolios_from_prices(
    prices: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    lookback_years: int = 4,
    freq: str = "BMS",
    cov_method: str | None = None,
    cov_params: dict | None = None,
    risk_free_rate: float | pd.Series = 0.0,
    min_weight: float = 0.10,
    max_weight: float = 0.40,
    expected_return_method: str = "mean",
) -> dict[str, pd.DataFrame]:
    """
    Build dynamic portfolios using provided adjusted-close price history (no fetching).

    - prices: DataFrame [dates x tickers] of adjusted-close prices
    - start/end: rebalancing window (inclusive)
    - lookback_years: rolling window length used for estimation
    - freq: rebalancing frequency (e.g., 'BMS' business-month start)
    - cov_method/cov_params: covariance configuration passed to the optimizer
    - risk_free_rate: annual decimal rate or date-indexed annual decimal Series

    Returns target-weight schedules with keys: 'min_variance', 'max_sharpe',
    'market_cap' (100% VT), and 'equal_weight' (1/N).
    """
    px = prices.sort_index().astype(float).dropna()

    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    rebalance_dates = pd.date_range(start, end, freq=freq)
    if start not in rebalance_dates:
        rebalance_dates = rebalance_dates.insert(0, start)

    if cov_method is None:
        cov_method = "oas"
    if cov_params is None:
        cov_params = {"jitter": 1e-8}

    tickers = list(px.columns)
    _validate_weight_bounds(len(tickers), min_weight, max_weight)
    # Ensure VT column exists in weights output for market_cap
    cols = sorted(set(tickers) | {"VT"})
    market_cap_row = pd.Series({ticker: float(ticker == "VT") for ticker in cols})
    equal_weight_row = pd.Series(
        {ticker: 1.0 / len(tickers) if ticker in tickers else 0.0 for ticker in cols}
    )
    mv: dict[pd.Timestamp, pd.Series] = {}
    ms: dict[pd.Timestamp, pd.Series] = {}

    if isinstance(risk_free_rate, pd.Series):
        risk_free_rate = risk_free_rate.sort_index()

    for dt in rebalance_dates:
        window_start = dt - pd.DateOffset(years=lookback_years)
        win = px.loc[(px.index >= window_start) & (px.index <= dt)]
        if win.shape[0] < 2:
            continue

        rets = win.pct_change().dropna(how="any")
        if rets.shape[0] < 2:
            continue

        mu = _expected_returns(win, method=expected_return_method)
        sigma = _compute_covariance(rets, method=cov_method, params=cov_params)
        ef_mv = EfficientFrontier(
            expected_returns=mu,
            cov_matrix=sigma,
            weight_bounds=(min_weight, max_weight),
        )
        ef_mv.min_volatility()
        w_mv = ef_mv.clean_weights()
        if isinstance(risk_free_rate, pd.Series):
            rf = risk_free_rate.reindex([dt], method="ffill").iloc[0]
            if pd.isna(rf):
                raise ValueError(f"Missing risk-free rate on {dt.date()}")
        else:
            rf = risk_free_rate
        w_ms = _solve_max_sharpe(
            mu,
            sigma,
            rf=float(rf),
            weight_bounds=(min_weight, max_weight),
        )

        mv[dt] = pd.Series({t: float(w_mv.get(t, 0.0)) for t in tickers}).reindex(cols, fill_value=0.0)
        ms[dt] = pd.Series({t: float(w_ms.get(t, 0.0)) for t in tickers}).reindex(cols, fill_value=0.0)

    def to_schedule(d: dict[pd.Timestamp, pd.Series], name: str) -> pd.DataFrame:
        if not d:
            raise ValueError(f"Insufficient price history to build {name} weights")
        return pd.DataFrame(d).T.sort_index()

    static_schedules = {
        "market_cap": {dt: market_cap_row for dt in rebalance_dates},
        "equal_weight": {dt: equal_weight_row for dt in rebalance_dates},
    }

    return {
        "min_variance": to_schedule(mv, "min_variance"),
        "max_sharpe": to_schedule(ms, "max_sharpe"),
        "market_cap": to_schedule(static_schedules["market_cap"], "market_cap"),
        "equal_weight": to_schedule(static_schedules["equal_weight"], "equal_weight"),
    }
    