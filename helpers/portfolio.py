import numpy as np
import pandas as pd
from datetime import date
from dateutil.relativedelta import relativedelta
from pypfopt import risk_models, expected_returns
from pypfopt.efficient_frontier import EfficientFrontier

from helpers.fetch import load_data
from helpers.stats import log_returns


def _solve_max_sharpe(mu: pd.Series, sigma: pd.DataFrame, rf: float = 0.0) -> dict[str, float]:
    ef = EfficientFrontier(expected_returns=mu, cov_matrix=sigma, weight_bounds=(0.0, 1.0))
    ef.max_sharpe(risk_free_rate=rf)
    return ef.clean_weights()

def _returns(tickers: list[str], as_of: date, timeframe_years: int) -> pd.DataFrame:
    return log_returns(_prices(tickers, as_of, timeframe_years))


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
) -> pd.Series:
    if as_of is None:
        as_of = date.today()

    rets = _returns(tickers, as_of, timeframe_years)
    sigma = _compute_covariance(rets, method=cov_method, params=cov_params)
    prices = _prices(tickers, as_of, timeframe_years)
    mu = expected_returns.mean_historical_return(prices, frequency=252, log_returns=False)
    ef = EfficientFrontier(expected_returns=mu, cov_matrix=sigma, weight_bounds=(0.0, 1.0))
    ef.min_volatility()
    return pd.Series(ef.clean_weights()).reindex(tickers, fill_value=0.0)


def max_sharpe(
    tickers: list[str],
    as_of: date = None,
    timeframe_years: int = 3,
    rf: float = 0.0,
    cov_method: str = "shrunk",  # 'shrunk' | 'empirical' | 'oas' | 'ewma' | 'factor'
    cov_params: dict | None = None,
) -> pd.Series:
    if as_of is None:
        as_of = date.today()

    rets = _returns(tickers, as_of, timeframe_years)
    sigma = _compute_covariance(rets, method=cov_method, params=cov_params)
    prices = _prices(tickers, as_of, timeframe_years)
    mu = expected_returns.mean_historical_return(prices, frequency=252, log_returns=False)
    return pd.Series(_solve_max_sharpe(mu, sigma, rf=rf)).reindex(tickers, fill_value=0.0)


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



def build_portfolios(tickers, start, end, lookback_years=4, freq="MS", cov_method=None, cov_params=None):
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    load_start = start_ts - pd.DateOffset(years=lookback_years)
    px = pd.concat(
        [load_data(ticker, load_start.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d"))["adj close"].rename(ticker) for ticker in tickers],
        axis=1,
        join="inner",
    ).sort_index()
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
    # Ensure VT column exists in weights output for market_cap
    cols = sorted(set(tickers) | {"VT"})
    mv, ms, mc, ew = {}, {}, {}, {}

    for dt in rebalance_dates:
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

        rets = win.pct_change().dropna(how="any")
        if rets.shape[0] < 2:
            if mv:
                mv[dt] = mv[max(mv.keys())]
            if ms:
                ms[dt] = ms[max(ms.keys())]
            if mc:
                mc[dt] = mc[max(mc.keys())]
            continue

        mu = expected_returns.mean_historical_return(win, frequency=252, log_returns=False)
        sigma = _compute_covariance(rets, method=cov_method, params=cov_params)
        ef_mv = EfficientFrontier(expected_returns=mu, cov_matrix=sigma, weight_bounds=(0.0, 1.0))
        ef_mv.min_volatility()
        w_mv = ef_mv.clean_weights()
        w_ms = _solve_max_sharpe(mu, sigma, rf=0.0)

        w_mc = {t: (1.0 if t == "VT" else 0.0) for t in cols}
        mv[dt] = pd.Series({t: float(w_mv.get(t, 0.0)) for t in tickers}).reindex(cols, fill_value=0.0)
        ms[dt] = pd.Series({t: float(w_ms.get(t, 0.0)) for t in tickers}).reindex(cols, fill_value=0.0)
        mc[dt] = pd.Series(w_mc).reindex(cols, fill_value=0.0)
        ew_row = {t: (1.0 / len(tickers)) for t in tickers}
        for t in set(cols) - set(tickers):
            ew_row[t] = 0.0
        ew[dt] = pd.Series(ew_row).reindex(cols, fill_value=0.0)

    def to_daily(d: dict[pd.Timestamp, pd.Series]) -> pd.DataFrame:
        if not d:
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
    