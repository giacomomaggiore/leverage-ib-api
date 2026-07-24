from pathlib import Path

import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).parent.parent / "data"


def _ticker_path(ticker: str) -> Path:
    """Prefer the splice-extended history file (see `splice_with_proxy`) when one exists."""
    spliced = DATA_DIR / f"{ticker}_spliced.csv"
    return spliced if spliced.exists() else DATA_DIR / f"{ticker}.csv"


def load_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    prices = pd.read_csv(_ticker_path(ticker), index_col=0, parse_dates=True)
    return prices.loc[pd.Timestamp(start):pd.Timestamp(end)]


def common_window(tickers: list[str]) -> tuple[pd.Timestamp, pd.Timestamp]:
    windows = [pd.read_csv(_ticker_path(ticker), index_col=0, parse_dates=True).index for ticker in tickers]
    return max(index.min() for index in windows), min(index.max() for index in windows)


def download_ticker(ticker: str, start: str = "1900-01-01") -> pd.DataFrame:
    """Download full adjusted-close history for a ticker and cache it to data/<ticker>.csv."""
    close = yf.download(ticker, start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(close, pd.DataFrame):  # yfinance returns a MultiIndex even for a single ticker
        close = close.iloc[:, 0]
    prices = close.rename("adj close").to_frame()
    prices.index.name = "date"
    prices.to_csv(DATA_DIR / f"{ticker}.csv")
    return prices


def build_cash_index(rate_csv: str = "FRED_EFFR", rate_col: str = "EFFR", save_as: str = "EFFR_CASH") -> pd.DataFrame:
    """
    Compound a short-term rate (percent p.a., e.g. FRED EFFR) into a synthetic cash price
    series, actual/360 -- the same day-count convention `leverage_backtest` uses for margin
    debt. Gives a long, ETF-independent proxy for the liquidity/cash sleeve, since T-bill ETFs
    like SGOV/BIL only exist for a couple of decades but EFFR is published back to 2000 here.
    """
    rate = pd.read_csv(DATA_DIR / f"{rate_csv}.csv", index_col=0, parse_dates=True)[rate_col]
    day_counts = rate.index.to_series().diff().dt.days.fillna(1.0)
    daily_growth = 1.0 + (rate / 100.0) * day_counts / 360.0
    prices = daily_growth.cumprod().rename("adj close").to_frame()
    prices.to_csv(DATA_DIR / f"{save_as}.csv")
    return prices


def build_blended_index(weights: dict[str, float], save_as: str) -> pd.DataFrame:
    """
    Blend several proxies into one synthetic price series using fixed weights -- e.g.
    approximating a global market-cap-weighted fund (like VT) with a US + ex-US split,
    for periods before any single fund tracked that exact global blend. Weights are static,
    not the true time-varying market-cap split, so this is an approximation.
    """
    prices = pd.concat(
        [pd.read_csv(_ticker_path(t), index_col=0, parse_dates=True)["adj close"].rename(t) for t in weights],
        axis=1,
        join="inner",
    )
    returns = prices.pct_change().fillna(0)
    blended_returns = sum(returns[t] * w for t, w in weights.items())
    prices_out = (1 + blended_returns).cumprod().rename("adj close").to_frame()
    prices_out.to_csv(DATA_DIR / f"{save_as}.csv")
    return prices_out


def splice_with_proxy(ticker: str, proxy_ticker: str, ter: float = 0.0) -> pd.DataFrame:
    """
    Extend an ETF's price history backward using a longer-history proxy (index or fund).
    Before the ETF's first date, the proxy's daily returns (minus a TER drag, since the
    proxy itself carries no fee) are compounded backward and rebased so both series
    match exactly on the splice date. Saves the result to data/<ticker>_spliced.csv.

    Reads through `_ticker_path`, so splicing the same ticker against a second, even
    longer-history proxy chains onto the previous splice instead of overwriting it.
    """
    etf = pd.read_csv(_ticker_path(ticker), index_col=0, parse_dates=True)
    proxy = pd.read_csv(_ticker_path(proxy_ticker), index_col=0, parse_dates=True)

    first_date = etf.index.min()
    pre = proxy.loc[:first_date, "adj close"]
    if pre.empty:
        return etf  # proxy has no history before the ETF started, nothing to splice

    daily_drag = ter / 252
    pre_returns = pre.pct_change().fillna(0) - daily_drag
    cum_growth = (1 + pre_returns).cumprod()
    synthetic = cum_growth * (etf["adj close"].iloc[0] / cum_growth.iloc[-1])

    spliced = pd.concat([synthetic.iloc[:-1].rename("adj close").to_frame(), etf])
    spliced.to_csv(DATA_DIR / f"{ticker}_spliced.csv")
    return spliced
