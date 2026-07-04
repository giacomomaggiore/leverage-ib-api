"""Download OHLCV data via yfinance, updating existing CSVs incrementally."""

import pandas as pd
import yfinance as yf
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "data"


def load_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    path = DATA_DIR / f"{ticker}.csv"
    req_start, req_end = pd.Timestamp(start), pd.Timestamp(end)

    if path.exists():
        existing = pd.read_csv(path, index_col=0, parse_dates=True)
        ex_start, ex_end = existing.index.min(), existing.index.max()

        # fetch only the gaps outside the already-stored range
        chunks = [existing]
        if req_start < ex_start:
            chunks.insert(0, _fetch(ticker, start, (ex_start - pd.Timedelta(days=1)).strftime("%Y-%m-%d")))
        if req_end > ex_end:
            chunks.append(_fetch(ticker, (ex_end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), end))

        if len(chunks) > 1:
            existing = pd.concat(chunks).sort_index()
            existing.to_csv(path)
    else:
        existing = _fetch(ticker, start, end)
        existing.to_csv(path)

    return existing.loc[req_start:req_end]


def common_window(tickers: list[str]) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return (start, end) of the largest date range covered by all tickers in /data/."""
    starts, ends = [], []
    for t in tickers:
        df = pd.read_csv(DATA_DIR / f"{t}.csv", index_col=0, parse_dates=True)
        starts.append(df.index.min())
        ends.append(df.index.max())
    return max(starts), min(ends)


def _fetch(ticker: str, start: str, end: str) -> pd.DataFrame:
    end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(ticker, start=start, end=end_exclusive, auto_adjust=True, progress=False)
    df = raw[["Open", "High", "Low", "Close", "Volume"]]
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index.name = "date"
    return df
