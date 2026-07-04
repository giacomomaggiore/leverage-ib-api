"""Download adjusted close data via yfinance, updating existing CSVs incrementally.

This module standardizes all /data/*.csv files to contain a single column named
"adj close" (adjusted close prices) indexed by date.
"""

import pandas as pd
import yfinance as yf
from pathlib import Path

# NOTE: helpers/fetch.py lives at repo_root/helpers/fetch.py → data is at repo_root/data
DATA_DIR = Path(__file__).parent.parent / "data"


def load_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    path = DATA_DIR / f"{ticker}.csv"
    req_start, req_end = pd.Timestamp(start), pd.Timestamp(end)

    if path.exists():
        try:
            existing = pd.read_csv(path, index_col=0, parse_dates=True)
        except pd.errors.EmptyDataError:
            # Existing file is empty — fetch fresh data for the requested window
            existing = _fetch(ticker, start, end)
            existing.to_csv(path)
        else:
            existing = _ensure_adj_close_only(existing, inplace_path=path)

        # If after normalization the file is effectively empty, refetch for window
        if existing is None or existing.shape[0] == 0 or existing.shape[1] == 0:
            existing = _fetch(ticker, start, end)
            existing.to_csv(path)

        ex_start, ex_end = existing.index.min(), existing.index.max()

        # fetch only the gaps outside the already-stored range
        chunks = [existing]
        if req_start < ex_start:
            chunks.insert(0, _fetch(ticker, start, (ex_start - pd.Timedelta(days=1)).strftime("%Y-%m-%d")))
        if req_end > ex_end:
            chunks.append(_fetch(ticker, (ex_end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), end))

        if len(chunks) > 1:
            existing = pd.concat(chunks).sort_index()
            existing = existing[~existing.index.duplicated(keep="last")]
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
    # With auto_adjust=True, 'Close' is adjusted close
    s = raw["Close"]
    if isinstance(s, pd.DataFrame):
        # yfinance may return a DataFrame for single-ticker; take the first column
        s = s.iloc[:, 0]
    s.name = "adj close"
    ser = s.to_frame()
    ser.index.name = "date"
    return ser


def _ensure_adj_close_only(df: pd.DataFrame, inplace_path: Path | None = None) -> pd.DataFrame:
    """Ensure a DataFrame has a single 'adj close' column; optionally persist changes.

    Accepted input columns (priority order): 'Adj Close', 'adj close', 'adj_close',
    'close', 'Close'. If none found, returns the input as-is.
    """
    col = None
    cols_lower = {c.lower(): c for c in df.columns}

    for cand in ["adj close", "adj_close", "adjclose", "adj. close", "adj. close.", "adjcls", "adjclose*", "adjclose ", "adj close ", "adj. close ", "adjclose-adjusted", "adj_close_price", "adj_close_px", "adj_closevalue", "adjcloseprice", "adjclose_px", "adj close price", "adj close px", "adj price", "adj cls", "adjcls price", "adjcls px", "adj","adj.", "adj. cls", "adj-cls", "adjClose", "adjclose"]:
        if cand in cols_lower:
            col = cols_lower[cand]
            break
    if col is None:
        for cand in ["Adj Close", "Close", "close"]:
            if cand in df.columns:
                col = cand
                break

    if col is None:
        return df

    out = df[[col]].rename(columns={col: "adj close"})
    if inplace_path is not None:
        out.to_csv(inplace_path)
    return out
