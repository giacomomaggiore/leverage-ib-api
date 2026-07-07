"""Download adjusted close data via yfinance, updating existing CSVs incrementally.

This module standardizes all /data/*.csv files to contain a single column named
"adj close" (adjusted close prices) indexed by date.
"""

import logging
import pandas as pd
import yfinance as yf
from pathlib import Path
import contextlib
import io

# NOTE: helpers/fetch.py lives at repo_root/helpers/fetch.py → data is at repo_root/data
DATA_DIR = Path(__file__).parent.parent / "data"
logger = logging.getLogger(__name__)


def load_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    path = DATA_DIR / f"{ticker}.csv"
    req_start, req_end = pd.Timestamp(start), pd.Timestamp(end)

    if path.exists():
        try:
            existing = pd.read_csv(path, index_col=0, parse_dates=True)
        except pd.errors.EmptyDataError:
            # Existing file is empty — fetch fresh data for the requested window
            existing = _fetch(ticker, start, end)
            if existing is not None and not existing.empty:
                existing.to_csv(path)
        
        # If after normalization the file is effectively empty, refetch for window
        if existing is None or existing.shape[0] == 0 or existing.shape[1] == 0:
            existing = _fetch(ticker, start, end)
            if existing is not None and not existing.empty:
                existing.to_csv(path)

        ex_start, ex_end = existing.index.min(), existing.index.max()

        # fetch only the gaps outside the already-stored range
        chunks = [existing]
        if req_start < ex_start:
            chunks.insert(0, _fetch(ticker, start, (ex_start - pd.Timedelta(days=1)).strftime("%Y-%m-%d")))
        if req_end > ex_end:
            chunks.append(_fetch(ticker, (ex_end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), end))

        # filter out any empty fetch results before concatenation
        chunks = [c for c in chunks if c is not None and not c.empty]
        if len(chunks) > 1:
            existing = pd.concat(chunks).sort_index()
            existing = existing[~existing.index.duplicated(keep="last")]
            if not existing.empty:
                existing.to_csv(path)
    else:
        existing = _fetch(ticker, start, end)
        if existing is not None and not existing.empty:
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
    try:
        # suppress noisy yfinance/pandas output printed to stdout/stderr
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            raw = yf.download(ticker, start=start, end=end_exclusive, auto_adjust=True, progress=False)
    except Exception as e:
        logger.warning("Failed download for %s (%s -> %s): %s", ticker, start, end, e)
        return pd.DataFrame(columns=["adj close"])

    if raw is None or raw.empty:
        logger.info("No data returned for %s (%s -> %s)", ticker, start, end)
        return pd.DataFrame(columns=["adj close"]) 

    # With auto_adjust=True, 'Close' is adjusted close
    try:
        s = raw["Close"]
    except Exception:
        # Structure not as expected
        logger.warning("Unexpected data format for %s: %r", ticker, getattr(raw, "columns", None))
        return pd.DataFrame(columns=["adj close"]) 

    if isinstance(s, pd.DataFrame):
        # yfinance may return a DataFrame for single-ticker; take the first column
        s = s.iloc[:, 0]
    s.name = "adj close"
    ser = s.to_frame()
    ser.index.name = "date"
    return ser
