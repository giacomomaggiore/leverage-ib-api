from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"


def _ticker_path(ticker: str) -> Path:
    """Prefer the committed splice-extended history when available."""
    spliced = DATA_DIR / f"{ticker}_spliced.csv"
    return spliced if spliced.exists() else DATA_DIR / f"{ticker}.csv"


def load_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    prices = pd.read_csv(_ticker_path(ticker), index_col=0, parse_dates=True)
    return prices.loc[pd.Timestamp(start):pd.Timestamp(end)]
