from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"


def load_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    prices = pd.read_csv(DATA_DIR / f"{ticker}.csv", index_col=0, parse_dates=True)
    return prices.loc[pd.Timestamp(start):pd.Timestamp(end)]


def common_window(tickers: list[str]) -> tuple[pd.Timestamp, pd.Timestamp]:
    windows = [pd.read_csv(DATA_DIR / f"{ticker}.csv", index_col=0, parse_dates=True).index for ticker in tickers]
    return max(index.min() for index in windows), min(index.max() for index in windows)
