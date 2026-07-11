import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def log_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna()


def sharpe(prices: pd.Series, rf: float = 0.0, periods: int = 252) -> float:
    r = log_returns(prices)
    return (r.mean() - rf / periods) / r.std() * np.sqrt(periods)


def covariance(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.cov()


def covariance_shrunk(returns: pd.DataFrame) -> pd.DataFrame:
    if returns is None or returns.shape[0] == 0:
        raise ValueError(
            f"No return observations provided to covariance_shrunk (shape={None if returns is None else returns.shape})."
            " Check the requested date window and source data files."
        )

    lw = LedoitWolf().fit(returns)
    return pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns)



def quantiles_df(
    df: pd.DataFrame,
    quantiles: list[float] = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99],
) -> pd.Series:
    """
    Compute quantiles of the terminal portfolio value across simulation columns.

    Expected input format:
    - index: date
    - columns: simulation paths
    - values: portfolio value at each date for each simulation

    Returns a Series keyed by quantile labels plus mean/min/max/std for the final row.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("df must be a non-empty DataFrame")

    terminal = pd.to_numeric(df.iloc[-1], errors="coerce").dropna()
    if terminal.empty:
        raise ValueError("df must contain numeric terminal simulation values")

    result = pd.Series(dtype=float)
    for q in quantiles:
        result[f"q{int(q * 100):02d}"] = float(terminal.quantile(q))

    result["mean"] = float(terminal.mean())
    result["min"] = float(terminal.min())
    result["max"] = float(terminal.max())
    result["std"] = float(terminal.std())
    return result