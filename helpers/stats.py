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
    terminal = df.iloc[-1]

    result = pd.Series(dtype=float)
    for q in quantiles:
        # round quantiles to nearest integer (avoid truncation)
        qval = float(terminal.quantile(q))
        result[f"q{int(q * 100):02d}"] = int(round(qval))

    # add CAGR and max drawdown metrics
    result["cagr"] = round(float((terminal.iloc[-1] / terminal.iloc[0]) ** (1 / (len(terminal) / 252)) - 1), 4)
    result["max_drawdown"] = round(float((df / df.cummax()).min().min() - 1), 4)


    result["mean"] = int(round(terminal.mean()))
    result["min"] = int(round(terminal.min()))
    result["max"] = int(round(terminal.max()))

    # Keep terminal distribution std (not a volatility measure)
    result["terminal_std"] = round(float(terminal.std()),3)

    returns = df.pct_change().dropna(how="all")
    per_sim_annualized = returns.std(ddof=1) * np.sqrt(252)
    result["ann_std_median"] = round(float(per_sim_annualized.median()),3)
    result["ann_std_mean"] = round(float(per_sim_annualized.mean()),3)
    return result