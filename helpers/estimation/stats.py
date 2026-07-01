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
