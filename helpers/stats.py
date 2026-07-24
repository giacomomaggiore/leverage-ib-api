import numpy as np
import pandas as pd

def quantiles_df(
    df: pd.DataFrame,
    quantiles: list[float] | None = None,
    start_value: float | None = None,
    periods_per_year: int = 252,
) -> pd.Series:
    """Summarize portfolio-value paths with pathwise risk and ruin diagnostics."""
    if df.empty or df.shape[1] == 0:
        raise ValueError("df must contain at least one simulation path")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")

    values = df.astype(float)
    initial = pd.Series(start_value, index=values.columns) if start_value is not None else values.iloc[0]
    n_periods = len(values) if start_value is not None else len(values) - 1
    if n_periods <= 0:
        raise ValueError("At least one return observation is required")

    terminal = values.iloc[-1]
    ruined = values.le(0).any(axis=0) | terminal.le(0)
    years = n_periods / periods_per_year

    cagr = pd.Series(np.nan, index=values.columns, dtype=float)
    survivors = ~ruined & initial.gt(0)
    cagr.loc[survivors] = (terminal.loc[survivors] / initial.loc[survivors]) ** (1.0 / years) - 1.0

    path_with_initial = (
        pd.concat([initial.to_frame().T, values], ignore_index=True)
        if start_value is not None
        else values.reset_index(drop=True)
    )
    drawdowns = path_with_initial.div(path_with_initial.cummax()).sub(1.0)
    max_drawdown = drawdowns.min(axis=0)
    max_drawdown.loc[ruined] = -1.0

    path_returns = path_with_initial.pct_change().replace([np.inf, -np.inf], np.nan)
    ann_std = path_returns.std(ddof=1) * np.sqrt(periods_per_year)
    ann_std.loc[ruined] = np.nan

    result = pd.Series(dtype=float)
    terminal_quantiles = quantiles or [0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99]
    for q in terminal_quantiles:
        result[f"terminal_q{int(q * 100):02d}"] = terminal.quantile(q)
    result["terminal_mean"] = terminal.mean()
    result["terminal_min"] = terminal.min()
    result["terminal_max"] = terminal.max()
    result["terminal_survival_q50"] = terminal.loc[survivors].quantile(0.50)
    result["terminal_survival_mean"] = terminal.loc[survivors].mean()

    for q in [0.01, 0.05, 0.10, 0.50, 0.90]:
        result[f"cagr_q{int(q * 100):02d}"] = cagr.quantile(q)
    result["cagr_mean"] = cagr.mean()

    for q in [0.01, 0.05, 0.10, 0.50]:
        result[f"max_drawdown_q{int(q * 100):02d}"] = max_drawdown.quantile(q)
    result["max_drawdown_mean"] = max_drawdown.mean()
    result["max_drawdown_min"] = max_drawdown.min()

    result["ann_std_q50"] = ann_std.quantile(0.50)
    result["ann_std_mean"] = ann_std.mean()
    result["ann_std_q90"] = ann_std.quantile(0.90)
    result["n_paths"] = len(values.columns)
    result["n_ruined_or_non_positive"] = ruined.sum()
    result["ruin_rate"] = ruined.mean()
    result["min_value_observed"] = values.min().min()

    # Keep monetary quantiles readable while retaining precision for risk metrics.
    result = result.round(3)
    terminal_quantile_keys = [key for key in result.index if key.startswith("terminal_q")]
    result.loc[terminal_quantile_keys] = result.loc[terminal_quantile_keys].round(0)
    return result