from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd
from pypfopt.exceptions import OptimizationError

from helpers.backtest import portfolio_returns
from helpers.leverage import leverage_backtest
from helpers.montecarlo import (
    _historical_returns,
    save_simulations_parquet,
    simulate_bootstrap,
    simulation_weights,
)
from helpers.portfolio import build_portfolios_from_prices
from helpers.stats import quantiles_df


@dataclass
class ResearchConfig:
    tickers: list[str] = field(default_factory=lambda: ["VT", "BND", "SGOV", "GLD", "GSG"])
    start_value: float = 100_000.0
    simulation_years: int = 5
    n_sims: int = 1000
    optimisation_lookback_years: int = 3
    simulation_history_years: int = 20
    block_size: int = 60
    leverage: float = 2.0
    leverage_rebalance_modes: list[str | None] = field(
        default_factory=lambda: ["daily", "weekly", "monthly", None]
    )
    band: float = 0.10
    hard_cap: float = 4.0
    spread: float = 0.01
    min_weight: float = 0.10
    max_weight: float = 0.40
    cov_method: str = "oas"
    cov_params: dict[str, float] = field(default_factory=lambda: {"jitter": 1e-8})
    expected_return_method: str = "mean"
    seed: int = 1

    @property
    def n_days(self) -> int:
        return 252 * self.simulation_years


def _combine_paths(paths: list[pd.Series]) -> pd.DataFrame:
    values = pd.concat(paths, axis=1)
    values.columns = pd.MultiIndex.from_tuples(values.columns, names=["sim", "strategy"])
    values.index.name = "date"
    return values


def _summary(values: pd.DataFrame, config: ResearchConfig, mode: str) -> pd.DataFrame:
    strategies = values.columns.get_level_values("strategy").unique()
    summary = pd.DataFrame(
        {
            strategy: quantiles_df(
                values.xs(strategy, level="strategy", axis=1),
                start_value=config.start_value,
            )
            for strategy in strategies
        }
    ).T
    summary.index.name = "strategy"
    assumptions = {
        "simulation_years": config.simulation_years,
        "n_days": config.n_days,
        "start_value": config.start_value,
        "leverage": 1.0 if mode == "unlevered" else config.leverage,
        "rebalance_mode": mode,
        "simulation_history_years": config.simulation_history_years,
        "optimisation_lookback_years": config.optimisation_lookback_years,
        "block_size": config.block_size,
        "n_simulations_requested": config.n_sims,
    }
    return summary.assign(**assumptions)


def run(config: ResearchConfig, output_dir: Path = Path("output")) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Simulate assets and EFFR jointly so financing conditions share the same history.
    simulation_tickers = list(dict.fromkeys(config.tickers + ["VT"]))
    simulations = simulate_bootstrap(
        simulation_tickers,
        n_days=config.n_days,
        n_sims=config.n_sims,
        simulation_history_years=config.simulation_history_years,
        block_size=config.block_size,
        seed=config.seed,
    )
    warmup_returns = _historical_returns(
        config.tickers,
        lookback_years=config.optimisation_lookback_years,
    )

    # Keep every strategy and leverage mode paired to the same simulated path.
    strategies = ["min_variance", "max_sharpe", "market_cap", "equal_weight"]
    unlevered_paths: list[pd.Series] = []
    leveraged_paths = {mode: [] for mode in config.leverage_rebalance_modes}
    weight_rows: list[pd.DataFrame] = []
    failures: list[str] = []

    for simulation_number, simulation in enumerate(simulations, start=1):
        index = pd.bdate_range("2000-01-03", periods=len(simulation))
        returns = simulation.drop(columns="EFFR").set_axis(index)
        effr = simulation["EFFR"].set_axis(index)

        try:
            weights = simulation_weights(
                returns=returns[config.tickers],
                lookback_years=config.optimisation_lookback_years,
                freq="BMS",
                cov_method=config.cov_method,
                cov_params=config.cov_params,
                warmup_returns=warmup_returns,
                risk_free_rate=effr / 100.0,
                min_weight=config.min_weight,
                max_weight=config.max_weight,
                expected_return_method=config.expected_return_method,
            )
        except (OptimizationError, ValueError) as error:
            failures.append(f"simulation {simulation_number}: {error}")
            continue

        for strategy in strategies:
            strategy_weights = weights[strategy]

            # Compute the drifting asset portfolio once, then reuse it for every leverage mode.
            strategy_returns = portfolio_returns(strategy_weights, returns)

            # Store average allocations to expose concentration and cash exposure.
            average_weights = strategy_weights.mean().rename("average_weight").to_frame()
            average_weights["simulation"] = simulation_number
            average_weights["strategy"] = strategy
            average_weights.index.name = "ticker"
            weight_rows.append(average_weights.reset_index())

            values = config.start_value * (1.0 + strategy_returns).cumprod()
            values.name = "portfolio_value"
            unlevered_paths.append(values.rename((simulation_number, strategy)))

            for mode in config.leverage_rebalance_modes:
                leveraged = leverage_backtest(
                    strategy_returns,
                    effr,
                    leverage=config.leverage,
                    start_value=config.start_value,
                    freq=mode,
                    band=config.band,
                    hard_cap=config.hard_cap,
                    spread=config.spread,
                )
                leveraged_paths[mode].append(
                    leveraged["portfolio_value"].rename((simulation_number, strategy))
                )

    if not unlevered_paths:
        raise RuntimeError(f"No simulations completed. First failure: {failures[0]}")
    if failures:
        warnings.warn(f"Skipped {len(failures)} optimizer failures. First: {failures[0]}", stacklevel=2)

    # Save path-level values before deriving summary statistics.
    unlevered = _combine_paths(unlevered_paths)
    save_simulations_parquet(unlevered, str(output_dir / "mc_values_unlevered.parquet"))
    _summary(unlevered, config, "unlevered").to_csv(output_dir / "summary_unlevered.csv")

    for mode, paths in leveraged_paths.items():
        tag = mode or "band"
        values = _combine_paths(paths)
        save_simulations_parquet(values, str(output_dir / f"mc_values_leveraged_{tag}.parquet"))
        _summary(values, config, tag).to_csv(output_dir / f"summary_leveraged_{tag}.csv")

    weight_diagnostics = pd.concat(weight_rows, ignore_index=True)
    weight_diagnostics.to_csv(output_dir / "average_weights_by_simulation.csv", index=False)
    weight_diagnostics.groupby(["strategy", "ticker"], as_index=False)["average_weight"].mean().to_csv(
        output_dir / "average_weights_summary.csv", index=False
    )

    # Test whether max-Sharpe weights depend on the expected-return estimator.
    sensitivity_prices = (1.0 + warmup_returns).cumprod()
    sensitivity_rows = []
    for return_method in ["mean", "ema", "equal"]:
        sensitivity_weights = build_portfolios_from_prices(
            prices=sensitivity_prices,
            start=sensitivity_prices.index.max(),
            end=sensitivity_prices.index.max(),
            lookback_years=config.optimisation_lookback_years,
            freq="BMS",
            cov_method=config.cov_method,
            cov_params=config.cov_params,
            min_weight=config.min_weight,
            max_weight=config.max_weight,
            expected_return_method=return_method,
        )["max_sharpe"].iloc[-1]
        for ticker, weight in sensitivity_weights.items():
            sensitivity_rows.append(
                {"expected_return_method": return_method, "ticker": ticker, "weight": weight}
            )
    pd.DataFrame(sensitivity_rows).to_csv(output_dir / "max_sharpe_sensitivity.csv", index=False)

    # Record the actual common history and all assumptions used for this run.
    history = _historical_returns(
        simulation_tickers,
        lookback_years=config.simulation_history_years,
    )
    run_config = asdict(config) | {
        "n_days": config.n_days,
        "n_simulations_completed": unlevered.columns.get_level_values("sim").nunique(),
        "optimizer_failures": failures,
        "actual_simulation_history_start": str(history.index.min().date()),
        "actual_simulation_history_end": str(history.index.max().date()),
        "actual_simulation_history_rows": len(history),
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run corrected leverage portfolio Monte Carlo research.")
    parser.add_argument("--n-sims", type=int, default=None, help="Override the configured simulation count.")
    args = parser.parse_args()
    config = ResearchConfig()
    if args.n_sims is not None:
        config.n_sims = args.n_sims
    run(config)


if __name__ == "__main__":
    main()