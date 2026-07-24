import unittest

import numpy as np
import pandas as pd

from helpers.backtest import portfolio_returns
from helpers.leverage import leverage_backtest
from helpers.montecarlo import _terminal_quantile_paths, plot_monthly_leverage_comparison
from helpers.portfolio import build_portfolios_from_prices
from helpers.stats import covariance, quantiles_df


class CorrectnessTests(unittest.TestCase):
    def test_monthly_comparison_uses_requested_runs_and_terminal_paths(self):
        unlevered = pd.DataFrame(
            {
                (1, "market_cap"): [100.0, 90.0],
                (2, "market_cap"): [100.0, 110.0],
                (3, "market_cap"): [100.0, 130.0],
            }
        )
        leveraged = pd.DataFrame(
            {
                (1, "equal_weight"): [100.0, 80.0],
                (2, "equal_weight"): [105.0, 120.0],
                (3, "equal_weight"): [110.0, 140.0],
                (1, "max_sharpe"): [100.0, 100.0],
                (2, "max_sharpe"): [115.0, 130.0],
                (3, "max_sharpe"): [120.0, 160.0],
            }
        )
        unlevered.columns.names = ["sim", "strategy"]
        leveraged.columns.names = ["sim", "strategy"]

        ax = plot_monthly_leverage_comparison(unlevered, leveraged)

        self.assertEqual(
            [line.get_label() for line in ax.lines],
            ["VT unleveraged", "Equal weight 2x monthly", "Max Sharpe 2x monthly"],
        )
        np.testing.assert_allclose(ax.lines[0].get_ydata(), [100.0, 110.0])
        np.testing.assert_allclose(ax.lines[1].get_ydata(), [105.0, 120.0])
        np.testing.assert_allclose(ax.lines[2].get_ydata(), [115.0, 130.0])

    def test_terminal_quantile_selects_a_complete_simulation_path(self):
        paths = pd.DataFrame(
            {
                (1, "strategy"): [1000.0, 100.0],
                (2, "strategy"): [0.0, 200.0],
                (3, "strategy"): [500.0, 300.0],
            }
        )
        paths.columns.names = ["sim", "strategy"]

        selected = _terminal_quantile_paths(paths, 0.50)

        self.assertEqual(selected["strategy"].tolist(), [0.0, 200.0])

    def test_weights_are_applied_after_decision_date(self):
        weights = pd.DataFrame({"A": [1.0]}, index=pd.to_datetime(["2020-01-01"]))
        returns = pd.DataFrame(
            {"A": [0.10, 0.20]},
            index=pd.to_datetime(["2020-01-02", "2020-01-03"]),
        )
        np.testing.assert_allclose(portfolio_returns(weights, returns), [0.10, 0.20])

    def test_holdings_drift_between_rebalances(self):
        weights = pd.DataFrame(
            {"A": [0.50], "B": [0.50]},
            index=pd.to_datetime(["2020-01-01"]),
        )
        returns = pd.DataFrame(
            {"A": [0.10, 0.10], "B": [0.0, 0.0]},
            index=pd.to_datetime(["2020-01-02", "2020-01-03"]),
        )

        result = portfolio_returns(weights, returns)

        np.testing.assert_allclose(result, [0.05, 5.5 / 105.0])

    def test_repeated_target_rebalances_before_next_return(self):
        weights = pd.DataFrame(
            {"A": [0.50, 0.50], "B": [0.50, 0.50]},
            index=pd.to_datetime(["2020-01-01", "2020-01-02"]),
        )
        returns = pd.DataFrame(
            {"A": [0.10, 0.10], "B": [0.0, 0.0]},
            index=pd.to_datetime(["2020-01-02", "2020-01-03"]),
        )

        result = portfolio_returns(weights, returns)

        np.testing.assert_allclose(result, [0.05, 0.05])

    def test_cagr_is_pathwise_and_ruin_is_explicit(self):
        paths = pd.DataFrame(
            {
                "double": np.linspace(100.0, 200.0, 253)[1:],
                "ruin": np.r_[np.full(251, 100.0), 0.0],
            }
        )
        summary = quantiles_df(paths, start_value=100.0)
        self.assertAlmostEqual(summary["cagr_q90"], 1.0)
        self.assertEqual(summary["n_ruined_or_non_positive"], 1)
        self.assertEqual(summary["ruin_rate"], 0.5)
        self.assertEqual(summary["max_drawdown_min"], -1.0)

    def test_leverage_liquidation_stays_at_zero(self):
        weights = pd.DataFrame({"A": [1.0]}, index=pd.to_datetime(["2020-01-01"]))
        returns = pd.DataFrame(
            {"A": [-0.60, 0.50]},
            index=pd.to_datetime(["2020-01-02", "2020-01-03"]),
        )
        effr = pd.Series([0.0, 0.0], index=returns.index)
        result = leverage_backtest(weights, 2.0, returns=returns, effr=effr, freq="daily")
        self.assertEqual(result["portfolio_value"].tolist(), [0.0, 0.0])
        self.assertEqual(result["ruined"].tolist(), [True, True])

    def test_leverage_reuses_the_same_drifting_portfolio_returns(self):
        weights = pd.DataFrame(
            {"A": [0.50], "B": [0.50]},
            index=pd.to_datetime(["2020-01-01"]),
        )
        returns = pd.DataFrame(
            {"A": [0.10, 0.10], "B": [0.0, 0.0]},
            index=pd.to_datetime(["2020-01-02", "2020-01-03"]),
        )
        effr = pd.Series([0.0, 0.0], index=returns.index)
        drifting_returns = portfolio_returns(weights, returns)

        direct = leverage_backtest(weights, 2.0, returns=returns, effr=effr, freq="daily")
        reused = leverage_backtest(
            weights,
            2.0,
            returns=returns,
            effr=effr,
            freq="daily",
            portfolio_return_series=drifting_returns,
        )

        pd.testing.assert_series_equal(direct["portfolio_value"], reused["portfolio_value"])

    def test_public_covariance_is_annualized(self):
        returns = pd.DataFrame({"A": [-0.01, 0.0, 0.01]})
        self.assertAlmostEqual(covariance(returns).iloc[0, 0], returns.var().iloc[0] * 252)

    def test_inferred_start_does_not_add_a_zero_return(self):
        values = pd.DataFrame({"path": [100.0, 110.0, 100.0]})
        expected = values.pct_change().std().iloc[0] * np.sqrt(252)
        self.assertEqual(quantiles_df(values)["ann_std_mean"], round(expected, 3))

    def test_market_cap_is_always_vt(self):
        index = pd.bdate_range("2020-01-01", periods=300)
        returns = pd.DataFrame(
            np.random.default_rng(1).normal(0.0002, 0.01, (300, 5)),
            index=index,
            columns=["VT", "BND", "SGOV", "GLD", "GSG"],
        )
        prices = 100.0 * (1.0 + returns).cumprod()
        portfolios = build_portfolios_from_prices(
            prices,
            start=index[-1],
            end=index[-1],
            lookback_years=1,
        )
        market_cap = portfolios["market_cap"].iloc[-1]
        self.assertEqual(market_cap["VT"], 1.0)
        self.assertEqual(market_cap.drop("VT").sum(), 0.0)


if __name__ == "__main__":
    unittest.main()