Leverage Portfolio
==================

Build and backtest a leveraged long-only portfolio of US ETFs, then run it live via IB Gateway.

---

Strategy
--------
Three portfolios are compared in backtest and Monte Carlo:
- Min-variance (Ledoit-Wolf covariance shrinkage)
- Max-Sharpe
- Market-cap weighted

Target leverage: 1.7x. The best performer is used live.

Rebalancing triggers:
- Monthly (calendar)
- Threshold: rebalance early if actual leverage drifts ±10% from target

---

Pipeline
--------
1. Download 5y daily OHLCV via yfinance → data/
2. Compute log-returns and Ledoit-Wolf covariance
3. Optimize weights (min-var, max-Sharpe, market-cap)
4. Backtest: monthly rebalance loop, 2x leverage, margin interest deducted
5. Monte Carlo: block bootstrap, collect return distribution
6. Pick best strategy
7. Live: query IB positions, compute delta to target weights, place orders

---

Live trading notes
------------------
- Run manually from laptop with IB Gateway open (paper account, port 4002)
- Before placing orders: query current IB positions and compute delta
- Circuit breaker: if leverage drifts >10% from target, rebalance immediately

---

Repo structure
--------------
    data/                   downloaded CSVs (gitignored)
    helpers/
        data/               yfinance download
        estimation/         returns, covariance
        optimization/       min-variance, max-Sharpe, market-cap
        backtest/           rebalance loop, leverage tracking
        montecarlo/         block bootstrap
        ib/                 IB connector, position reconciliation, order placement
    main.ipynb              orchestration notebook
    requirements.txt

---

Setup
-----
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

For live trading: start IB Gateway (paper, port 4002) before running the notebook.
