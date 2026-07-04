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

Covariance Models (Theory, Intuition, When to Use)
--------------------------------------------------
Portfolio optimizers depend critically on the return covariance matrix. This repo supports multiple estimators via the `cov_method` parameter in `min_variance` and `max_sharpe`:

- `shrunk` (default): Ledoit–Wolf shrinkage toward a scaled identity
- `empirical`: plain sample covariance
- `oas`: Oracle Approximating Shrinkage (sklearn)
- `ewma`: exponentially weighted covariance
- `factor`: simple PCA factor model

Each method trades bias vs variance differently. Below is a practical guide.

Empirical (sample) covariance — `cov_method='empirical'`
- Definition: S = (1/(T-1)) Σ_t (r_t − μ)(r_t − μ)^T
- Intuition: Uses only in-sample co-movements; no regularization.
- Pros: Preserves true cross-asset structure; differentiates assets strongly.
- Cons: Noisy/ill-conditioned when lookback T is short relative to number of assets N; can produce unstable/extreme weights.
- Use when: You have ample history vs asset count, accept more variability for higher fidelity to observed correlations.

Ledoit–Wolf shrinkage — `cov_method='shrunk'`
- Definition: Σ̂ = (1 − α) S + α μ I (α auto-chosen; μ often trace(S)/N)
- Intuition: Pulls estimates toward a spherical target to reduce sampling noise and improve conditioning.
- Pros: Stabler optimizer, fewer pathologies from noisy S; no hyperparameters.
- Cons: Can “wash out” correlation structure; when off-diagonals are small and variances similar, min-variance tends toward equal-weight under long-only constraints.
- Use when: Moderate history, many assets, need stability with minimal tuning.

OAS (Oracle Approximating Shrinkage) — `cov_method='oas'`
- Definition: Like LW but with an analytically derived shrinkage intensity optimal for Gaussian data.
- Intuition: Often shrinks slightly differently than LW; can be milder in practice.
- Pros: Strong conditioning with competitive bias/variance trade-off; no knobs to tune.
- Cons: Same structural limitation as LW if assets look similar.
- Use when: You want a robust, parameter-free shrinker; try this if LW produces overly uniform weights.

EWMA (Exponentially Weighted) — `cov_method='ewma'` with `cov_params={'alpha': a}` or `{'span': s}`
- Definition (conceptually): recent returns get more weight; effective Cov = Σ_t w_t (r_t − μ)(r_t − μ)^T with decaying weights.
- Intuition: Emphasizes the recent regime; adapts to changing vol/correlation faster than static windows.
- Pros: Captures time-variation; can reduce stale-risk during regime shifts.
- Cons: More parameter sensitivity (α or span); can underuse older data and be noisier if span is too short.
- Use when: You want responsiveness to recent behavior; start with `span=60`–`120` trading days.

Factor (PCA) model — `cov_method='factor'` with `cov_params={'n_factors': k}`
- Construction (outline):
    - Demean returns and take SVD/PCA → top k factors (scores) and loadings.
    - Rebuild Σ̂ = B Σ_F Bᵀ + D, where B are loadings, Σ_F factor covariance, D diagonal of specific variances.
- Intuition: Most co-movement is driven by a few latent factors; model common risk parsimoniously, reduce noise elsewhere.
- Pros: Good stability with retained structure; tunable complexity via `n_factors`.
- Cons: Choice of k matters; purely statistical (no economic labeling by default).
- Use when: N is large vs T and you want structure without heavy shrinkage; try `n_factors=2–4`.

Choosing a method (practical tips)
- Seeing equal weights with shrinkage? Try `empirical`, `oas`, or `ewma` to re-introduce structure.
- If weights look jumpy, prefer `oas`, `shrunk`, or `factor` (small `n_factors`).
- Increase lookback or rebalance on business-month-start (`BMS`) to improve estimation and reduce NaNs.
- Check constraints: tight per-asset caps push solutions toward uniform weights when assets look similar.

Code usage
```
# Min-variance with different covariances
min_variance(tickers, as_of=dt, timeframe_years=3, cov_method='empirical')
min_variance(tickers, as_of=dt, cov_method='oas')
min_variance(tickers, as_of=dt, cov_method='ewma',   cov_params={'span': 60})
min_variance(tickers, as_of=dt, cov_method='factor', cov_params={'n_factors': 3})

# Max-Sharpe mirrors the same interface
max_sharpe(tickers, as_of=dt, cov_method='shrunk')
```

Troubleshooting equal-weight outcomes
- Cause: With strong shrinkage and similar variances, Σ̂ ≈ c·I; under long-only + sum-to-1, min-variance → equal-weight.
- Remedies: Switch to `empirical`/`oas`/`ewma`, increase lookback, relax caps, or use `factor` with small `n_factors`.

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
