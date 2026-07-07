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
2. Compute covariances from returns (PyPortfolioOpt risk models)
3. Optimize weights with PyPortfolioOpt (min-vol, max-Sharpe); expected returns via mean_historical_return on prices (simple returns)
4. Backtest: monthly rebalance loop, 2x leverage, margin interest deducted
5. Monte Carlo: block bootstrap, collect return distribution
6. Pick best strategy
7. Live: query IB positions, compute delta to target weights, place orders

---

Covariance Models (Theory, Intuition, When to Use)
--------------------------------------------------
Portfolio optimizers depend critically on the return covariance matrix. This repo supports multiple estimators via the `cov_method` parameter in `min_variance` and `max_sharpe` (backed by PyPortfolioOpt):

- `shrunk` (default): Ledoit–Wolf shrinkage toward a scaled identity
- `empirical`: plain sample covariance
- `oas`: Oracle Approximating Shrinkage (sklearn)
- `ewma`: exponentially weighted covariance
- `factor`: simple PCA factor model

Each method trades bias vs variance differently. Below is a practical guide.

Empirical (sample) covariance — `cov_method='empirical'`
 - Definition:

    $$S = \frac{1}{T-1}\sum_t (r_t - \mu)(r_t - \mu)^\top$$
- Intuition: Uses only in-sample co-movements; no regularization.
- Pros: Preserves true cross-asset structure; differentiates assets strongly.
- Cons: Noisy/ill-conditioned when lookback T is short relative to number of assets N; can produce unstable/extreme weights.
- Use when: You have ample history vs asset count, accept more variability for higher fidelity to observed correlations.

Ledoit–Wolf shrinkage — `cov_method='shrunk'`
 - Definition:

    $$\hat{\Sigma} = (1-\alpha) S + \alpha \mu I \quad (\alpha\ \text{auto-chosen};\ \mu = \mathrm{tr}(S)/N)$$
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
 - Definition (conceptually): recent returns get more weight; effective covariance:

    $$\mathrm{Cov} = \sum_t w_t (r_t - \mu)(r_t - \mu)^\top$$

    with decaying weights.
- Intuition: Emphasizes the recent regime; adapts to changing vol/correlation faster than static windows.
- Pros: Captures time-variation; can reduce stale-risk during regime shifts.
- Cons: More parameter sensitivity (α or span); can underuse older data and be noisier if span is too short.
- Use when: You want responsiveness to recent behavior; start with `span=60`–`120` trading days.

Factor (PCA) model — `cov_method='factor'` with `cov_params={'n_factors': k}`
- Construction (outline):
    - Demean returns and take SVD/PCA → top k factors (scores) and loadings.
        - Rebuild

            $$\hat{\Sigma} = B \Sigma_F B^\top + D$$

            where $B$ are loadings, $\Sigma_F$ factor covariance, $D$ diagonal of specific variances.
- Intuition: Most co-movement is driven by a few latent factors; model common risk parsimoniously, reduce noise elsewhere.
- Pros: Good stability with retained structure; tunable complexity via `n_factors`.
- Cons: Choice of k matters; purely statistical (no economic labeling by default).
- Use when: N is large vs T and you want structure without heavy shrinkage; try `n_factors=2–4`.

Choosing a method (practical tips)
- Seeing equal weights with shrinkage? Try `empirical`, `oas`, or `ewma` to re-introduce structure.
- If weights look jumpy, prefer `oas`, `shrunk`, or `factor` (small `n_factors`).
- Increase lookback or rebalance on business-month-start (`BMS`) to improve estimation and reduce NaNs.
- Check constraints: tight per-asset caps push solutions toward uniform weights when assets look similar.
 - Prefer well-conditioned Σ: set `cov_method='oas'` or `'shrunk'`; or use `'factor'` with `cov_params={'n_factors': 2–4}` to retain structure while reducing noise.
 - Optional ridge (jitter): improve conditioning by adding a tiny diagonal term via `cov_params={'jitter': 1e-8}`.
 - Remove near-duplicates: avoid including aggregate ETFs (e.g., `AGG`) together with their sleeve constituents (e.g., `VGSH`, `IEF`, `VGLT`, `TLH`).

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

PyPortfolioOpt specifics
- Expected returns: `pypfopt.expected_returns.mean_historical_return(prices, frequency=252, log_returns=False)` (simple returns on adjusted-close prices, annualized).
- Covariance (returns): `pypfopt.risk_models.sample_cov`, `CovarianceShrinkage(...).ledoit_wolf()`, `CovarianceShrinkage(...).oracle_approximating()`, or `exp_cov(..., span=s)` depending on `cov_method`.
- Optimization: `EfficientFrontier.min_volatility()` and `EfficientFrontier.max_sharpe(risk_free_rate=rf)` under long-only, fully-invested constraints.
 - Extras: You can pass `cov_params` to tweak estimators; this repo supports `{'span': s}` (EWMA), `{'n_factors': k}` (factor), and `{'jitter': 1e-8}` to add a small diagonal ridge for numerical stability.

Troubleshooting equal-weight outcomes
 - Cause: With strong shrinkage and similar variances, $\hat{\Sigma} \approx cI$; under long-only + sum-to-1, min-variance → equal-weight.
 - Remedies: Switch to `empirical`/`oas`/`ewma`, increase lookback, relax caps, or use `factor` with small `n_factors`.

---

Clustering Methodology
----------------------
Before optimization, the ETF universe can be reduced by clustering highly similar ETFs and selecting one representative per cluster.

Inputs:
- Universe CSV: first column is the ticker, second column is AUM in millions of USD.
- Price files: one CSV per ticker in `data/`, named `{ticker}.csv`, with columns `date` and `adj close`.

Method:
1. Load the universe mapping from `data/etf_universe.csv`.
2. Load each ticker price history from `data/{ticker}.csv`.
3. Filter out ETFs with less than 6 years of history by default.
4. Align all remaining ETFs on common dates.
5. Compute daily simple returns.
6. Compute the return correlation matrix.
7. Convert correlation into distance using `1 - |corr|` by default.
8. Run hierarchical clustering with average linkage.
9. Select the largest-AUM ETF inside each cluster.

Rationale:
- Reduces duplicate exposures such as overlapping equity or bond ETFs.
- Improves covariance conditioning by removing near-identical assets.
- Keeps the universe investable by preferring larger, more liquid ETFs.
- Preserves broad market coverage while reducing optimizer noise.

Usage:
```python
from helpers.clustering import cluster_select_representatives_from_csv

selected_tickers, clusters = cluster_select_representatives_from_csv(
    universe_csv_path="data/etf_universe.csv",
    data_dir="data",
    n_clusters=20,
    min_history_years=6,
)

selected_tickers

---

Recent Work (what I changed)
----------------------------
- `helpers/fetch.py`
    - Improved robustness when downloading with `yfinance`: catches download errors, returns empty DataFrames on failure, and avoids overwriting existing CSVs with empty results.
    - Suppresses noisy `yfinance` stdout/stderr messages during download to keep notebook output clean.
    - Filters empty fetch chunks before concatenation when updating local CSVs.

- `helpers/clustering.py`
    - Added `visualize_clusters(clusters, corr=None, returns=None, ...)` which computes a 2D embedding of tickers and plots them colored by cluster. Uses MDS (via `sklearn.manifold.MDS`) on a distance matrix derived from correlations (default `1 - |corr|`). Returns `(fig, ax)` for further customization or saving.

- `main.ipynb`
    - Inserted an example notebook cell that runs `cluster_select_representatives_from_csv(...)`, builds aligned returns for the clustered tickers and calls `visualize_clusters(...)` to render the plot.

Why these changes?
- Make data fetching more robust and idempotent so repeated notebook runs don't clobber existing data or print confusing warnings from upstream libraries.
- Provide a lightweight visual tool to inspect cluster structure before/after representative selection so you can study which ETFs were grouped together and why.

How to reproduce the visualization quickly
---------------------------------------
In a notebook cell (this is the example I added to `main.ipynb`):

```python
from helpers.clustering import cluster_select_representatives_from_csv, visualize_clusters
from helpers.fetch import load_data
import pandas as pd

ETF_UNIVERSE_CSV_PATH = "data/etf_universe.csv"
selected, clusters = cluster_select_representatives_from_csv(ETF_UNIVERSE_CSV_PATH)

# build aligned price matrix for tickers in the clusters
tickers = sorted([t for members in clusters.values() for t in members])
start_date = '2012-01-01'
end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
prices = pd.concat([load_data(t, start_date, end_date)['adj close'].rename(t) for t in tickers], axis=1, join='inner')
returns = prices.pct_change().dropna()

# show the cluster plot
fig, ax = visualize_clusters(clusters, returns=returns)
fig
```

Technical: Multi-Dimensional Scaling (MDS)
---------------------------------------
Purpose
- MDS is a family of techniques that embed objects described by pairwise dissimilarities (distances) into a low-dimensional Euclidean space so that the pairwise Euclidean distances in the embedding approximate the original dissimilarities.

Two common variants
- Classical MDS (metric / Torgerson scaling): computes an exact embedding from a Euclidean distance matrix using linear algebra (double-centering). If the input dissimilarities are true Euclidean distances, classical MDS recovers coordinates (up to rotation/reflection and translation).
    - Algorithm (brief): given squared distance matrix D^{(2)}, compute B = -0.5 * J D^{(2)} J where J = I - (1/n)11^T. Eigen-decompose B = V Λ V^T and take embedding X = V_k Λ_k^{1/2} using the top k positive eigenvalues.

- Metric MDS (stress-minimizing, as implemented in `sklearn.manifold.MDS`): iteratively optimizes a low-dimensional configuration to minimize a stress function (discrepancy between original dissimilarities and embedding distances). It is more flexible (works with non-Euclidean dissimilarities) but is iterative and can be slower.

Practical notes for this project
- We convert correlation → distance using either `1 - corr` or `1 - |corr|`. This produces a symmetric dissimilarity matrix with zeros on the diagonal and values in [0,2]. Treating correlation-based distances with MDS gives a geometric view of similarity: nearby points have stronger (absolute) correlations.
- The implementation uses `sklearn.manifold.MDS` with `dissimilarity='precomputed'`. That performs metric MDS that minimizes stress. By default it initializes randomly; consider `init='classical'` or setting `random_state` for reproducible layouts.
- Complexity: constructing the full distance matrix is O(n^2) memory/time. MDS itself scales poorly beyond a few hundred points (stress minimization is iterative). For very large universes prefer dimensionality reduction by PCA on features or classical MDS on a landmark subset.
- Interpretation: MDS coordinates are useful for visualization and exploratory analysis (cluster coherence, outliers) but should not be treated as factor returns or direct inputs to optimization without careful validation.

Tuning and alternatives
- If plotting many tickers produces clutter, use `annotate=False` in `visualize_clusters` and hover-enabled plotting (e.g., Plotly) to inspect points interactively.
- Alternatives:
    - PCA on returns or on the double-centered Gram matrix (fast, linear algebra based).
    - t-SNE or UMAP for non-linear neighborhood-preserving embeddings (better local clustering but more parameters and interpretation challenges).

References
- Borg, I., & Groenen, P. J. F. (2005). Modern Multidimensional Scaling: Theory and Applications.
- sklearn.manifold.MDS documentation: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html

---
If you want, I can also:
- add PCA / t-SNE options to `visualize_clusters`,
- add an option to save the plot to file from the helper, or
- add an interactive Plotly-based visualization cell to the notebook.
