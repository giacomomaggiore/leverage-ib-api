from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from helpers.fetch import load_data
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
# use pyplot's get_cmap to avoid relying on the cm module implementation


def cluster_select_representatives_from_csv(
    universe_csv_path: str | Path,
    data_dir: str | Path = "data",
    n_clusters: Optional[int] = None,
    distance_metric: str = "1-abs_corr",
    linkage_method: str = "average",
    distance_threshold: Optional[float] = None,
    min_history_years: float = 6.0,
) -> Tuple[List[str], Dict[int, List[str]]]:
    """
    Cluster ETFs by return correlation and select the largest-AUM ETF per cluster.

    Inputs:
    - universe_csv_path: CSV where column 1 is ticker and column 2 is AUM in millions USD.
    - data_dir: folder containing ticker CSV files with 'date' and 'adj close' columns.
    - n_clusters: optional number of clusters.
    - distance_metric: '1-abs_corr' or '1-corr'.
    - linkage_method: hierarchical clustering linkage method.
    - distance_threshold: optional distance cut threshold.
    - min_history_years: minimum required history length, default 6 years.

    Returns:
    - selected: one representative ETF per cluster.
    - clusters: mapping from cluster label to tickers in the cluster.
    """
    # Load the ticker-to-AUM universe mapping.
    universe = pd.read_csv(universe_csv_path)
    if universe.shape[1] < 2:
        raise ValueError("Universe CSV must have at least two columns: ticker and AUM.")

    # Standardize the first two columns and clean data
    universe = universe.iloc[:, :2].copy()
    universe.columns = ["ticker", "aum_millions"]
    universe["ticker"] = universe["ticker"].astype(str).str.strip().str.upper()
    universe["aum_millions"] = pd.to_numeric(universe["aum_millions"], errors="coerce").fillna(0.0)

    # Build a ticker-to-AUM map, clean data
    aum: Dict[str, float] = dict(zip(universe["ticker"], universe["aum_millions"]))

    # Load or update adjusted-close price histories using the shared fetch helper.
    prices_by_ticker: Dict[str, pd.Series] = {}
    end_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    
    start_date = (pd.Timestamp.today() - pd.DateOffset(years=min_history_years + 5)).strftime("%Y-%m-%d")

    for ticker in universe["ticker"]:
        try:
            # use function in fetch.py to load data from csv locally of YF
            df_prices = load_data(ticker, start_date, end_date)
        except Exception:
            continue

        if "adj close" not in df_prices.columns:
            continue

        series = df_prices["adj close"].dropna().rename(ticker)
        if len(series) > 0:
            prices_by_ticker[ticker] = series

    # Filter out ETFs with less than the required history.
    min_calendar_days = int(min_history_years * 365)
    min_observations = int(min_history_years * 252)
    kept_tickers: List[str] = []
    for ticker, series in prices_by_ticker.items():
        if len(series) < min_observations:
            continue
        if (series.index.max() - series.index.min()).days < min_calendar_days:
            continue
        
        # if everything ok: keep the ticker and append it to the kept_tickers list
        kept_tickers.append(ticker)

    # Stop early if nothing survives the history filter.
    if not kept_tickers:
        raise ValueError(f"No ETFs have at least {min_history_years} years of history.")

    # Return the only ticker directly if just one survives.
    if len(kept_tickers) == 1:
        return kept_tickers, {1: kept_tickers}

    # Align all price series on common dates.
    prices = pd.concat([prices_by_ticker[ticker] for ticker in kept_tickers], axis=1, join="inner")
    prices.columns = kept_tickers

    # Compute daily simple returns.
    returns = prices.pct_change().dropna(how="any")
    if returns.shape[0] < 2:
        raise ValueError("Not enough common return observations after alignment.")

    # Compute the ETF correlation matrix.
    corr = returns.corr().fillna(0.0)

    # Convert correlations into a distance matrix.
    if distance_metric == "1-corr":
        dist_matrix = 1.0 - corr.values
    elif distance_metric == "1-abs_corr":
        dist_matrix = 1.0 - np.abs(corr.values)
    else:
        raise ValueError("distance_metric must be '1-abs_corr' or '1-corr'.")

    # Ensure the distance matrix is valid for hierarchical clustering.
    # sets the diagonal to zero to ensure that self-distance is zero
    np.fill_diagonal(dist_matrix, 0.0)
    
    # Convert the square distance matrix to condensed form for linkage function.
    condensed_dist = squareform(dist_matrix, checks=False)

    # Run hierarchical clustering.
    tree = linkage(condensed_dist, method=linkage_method)

    # Cut the tree into clusters.
    if n_clusters is not None:
        labels = fcluster(tree, t=n_clusters, criterion="maxclust")
    elif distance_threshold is not None:
        labels = fcluster(tree, t=distance_threshold, criterion="distance")
    else:
        labels = fcluster(tree, t=max(2, int(len(kept_tickers) / 4)), criterion="maxclust")

    # Group tickers by cluster label.
    clusters: Dict[int, List[str]] = {}
    for ticker, label in zip(kept_tickers, labels):
        clusters.setdefault(int(label), []).append(ticker)

    # Select the largest-AUM ETF inside each cluster.
    selected: List[str] = []
    for members in clusters.values():
        representative = max(members, key=lambda ticker: aum.get(ticker, 0.0))
        selected.append(representative)

    return selected, clusters


def visualize_clusters(
    clusters: Dict[int, List[str]],
    corr: Optional[pd.DataFrame] = None,
    returns: Optional[pd.DataFrame] = None,
    distance_metric: str = "1-abs_corr",
    method: str = "mds",
    figsize: Tuple[int, int] = (10, 8),
    annotate: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Visualize clustering groups in 2D.

    - `clusters`: mapping label -> list of tickers
    - Provide either `corr` (DataFrame of correlations) or `returns` (DataFrame of aligned returns)
    - `distance_metric`: '1-abs_corr' or '1-corr'
    - `method`: currently only 'mds' is supported (classical MDS via sklearn)

    Returns (fig, ax).
    """
    # collect tickers in stable order
    tickers = []
    label_of = {}
    for lbl, members in sorted(clusters.items()):
        for t in members:
            label_of[t] = int(lbl)
            tickers.append(t)

    if corr is None:
        if returns is None:
            raise ValueError("Provide either 'corr' or 'returns' to compute embedding")
        corr = returns.corr().reindex(index=tickers, columns=tickers).fillna(0.0)

    # Ensure correlation matrix covers all tickers in the clusters
    corr = corr.reindex(index=tickers, columns=tickers).fillna(0.0)

    if distance_metric == "1-corr":
        dist = 1.0 - corr.values
    else:
        dist = 1.0 - np.abs(corr.values)

    np.fill_diagonal(dist, 0.0)

    if method == "mds":
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=0)
        coords = mds.fit_transform(dist)
    else:
        raise ValueError("Unsupported embedding method: %s" % method)

    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.get_cmap("tab20")
    labels = [label_of[t] for t in tickers]
    unique_labels = sorted(set(labels))
    color_for = {lbl: cmap(i % 20) for i, lbl in enumerate(unique_labels)}

    for t, (x, y) in zip(tickers, coords):
        lbl = label_of[t]
        ax.scatter(x, y, color=color_for[lbl], s=70, edgecolors="k", linewidths=0.4)
        if annotate:
            ax.text(x + 1e-6, y + 1e-6, t, fontsize=8)

    # build legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=str(lbl),
                          markerfacecolor=color_for[lbl], markersize=8, markeredgecolor='k')
               for lbl in unique_labels]
    ax.legend(handles=handles, title="cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_title("Cluster visualization")
    ax.set_xlabel("dim1")
    ax.set_ylabel("dim2")
    plt.tight_layout()
    return fig, ax