from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

import matplotlib.pyplot as plt
from sklearn.manifold import MDS

from helpers.fetch import load_data


def cluster_select_representatives_from_csv(
    universe_csv_path: str | Path,
    n_clusters: Optional[int] = None,
    distance_metric: str = "1-abs_corr",
    linkage_method: str = "average",
    distance_threshold: Optional[float] = None,
    min_history_years: float = 20.0,
) -> Tuple[List[str], Dict[int, List[str]]]:
    """
    Cluster ETFs by return correlation and select the largest-AUM ETF per cluster.

    Inputs:
    - universe_csv_path: CSV where column 1 is ticker and column 2 is AUM in millions USD.
    - n_clusters: optional number of clusters.
    - distance_metric: '1-abs_corr' or '1-corr'.
    - linkage_method: hierarchical clustering linkage method.
    - distance_threshold: optional distance cut threshold.
    - min_history_years: minimum required history length, default 20 years. `load_data`
      transparently uses splice-extended history (see `helpers.fetch.splice_with_proxy`)
      when available, so this filter counts synthetic pre-inception history too.

    Returns:
    - selected: one representative ETF per cluster.
    - clusters: mapping from cluster label to tickers in the cluster.
    """
    # Load the ticker-to-AUM universe mapping.
    universe = pd.read_csv(universe_csv_path)
    universe = universe.iloc[:, :2].copy()
    universe.columns = ["ticker", "aum_millions"]
    universe["ticker"] = universe["ticker"].astype(str).str.strip().str.upper()
    universe["aum_millions"] = pd.to_numeric(universe["aum_millions"], errors="coerce").fillna(0.0)

    # Build a ticker-to-AUM map, clean data
    aum: Dict[str, float] = dict(zip(universe["ticker"], universe["aum_millions"]))

    end_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    start_date = (pd.Timestamp.today() - pd.DateOffset(years=min_history_years + 5)).strftime("%Y-%m-%d")
    prices_by_ticker = {
        ticker: load_data(ticker, start_date, end_date)["adj close"].rename(ticker)
        for ticker in universe["ticker"]
    }

    # Filter out ETFs with less than the required history.
    min_calendar_days = int(min_history_years * 365)
    min_observations = int(min_history_years * 252)
    kept_tickers = [
        ticker
        for ticker, series in prices_by_ticker.items()
        if len(series) >= min_observations
        and (series.index.max() - series.index.min()).days >= min_calendar_days
    ]

    if not kept_tickers:
        raise ValueError(f"No tickers have at least {min_history_years:g} years of history")

    # Return the only ticker directly if just one survives.
    if len(kept_tickers) == 1:
        return kept_tickers, {1: kept_tickers}

    # Align all price series on common dates.
    prices = pd.concat([prices_by_ticker[ticker] for ticker in kept_tickers], axis=1, join="inner")
    prices.columns = kept_tickers

    returns = prices.pct_change().dropna(how="any")
    corr = returns.corr()

    # Convert correlations into a distance matrix.
    if distance_metric == "1-corr":
        dist_matrix = 1.0 - corr.values
    elif distance_metric == "1-abs_corr":
        dist_matrix = 1.0 - np.abs(corr.values)
    else:
        raise ValueError("distance_metric must be '1-abs_corr' or '1-corr'.")

    np.fill_diagonal(dist_matrix, 0.0)
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
    figsize: Tuple[int, int] = (8, 8),
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
        corr = returns.corr().reindex(index=tickers, columns=tickers)

    # Ensure correlation matrix covers all tickers in the clusters
    corr = corr.reindex(index=tickers, columns=tickers)

    if distance_metric == "1-corr":
        dist = 1.0 - corr.values
    else:
        dist = 1.0 - np.abs(corr.values)

    np.fill_diagonal(dist, 0.0)

    if method == "mds":
        mds = MDS(n_components=2, metric="precomputed", init="random", random_state=0)
        coords = mds.fit_transform(dist)
    else:
        raise ValueError("Unsupported embedding method: %s" % method)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    cmap = plt.get_cmap("tab10")
    labels = [label_of[t] for t in tickers]
    unique_labels = sorted(set(labels))
    color_for = {lbl: cmap(i % 10) for i, lbl in enumerate(unique_labels)}

    for label_index, lbl in enumerate(unique_labels):
        member_indexes = [index for index, ticker in enumerate(tickers) if label_of[ticker] == lbl]
        member_coords = coords[member_indexes]
        color = color_for[lbl]
        ax.scatter(
            member_coords[:, 0],
            member_coords[:, 1],
            color=color,
            s=55,
            edgecolors="white",
            linewidths=0.8,
            alpha=0.9,
            label=f"Cluster {lbl} ({len(member_indexes)})",
            zorder=2,
        )

        centroid = member_coords.mean(axis=0)
        spread = member_coords.std(axis=0)
        radius = max(spread.max() * 2.5, 0.03)
        ax.add_patch(plt.Circle(centroid, radius, color=color, alpha=0.08, linewidth=0, zorder=1))

        if annotate:
            angles = np.linspace(0, 2 * np.pi, len(member_indexes), endpoint=False)
            for ticker_index, angle in zip(member_indexes, angles + label_index * 0.35):
                x, y = coords[ticker_index]
                offset = (10 * np.cos(angle), 10 * np.sin(angle))
                ax.annotate(
                    tickers[ticker_index],
                    (x, y),
                    xytext=offset,
                    textcoords="offset points",
                    fontsize=8,
                    ha="left" if offset[0] >= 0 else "right",
                    va="center",
                    arrowprops={"arrowstyle": "-", "color": "0.5", "lw": 0.5},
                    zorder=4,
                )

    ax.legend(title="Cluster (members)", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    ax.set_title("Correlation-Based ETF Clusters", pad=12)
    ax.set_xlabel("MDS dimension 1")
    ax.set_ylabel("MDS dimension 2")
    ax.grid(True, color="0.9", linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax