from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.manifold import MDS

from helpers.fetch import load_data


def cluster_select_representatives_from_csv(
    universe_csv_path: str | Path,
    n_clusters: int,
    distance_metric: str = "1-abs_corr",
    linkage_method: str = "average",
    min_history_years: float = 20.0,
) -> tuple[list[str], dict[int, list[str]]]:
    """
    Cluster ETFs by return correlation and select the largest-AUM ETF per cluster.

    Inputs:
    - universe_csv_path: CSV where column 1 is ticker and column 2 is AUM in millions USD.
        - n_clusters: number of clusters.
    - distance_metric: '1-abs_corr' or '1-corr'.
    - linkage_method: hierarchical clustering linkage method.
        - min_history_years: minimum required history length. Spliced CSVs count as history.

    Returns:
    - selected: one representative ETF per cluster.
    - clusters: mapping from cluster label to tickers in the cluster.
    """
    universe = pd.read_csv(universe_csv_path)
    universe = universe.iloc[:, :2].copy()
    universe.columns = ["ticker", "aum_millions"]
    universe["ticker"] = universe["ticker"].astype(str).str.strip().str.upper()
    universe["aum_millions"] = pd.to_numeric(universe["aum_millions"], errors="coerce").fillna(0.0)

    aum = dict(zip(universe["ticker"], universe["aum_millions"]))

    end_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    start_date = (pd.Timestamp.today() - pd.DateOffset(years=min_history_years + 5)).strftime("%Y-%m-%d")
    prices_by_ticker = {
        ticker: load_data(ticker, start_date, end_date)["adj close"].rename(ticker)
        for ticker in universe["ticker"]
    }

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

    if len(kept_tickers) == 1:
        return kept_tickers, {1: kept_tickers}

    prices = pd.concat([prices_by_ticker[ticker] for ticker in kept_tickers], axis=1, join="inner")
    prices.columns = kept_tickers

    returns = prices.pct_change().dropna(how="any")
    corr = returns.corr()

    if distance_metric == "1-corr":
        dist_matrix = 1.0 - corr.values
    elif distance_metric == "1-abs_corr":
        dist_matrix = 1.0 - np.abs(corr.values)
    else:
        raise ValueError("distance_metric must be '1-abs_corr' or '1-corr'.")

    np.fill_diagonal(dist_matrix, 0.0)
    condensed_dist = squareform(dist_matrix, checks=False)

    tree = linkage(condensed_dist, method=linkage_method)
    labels = fcluster(tree, t=n_clusters, criterion="maxclust")

    clusters: dict[int, list[str]] = {}
    for ticker, label in zip(kept_tickers, labels):
        clusters.setdefault(int(label), []).append(ticker)

    selected: list[str] = []
    for members in clusters.values():
        representative = max(members, key=lambda ticker: aum.get(ticker, 0.0))
        selected.append(representative)

    return selected, clusters


def visualize_clusters(
    clusters: dict[int, list[str]],
    corr: pd.DataFrame | None = None,
    returns: pd.DataFrame | None = None,
    distance_metric: str = "1-abs_corr",
    figsize: tuple[int, int] = (8, 8),
    annotate: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Visualize clustering groups in 2D.

    - `clusters`: mapping label -> list of tickers
    - Provide either `corr` (DataFrame of correlations) or `returns` (DataFrame of aligned returns)
    - `distance_metric`: '1-abs_corr' or '1-corr'

    Returns (fig, ax).
    """
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

    corr = corr.reindex(index=tickers, columns=tickers)

    if distance_metric == "1-corr":
        dist = 1.0 - corr.values
    elif distance_metric == "1-abs_corr":
        dist = 1.0 - np.abs(corr.values)
    else:
        raise ValueError("distance_metric must be '1-abs_corr' or '1-corr'.")

    np.fill_diagonal(dist, 0.0)

    mds = MDS(n_components=2, metric="precomputed", init="random", random_state=0)
    coords = mds.fit_transform(dist)

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