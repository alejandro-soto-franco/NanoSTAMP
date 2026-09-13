"""Figure 1f/1g: SM-102 LNP-treated spleen spatial neighbourhoods.

Ported from ``Code/Figure_1f_1g_Spleen_Neighborhood_Analysis.ipynb``.
Restricted to ``reg001`` (SM-102 LNP-treated spleen). For each cell, counts
cell types in its ``k``-nearest-cell window (including the index cell,
matching the source notebook), clusters those windows into
``n_neighborhoods`` neighbourhoods with ``MiniBatchKMeans``, and names each
neighbourhood by its dominant cell type (collapsing same-named clusters by
cell-count-weighted average).
"""

from __future__ import annotations

from collections.abc import Hashable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

#: Nearest-cell window size (including the index cell) for Figure 1f/1g.
K_NEIGHBORS = 10
#: Number of MiniBatchKMeans neighbourhoods for Figure 1f/1g.
N_NEIGHBORHOODS = 8


def load_neighborhood_cells(data_root: Path) -> pd.DataFrame:
    """Load the frozen ``reg001`` cell table for Figure 1f/1g.

    Parameters
    ----------
    data_root
        ``Figure_1f_1g_Spleen_Neighborhoods`` directory (containing
        ``Precomputed_Analysis_Input/``).

    Returns
    -------
    One row per cell, with ``x``, ``y`` (float), ``lnp_positive``/
    ``lnp_negative`` (bool), ``Cell Type`` and ``unique_region`` columns
    added.

    Raises
    ------
    ValueError
        If any row is outside ``reg001``, or ``(region, cell)`` is not unique.
    """
    cells = pd.read_csv(
        Path(data_root) / "Precomputed_Analysis_Input" / "reg001_neighborhood_cells.csv.gz",
        dtype={"region": str, "cell_type": str},
    )
    cells["cell"] = pd.to_numeric(cells["cell"], errors="raise").astype("int64")
    cells["x"] = pd.to_numeric(cells["x"], errors="raise")
    cells["y"] = pd.to_numeric(cells["y"], errors="raise")
    cells["lnp_positive"] = cells["lnp_positive"].astype(bool)
    cells["lnp_negative"] = ~cells["lnp_positive"]
    cells["Cell Type"] = cells["cell_type"].astype(str)
    cells["unique_region"] = cells["region"]

    if not cells["region"].eq("reg001").all():
        raise ValueError("Figure 1f/1g expects reg001 treated-spleen cells only")
    if cells.duplicated(["region", "cell"]).any():
        raise ValueError("Duplicate region/cell identifiers")
    return cells


def _k_nearest_window(
    tissue: pd.DataFrame, indices: np.ndarray, n_neighbors: int, x_col: str, y_col: str
) -> np.ndarray:
    """Return, for each row in ``indices``, its ``n_neighbors`` nearest row labels (by distance) within ``tissue``."""
    to_fit = tissue.loc[indices, [x_col, y_col]].to_numpy()
    fit = NearestNeighbors(n_neighbors=n_neighbors).fit(tissue[[x_col, y_col]].to_numpy())
    distances, neighbor_indices = fit.kneighbors(to_fit)
    order = distances.argsort(axis=1)
    offsets = np.arange(neighbor_indices.shape[0]) * neighbor_indices.shape[1]
    sorted_indices = neighbor_indices.ravel()[order + offsets[:, None]]
    neighbors = tissue.index.to_numpy()[sorted_indices]
    return neighbors.astype(np.int32)


def build_region_windows(
    cells_df: pd.DataFrame,
    cluster_col: str = "Cell Type",
    k: int = K_NEIGHBORS,
    x_col: str = "x",
    y_col: str = "y",
    region_col: str = "unique_region",
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """Count cell types in each cell's ``k``-nearest-cell window, per region.

    Parameters
    ----------
    cells_df
        One row per cell with ``x``, ``y``, ``region_col`` and
        ``cluster_col`` columns.
    cluster_col
        Column naming each cell's type.
    k
        Window size (including the index cell).
    x_col, y_col
        Centroid coordinate columns.
    region_col
        Column partitioning cells into independent spatial regions; windows
        never cross a region boundary.

    Returns
    -------
    ``(cells_df, cell_types, windows)``: the input frame reset to a
    contiguous index with one-hot cell-type dummy columns appended, the list
    of cell-type names, and a same-indexed frame of per-cell window counts.
    """
    cells_df = cells_df.reset_index(drop=True).copy()
    cells_df[cluster_col] = cells_df[cluster_col].astype(str).str.strip()
    dummies = pd.get_dummies(cells_df[cluster_col], dtype=np.int8)
    metadata = cells_df[
        [x_col, y_col, region_col, cluster_col, "lnp_positive", "lnp_negative"]
    ].copy()
    cells_df = pd.concat([metadata, dummies], axis=1)
    cell_types = dummies.columns.to_list()
    values = dummies.to_numpy(dtype=np.float32)

    tissue_group = cells_df.groupby(region_col)
    window_values = np.zeros((len(cells_df), len(cell_types)), dtype=np.float32)
    for region in cells_df[region_col].astype(str).unique():
        tissue = tissue_group.get_group(region)
        neighbors = _k_nearest_window(tissue, tissue.index.to_numpy(), k, x_col, y_col)
        index_cells = neighbors[:, 0]
        window_values[index_cells] = values[neighbors[:, :k]].sum(axis=1)

    windows = pd.DataFrame(window_values, columns=cell_types, index=cells_df.index)
    return cells_df, cell_types, windows


def cluster_neighborhoods(
    cells_df: pd.DataFrame,
    windows: pd.DataFrame,
    cell_types: list[str],
    n_neighborhoods: int = N_NEIGHBORHOODS,
    random_state: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Cluster per-cell neighbourhood windows with ``MiniBatchKMeans``.

    Parameters
    ----------
    cells_df
        Output of :func:`build_region_windows`.
    windows
        Per-cell window cell-type counts, same index as ``cells_df``.
    cell_types
        Column order of ``windows``.
    n_neighborhoods
        Number of clusters.
    random_state
        ``MiniBatchKMeans`` random state (``n_init=10`` is fixed to match the
        source notebook).

    Returns
    -------
    ``(clustered, percent, fold_change, summary)``:
    ``clustered`` is ``cells_df`` with a ``neighborhood`` label column;
    ``percent`` is each cluster centroid normalised to a composition
    percentage; ``fold_change`` is the log2 fold-change of each cluster
    centroid (plus the pooled window mean, to avoid a zero denominator)
    relative to the pooled per-cell-type window mean; ``summary`` reports
    cluster sizes, sorted largest first.
    """
    model = MiniBatchKMeans(n_clusters=n_neighborhoods, random_state=random_state, n_init=10)
    labels = model.fit_predict(windows[cell_types].to_numpy())
    clustered = cells_df.copy()
    clustered["neighborhood"] = labels

    centroids = pd.DataFrame(
        model.cluster_centers_,
        columns=cell_types,
        index=pd.Index(range(n_neighborhoods), name="neighborhood"),
    )
    percent = centroids.div(centroids.sum(axis=1), axis=0).fillna(0) * 100
    baseline = windows[cell_types].to_numpy().mean(axis=0)
    fold_change = np.log2(
        (centroids.to_numpy() + baseline)
        / (centroids.to_numpy() + baseline).sum(axis=1, keepdims=True)
        / baseline
    )
    fold_change = pd.DataFrame(fold_change, index=centroids.index, columns=cell_types)
    summary = (
        clustered.groupby("neighborhood", observed=False)
        .agg(n_cells=("neighborhood", "size"))
        .sort_values("n_cells", ascending=False)
    )
    return clustered, percent, fold_change, summary


def collapse_neighborhood_rows(
    table: pd.DataFrame, label_map: dict, weights: pd.Series
) -> pd.DataFrame:
    """Collapse rows sharing a mapped label into their cell-count-weighted average.

    Parameters
    ----------
    table
        Per-neighbourhood-index rows to collapse (e.g. composition percent or
        log2 fold-change).
    label_map
        Neighbourhood index -> collapsed label (e.g. ``"B enriched"``).
    weights
        Neighbourhood index -> cell count, used as the averaging weight.

    Returns
    -------
    One row per distinct label, in first-appearance order.
    """
    labeled = table.copy()
    labeled.index = pd.Index(
        [label_map.get(index, index) for index in labeled.index], name=table.index.name
    )
    weights = pd.Series(weights).reindex(table.index).astype(float).fillna(0)
    weighted = labeled.mul(weights.to_numpy(), axis=0)
    grouped_values = weighted.groupby(level=0, sort=False).sum()
    grouped_weights = (
        pd.Series(weights.to_numpy(), index=labeled.index).groupby(level=0, sort=False).sum()
    )
    return grouped_values.div(grouped_weights, axis=0).fillna(0)


def name_neighborhoods(
    neighborhood_percent: pd.DataFrame, neighborhood_summary: pd.DataFrame
) -> tuple[dict[Hashable, str], pd.Series]:
    """Derive each neighbourhood's ``"<dominant cell type> enriched"`` label.

    Parameters
    ----------
    neighborhood_percent
        Output of :func:`cluster_neighborhoods` (``percent``).
    neighborhood_summary
        Output of :func:`cluster_neighborhoods` (``summary``), for weights.

    Returns
    -------
    ``(label_map, weights)`` ready for :func:`collapse_neighborhood_rows`.
    """
    dominant_types = neighborhood_percent.idxmax(axis=1)
    label_map = {index: f"{cell_type} enriched" for index, cell_type in dominant_types.items()}
    weights = neighborhood_summary["n_cells"]
    return label_map, weights


def compute_tissue_enrichment(
    cells: pd.DataFrame,
    named_percent: pd.DataFrame,
    cell_type_order: list[str],
) -> pd.DataFrame:
    """Compute Figure 1f log2 enrichment of each cell type per neighbourhood.

    Parameters
    ----------
    cells
        The full ``reg001`` cell table (for the whole-tissue baseline
        composition).
    named_percent
        Collapsed, named neighbourhood composition percentages.
    cell_type_order
        Column order to reindex to (matching the source notebook's fixed
        display order); cell types absent from the data get 0%.

    Returns
    -------
    Neighbourhood x cell-type table of ``log2(neighbourhood % / tissue %)``.
    """
    plot_composition = named_percent.reindex(columns=cell_type_order)
    tissue_percent = (
        cells["Cell Type"].value_counts(normalize=True).reindex(cell_type_order).fillna(0) * 100
    )
    ratio = plot_composition.div(tissue_percent, axis=1)
    return pd.DataFrame(np.log2(ratio), index=ratio.index, columns=ratio.columns)


def compute_neighborhood_abundance(clustered_cells: pd.DataFrame) -> pd.DataFrame:
    """Compute Figure 1g neighbourhood abundance among LNP− versus LNP+ cells.

    Parameters
    ----------
    clustered_cells
        Output of :func:`cluster_neighborhoods` (``clustered``) with a
        ``neighborhood_name`` column added.

    Returns
    -------
    One row per neighbourhood name, with ``Decoded LNP-``/``Decoded LNP+``
    percentages (each column summing to 100%) and their ``Difference``,
    sorted ascending by ``Difference``.
    """
    comparison_frames = []
    for label, mask in [
        ("Decoded LNP-", clustered_cells["lnp_negative"]),
        ("Decoded LNP+", clustered_cells["lnp_positive"]),
    ]:
        part = clustered_cells.loc[mask, ["neighborhood_name"]].copy()
        part["Comparison group"] = label
        comparison_frames.append(part)

    comparison = pd.concat(comparison_frames, ignore_index=True)
    comparison["Comparison group"] = pd.Categorical(
        comparison["Comparison group"], categories=["Decoded LNP-", "Decoded LNP+"], ordered=True
    )
    abundance = (
        pd.crosstab(
            comparison["neighborhood_name"], comparison["Comparison group"], normalize="columns"
        )
        * 100
    )
    abundance = abundance.reindex(columns=["Decoded LNP-", "Decoded LNP+"]).fillna(0)
    abundance["Difference"] = abundance["Decoded LNP+"] - abundance["Decoded LNP-"]
    return abundance.sort_values("Difference", ascending=True)
