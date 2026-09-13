"""Figure 2 / Supplementary Figures 9-11, Sections 8-12: neighbour-distance and marker-state analyses.

Ported from the second half of
``Code/Figure_2_and_Supplementary_9_11_Multiplex_LNP_Spatial_Neighborhood_Analysis.ipynb``.
Sections 8-11 repeat the same two patterns roughly fifteen times, against
different target/reference cell-type pairs and marker lists: a
self-excluding k-nearest-neighbour or radius search from a target population
to a reference cell type, and a paired region-level significance test
comparing two LNP calls. This module implements each pattern once,
parameterised by cell type, marker list and k/radius settings, so every
upstream table is producible by calling the same handful of functions with
different arguments (see ``workflow/scripts/run_spatial_neighborhood_distance.py``
for the sixteen call sites that reproduce Sections 8-11's ~30 tables, and
Section 12's 8 final paired-panel re-derivations).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from nanostamp.spatial_neighborhood import format_pvalue, paired_region_ttest

__all__ = [
    "dedup_nearest_pairs",
    "focused_neighbor_composition",
    "format_pvalue",
    "hybrid_neighbor_metrics",
    "multiscale_neighbor_metrics",
    "neighbor_state_summary",
    "own_state_summary",
    "paired_region_ttest",
    "paired_ttest_table",
    "positive_subset_neighborhood_distribution",
    "radius_neighbor_metrics",
    "summarize_neighbor_metrics_by_region",
]


def _self_excluded_neighbors(
    target_positions: np.ndarray,
    all_xy: np.ndarray,
    target_global_index: np.ndarray,
    n_neighbors: int,
) -> tuple[np.ndarray, np.ndarray]:
    """K-nearest-neighbour search over ``all_xy``, excluding each query's own point by global index match.

    Returns ``(distances, global_indices)``, both sorted ascending by distance,
    with the self match removed (so both arrays have ``n_neighbors - 1`` columns).
    """
    fit = NearestNeighbors(n_neighbors=min(n_neighbors, len(all_xy))).fit(all_xy)
    distances, local_indices = fit.kneighbors(target_positions)
    global_indices = np.arange(len(all_xy))[local_indices]
    is_self = global_indices == target_global_index[:, None]
    distances = np.where(is_self, np.inf, distances)
    order = distances.argsort(axis=1)
    distances = np.take_along_axis(distances, order, axis=1)
    global_indices = np.take_along_axis(global_indices, order, axis=1)
    keep = distances.shape[1] - 1
    return distances[:, :keep], global_indices[:, :keep]


def multiscale_neighbor_metrics(
    targets: pd.DataFrame,
    reference_cell_type: str,
    all_cells: pd.DataFrame,
    ks: Sequence[int],
    metric_name: str = "reference",
    um_per_coordinate_unit: float = 0.5,
) -> pd.DataFrame:
    """Per-target, multi-k count of ``reference_cell_type`` neighbours (generalises ``add_multiscale_cd8_metrics``).

    Per region, fits a self-excluded k-nearest-neighbour search over every
    cell (not only ``reference_cell_type``), then for each ``k`` in ``ks``
    counts how many of the nearest ``k`` are ``reference_cell_type``.

    Parameters
    ----------
    targets
        Target cells, with ``lnp_region``, ``x``, ``y`` and (matching
        ``all_cells``) a positional index into it.
    reference_cell_type
        ``cell_type`` value to count among neighbours (e.g. ``"CD8+ T"``).
    all_cells
        The full per-region cell table ``targets`` is drawn from; row
        position in this frame is the "global index" used for self-exclusion.
    ks
        Neighbour-window sizes to evaluate.
    metric_name
        Column-name stem, so ``n_{metric_name}_neighbors_k{k}`` etc.
    um_per_coordinate_unit
        Distance-unit conversion for the ``mean_distance`` columns.

    Returns
    -------
    ``targets`` with added ``n_{metric_name}_neighbors_k{k},
    has_{metric_name}_neighbor_k{k}, mean_{metric_name}_distance_k{k}``
    (native units) columns per ``k``.
    """
    max_k = max(ks)
    result = targets.copy()
    for k in ks:
        result[f"n_{metric_name}_neighbors_k{k}"] = 0
        result[f"has_{metric_name}_neighbor_k{k}"] = False
        result[f"mean_{metric_name}_distance_k{k}"] = np.nan

    # `targets` must be a row subset of `all_cells` sharing the same index,
    # so a target's position within its region's `all_cells` slice can be
    # found by index lookup rather than a fragile (x, y) coordinate match.
    for region in targets["lnp_region"].unique():
        region_all = all_cells.loc[all_cells["lnp_region"] == region]
        region_targets_index = targets.index[targets["lnp_region"] == region]
        query_global_index = region_all.index.get_indexer(region_targets_index)
        if (query_global_index < 0).any():
            raise ValueError("Every target must be present in all_cells within its region")

        all_xy = region_all[["x", "y"]].to_numpy()
        is_reference = (region_all["cell_type"].astype(str) == reference_cell_type).to_numpy()

        distances, neighbor_global = _self_excluded_neighbors(
            all_xy[query_global_index], all_xy, query_global_index, max_k + 1
        )
        neighbor_is_reference = is_reference[neighbor_global]

        for k in ks:
            k_distances = distances[:, :k]
            k_is_reference = neighbor_is_reference[:, :k]
            counts = k_is_reference.sum(axis=1)
            reference_distances = np.where(k_is_reference, k_distances, np.nan)
            with np.errstate(invalid="ignore"):
                mean_distance = np.nanmean(reference_distances, axis=1)
            for row_offset, target_index in enumerate(region_targets_index):
                result.loc[target_index, f"n_{metric_name}_neighbors_k{k}"] = int(
                    counts[row_offset]
                )
                result.loc[target_index, f"has_{metric_name}_neighbor_k{k}"] = bool(
                    counts[row_offset] > 0
                )
                result.loc[target_index, f"mean_{metric_name}_distance_k{k}"] = (
                    float(mean_distance[row_offset]) * um_per_coordinate_unit
                    if np.isfinite(mean_distance[row_offset])
                    else np.nan
                )
    return result


def radius_neighbor_metrics(
    targets: pd.DataFrame,
    reference_cell_type: str,
    all_cells: pd.DataFrame,
    radii: Sequence[float],
    metric_name: str = "reference",
    um_per_coordinate_unit: float = 0.5,
) -> pd.DataFrame:
    """Per-target, multi-radius count of ``reference_cell_type`` neighbours (generalises ``add_radius_cd8_metrics``).

    The reference pool is restricted to ``reference_cell_type`` cells only
    (no self-exclusion, matching the source notebook: targets are never of
    the reference type in any of its call sites).

    Returns
    -------
    ``targets`` with added ``n_{metric_name}_within_r{r},
    has_{metric_name}_within_r{r}, mean_{metric_name}_distance_within_r{r}``
    columns per radius.
    """
    result = targets.copy()
    max_radius = max(radii)
    for radius in radii:
        result[f"n_{metric_name}_within_r{radius:g}"] = 0
        result[f"has_{metric_name}_within_r{radius:g}"] = False
        result[f"mean_{metric_name}_distance_within_r{radius:g}"] = np.nan

    for region, region_targets in targets.groupby("lnp_region", observed=False):
        reference_pool = all_cells.loc[
            (all_cells["lnp_region"] == region)
            & (all_cells["cell_type"].astype(str) == reference_cell_type)
        ]
        if reference_pool.empty:
            continue
        fit = NearestNeighbors(radius=max_radius).fit(reference_pool[["x", "y"]].to_numpy())
        distances_list, _ = fit.radius_neighbors(
            region_targets[["x", "y"]].to_numpy(), sort_results=True
        )
        for target_index, distances in zip(region_targets.index, distances_list, strict=True):
            distances = np.asarray(distances)
            for radius in radii:
                within = distances[distances <= radius]
                result.loc[target_index, f"n_{metric_name}_within_r{radius:g}"] = int(len(within))
                result.loc[target_index, f"has_{metric_name}_within_r{radius:g}"] = bool(
                    len(within) > 0
                )
                if len(within):
                    result.loc[target_index, f"mean_{metric_name}_distance_within_r{radius:g}"] = (
                        float(within.mean()) * um_per_coordinate_unit
                    )
    return result


def hybrid_neighbor_metrics(
    targets: pd.DataFrame,
    reference_cell_type: str,
    all_cells: pd.DataFrame,
    k: int,
    radius: float,
    metric_name: str = "reference",
    um_per_coordinate_unit: float = 0.5,
) -> pd.DataFrame:
    """Per-target hybrid k-and-radius count of ``reference_cell_type`` neighbours.

    Generalises ``add_hybrid_knn_radius_cd8_metrics``: a self-excluded
    k-nearest-neighbour search over all cells, retaining only neighbours
    within the top ``k`` AND strictly inside ``radius``.

    Returns
    -------
    ``targets`` with added ``hybrid_n_retained_neighbors,
    hybrid_n_{metric_name}_neighbors, hybrid_has_{metric_name}_neighbor,
    hybrid_mean_{metric_name}_distance, hybrid_median_{metric_name}_distance``.
    """
    result = targets.copy()
    result["hybrid_n_retained_neighbors"] = 0
    result[f"hybrid_n_{metric_name}_neighbors"] = 0
    result[f"hybrid_has_{metric_name}_neighbor"] = False
    result[f"hybrid_mean_{metric_name}_distance"] = np.nan
    result[f"hybrid_median_{metric_name}_distance"] = np.nan

    for region in targets["lnp_region"].unique():
        region_all = all_cells.loc[all_cells["lnp_region"] == region]
        region_targets_index = targets.index[targets["lnp_region"] == region]
        query_global_index = region_all.index.get_indexer(region_targets_index)
        if (query_global_index < 0).any():
            raise ValueError("Every target must be present in all_cells within its region")

        all_xy = region_all[["x", "y"]].to_numpy()
        is_reference = (region_all["cell_type"].astype(str) == reference_cell_type).to_numpy()

        distances, neighbor_global = _self_excluded_neighbors(
            all_xy[query_global_index], all_xy, query_global_index, k + 1
        )
        within_radius = distances < radius
        neighbor_is_reference = is_reference[neighbor_global] & within_radius

        n_retained = within_radius.sum(axis=1)
        n_reference = neighbor_is_reference.sum(axis=1)
        reference_distances = np.where(neighbor_is_reference, distances, np.nan)
        for row_offset, target_index in enumerate(region_targets_index):
            row_distances = reference_distances[row_offset]
            valid = row_distances[np.isfinite(row_distances)]
            result.loc[target_index, "hybrid_n_retained_neighbors"] = int(n_retained[row_offset])
            result.loc[target_index, f"hybrid_n_{metric_name}_neighbors"] = int(
                n_reference[row_offset]
            )
            result.loc[target_index, f"hybrid_has_{metric_name}_neighbor"] = bool(
                n_reference[row_offset] > 0
            )
            if len(valid):
                result.loc[target_index, f"hybrid_mean_{metric_name}_distance"] = (
                    float(valid.mean()) * um_per_coordinate_unit
                )
                result.loc[target_index, f"hybrid_median_{metric_name}_distance"] = (
                    float(np.median(valid)) * um_per_coordinate_unit
                )
    return result


def summarize_neighbor_metrics_by_region(
    per_cell: pd.DataFrame,
    group_col: str,
    count_cols: Sequence[str],
    has_cols: Sequence[str],
    distance_cols: Sequence[str],
) -> pd.DataFrame:
    """Region-level summary of :func:`multiscale_neighbor_metrics` / friends' per-cell output.

    Parameters
    ----------
    per_cell
        Output of one of the per-cell neighbour-metric functions above.
    group_col
        Column distinguishing comparison groups (e.g. ``lnp_call``).
    count_cols, has_cols, distance_cols
        Columns to aggregate as, respectively, ``mean`` (average neighbour
        count), ``mean * 100`` (percentage with at least one neighbour), and
        ``mean`` (average distance).

    Returns
    -------
    One row per ``(lnp_region, group_col)``, with ``n_target_cells`` plus
    one ``mean_<col>`` (or ``pct_<col>``, for ``has_cols``) per input column.
    """
    grouped = per_cell.groupby(["lnp_region", group_col], observed=False)
    result = grouped.size().rename("n_target_cells").reset_index()
    for col in count_cols:
        result[f"mean_{col}"] = grouped[col].mean().to_numpy()
    for col in has_cols:
        result[f"pct_{col}"] = grouped[col].mean().to_numpy() * 100
    for col in distance_cols:
        result[f"mean_{col}"] = grouped[col].mean().to_numpy()
        result[f"median_{col}"] = grouped[col].median().to_numpy()
    return result


def paired_ttest_table(
    region_df: pd.DataFrame,
    group_col: str,
    groups: tuple[str, str],
    metrics: Sequence[str],
    alternative: str = "two-sided",
) -> pd.DataFrame:
    """Apply :func:`paired_region_ttest` across many metrics, one output row each."""
    return pd.DataFrame(
        [
            paired_region_ttest(region_df, group_col, groups, metric, alternative)
            for metric in metrics
        ]
    )


def dedup_nearest_pairs(pairs: pd.DataFrame, reference_col: str = "neighbor_index") -> pd.DataFrame:
    """Keep one row per unique reference cell: its nearest paired target, by distance.

    Ports the notebook's ``sort_values('distance').drop_duplicates(...)``
    dedup step (Sections 8-10), which turns "every (target, reference) pair
    within range" into "every reference cell's single nearest target", so a
    reference cell's marker state is not double-counted across several
    nearby targets.
    """
    return pairs.sort_values("distance").drop_duplicates(subset=[reference_col], keep="first")


def own_state_summary(
    cells: pd.DataFrame, markers: Sequence[str], group_cols: Sequence[str]
) -> pd.DataFrame:
    """Mean/median marker expression read directly off a target population (no neighbour lookup).

    Generalises the "own-state" marker blocks (e.g.
    ``SECTION8A_DC_STATE_MARKERS`` read directly off the DC targets).

    Returns
    -------
    Long-form table: one row per ``(*group_cols, marker)`` with ``n_cells,
    mean_expression, median_expression``.
    """
    long = cells[[*group_cols, *markers]].melt(
        id_vars=group_cols, var_name="marker", value_name="value"
    )
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    return (
        long.groupby([*group_cols, "marker"], observed=False)["value"]
        .agg(n_cells="size", mean_expression="mean", median_expression="median")
        .reset_index()
    )


def neighbor_state_summary(
    pairs: pd.DataFrame, all_cells: pd.DataFrame, markers: Sequence[str], group_cols: Sequence[str]
) -> pd.DataFrame:
    """Mean/median marker expression of the (deduplicated) neighbour cells in ``pairs``.

    Generalises the "neighbour-state" marker blocks (e.g.
    ``SECTION8A_CD8_STATE_MARKERS`` read off the CD8 cells nearest each DC
    target): deduplicates ``pairs`` to one row per unique neighbour, attaches
    that neighbour's raw marker values from ``all_cells``, then aggregates
    exactly as :func:`own_state_summary`.

    Parameters
    ----------
    pairs
        Output of :func:`nanostamp.spatial_neighborhood.collect_neighbor_pairs`,
        with a ``neighbor_index`` column indexing into ``all_cells``.
    all_cells
        The full cell table ``neighbor_index`` refers into.
    markers
        Marker columns to summarise.
    group_cols
        Grouping columns present on ``pairs`` (e.g. ``["lnp_region",
        "lnp_call"]``).

    Returns
    -------
    Long-form table as in :func:`own_state_summary`, plus an
    ``n_unique_neighbors`` alias of ``n_cells``.
    """
    deduped = dedup_nearest_pairs(pairs)
    marker_values = all_cells.loc[deduped["neighbor_index"], markers].reset_index(drop=True)
    combined = pd.concat([deduped[group_cols].reset_index(drop=True), marker_values], axis=1)
    summary = own_state_summary(combined, markers, group_cols)
    summary["n_unique_neighbors"] = summary["n_cells"]
    return summary


def focused_neighbor_composition(
    targets: pd.DataFrame,
    neighbor_composition: pd.DataFrame,
    group_col: str,
    group_order: Sequence[str],
    ordered_names: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Pooled and per-region neighbourhood/composition summaries for a target subset (Section 7 ``summarize_focused_neighbors``).

    Parameters
    ----------
    targets
        Target cells with ``lnp_region``, ``group_col`` and
        ``neighborhood`` columns.
    neighbor_composition
        Per-cell neighbour-type fractions (same index as ``targets``).
    group_col
        Column distinguishing comparison groups (e.g. ``lnp_call``).
    group_order
        Group display order.
    ordered_names
        Neighbourhood display order (columns of the pooled/region tables).

    Returns
    -------
    ``(pooled, by_region, direct_pooled, direct_by_region)``: pooled and
    per-region neighbourhood-membership percentages, and pooled and
    per-region mean direct neighbour-composition percentages.
    """
    counts = (
        targets.groupby([group_col, "neighborhood"], observed=False)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    totals = counts.groupby(group_col)["n_cells"].transform("sum")
    counts["percent"] = 100 * counts["n_cells"] / totals
    pooled = counts.pivot(index=group_col, columns="neighborhood", values="percent").reindex(
        # pyrefly: ignore [bad-argument-type]
        index=group_order
    )
    if ordered_names is not None:
        # pyrefly: ignore [bad-argument-type]
        pooled = pooled.reindex(columns=ordered_names)
    pooled = pooled.fillna(0)

    region_counts = (
        targets.groupby(["lnp_region", group_col, "neighborhood"], observed=False)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    region_totals = region_counts.groupby(["lnp_region", group_col])["n_cells"].transform("sum")
    region_counts["percent"] = 100 * region_counts["n_cells"] / region_totals
    by_region = region_counts.pivot_table(
        index=["lnp_region", group_col], columns="neighborhood", values="percent"
    ).fillna(0)

    direct = pd.concat(
        [targets[["lnp_region", group_col]], neighbor_composition.loc[targets.index] * 100], axis=1
    )
    direct_pooled = (
        # pyrefly: ignore [bad-argument-type]
        direct.groupby(group_col, observed=False).mean(numeric_only=True).reindex(group_order)
    )
    direct_by_region = direct.groupby(["lnp_region", group_col], observed=False).mean(
        numeric_only=True
    )

    return pooled, by_region, direct_pooled, direct_by_region


def positive_subset_neighborhood_distribution(
    neighbor_cells: pd.DataFrame,
    positive_mask: pd.Series,
    group_col: str = "lnp_call",
    group_order: Sequence[str] | None = None,
    ordered_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Neighbourhood-membership percentages within an arbitrary positive subset (generalises Step 5 to Step 11's OVA+ subset).

    Parameters
    ----------
    neighbor_cells
        Per-cell frame with ``group_col`` and ``neighborhood`` columns.
    positive_mask
        Boolean mask selecting the subset to summarise (e.g.
        ``lnp_positive & ova_positive`` for Section 11, rather than Step 5's
        ``lnp_positive`` alone).
    group_col
        Column distinguishing comparison groups.
    group_order
        Group display order; inferred from the data if omitted.
    ordered_names
        Neighbourhood display order.

    Returns
    -------
    ``group_col x neighborhood`` percentage table, each row summing to 100.
    """
    subset = neighbor_cells.loc[positive_mask]
    if group_order is None:
        group_order = sorted(subset[group_col].unique())
    counts = (
        subset.groupby([group_col, "neighborhood"], observed=False)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    totals = counts.groupby(group_col)["n_cells"].transform("sum")
    counts["percent"] = 100 * counts["n_cells"] / totals
    wide = counts.pivot(index=group_col, columns="neighborhood", values="percent").reindex(
        # pyrefly: ignore [bad-argument-type]
        index=group_order
    )
    if ordered_names is not None:
        # pyrefly: ignore [bad-argument-type]
        wide = wide.reindex(columns=ordered_names)
    return wide.fillna(0)
