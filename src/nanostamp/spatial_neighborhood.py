"""Figure 2 / Supplementary Figures 9-11: multiplex LNP spatial neighbourhood analysis.

Ported from
``Code/Figure_2_and_Supplementary_9_11_Multiplex_LNP_Spatial_Neighborhood_Analysis.ipynb``.
Builds a region-restricted 10-nearest-neighbour cell-type composition vector
per cell from the merged annotated table produced by
:mod:`nanostamp.cell_functional`, clusters those vectors into recurrent
spatial neighbourhoods, and reports how LNP-positive and marker-positive cell
subsets distribute across them.

Scope note: the source notebook continues (Sections 8-11) into an extensive
set of region-paired distance/marker analyses between LNP_08- and
LNP_10-positive dendritic, CD8, CD4 and B cells, each built from two shared
patterns: a self-excluding k-nearest-neighbour or radius search, and a paired
region-level significance test. This module ports those two shared patterns
(:func:`collect_neighbor_pairs` and :func:`paired_region_ttest`) as general,
reusable functions rather than transcribing each of the notebook's roughly
fifteen near-identical call sites into its own named function; see the
README "Differences from upstream" section.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

RANDOM_STATE = 0
K_NEIGHBORS = 10
N_NEIGHBORHOODS = 12
ELBOW_CLUSTER_VALUES: list[int] = [5, 8, 10, 12, 15, 20, 25, 30]
ELBOW_MAX_CELLS = 250_000
LNP_ORDER: list[str] = [f"LNP_{i:02d}" for i in range(1, 11)]
GATE_REFERENCE_REGIONS: list[str] = ["S2_reg004"]
GATE_QUANTILE = 0.95
SIINFEKL_GATE_QUANTILE = 0.9
LNP_REGIONS: list[str] = [
    f"S{slide}_reg{region:03d}" for slide in range(1, 5) for region in range(4)
]
REQUIRED_OBS_COLUMNS: set[str] = {
    "x",
    "y",
    "lnp_region",
    "condition",
    "cell_type",
    "lnp_call",
    "lnp_positive",
    "Fluc",
    "OVA",
    "SIINFEKL_H-2Kb",
    "CD86",
}


def pool_cell_types(cell_type: pd.Series) -> pd.Series:
    """Collapse any cell type containing ``"epithelial"`` (case-insensitive) to ``"Epithelial"``."""
    cleaned = cell_type.astype("string").str.strip()
    return cleaned.where(~cleaned.str.contains("epithelial", case=False, na=False), "Epithelial")


def load_and_gate_cells(
    obs: pd.DataFrame,
    lnp_regions: list[str] = LNP_REGIONS,
    gate_reference_regions: list[str] = GATE_REFERENCE_REGIONS,
    gate_quantile: float = GATE_QUANTILE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean, filter and Luc/OVA-gate the merged annotated cell table (Step 1).

    Parameters
    ----------
    obs
        ``.obs`` of the merged AnnData written by
        :func:`nanostamp.cell_functional.merge_cell_annotations_with_lnp_calls`.
    lnp_regions
        The 16 canonical LNP imaging regions to retain, in addition to
        ``gate_reference_regions``.
    gate_reference_regions
        Region(s) whose values define the Luc/OVA positivity threshold.
    gate_quantile
        Quantile of the reference-region marker value used as the threshold.

    Returns
    -------
    ``(cells, gate_table)``.

    Raises
    ------
    KeyError
        If any of :data:`REQUIRED_OBS_COLUMNS` is missing.
    """
    missing = REQUIRED_OBS_COLUMNS - set(obs.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    cells = obs.copy()
    for column in ["x", "y", "Fluc", "OVA", "SIINFEKL_H-2Kb", "CD86"]:
        cells[column] = pd.to_numeric(cells[column], errors="coerce")
    cells["cell_type"] = cells["cell_type"].astype("string").str.strip().fillna("Unknown")
    cells["lnp_call"] = cells["lnp_call"].fillna("no_barcode").astype(str)
    cells["lnp_positive"] = cells["lnp_positive"].fillna(False).astype(bool)

    keep_regions = lnp_regions + gate_reference_regions
    cells = cells.loc[cells["lnp_region"].isin(keep_regions)]
    cells = cells.dropna(subset=["x", "y", "lnp_region"])

    gate_rows = []
    for marker_name, column in [("Luc", "Fluc"), ("OVA", "OVA")]:
        reference_values = cells.loc[
            cells["lnp_region"].isin(gate_reference_regions), column
        ].dropna()
        threshold = (
            float(reference_values.quantile(gate_quantile))
            if len(reference_values)
            else float("nan")
        )
        cells[f"{marker_name.lower()}_positive"] = cells[column] > threshold
        gate_rows.append(
            {
                "marker": marker_name,
                "column": column,
                "threshold": threshold,
                "quantile": gate_quantile,
                "reference_regions": ",".join(gate_reference_regions),
                "n_reference_cells": int(len(reference_values)),
            }
        )
    return cells, pd.DataFrame(gate_rows)


def build_neighbor_composition(
    df: pd.DataFrame,
    k: int = K_NEIGHBORS,
    region_col: str = "lnp_region",
    type_col: str = "cell_type_pooled",
    robust_self_exclusion: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build each cell's k-nearest-neighbour cell-type composition, within its own region (Step 2).

    Parameters
    ----------
    df
        Cell table with ``x``, ``y``, ``region_col`` and ``type_col`` columns.
    k
        Neighbourhood window size, excluding the index cell.
    region_col, type_col
        Column names for the spatial partition and the cell-type label.
    robust_self_exclusion
        The source notebook drops the first (assumed-self) column of the
        k+1-nearest-neighbour search positionally, which can silently keep a
        duplicate-location neighbour and drop a true nearest neighbour when
        two cells share exact coordinates (flagged as a bug in the porting
        spec; every later helper in the same notebook instead matches the
        query's own global index and excludes that one explicitly). When
        ``True`` (the default), this robust index-matching exclusion is used
        uniformly; when ``False``, the positional drop is used instead, for
        exact upstream reproduction.

    Returns
    -------
    ``(work, composition)``: ``work`` has ``source_obs_name``,
    ``mean_neighbor_distance`` (excluding self) indexed like ``df``;
    ``composition`` has one column per category in ``type_col`` with each
    cell's row-normalised neighbour-type fractions (0 for cells with no
    neighbours).
    """
    df = df.reset_index(drop=False).rename(columns={"index": "source_obs_name"})
    categories = sorted(df[type_col].astype(str).unique())
    dummies = pd.get_dummies(df[type_col].astype(str), dtype=np.float32).reindex(
        columns=categories, fill_value=0
    )

    counts = np.zeros((len(df), len(categories)), dtype=np.float32)
    mean_distance = np.full(len(df), np.nan)

    for _region, region_df in df.groupby(region_col, observed=False):
        if len(region_df) < 2:
            continue
        n_neighbors = min(k, len(region_df) - 1) + 1
        xy = region_df[["x", "y"]].to_numpy()
        fit = NearestNeighbors(n_neighbors=n_neighbors).fit(xy)
        distances, indices = fit.kneighbors(xy)
        local_positions = region_df.index.to_numpy()

        if robust_self_exclusion:
            query_positions = np.arange(len(region_df))
            is_self = indices == query_positions[:, None]
            distances = np.where(is_self, np.inf, distances)
            order = distances.argsort(axis=1)
            distances = np.take_along_axis(distances, order, axis=1)
            indices = np.take_along_axis(indices, order, axis=1)
            neighbor_distances = distances[:, : n_neighbors - 1]
            neighbor_indices = indices[:, : n_neighbors - 1]
        else:
            neighbor_distances = distances[:, 1:]
            neighbor_indices = indices[:, 1:]

        neighbor_global_positions = local_positions[neighbor_indices]
        region_counts = dummies.to_numpy()[neighbor_global_positions].sum(axis=1)
        counts[local_positions] = region_counts
        with np.errstate(invalid="ignore"):
            mean_distance[local_positions] = np.nanmean(
                np.where(np.isinf(neighbor_distances), np.nan, neighbor_distances), axis=1
            )

    row_sums = counts.sum(axis=1, keepdims=True)
    fractions = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums > 0)

    work = pd.DataFrame(
        {"source_obs_name": df["source_obs_name"], "mean_neighbor_distance": mean_distance}
    )
    composition = pd.DataFrame(fractions, columns=categories, index=df.index)
    return work, composition


def run_elbow_analysis(
    composition: pd.DataFrame,
    valid_mask: pd.Series,
    cluster_values: list[int] = ELBOW_CLUSTER_VALUES,
    max_cells: int = ELBOW_MAX_CELLS,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Fit MiniBatchKMeans at each candidate cluster count and report inertia (Step 3).

    Parameters
    ----------
    composition
        Full neighbour-composition matrix.
    valid_mask
        Rows with at least one neighbour.
    cluster_values
        Candidate cluster counts to fit.
    max_cells
        Subsample cap for the elbow fits (uniform, without replacement).
    random_state
        Seed for both the subsample and every ``MiniBatchKMeans`` fit.

    Returns
    -------
    One row per candidate ``k`` with ``inertia``, ``inertia_per_sample`` and
    ``relative_inertia_reduction`` (versus the previous ``k``).
    """
    valid_rows = composition.loc[valid_mask]
    if len(valid_rows) > max_cells:
        rng = np.random.default_rng(random_state)
        sampled_positions = np.sort(rng.choice(len(valid_rows), size=max_cells, replace=False))
        valid_rows = valid_rows.iloc[sampled_positions]
    n_sampled = len(valid_rows)
    matrix = valid_rows.to_numpy(dtype=np.float32)

    rows = []
    for k in cluster_values:
        if not (1 < k < n_sampled):
            continue
        model = MiniBatchKMeans(n_clusters=k, random_state=random_state, n_init=10, batch_size=4096)
        model.fit(matrix)
        rows.append(
            {
                "n_clusters": k,
                "inertia": model.inertia_,
                "inertia_per_sample": model.inertia_ / n_sampled,
            }
        )

    result = pd.DataFrame(rows)
    if not result.empty:
        result["relative_inertia_reduction"] = -result["inertia_per_sample"].diff() / result[
            "inertia_per_sample"
        ].shift(1)
    return result


def fit_and_name_neighborhoods(
    composition: pd.DataFrame,
    valid_mask: pd.Series,
    neighbor_cells: pd.DataFrame,
    n_clusters: int = N_NEIGHBORHOODS,
    random_state: int = RANDOM_STATE,
    manual_names: dict[int, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Cluster neighbour-composition vectors and name each cluster by its top categories (Step 4).

    Parameters
    ----------
    composition
        Full neighbour-composition matrix.
    valid_mask
        Rows with at least one neighbour; only these are clustered.
    neighbor_cells
        Per-cell frame to annotate with a ``neighborhood`` label column.
    n_clusters
        Target cluster count (capped at the number of valid rows).
    random_state
        ``MiniBatchKMeans`` random state (``n_init=20``, matching the source
        notebook's final fit).
    manual_names
        Optional cluster-id -> label override. Clusters that end up sharing a
        name are merged into one cell-count-weighted-average row.

    Returns
    -------
    ``(centroid_percent, name_table, neighbor_cells, ordered_names)``.
    """
    n_clusters = min(n_clusters, int(valid_mask.sum()))
    model = MiniBatchKMeans(
        n_clusters=n_clusters, random_state=random_state, n_init=20, batch_size=4096
    )
    labels = model.fit_predict(composition.loc[valid_mask].to_numpy(dtype=np.float32))

    neighbor_cells = neighbor_cells.copy()
    neighbor_cells["neighborhood_id"] = pd.array([pd.NA] * len(neighbor_cells), dtype="Int64")
    neighbor_cells.loc[valid_mask, "neighborhood_id"] = labels

    categories = list(composition.columns)
    centroids = pd.DataFrame(model.cluster_centers_, columns=categories, index=range(n_clusters))
    automatic_names = {}
    for cluster_id, row in centroids.iterrows():
        # pyrefly: ignore [bad-argument-type]
        cluster_id = int(cluster_id)
        top_two = row.sort_values(ascending=False).index[:2]
        automatic_names[cluster_id] = f"N{cluster_id + 1:02d}: {top_two[0]} + {top_two[1]}-rich"

    cluster_sizes_by_id = pd.Series(labels).value_counts().reindex(range(n_clusters), fill_value=0)

    manual_names = manual_names or {}
    final_names = {
        cluster_id: (manual_names.get(cluster_id) or automatic_names[cluster_id])
        for cluster_id in range(n_clusters)
    }
    name_table = pd.DataFrame(
        {
            "source_cluster_id": list(range(n_clusters)),
            "merged_neighborhood_name": [final_names[i] for i in range(n_clusters)],
            "n_index_cells": cluster_sizes_by_id.to_numpy(),
        }
    )

    weights = cluster_sizes_by_id.to_numpy().astype(float)
    weighted_centroids = centroids.to_numpy() * weights[:, None]
    merged_names_order: list[str] = []
    merged_rows: dict[str, np.ndarray] = {}
    merged_weights: dict[str, float] = {}
    for cluster_id in range(n_clusters):
        name = final_names[cluster_id]
        if name not in merged_rows:
            merged_rows[name] = np.zeros(len(categories))
            merged_weights[name] = 0.0
            merged_names_order.append(name)
        merged_rows[name] += weighted_centroids[cluster_id]
        merged_weights[name] += weights[cluster_id]

    centroid_percent_rows = []
    for name in merged_names_order:
        weight = merged_weights[name]
        centroid = merged_rows[name] / weight if weight > 0 else merged_rows[name]
        percent = 100 * centroid / centroid.sum() if centroid.sum() > 0 else centroid
        centroid_percent_rows.append(percent)
    centroid_percent = pd.DataFrame(
        centroid_percent_rows, columns=categories, index=merged_names_order
    )

    neighbor_cells["neighborhood"] = neighbor_cells["neighborhood_id"].map(
        lambda cluster_id: final_names.get(cluster_id) if pd.notna(cluster_id) else pd.NA
    )

    size_by_name = pd.Series(merged_weights)
    ordered_names = size_by_name.sort_values(ascending=False).index.tolist()
    centroid_percent = centroid_percent.reindex(ordered_names)

    return centroid_percent, name_table, neighbor_cells, ordered_names


def summarize_lnp_neighborhood_distribution(
    neighbor_cells: pd.DataFrame,
    lnp_order: list[str] = LNP_ORDER,
    ordered_names: list[str] | None = None,
) -> pd.DataFrame:
    """Pivot LNP-positive neighbourhood membership to an ``lnp_call x neighborhood`` percentage table (Step 5)."""
    positive = neighbor_cells.loc[
        neighbor_cells["lnp_positive"] & neighbor_cells["lnp_call"].isin(lnp_order)
    ]
    counts = (
        positive.groupby(["lnp_call", "neighborhood"], observed=False)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    totals = counts.groupby("lnp_call")["n_cells"].transform("sum")
    counts["n_lnp_cells"] = totals
    counts["percent"] = 100 * counts["n_cells"] / counts["n_lnp_cells"]
    wide = counts.pivot(index="lnp_call", columns="neighborhood", values="percent").reindex(
        index=lnp_order
    )
    if ordered_names is not None:
        wide = wide.reindex(columns=ordered_names)
    return wide.fillna(0)


def compute_local_enrichment(
    neighbor_cells: pd.DataFrame, neighbor_composition: pd.DataFrame, pseudocount: float = 0.0001
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute log2 neighbour-composition enrichment around LNP-positive versus LNP-negative cells (Step 6).

    Parameters
    ----------
    neighbor_cells
        Per-cell frame with ``lnp_region``, ``condition``, ``lnp_positive``.
    neighbor_composition
        Per-cell neighbour-type fractions, same index as ``neighbor_cells``.
    pseudocount
        Added to both the LNP-positive and LNP-negative mean fraction before
        the log2 ratio, avoiding a division by (or log of) zero.

    Returns
    -------
    ``(region_composition, region_enrichment)``: mean composition per
    ``(lnp_region, condition, lnp_positive)``, and the per-region,
    per-cell-type log2 enrichment.
    """
    metadata = neighbor_cells[["lnp_region", "condition", "lnp_positive"]]
    combined = pd.concat([metadata, neighbor_composition.add_prefix("neighbor__")], axis=1)
    region_composition = combined.groupby(
        ["lnp_region", "condition", "lnp_positive"], observed=False
    ).mean(numeric_only=True)

    long = region_composition.reset_index().melt(
        id_vars=["lnp_region", "condition", "lnp_positive"],
        var_name="cell_type",
        value_name="fraction",
    )
    long["cell_type"] = long["cell_type"].str.removeprefix("neighbor__")
    wide = long.pivot_table(
        index=["lnp_region", "cell_type"], columns="lnp_positive", values="fraction"
    )
    wide = wide.rename(columns={True: "pos", False: "neg"}).reset_index()
    wide["log2_enrichment_lnp_pos_vs_neg"] = np.log2(
        (wide.get("pos", 0) + pseudocount) / (wide.get("neg", 0) + pseudocount)
    )
    region_enrichment = wide.rename(
        columns={"pos": "lnp_positive_fraction", "neg": "lnp_negative_fraction"}
    )
    return region_composition.reset_index(), region_enrichment


def format_pvalue(pvalue: float) -> str:
    """Bucket a p-value into a display string (``"P < 0.0001"``, scientific, or 3dp)."""
    if pd.isna(pvalue):
        return "P = NA"
    if pvalue < 0.0001:
        return "P < 0.0001"
    if pvalue < 0.001:
        return f"P = {pvalue:.2e}"
    return f"P = {pvalue:.3f}"


def paired_region_ttest(
    region_df: pd.DataFrame,
    group_col: str,
    groups: tuple[str, str],
    metric: str,
    alternative: str = "two-sided",
) -> dict[str, object]:
    """Paired t-test of ``metric`` between two groups, matched by region (Section 7 shared helper).

    Parameters
    ----------
    region_df
        Region-level table with ``lnp_region``, ``group_col`` and ``metric``.
    group_col
        Column naming the two groups to compare (e.g. ``lnp_call``).
    groups
        The two group labels to compare, e.g. ``("LNP_08", "LNP_10")``.
    metric
        Column to test.
    alternative
        ``"two-sided"``, ``"less"`` or ``"greater"``, passed to
        ``scipy.stats.ttest_rel``.

    Returns
    -------
    Dict with ``test, metric, group_1, group_2, n_paired_regions,
    degrees_of_freedom, t_statistic, pvalue, mean_<group0>, mean_<group1>``.
    All-NaN when fewer than 2 complete-case region pairs exist.
    """
    wide = region_df.pivot(index="lnp_region", columns=group_col, values=metric)
    wide = wide.reindex(columns=list(groups)).dropna()
    n = len(wide)
    if n < 2:
        return {
            "test": "paired_ttest",
            "metric": metric,
            "group_1": groups[0],
            "group_2": groups[1],
            "n_paired_regions": n,
            "degrees_of_freedom": np.nan,
            "t_statistic": np.nan,
            "pvalue": np.nan,
            f"mean_{groups[0]}": wide[groups[0]].mean() if n else np.nan,
            f"mean_{groups[1]}": wide[groups[1]].mean() if n else np.nan,
        }
    t_statistic, pvalue = stats.ttest_rel(
        wide[groups[0]], wide[groups[1]], nan_policy="omit", alternative=alternative
    )
    return {
        "test": "paired_ttest",
        "metric": metric,
        "group_1": groups[0],
        "group_2": groups[1],
        "n_paired_regions": n,
        "degrees_of_freedom": n - 1,
        "t_statistic": float(t_statistic),
        "pvalue": float(pvalue),
        f"mean_{groups[0]}": float(wide[groups[0]].mean()),
        f"mean_{groups[1]}": float(wide[groups[1]].mean()),
    }


def collect_neighbor_pairs(
    target_df: pd.DataFrame,
    reference_cell_type: str,
    all_cells: pd.DataFrame,
    group_col: str,
    k: int = 10,
    radius: float = 25.0,
    um_per_coordinate_unit: float = 0.5,
) -> pd.DataFrame:
    """Self-excluded k-nearest-neighbour search from ``target_df`` cells to ``reference_cell_type`` cells.

    Generalises the notebook's ``collect_target_cd8_neighbor_pairs`` /
    ``collect_target_cd4_neighbor_pairs`` (Sections 8-10): per region, finds
    each target cell's neighbours among cells of ``reference_cell_type``
    within the same region, keeping only those within ``k`` nearest and
    strictly inside ``radius`` (native coordinate units).

    Parameters
    ----------
    target_df
        Target cells (e.g. LNP-positive, marker-positive DCs), with ``x``,
        ``y``, ``lnp_region``, ``group_col``.
    reference_cell_type
        ``cell_type`` value the neighbours are drawn from (e.g. ``"CD8+ T"``).
    all_cells
        The full per-region cell table to search within.
    group_col
        Column distinguishing the comparison groups (e.g. ``lnp_call``).
    k
        Number of nearest same-region reference cells to consider per target.
    radius
        Distance cutoff (native coordinate units); a neighbour further than
        this is dropped (strict ``<``).
    um_per_coordinate_unit
        Conversion factor applied to produce a ``distance_um`` column
        alongside ``distance`` (native units).

    Returns
    -------
    One row per retained (target, neighbour) pair, with ``lnp_region,
    <group_col>, target_index, neighbor_index, distance, distance_um``.
    """
    rows = []
    for region, region_targets in target_df.groupby("lnp_region", observed=False):
        region_reference = all_cells.loc[
            (all_cells["lnp_region"] == region)
            & (all_cells["cell_type"].astype(str) == reference_cell_type)
        ]
        if region_reference.empty:
            continue
        n_neighbors = min(k, len(region_reference))
        fit = NearestNeighbors(n_neighbors=n_neighbors).fit(region_reference[["x", "y"]].to_numpy())
        distances, indices = fit.kneighbors(region_targets[["x", "y"]].to_numpy())
        reference_positions = region_reference.index.to_numpy()

        for row_position, (target_index, target_row) in enumerate(region_targets.iterrows()):
            for distance, neighbor_position in zip(
                distances[row_position], indices[row_position], strict=True
            ):
                if distance >= radius:
                    continue
                rows.append(
                    {
                        "lnp_region": region,
                        group_col: target_row[group_col],
                        "target_index": target_index,
                        "neighbor_index": reference_positions[neighbor_position],
                        "distance": float(distance),
                        "distance_um": float(distance) * um_per_coordinate_unit,
                    }
                )
    return pd.DataFrame(rows)
