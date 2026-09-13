"""Tests for nanostamp.spatial_distance (Figure 2 spatial neighbourhood Sections 8-12 primitives)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nanostamp.spatial_distance import (
    dedup_nearest_pairs,
    focused_neighbor_composition,
    hybrid_neighbor_metrics,
    multiscale_neighbor_metrics,
    neighbor_state_summary,
    own_state_summary,
    paired_ttest_table,
    positive_subset_neighborhood_distribution,
    radius_neighbor_metrics,
    summarize_neighbor_metrics_by_region,
)
from nanostamp.spatial_neighborhood import collect_neighbor_pairs


@pytest.fixture
def all_cells() -> pd.DataFrame:
    # One region: a DC at the origin, three CD8+ T cells at increasing
    # distance, and one unrelated B cell far away.
    return pd.DataFrame(
        {
            "lnp_region": ["r1"] * 5,
            "x": [0.0, 5.0, 15.0, 40.0, 500.0],
            "y": [0.0, 0.0, 0.0, 0.0, 0.0],
            "cell_type": ["DC", "CD8+ T", "CD8+ T", "CD8+ T", "B"],
            "lnp_call": ["LNP_08", "x", "x", "x", "x"],
        }
    )


def test_multiscale_neighbor_metrics_counts_reference_type(all_cells: pd.DataFrame) -> None:
    targets = all_cells.loc[all_cells["cell_type"] == "DC"]
    result = multiscale_neighbor_metrics(targets, "CD8+ T", all_cells, ks=[1, 3], metric_name="cd8")
    row = result.iloc[0]
    # Nearest neighbour of the DC is a CD8+ T cell (distance 5).
    assert row["n_cd8_neighbors_k1"] == 1
    assert row["has_cd8_neighbor_k1"]
    # Within k=3 (the 3 closest cells), all three CD8+ T cells are included.
    assert row["n_cd8_neighbors_k3"] == 3
    assert row["mean_cd8_distance_k3"] > 0


def test_radius_neighbor_metrics_respects_radius(all_cells: pd.DataFrame) -> None:
    targets = all_cells.loc[all_cells["cell_type"] == "DC"]
    result = radius_neighbor_metrics(targets, "CD8+ T", all_cells, radii=[10.0, 20.0])
    row = result.iloc[0]
    assert row["n_reference_within_r10"] == 1  # only the CD8 at distance 5
    assert row["n_reference_within_r20"] == 2  # distances 5 and 15


def test_hybrid_neighbor_metrics_intersects_k_and_radius(all_cells: pd.DataFrame) -> None:
    targets = all_cells.loc[all_cells["cell_type"] == "DC"]
    result = hybrid_neighbor_metrics(
        targets, "CD8+ T", all_cells, k=2, radius=10.0, metric_name="cd8"
    )
    row = result.iloc[0]
    # k=2 nearest (excluding self) are at distance 5 and 15; radius 10 keeps only the first.
    assert row["hybrid_n_retained_neighbors"] == 1
    assert row["hybrid_n_cd8_neighbors"] == 1
    assert row["hybrid_has_cd8_neighbor"]


def test_summarize_neighbor_metrics_by_region_aggregates() -> None:
    per_cell = pd.DataFrame(
        {
            "lnp_region": ["r1", "r1", "r2"],
            "lnp_call": ["LNP_08", "LNP_08", "LNP_10"],
            "n_x_neighbors_k5": [2, 4, 0],
            "has_x_neighbor_k5": [True, True, False],
            "mean_x_distance_k5": [10.0, 20.0, np.nan],
        }
    )
    summary = summarize_neighbor_metrics_by_region(
        per_cell,
        "lnp_call",
        count_cols=["n_x_neighbors_k5"],
        has_cols=["has_x_neighbor_k5"],
        distance_cols=["mean_x_distance_k5"],
    )
    row = summary.loc[summary["lnp_region"] == "r1"].iloc[0]
    assert row["n_target_cells"] == 2
    assert row["mean_n_x_neighbors_k5"] == pytest.approx(3.0)
    assert row["pct_has_x_neighbor_k5"] == pytest.approx(100.0)


def test_paired_ttest_table_one_row_per_metric() -> None:
    region_df = pd.DataFrame(
        {
            "lnp_region": ["r1", "r2", "r1", "r2"],
            "lnp_call": ["LNP_08", "LNP_08", "LNP_10", "LNP_10"],
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [5.0, 6.0, 1.0, 1.0],
        }
    )
    table = paired_ttest_table(region_df, "lnp_call", ("LNP_08", "LNP_10"), ["a", "b"])
    assert list(table["metric"]) == ["a", "b"]
    assert len(table) == 2


def test_dedup_nearest_pairs_keeps_closest() -> None:
    pairs = pd.DataFrame(
        {"neighbor_index": [10, 10, 11], "distance": [5.0, 2.0, 1.0], "target_index": [0, 1, 2]}
    )
    deduped = dedup_nearest_pairs(pairs)
    assert len(deduped) == 2
    row_10 = deduped.loc[deduped["neighbor_index"] == 10].iloc[0]
    assert row_10["distance"] == pytest.approx(2.0)


def test_own_state_summary_aggregates_per_group() -> None:
    cells = pd.DataFrame(
        {
            "lnp_call": ["LNP_08", "LNP_08", "LNP_10"],
            "FOXP3": [1.0, 3.0, 5.0],
            "CD86": [2.0, 4.0, 6.0],
        }
    )
    summary = own_state_summary(cells, ["FOXP3", "CD86"], ["lnp_call"])
    row = summary.loc[(summary["lnp_call"] == "LNP_08") & (summary["marker"] == "FOXP3")].iloc[0]
    assert row["mean_expression"] == pytest.approx(2.0)
    assert row["n_cells"] == 2


def test_neighbor_state_summary_dedups_before_aggregating() -> None:
    all_cells_df = pd.DataFrame({"FOXP3": [10.0, 20.0, 30.0]}, index=[100, 101, 102])
    pairs = pd.DataFrame(
        {
            "lnp_region": ["r1", "r1", "r1"],
            "lnp_call": ["LNP_08", "LNP_08", "LNP_08"],
            "target_index": [0, 1, 2],
            "neighbor_index": [100, 100, 101],  # cell 100 is nearest to two different targets
            "distance": [5.0, 1.0, 3.0],
        }
    )
    summary = neighbor_state_summary(pairs, all_cells_df, ["FOXP3"], ["lnp_region", "lnp_call"])
    row = summary.iloc[0]
    # After dedup, cell 100 (paired at its closest distance 1.0) and cell 101
    # each count once: mean of [10.0, 20.0].
    assert row["n_cells"] == 2
    assert row["mean_expression"] == pytest.approx(15.0)


def test_focused_neighbor_composition_pooled_sums_to_100() -> None:
    targets = pd.DataFrame(
        {
            "lnp_region": ["r1", "r1", "r1", "r2"],
            "lnp_call": ["LNP_08", "LNP_08", "LNP_10", "LNP_10"],
            "neighborhood": ["A", "B", "A", "A"],
        }
    )
    neighbor_composition = pd.DataFrame({"A_type": [0.5, 0.5, 0.2, 0.3]}, index=targets.index)
    pooled, by_region, direct_pooled, direct_by_region = focused_neighbor_composition(
        targets, neighbor_composition, "lnp_call", group_order=["LNP_08", "LNP_10"]
    )
    assert pooled.loc["LNP_08"].sum() == pytest.approx(100.0)
    assert pooled.loc["LNP_10"].sum() == pytest.approx(100.0)
    assert direct_pooled.loc["LNP_10", "A_type"] == pytest.approx(25.0)
    assert not by_region.empty
    assert not direct_by_region.empty


def test_positive_subset_neighborhood_distribution_generalizes_mask() -> None:
    neighbor_cells = pd.DataFrame(
        {
            "lnp_call": ["LNP_01", "LNP_01", "LNP_02"],
            "lnp_positive": [True, True, True],
            "ova_positive": [True, False, True],
            "neighborhood": ["N1", "N2", "N1"],
        }
    )
    mask = neighbor_cells["lnp_positive"] & neighbor_cells["ova_positive"]
    result = positive_subset_neighborhood_distribution(
        neighbor_cells, mask, group_order=["LNP_01", "LNP_02"]
    )
    assert result.loc["LNP_01"].sum() == pytest.approx(100.0)
    # Only one LNP_01 cell (index 0) satisfies both masks, and it is in N1.
    assert result.loc["LNP_01", "N1"] == pytest.approx(100.0)


def test_collect_neighbor_pairs_feeds_neighbor_state_summary(all_cells: pd.DataFrame) -> None:
    targets = all_cells.loc[all_cells["cell_type"] == "DC"].assign(lnp_call="LNP_08")
    all_cells_with_marker = all_cells.assign(FOXP3=[0.0, 1.0, 2.0, 3.0, 4.0])
    pairs = collect_neighbor_pairs(
        targets, "CD8+ T", all_cells_with_marker, group_col="lnp_call", k=3, radius=100.0
    )
    assert not pairs.empty
    summary = neighbor_state_summary(pairs, all_cells_with_marker, ["FOXP3"], ["lnp_call"])
    assert summary.iloc[0]["n_cells"] == 3
