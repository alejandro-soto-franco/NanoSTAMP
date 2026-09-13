"""Tests for nanostamp.spatial_neighborhood."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nanostamp.spatial_neighborhood import (
    build_neighbor_composition,
    collect_neighbor_pairs,
    compute_local_enrichment,
    fit_and_name_neighborhoods,
    format_pvalue,
    load_and_gate_cells,
    paired_region_ttest,
    pool_cell_types,
    run_elbow_analysis,
    summarize_lnp_neighborhood_distribution,
)


def test_pool_cell_types_collapses_epithelial_variants() -> None:
    result = pool_cell_types(pd.Series(["Gut Epithelial", "epithelial cell", "B"]))
    assert result.tolist() == ["Epithelial", "Epithelial", "B"]


@pytest.fixture
def gated_obs() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n_per_region = 40
    rows = []
    for region in ["S1_reg000", "S2_reg004"]:
        for i in range(n_per_region):
            rows.append(
                {
                    "x": rng.normal(0, 5),
                    "y": rng.normal(0, 5),
                    "lnp_region": region,
                    "condition": "LNP" if region != "S2_reg004" else "reference",
                    "cell_type": "B" if i % 2 == 0 else "T",
                    "lnp_call": "LNP_01" if i % 3 == 0 else "no_barcode",
                    "lnp_positive": i % 3 == 0,
                    "Fluc": rng.normal(5, 1),
                    "OVA": rng.normal(5, 1),
                    "SIINFEKL_H-2Kb": rng.normal(5, 1),
                    "CD86": rng.normal(5, 1),
                }
            )
    return pd.DataFrame(rows)


def test_load_and_gate_cells_computes_threshold_from_reference_only(
    gated_obs: pd.DataFrame,
) -> None:
    cells, gate_table = load_and_gate_cells(
        gated_obs,
        lnp_regions=["S1_reg000"],
        gate_reference_regions=["S2_reg004"],
        gate_quantile=0.5,
    )
    assert set(cells["lnp_region"]) == {"S1_reg000", "S2_reg004"}
    luc_row = gate_table.loc[gate_table["marker"] == "Luc"].iloc[0]
    assert luc_row["n_reference_cells"] == 40


def test_load_and_gate_cells_raises_on_missing_column(gated_obs: pd.DataFrame) -> None:
    with pytest.raises(KeyError):
        load_and_gate_cells(gated_obs.drop(columns=["CD86"]))


def test_build_neighbor_composition_sums_to_one(gated_obs: pd.DataFrame) -> None:
    cells, _ = load_and_gate_cells(
        gated_obs,
        lnp_regions=["S1_reg000"],
        gate_reference_regions=["S2_reg004"],
        gate_quantile=0.5,
    )
    cells["cell_type_pooled"] = pool_cell_types(cells["cell_type"])
    work, composition = build_neighbor_composition(cells, k=5)
    row_sums = composition.sum(axis=1)
    # Every region here has >5 cells, so every row should have exactly 5 neighbours.
    assert (row_sums.round(5) == 1.0).all()
    assert len(work) == len(cells)


def test_build_neighbor_composition_robust_vs_positional_self_exclusion() -> None:
    # Two cells share exact coordinates: a positional self-exclusion drop can
    # keep the duplicate-location cell as a "neighbour" for itself, while the
    # robust index-matched exclusion always removes the true self match.
    cells = pd.DataFrame(
        {
            "x": [0.0, 0.0, 10.0, 20.0],
            "y": [0.0, 0.0, 10.0, 20.0],
            "lnp_region": ["r1"] * 4,
            "cell_type_pooled": ["A", "B", "A", "A"],
        }
    )
    _, robust_composition = build_neighbor_composition(cells, k=1, robust_self_exclusion=True)
    _, positional_composition = build_neighbor_composition(cells, k=1, robust_self_exclusion=False)
    # Cell 0's single nearest neighbour under robust exclusion must be cell 1
    # (type B, at distance 0), not itself.
    assert robust_composition.loc[0, "B"] == pytest.approx(1.0)
    # Both methods should differ or agree deterministically; simply confirm both run and are valid fractions.
    assert positional_composition.sum(axis=1).round(5).isin([0.0, 1.0]).all()


def test_run_elbow_analysis_produces_monotonic_columns() -> None:
    rng = np.random.default_rng(0)
    composition = pd.DataFrame(rng.random((200, 3)), columns=["A", "B", "C"])
    valid_mask = pd.Series(True, index=composition.index)
    result = run_elbow_analysis(
        composition, valid_mask, cluster_values=[2, 3, 4], max_cells=1000, random_state=0
    )
    assert set(result["n_clusters"]) == {2, 3, 4}
    assert (result["inertia_per_sample"] > 0).all()


def test_fit_and_name_neighborhoods_merges_same_named_clusters() -> None:
    n = 60
    composition = pd.DataFrame(
        {
            "A": [1.0] * 20 + [0.0] * 20 + [1.0] * 20,
            "B": [0.0] * 20 + [1.0] * 20 + [0.0] * 20,
        }
    )
    valid_mask = pd.Series(True, index=composition.index)
    neighbor_cells = pd.DataFrame({"lnp_positive": [False] * n, "lnp_call": ["no_barcode"] * n})
    # Force clusters 0 and 2 (both dominated by "A") to share a manual name.
    centroid_percent, name_table, annotated, ordered_names = fit_and_name_neighborhoods(
        composition,
        valid_mask,
        neighbor_cells,
        n_clusters=3,
        random_state=0,
        manual_names={0: "A-rich", 2: "A-rich"},
    )
    assert "A-rich" in ordered_names
    assert len(centroid_percent) == len(ordered_names)
    assert annotated["neighborhood"].notna().all()


def test_summarize_lnp_neighborhood_distribution_sums_to_100_per_call() -> None:
    neighbor_cells = pd.DataFrame(
        {
            "lnp_call": ["LNP_01", "LNP_01", "LNP_01", "LNP_02"],
            "lnp_positive": [True, True, True, True],
            "neighborhood": ["N1", "N1", "N2", "N1"],
        }
    )
    result = summarize_lnp_neighborhood_distribution(neighbor_cells, lnp_order=["LNP_01", "LNP_02"])
    assert result.loc["LNP_01"].sum() == pytest.approx(100.0)
    assert result.loc["LNP_02"].sum() == pytest.approx(100.0)


def test_compute_local_enrichment_positive_for_enriched_type() -> None:
    neighbor_cells = pd.DataFrame(
        {
            "lnp_region": ["r1"] * 4,
            "condition": ["LNP"] * 4,
            "lnp_positive": [True, True, False, False],
        }
    )
    neighbor_composition = pd.DataFrame({"A": [0.9, 0.8, 0.1, 0.2], "B": [0.1, 0.2, 0.9, 0.8]})
    _, enrichment = compute_local_enrichment(neighbor_cells, neighbor_composition)
    a_row = enrichment.loc[enrichment["cell_type"] == "A"].iloc[0]
    assert a_row["log2_enrichment_lnp_pos_vs_neg"] > 0


def test_format_pvalue_buckets() -> None:
    assert format_pvalue(0.00001) == "P < 0.0001"
    assert format_pvalue(0.0005).startswith("P = ")
    assert format_pvalue(0.2) == "P = 0.200"
    assert format_pvalue(float("nan")) == "P = NA"


def test_paired_region_ttest_requires_two_regions() -> None:
    region_df = pd.DataFrame({"lnp_region": ["r1"], "lnp_call": ["LNP_08"], "metric": [1.0]})
    result = paired_region_ttest(region_df, "lnp_call", ("LNP_08", "LNP_10"), "metric")
    assert result["n_paired_regions"] == 0
    # pyrefly: ignore [no-matching-overload]
    assert np.isnan(result["pvalue"])


def test_paired_region_ttest_on_consistent_difference_is_significant() -> None:
    # LNP_08 is uniformly 2 units higher than LNP_10 in every paired region:
    # a small, consistent difference should yield a low (significant) p-value.
    region_df = pd.DataFrame(
        {
            "lnp_region": ["r1", "r2", "r3", "r1", "r2", "r3"],
            "lnp_call": ["LNP_08"] * 3 + ["LNP_10"] * 3,
            "metric": [3.0, 4.0, 5.0, 1.0, 2.0, 3.0],
        }
    )
    result = paired_region_ttest(region_df, "lnp_call", ("LNP_08", "LNP_10"), "metric")
    assert result["n_paired_regions"] == 3
    assert result["degrees_of_freedom"] == 2
    # pyrefly: ignore [unsupported-operation]
    assert result["pvalue"] < 0.05
    assert result["mean_LNP_08"] == pytest.approx(4.0)
    assert result["mean_LNP_10"] == pytest.approx(2.0)


def test_paired_region_ttest_identical_groups_is_nan_not_an_error() -> None:
    # Zero variance in the paired differences is a genuine 0/0 case for a
    # paired t-test; scipy reports it as NaN rather than raising.
    region_df = pd.DataFrame(
        {
            "lnp_region": ["r1", "r2", "r3", "r1", "r2", "r3"],
            "lnp_call": ["LNP_08"] * 3 + ["LNP_10"] * 3,
            "metric": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        }
    )
    result = paired_region_ttest(region_df, "lnp_call", ("LNP_08", "LNP_10"), "metric")
    assert result["n_paired_regions"] == 3
    # pyrefly: ignore [no-matching-overload]
    assert np.isnan(result["pvalue"])


def test_collect_neighbor_pairs_respects_radius() -> None:
    targets = pd.DataFrame({"x": [0.0], "y": [0.0], "lnp_region": ["r1"], "lnp_call": ["LNP_08"]})
    all_cells = pd.DataFrame(
        {
            "x": [1.0, 100.0],
            "y": [0.0, 0.0],
            "lnp_region": ["r1", "r1"],
            "cell_type": ["CD8+ T", "CD8+ T"],
        }
    )
    pairs = collect_neighbor_pairs(
        targets, "CD8+ T", all_cells, group_col="lnp_call", k=2, radius=25.0
    )
    # Only the neighbour at distance 1 is within radius 25; the one at distance 100 is not.
    assert len(pairs) == 1
    assert pairs.iloc[0]["distance"] == pytest.approx(1.0)
    assert pairs.iloc[0]["distance_um"] == pytest.approx(0.5)
