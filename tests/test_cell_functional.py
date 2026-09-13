"""Tests for nanostamp.cell_functional."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from nanostamp.cell_functional import (
    LNP_ORDER,
    build_lnp_detail_table,
    compute_bcell_uptake_vs_ova_correlation,
    compute_celltype_lnp_uptake,
    compute_celltype_relative_enrichment,
    compute_gene_expression_gates,
    compute_lnp_positive_celltype_composition,
    compute_region_level_lnp_uptake,
    compute_siinfekl_gate_and_dc_summary,
    merge_cell_annotations_with_lnp_calls,
    reference_quantile_gate,
    sem,
    summarize_gene_group,
)


def test_build_lnp_detail_table_swaps_lnp_01_and_02() -> None:
    table, raw_to_corrected = build_lnp_detail_table()
    assert len(table) == 10
    assert raw_to_corrected["LNP_02"] == "LNP_01"
    assert raw_to_corrected["LNP_01"] == "LNP_02"
    assert raw_to_corrected["LNP_05"] == "LNP_05"
    row01 = table.loc[table["lnp_call"] == "LNP_01"].iloc[0]
    assert row01["formulation"] == "SM-102 + DOTAP"


def test_sem_zero_for_small_n() -> None:
    assert sem([]) == 0.0
    assert sem([5.0]) == 0.0
    assert sem([1.0, 2.0, 3.0]) == pytest.approx(np.std([1, 2, 3], ddof=1) / np.sqrt(3))


def test_reference_quantile_gate_nan_when_empty() -> None:
    assert np.isnan(reference_quantile_gate([], 0.95))
    assert reference_quantile_gate([1, 2, 3, 4, 5], 0.5) == pytest.approx(3.0)


@pytest.fixture
def tiny_adata() -> ad.AnnData:
    n = 6
    obs = pd.DataFrame(
        {
            "label": [1, 2, 3, 1, 2, 3],
            "slide_name": ["foo_registered_S1"] * 3 + ["bar_registered_S2"] * 3,
            "region": ["reg000"] * 3 + ["reg004"] * 3,
            "cell_type": ["B", "T", "DC", "B", "T", "DC"],
            "Fluc": [1.0, 2.0, 3.0, 10.0, 11.0, 12.0],
            "OVA": [1.0, 2.0, 3.0, 10.0, 11.0, 12.0],
            "SIINFEKL_H-2Kb": [1.0, 2.0, 3.0, 10.0, 11.0, 12.0],
            "CD86": [1.0, 2.0, 3.0, 10.0, 11.0, 12.0],
        },
        index=[f"cell{i}" for i in range(n)],
    )
    X = np.zeros((n, 1))
    return ad.AnnData(X=X, obs=obs)


@pytest.fixture
def tiny_spot() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "region": [
                "S1_reg000",
                "S1_reg000",
                "S1_reg000",
                "S2_reg004",
                "S2_reg004",
                "S2_reg004",
            ],
            "cell": [1, 2, 3, 1, 2, 3],
            "barcode": ["b"] * 6,
            "lnp_call": ["LNP_02", "LNP_03", None, None, None, None],
            "lnp_positive": [True, True, False, False, False, False],
            "barcode_in_library": [True] * 6,
            "barcode_match_distance": [0] * 6,
            "barcode_match_status": ["ok"] * 6,
            "barcode_excluded": [False] * 6,
            "total_barcode_spots": [1] * 6,
            "decoded_exact_spots": [1] * 6,
            "decoded_tolerant_spots": [1] * 6,
            "n_positive_bits": [1] * 6,
            "dominant_decoded_barcode": ["x"] * 6,
            "dominant_decoded_barcode_count": [1] * 6,
            "dominant_barcode_marker": ["x"] * 6,
            "dominant_barcode_count": [1] * 6,
            "dominant_barcode_fraction": [1.0] * 6,
            "barcode_confidence": [1.0] * 6,
            "decoded_spots": [1] * 6,
        }
    )


def test_merge_applies_raw_to_corrected_swap(
    tiny_adata: ad.AnnData, tiny_spot: pd.DataFrame
) -> None:
    lnp_detail_table, raw_to_corrected = build_lnp_detail_table()
    merged, report = merge_cell_annotations_with_lnp_calls(
        tiny_adata,
        tiny_spot,
        marker_cols=[],
        lnp_marker_cols=[],
        lnp_detail_table=lnp_detail_table,
        raw_to_corrected=raw_to_corrected,
        analysis_regions=["S1_reg000", "S2_reg004"],
        reference_regions=["S2_reg004"],
    )
    assert report["n_retained"] == 6
    obs = merged.obs
    # Raw LNP_02 for cell 1 in S1_reg000 becomes corrected LNP_01.
    # pyrefly: ignore [missing-attribute]
    row = obs.loc[(obs["lnp_region"] == "S1_reg000") & (obs["cell"] == 1)].iloc[0]
    assert row["lnp_call_raw"] == "LNP_02"
    assert row["lnp_call_corrected"] == "LNP_01"
    assert row["lnp_call"] == "LNP_01"
    assert row["condition"] == "LNP"
    # pyrefly: ignore [missing-attribute]
    reference_row = obs.loc[obs["lnp_region"] == "S2_reg004"].iloc[0]
    assert reference_row["condition"] == "reference"
    # Cell 3 in S1_reg000 has no barcode call.
    # pyrefly: ignore [missing-attribute]
    no_call_row = obs.loc[(obs["lnp_region"] == "S1_reg000") & (obs["cell"] == 3)].iloc[0]
    assert no_call_row["lnp_call"] == "no_barcode"
    assert no_call_row["lnp_call_positive_only"] == "no_barcode"


def test_merge_raises_when_expected_region_has_no_cells(
    tiny_adata: ad.AnnData, tiny_spot: pd.DataFrame
) -> None:
    # Filtering happens before the region-set check, so a region that is
    # merely absent from the fixture (rather than one present-but-unlisted)
    # is what trips the mismatch: it survives the keep_mask filter as an
    # expected-but-unobserved entry.
    lnp_detail_table, raw_to_corrected = build_lnp_detail_table()
    with pytest.raises(ValueError, match="lnp_region mismatch"):
        merge_cell_annotations_with_lnp_calls(
            tiny_adata,
            tiny_spot,
            marker_cols=[],
            lnp_marker_cols=[],
            lnp_detail_table=lnp_detail_table,
            raw_to_corrected=raw_to_corrected,
            analysis_regions=["S1_reg000", "S9_reg999"],
            reference_regions=[],
        )


@pytest.fixture
def merged_obs(tiny_adata: ad.AnnData, tiny_spot: pd.DataFrame) -> pd.DataFrame:
    lnp_detail_table, raw_to_corrected = build_lnp_detail_table()
    merged, _ = merge_cell_annotations_with_lnp_calls(
        tiny_adata,
        tiny_spot,
        marker_cols=[],
        lnp_marker_cols=[],
        lnp_detail_table=lnp_detail_table,
        raw_to_corrected=raw_to_corrected,
        analysis_regions=["S1_reg000", "S2_reg004"],
        reference_regions=["S2_reg004"],
    )
    # pyrefly: ignore [bad-return]
    return merged.obs


def test_region_level_uptake_counts_positive_cells(merged_obs: pd.DataFrame) -> None:
    uptake_by_region, uptake_summary = compute_region_level_lnp_uptake(
        merged_obs, lnp_order=LNP_ORDER, lnp_regions=["S1_reg000"], reference_regions=["S2_reg004"]
    )
    row = uptake_by_region.loc[
        (uptake_by_region["lnp_region"] == "S1_reg000") & (uptake_by_region["lnp_call"] == "LNP_01")
    ].iloc[0]
    assert row["n_lnp_positive"] == 1
    assert row["n_cells_total"] == 3
    assert row["pct_cells_lnp_positive"] == pytest.approx(100 / 3)
    assert not uptake_summary.empty


def test_celltype_uptake_and_top_celltypes(merged_obs: pd.DataFrame) -> None:
    celltype_uptake, celltype_summary, top_celltypes = compute_celltype_lnp_uptake(merged_obs)
    assert set(celltype_uptake["cell_type"]).issubset({"B", "T", "DC"})
    assert len(top_celltypes) <= 12


def test_composition_includes_tissue_average_row(merged_obs: pd.DataFrame) -> None:
    composition = compute_lnp_positive_celltype_composition(merged_obs)
    assert "Tissue_average" in set(composition["lnp_call"])
    # Percentages within each lnp_call should sum to ~100, or be all-NaN for
    # an lnp_call with zero positive cells (pandas .sum() over an all-NaN
    # series returns 0.0, not NaN, so check the NaN case explicitly first).
    for _lnp_call, group in composition.groupby("lnp_call"):
        if group["pct_of_lnp_positive_cells"].isna().all():
            continue
        assert group["pct_of_lnp_positive_cells"].sum() == pytest.approx(100.0)


def test_relative_enrichment_formula(merged_obs: pd.DataFrame) -> None:
    bias, _, best = compute_celltype_relative_enrichment(merged_obs)
    b_row = bias.loc[(bias["lnp_call"] == "LNP_01") & (bias["cell_type"] == "B")].iloc[0]
    expected_ratio = b_row["positive_fraction_within_lnp"] / b_row["background_fraction"]
    assert b_row["enrichment_vs_background"] == pytest.approx(expected_ratio)
    assert not best.empty


def test_gene_expression_gates_threshold_from_reference_only(merged_obs: pd.DataFrame) -> None:
    expression_obs, gate_table = compute_gene_expression_gates(
        merged_obs, reference_regions=["S2_reg004"], quantile=0.5
    )
    luc_row = gate_table.loc[gate_table["marker"] == "Luc"].iloc[0]
    # Reference region Fluc values are [10, 11, 12]; median = 11.
    assert luc_row["threshold"] == pytest.approx(11.0)
    assert luc_row["reference_n_cells"] == 3
    summary = summarize_gene_group(expression_obs, ["lnp_region"], "all_cells")
    assert (summary["analysis_population"] == "all_cells").all()


def test_siinfekl_gate_and_summary_smoke(merged_obs: pd.DataFrame) -> None:
    expression_obs, _ = compute_gene_expression_gates(merged_obs, reference_regions=["S2_reg004"])
    gate_table, by_region, summary, composition = compute_siinfekl_gate_and_dc_summary(
        expression_obs, reference_regions=["S2_reg004"], quantile=0.5
    )
    assert gate_table.iloc[0]["marker"] == "SIINFEKL-H-2Kb"
    assert set(summary["lnp_call"]) == set(LNP_ORDER)
    assert "n_siinfekl_positive_lnp_dc" in composition.columns


def test_bcell_uptake_vs_ova_correlation_denominator_fix_changes_values() -> None:
    uptake = pd.DataFrame(
        {
            "lnp_call": ["LNP_01", "LNP_02", "Tissue_average"],
            "cell_type": ["B", "B", "B"],
            "n_lnp_positive": [10, 30, 1000],
            "lnp_number": [1, 2, np.nan],
            "formulation": ["a", "b", "All cells (tissue average)"],
        }
    )
    transfection = pd.DataFrame(
        {
            "cell_type": ["B", "B"],
            "marker": ["OVA", "OVA"],
            "lnp_label_for_gene_positive": ["LNP_01", "LNP_02"],
            "n_positive_cells": [4, 6],
        }
    )
    fixed_df, fixed_stats = compute_bcell_uptake_vs_ova_correlation(
        uptake, transfection, fix_denominator_bug=True
    )
    buggy_df, buggy_stats = compute_bcell_uptake_vs_ova_correlation(
        uptake, transfection, fix_denominator_bug=False
    )
    # Fixed denominator excludes Tissue_average (10+30=40); buggy includes it (10+30+1000=1040).
    fixed_lnp01 = fixed_df.loc[fixed_df["lnp_call"] == "LNP_01", "uptake_composition_pct"].iloc[0]
    buggy_lnp01 = buggy_df.loc[buggy_df["lnp_call"] == "LNP_01", "uptake_composition_pct"].iloc[0]
    assert fixed_lnp01 == pytest.approx(100 * 10 / 40)
    assert buggy_lnp01 == pytest.approx(100 * 10 / 1040)
    assert fixed_lnp01 != buggy_lnp01
    assert "spearman_rho" in fixed_stats
