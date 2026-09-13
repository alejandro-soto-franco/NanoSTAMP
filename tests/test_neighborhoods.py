"""Tests for nanostamp.neighborhoods."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nanostamp.neighborhoods import (
    build_region_windows,
    cluster_neighborhoods,
    collapse_neighborhood_rows,
    compute_neighborhood_abundance,
    compute_tissue_enrichment,
    name_neighborhoods,
)


@pytest.fixture
def synthetic_cells() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n_per_cluster = 30
    # Two well-separated spatial blobs, each homogeneous in cell type, so
    # k-nearest windows and MiniBatchKMeans should cleanly recover 2 clusters.
    blob_a = pd.DataFrame(
        {
            "x": rng.normal(0, 5, n_per_cluster),
            "y": rng.normal(0, 5, n_per_cluster),
            "Cell Type": "B",
        }
    )
    blob_b = pd.DataFrame(
        {
            "x": rng.normal(1000, 5, n_per_cluster),
            "y": rng.normal(1000, 5, n_per_cluster),
            "Cell Type": "T",
        }
    )
    cells = pd.concat([blob_a, blob_b], ignore_index=True)
    cells["region"] = "reg001"
    cells["unique_region"] = "reg001"
    cells["lnp_positive"] = np.arange(len(cells)) % 3 == 0
    cells["lnp_negative"] = ~cells["lnp_positive"]
    return cells


def test_build_region_windows_shapes(synthetic_cells: pd.DataFrame) -> None:
    cells_df, cell_types, windows = build_region_windows(synthetic_cells, k=5)
    assert set(cell_types) == {"B", "T"}
    assert len(windows) == len(cells_df)
    # Every cell's 5-nearest window (same tight blob) sums to 5.
    assert windows[cell_types].sum(axis=1).eq(5).all()


def test_cluster_neighborhoods_recovers_two_pure_blobs(synthetic_cells: pd.DataFrame) -> None:
    cells_df, cell_types, windows = build_region_windows(synthetic_cells, k=5)
    clustered, percent, fold_change, summary = cluster_neighborhoods(
        cells_df, windows, cell_types, n_neighborhoods=2, random_state=0
    )
    # Each cluster should be near-purely one cell type (window composition).
    for neighborhood in percent.index:
        row = percent.loc[neighborhood]
        assert row.max() > 90
    assert set(summary.index) == {0, 1}
    assert summary["n_cells"].sum() == len(cells_df)


def test_name_neighborhoods_and_collapse_are_consistent(synthetic_cells: pd.DataFrame) -> None:
    cells_df, cell_types, windows = build_region_windows(synthetic_cells, k=5)
    clustered, percent, fold_change, summary = cluster_neighborhoods(
        cells_df, windows, cell_types, n_neighborhoods=2, random_state=0
    )
    label_map, weights = name_neighborhoods(percent, summary)
    named_percent = collapse_neighborhood_rows(percent, label_map, weights)
    # Two pure, distinctly named clusters should not collapse into one row.
    assert len(named_percent) == 2
    assert set(label_map.values()) == {"B enriched", "T enriched"}


def test_compute_tissue_enrichment_positive_for_dominant_type() -> None:
    named_percent = pd.DataFrame({"B": [80.0], "T": [20.0]}, index=["B enriched"])
    cells = pd.DataFrame({"Cell Type": ["B"] * 40 + ["T"] * 60})
    enrichment = compute_tissue_enrichment(cells, named_percent, cell_type_order=["B", "T"])
    # pyrefly: ignore [unsupported-operation]
    assert enrichment.loc["B enriched", "B"] > 0
    # pyrefly: ignore [unsupported-operation]
    assert enrichment.loc["B enriched", "T"] < 0


def test_compute_neighborhood_abundance_sums_to_100() -> None:
    clustered = pd.DataFrame(
        {
            "neighborhood_name": ["B enriched", "B enriched", "T enriched", "T enriched"],
            "lnp_positive": [True, False, True, False],
            "lnp_negative": [False, True, False, True],
        }
    )
    abundance = compute_neighborhood_abundance(clustered)
    assert abundance["Decoded LNP-"].sum() == pytest.approx(100.0)
    assert abundance["Decoded LNP+"].sum() == pytest.approx(100.0)
    assert set(abundance["Difference"].round(6)) <= {0.0}
