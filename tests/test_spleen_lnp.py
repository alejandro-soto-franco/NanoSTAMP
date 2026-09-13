"""Tests for nanostamp.spleen_lnp."""

from __future__ import annotations

import pandas as pd
import pytest

from nanostamp.spleen_lnp import assemble_obs, compute_cell_type_enrichment, compute_tile_summary


@pytest.fixture
def codebook() -> pd.DataFrame:
    # reg001: 4 cells, 2 LNP+ (cells 1 and 2), all within one 1500px tile.
    return pd.DataFrame(
        {
            "region": ["reg001", "reg001", "reg001", "reg001"],
            "cell": [1, 2, 3, 4],
            "x": [10.0, 20.0, 30.0, 40.0],
            "y": [10.0, 20.0, 30.0, 40.0],
            "lnp_positive": [True, True, False, False],
        }
    )


@pytest.fixture
def bit_1() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "region": ["reg001", "reg001", "reg001", "reg001"],
            "cell": [1, 2, 3, 4],
            "x": [10.0, 20.0, 30.0, 40.0],
            "y": [10.0, 20.0, 30.0, 40.0],
            "bit_1_positive": [True, False, False, False],
        }
    )


@pytest.fixture
def annotations() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "region": ["reg001", "reg001", "reg001", "reg001"],
            "cell": [1, 2, 3, 4],
            "cell_type": ["B", "B", "T", "Lymphatic endothelial"],
        }
    )


def test_assemble_obs_fills_unknown_and_renames(codebook: pd.DataFrame) -> None:
    annotations = pd.DataFrame(
        {"region": ["reg001"], "cell": [1], "cell_type": ["Lymphatic endothelial"]}
    )
    obs = assemble_obs(codebook, annotations)
    row = obs.loc[obs["cell"] == 1]
    assert row["cell_type"].iloc[0] == "Endothelial"
    unknown_rows = obs.loc[obs["cell"].isin([2, 3, 4])]
    assert (unknown_rows["cell_type"] == "Unknown").all()
    assert set(obs["region_label"]) == {"SM-102 LNP"}


def test_assemble_obs_rejects_unexpected_region(codebook: pd.DataFrame) -> None:
    codebook = codebook.copy()
    codebook.loc[0, "region"] = "reg999"
    annotations = pd.DataFrame({"region": ["reg999"], "cell": [1], "cell_type": ["B"]})
    with pytest.raises(ValueError, match="reg000 and reg001"):
        assemble_obs(codebook, annotations)


def test_compute_tile_summary_whole_region_percent(
    bit_1: pd.DataFrame, codebook: pd.DataFrame
) -> None:
    tile_summary, whole_summary = compute_tile_summary(
        bit_1, codebook, tile_size_px=1500, min_cells_per_tile=1
    )
    whole_bit1 = whole_summary.loc[whole_summary["strategy"] == "Single-oligo call"].iloc[0]
    assert whole_bit1["total_cells"] == 4
    assert whole_bit1["positive_cells"] == 1
    assert whole_bit1["percent_positive"] == pytest.approx(25.0)

    whole_codebook = whole_summary.loc[whole_summary["strategy"] == "Codebook-matched call"].iloc[0]
    assert whole_codebook["percent_positive"] == pytest.approx(50.0)

    # All four cells share one 1500px tile.
    assert tile_summary["total_cells"].eq(4).all()


def test_compute_tile_summary_excludes_sparse_tiles(
    bit_1: pd.DataFrame, codebook: pd.DataFrame
) -> None:
    tile_summary, _ = compute_tile_summary(bit_1, codebook, tile_size_px=1500, min_cells_per_tile=5)
    assert tile_summary.empty


def test_compute_cell_type_enrichment_ratio(
    codebook: pd.DataFrame, annotations: pd.DataFrame
) -> None:
    obs = assemble_obs(codebook, annotations)
    comparison = compute_cell_type_enrichment(obs, treated_region="reg001", min_percent=5)
    # Both LNP+ cells are cell_type "B": B is 100% of the LNP+ pool and 50% of
    # all cells, so its representation ratio is 2x.
    b_row = comparison.loc[comparison["plot_cell_type"] == "B"].iloc[0]
    assert b_row["representation_ratio"] == pytest.approx(2.0)
    assert b_row["log2_representation_ratio"] == pytest.approx(1.0)
