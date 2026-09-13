"""Figure 1d/1e: spleen LNP spatial-tile calls and cell-type composition.

Ported from ``Code/Figure_1d_1e_Spleen_LNP_Analysis.ipynb``. Reads the
compact, frozen Bit_1 and full-barcode (codebook-matched) cell calls plus the
frozen cell-type annotations for the PBS and SM-102 LNP-treated spleens, then
computes:

- Figure 1d: per-tile and whole-region percentage of LNP+ calls, for both the
  single-oligo (Bit_1) and codebook-matched detection strategies.
- Figure 1e: cell-type composition of codebook-matched LNP+ cells in the
  treated spleen versus all annotated cells, with the relative-enrichment
  ratio of each cell type.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

#: Region-code to treatment-label mapping used throughout this figure pair.
REGION_LABELS: dict[str, str] = {"reg000": "PBS", "reg001": "SM-102 LNP"}

#: Cell-type label collapses applied before any composition analysis.
CELL_TYPE_RENAMES: dict[str, str] = {"Lymphatic endothelial": "Endothelial"}


def load_frozen_tables(data_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three compact, frozen input tables for Figure 1d/1e.

    Parameters
    ----------
    data_root
        ``Figure_1d_1e_Spleen_LNP`` directory (containing
        ``Precomputed_Analysis_Input/``).

    Returns
    -------
    ``(codebook, bit_1, annotations)`` dataframes, with the ``cell`` column
    cast to ``int64`` in each.
    """
    input_dir = Path(data_root) / "Precomputed_Analysis_Input"
    codebook = pd.read_csv(input_dir / "codebook_cell_calls_spleen.csv.gz", dtype={"region": str})
    bit_1 = pd.read_csv(input_dir / "bit_1_cell_calls_spleen.csv.gz", dtype={"region": str})
    annotations = pd.read_csv(
        input_dir / "frozen_cell_annotations_spleen.csv.gz",
        dtype={"region": str, "cell_type": str},
    )
    for frame in (codebook, bit_1, annotations):
        frame["cell"] = pd.to_numeric(frame["cell"], errors="raise").astype("int64")
    return codebook, bit_1, annotations


def assemble_obs(codebook: pd.DataFrame, annotations: pd.DataFrame) -> pd.DataFrame:
    """Merge codebook calls with frozen cell-type annotations.

    Parameters
    ----------
    codebook
        Full-barcode codebook-matched cell calls, one row per cell.
    annotations
        Frozen cell-type annotations, one row per cell.

    Returns
    -------
    ``codebook`` left-joined on ``["region", "cell"]`` with an added
    ``cell_type`` (missing filled as ``"Unknown"``, then collapsed per
    :data:`CELL_TYPE_RENAMES`) and ``region_label`` column.

    Raises
    ------
    ValueError
        If any region is not one of :data:`REGION_LABELS`.
    """
    obs = codebook.merge(
        annotations[["region", "cell", "cell_type"]],
        on=["region", "cell"],
        how="left",
        validate="one_to_one",
    )
    obs["lnp_positive"] = obs["lnp_positive"].astype(bool)
    obs["cell_type"] = obs["cell_type"].fillna("Unknown").replace(CELL_TYPE_RENAMES)
    obs["region_label"] = obs["region"].map(REGION_LABELS)
    if obs["region_label"].isna().any():
        raise ValueError("Only reg000 and reg001 are expected in this package")
    return obs


def compute_tile_summary(
    bit_1: pd.DataFrame,
    codebook: pd.DataFrame,
    tile_size_px: int = 1500,
    min_cells_per_tile: int = 500,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute Figure 1d per-tile and whole-region LNP+ percentages.

    Cells are binned into ``tile_size_px``-pixel square tiles by their
    centroid; tiles with fewer than ``min_cells_per_tile`` segmented cells are
    excluded from the per-tile summary (the whole-region summary is
    unaffected by that cutoff).

    Parameters
    ----------
    bit_1
        Single-oligo Bit_1 calls with ``region``, ``x``, ``y``,
        ``bit_1_positive`` columns.
    codebook
        Codebook-matched calls with ``region``, ``x``, ``y``, ``lnp_positive``
        columns.
    tile_size_px
        Tile edge length in pixels.
    min_cells_per_tile
        Minimum segmented cells for a tile to be retained in the per-tile
        summary.

    Returns
    -------
    ``(tile_summary, whole_summary)``, each with a ``strategy`` column
    (``"Single-oligo call"`` or ``"Codebook-matched call"``) and a
    ``percent_positive`` column.
    """
    strategy_inputs = {
        "Single-oligo call": bit_1[["region", "x", "y", "bit_1_positive"]].rename(
            columns={"bit_1_positive": "positive"}
        ),
        "Codebook-matched call": codebook[["region", "x", "y", "lnp_positive"]].rename(
            columns={"lnp_positive": "positive"}
        ),
    }

    tile_rows = []
    whole_rows = []
    for strategy, tile_source in strategy_inputs.items():
        tile_source = tile_source.copy()
        tile_source["region_label"] = tile_source["region"].map(REGION_LABELS)
        tile_source["x"] = pd.to_numeric(tile_source["x"], errors="coerce")
        tile_source["y"] = pd.to_numeric(tile_source["y"], errors="coerce")
        tile_source["positive"] = tile_source["positive"].astype(bool)
        tile_source = tile_source.dropna(subset=["x", "y"])
        tile_source["tile_x"] = np.floor(tile_source["x"] / tile_size_px).astype(int)
        tile_source["tile_y"] = np.floor(tile_source["y"] / tile_size_px).astype(int)

        grouped = (
            tile_source.groupby(["region", "region_label", "tile_x", "tile_y"], observed=False)[
                "positive"
            ]
            .agg(positive_cells="sum", total_cells="count")
            .reset_index()
        )
        grouped = grouped.loc[grouped["total_cells"] >= min_cells_per_tile].copy()
        grouped["percent_positive"] = 100 * grouped["positive_cells"] / grouped["total_cells"]
        grouped["strategy"] = strategy
        tile_rows.append(grouped)

        whole = (
            tile_source.groupby(["region", "region_label"], observed=False)["positive"]
            .agg(positive_cells="sum", total_cells="count")
            .reset_index()
        )
        whole["percent_positive"] = 100 * whole["positive_cells"] / whole["total_cells"]
        whole["strategy"] = strategy
        whole_rows.append(whole)

    tile_summary = pd.concat(tile_rows, ignore_index=True)
    whole_summary = pd.concat(whole_rows, ignore_index=True)
    return tile_summary, whole_summary


def compute_cell_type_enrichment(
    obs: pd.DataFrame,
    treated_region: str = "reg001",
    min_percent: float = 5,
) -> pd.DataFrame:
    """Compute Figure 1e cell-type composition and relative enrichment.

    Compares each cell type's share of codebook-matched LNP+ cells against
    its share of all annotated cells in ``treated_region``. Cell types below
    ``min_percent`` in both populations are pooled as ``"Other"``.

    Parameters
    ----------
    obs
        Output of :func:`assemble_obs`.
    treated_region
        Region code to restrict the comparison to.
    min_percent
        Minimum percentage (in either group) for a cell type to be reported
        individually rather than pooled into ``"Other"``.

    Returns
    -------
    One row per (possibly pooled) cell type, with ``n_cells_*``,
    ``percent_*``, ``representation_ratio`` (LNP+ percent / all-cells
    percent) and ``log2_representation_ratio`` columns, sorted by descending
    ``representation_ratio``.
    """
    treated = obs.loc[obs["region"].eq(treated_region)].copy()
    groups = {"LNP+ cells": treated.loc[treated["lnp_positive"]], "All cells": treated}

    raw_rows = []
    for group_name, group_df in groups.items():
        counts = (
            group_df["cell_type"]
            .value_counts(dropna=False)
            .rename_axis("cell_type")
            .reset_index(name="n_cells")
        )
        counts["percent"] = 100 * counts["n_cells"] / counts["n_cells"].sum()
        counts["group"] = group_name
        raw_rows.append(counts)
    raw_composition = pd.concat(raw_rows, ignore_index=True)

    maximum_percent = raw_composition.groupby("cell_type", observed=False)["percent"].max()
    displayed_cell_types = maximum_percent.loc[maximum_percent >= min_percent].index
    raw_composition["plot_cell_type"] = raw_composition["cell_type"].where(
        raw_composition["cell_type"].isin(displayed_cell_types), "Other"
    )

    composition_rows = []
    for group_name, counts in raw_composition.groupby("group", sort=False, observed=False):
        collapsed = counts.groupby("plot_cell_type", observed=False)["n_cells"].sum().reset_index()
        collapsed["percent"] = 100 * collapsed["n_cells"] / collapsed["n_cells"].sum()
        collapsed["group"] = group_name
        composition_rows.append(collapsed)
    composition = pd.concat(composition_rows, ignore_index=True)

    comparison = composition.pivot(
        index="plot_cell_type", columns="group", values=["n_cells", "percent"]
    )
    comparison.columns = [
        f"{measure}_{group.lower().replace('+', 'positive').replace(' ', '_')}"
        for measure, group in comparison.columns
    ]
    comparison = comparison.reset_index()
    comparison["representation_ratio"] = (
        comparison["percent_lnppositive_cells"] / comparison["percent_all_cells"]
    )
    comparison["log2_representation_ratio"] = np.log2(comparison["representation_ratio"])
    return comparison.sort_values("representation_ratio", ascending=False).reset_index(drop=True)
