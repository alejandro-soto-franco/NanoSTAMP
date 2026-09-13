"""Figure 2 / Supplementary Figures 6-8: multiplex LNP cell and functional analysis.

Ported from
``Code/Figure_2_and_Supplementary_6_8_Multiplex_LNP_Cell_and_Functional_Analysis.ipynb``.
Merges the per-batch cell-segmentation annotation (AnnData) with the per-cell
full-barcode LNP decoding table (CSV) for 16 LNP-treated tissue regions plus
one OVA reference region, applies the raw-to-corrected LNP-label swap, and
computes region- and cell-type-level LNP uptake, LNP-positive cell-type
composition and enrichment, luciferase/OVA expression gates, and
SIINFEKL-H-2Kb/CD86 dendritic-cell functional readouts.

Scope note: this module ports the notebook's tables (Sections 1-8), which are
the analysis's numeric ground truth. Section 9 (a spatial overlay figure for
three selected regions) and Section 10 (plot-only re-derivations of tables
already computed here) add no new computation over what is implemented below
and are treated as workflow/plotting concerns rather than library functions;
see the README "Differences from upstream" section.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats

REFERENCE_REGIONS: list[str] = ["S2_reg004"]
LNP_REGIONS: list[str] = [
    f"S{slide}_reg{region:03d}" for slide in range(1, 5) for region in range(4)
]
ANALYSIS_REGIONS: list[str] = LNP_REGIONS + REFERENCE_REGIONS
LNP_ORDER: list[str] = [f"LNP_{i:02d}" for i in range(1, 11)]
TISSUE_AVERAGE_LABEL = "Tissue_average"
GENE_MARKERS: dict[str, str] = {"Luc": "Fluc", "OVA": "OVA"}
REFERENCE_GATE_QUANTILE = 0.95
SIINFEKL_COLUMN = "SIINFEKL_H-2Kb"
CD86_COLUMN = "CD86"
SIINFEKL_GATE_QUANTILE = 0.9

#: (lnp_call, source_raw_lnp_call, ionizable_lipid, helper_lipid, formulation).
#: Only LNP_01/LNP_02 differ between the raw and corrected label.
_LNP_DETAIL_ROWS: list[tuple[str, str, str, str]] = [
    ("LNP_02", "SM-102", "DSPC", "SM-102 + DSPC"),
    ("LNP_01", "SM-102", "DOTAP", "SM-102 + DOTAP"),
    ("LNP_03", "SM-102", "DDAB", "SM-102 + DDAB"),
    ("LNP_04", "SM-102", "DOPE", "SM-102 + DOPE"),
    ("LNP_05", "SM-102", "18PG", "SM-102 + 18PG"),
    ("LNP_06", "ALC-0315", "DSPC", "ALC-0315 + DSPC"),
    ("LNP_07", "ALC-0315", "DOTAP", "ALC-0315 + DOTAP"),
    ("LNP_08", "ALC-0315", "DDAB", "ALC-0315 + DDAB"),
    ("LNP_09", "ALC-0315", "DOPE", "ALC-0315 + DOPE"),
    ("LNP_10", "ALC-0315", "18PG", "ALC-0315 + 18PG"),
]
LNP_SWAP_NOTE = (
    "Display labels swap raw LNP_01 and raw LNP_02: corrected LNP_01 is raw "
    "LNP_02 (DSPC), corrected LNP_02 is raw LNP_01 (DOTAP)."
)


def build_lnp_detail_table() -> tuple[pd.DataFrame, dict[str, str]]:
    """Return the hardcoded LNP-identity metadata table and its raw->corrected map.

    Returns
    -------
    ``(table, raw_to_corrected)`` where ``table`` has one row per
    :data:`LNP_ORDER` entry with columns ``lnp_number, lnp_call,
    source_raw_lnp_call, ionizable_lipid, helper_lipid, formulation``, and
    ``raw_to_corrected`` maps each ``source_raw_lnp_call`` to its corrected
    ``lnp_call``.
    """
    rows = []
    for lnp_number, (lnp_call, ionizable_lipid, helper_lipid, formulation) in enumerate(
        _LNP_DETAIL_ROWS, start=1
    ):
        source_raw = (
            "LNP_02" if lnp_call == "LNP_01" else "LNP_01" if lnp_call == "LNP_02" else lnp_call
        )
        rows.append(
            {
                "lnp_number": lnp_number,
                "lnp_call": lnp_call,
                "source_raw_lnp_call": source_raw,
                "ionizable_lipid": ionizable_lipid,
                "helper_lipid": helper_lipid,
                "formulation": formulation,
            }
        )
    table = pd.DataFrame(rows)
    raw_to_corrected = dict(zip(table["source_raw_lnp_call"], table["lnp_call"], strict=True))
    return table, raw_to_corrected


def sem(values: Iterable[float]) -> float:
    """Sample standard error of the mean.

    Returns ``0.0`` (not ``NaN``) when fewer than 2 valid values remain after
    coercion and NaN-drop, matching every SEM column in the source notebook.
    """
    series = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    if len(series) <= 1:
        return 0.0
    return float(series.std(ddof=1) / np.sqrt(len(series)))


def reference_quantile_gate(values: Iterable[float], quantile: float) -> float:
    """Quantile of ``values`` after numeric coercion and NaN-drop.

    Returns ``NaN`` if no valid values remain.
    """
    series = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    if series.empty:
        return float("nan")
    return float(series.quantile(quantile))


def load_batches(
    batch_inputs: Mapping[str, Mapping[str, Path]],
    analysis_regions: list[str] = ANALYSIS_REGIONS,
) -> tuple[ad.AnnData, pd.DataFrame, pd.DataFrame]:
    """Load and concatenate the per-batch annotation, spot-cell and spot-summary inputs.

    Parameters
    ----------
    batch_inputs
        ``{batch_key: {"annotation": path, "spot_cell": path, "spot_summary": path}}``.
    analysis_regions
        Region codes to retain in the spot tables.

    Returns
    -------
    ``(adata, spot, spot_summary)``.
    """
    annotations = []
    spot_frames = []
    summary_frames = []
    for batch_key, paths in batch_inputs.items():
        annotations.append(ad.read_h5ad(paths["annotation"]))
        spot = pd.read_csv(paths["spot_cell"])
        spot["spot_batch"] = batch_key
        spot_frames.append(spot)
        summary = pd.read_csv(paths["spot_summary"])
        summary["spot_batch"] = batch_key
        summary_frames.append(summary)

    adata = ad.concat(
        annotations,
        join="outer",
        label="annotation_batch",
        index_unique="__",
        fill_value=np.nan,
        merge="same",
    )
    spot = pd.concat(spot_frames, ignore_index=True)
    spot = spot.loc[spot["region"].isin(analysis_regions)].copy()
    spot_summary = pd.concat(summary_frames, ignore_index=True)
    if "region" in spot_summary.columns:
        spot_summary = spot_summary.loc[spot_summary["region"].isin(analysis_regions)].copy()
    return adata, spot, spot_summary


#: The 37 fluorescence/marker intensity columns shared between the annotation
#: h5ad and the spot table (see the porting spec, Section 2.1/2.2).
MARKER_COLS: list[str] = [
    "DAPI",
    "Podoplanin",
    "aSMA",
    "CD4",
    "CD31",
    "SCA1",
    "CD152",
    "CD45.2",
    "Ly6C",
    "PD1",
    "CD3",
    "CD11c",
    "CD19",
    "F480",
    "CD138",
    "CD169",
    "CD62L",
    "TCRb",
    "GZMB",
    "FOXP3",
    "CD27",
    "CD8a",
    "CD11b",
    "Fluc",
    "MHCII",
    "CCR7",
    "CD86",
    "CD28",
    "B220",
    "CD103",
    "CD25",
    "MPO",
    "CD90",
    "SIINFEKL_H-2Kb",
    "CD44",
    "OVA",
    "NKp46",
]

#: The 15 codebook barcode marker names (Round 2's 12+3-channel panel).
LNP_MARKER_COLS: list[str] = [
    "A6",
    "A17",
    "A55",
    "A56",
    "A76",
    "A79",
    "A20",
    "A46",
    "A28",
    "A72",
    "A2",
    "A63",
    "A126",
    "A127",
    "A128",
]

#: Fixed spot-table columns always selected (Section 2.2 of the notebook spec).
SPOT_BASE_COLUMNS: list[str] = [
    "region",
    "cell",
    "barcode",
    "lnp_call",
    "lnp_positive",
    "barcode_in_library",
    "barcode_match_distance",
    "barcode_match_status",
    "barcode_excluded",
    "total_barcode_spots",
    "decoded_exact_spots",
    "decoded_tolerant_spots",
    "n_positive_bits",
    "dominant_decoded_barcode",
    "dominant_decoded_barcode_count",
    "dominant_barcode_marker",
    "dominant_barcode_count",
    "dominant_barcode_fraction",
    "barcode_confidence",
    "decoded_spots",
]


def merge_cell_annotations_with_lnp_calls(
    adata: ad.AnnData,
    spot: pd.DataFrame,
    marker_cols: list[str],
    lnp_marker_cols: list[str],
    lnp_detail_table: pd.DataFrame,
    raw_to_corrected: dict[str, str],
    analysis_regions: list[str] = ANALYSIS_REGIONS,
    reference_regions: list[str] = REFERENCE_REGIONS,
) -> tuple[ad.AnnData, dict[str, object]]:
    """Merge per-cell annotations with LNP-barcode calls (Section 3 of the notebook).

    Parameters
    ----------
    adata
        Concatenated per-batch annotation, with ``obs`` columns ``label``,
        ``slide_name``, ``region``, ``cell_type``, plus ``marker_cols``.
    spot
        Concatenated per-batch spot/barcode table.
    marker_cols, lnp_marker_cols
        Fluorescence marker names, and codebook barcode marker names, to
        carry across from ``spot`` (renamed ``spot_mean_<name>`` where they
        collide with an existing ``obs`` column).
    lnp_detail_table, raw_to_corrected
        Output of :func:`build_lnp_detail_table`.
    analysis_regions, reference_regions
        Region codes that define, respectively, the full analysis population
        and the OVA/Luc reference population.

    Returns
    -------
    ``(adata, merge_report)``: the merged AnnData, with ``.obs`` gaining
    ``cell``, ``lnp_region``, ``condition``, ``region_group``, corrected
    ``lnp_call``, ``lnp_call_raw``, ``lnp_call_corrected``, ``lnp_positive``,
    ``lnp_call_positive_only``, ``lnp_detail_*`` columns and every carried-over
    spot column; and a report dict with ``n_retained``, ``merge_counts``
    (``_merge`` indicator value counts) and ``extra_spot_rows`` for logging.

    Raises
    ------
    ValueError
        If the retained regions do not exactly equal ``analysis_regions``, or
        any annotated cell fails to find a matching spot row.
    """
    obs = adata.obs.copy()
    obs["_obs_name"] = obs.index.astype(str)
    obs["cell"] = obs["label"].astype(int)
    obs["lnp_region"] = (
        obs["slide_name"].astype(str).str.replace(r"^.*_registered_", "", regex=True)
        + "_"
        + obs["region"].astype(str)
    )

    keep_mask = obs["lnp_region"].isin(analysis_regions)
    # pyrefly: ignore [missing-attribute]
    obs = obs.loc[keep_mask].copy()
    adata = adata[keep_mask.to_numpy()].copy()

    observed = set(obs["lnp_region"].unique())
    expected = set(analysis_regions)
    if observed != expected:
        raise ValueError(
            f"lnp_region mismatch: missing {expected - observed}, unexpected {observed - expected}"
        )

    spot_cols = list(SPOT_BASE_COLUMNS)
    for name in (
        marker_cols
        + lnp_marker_cols
        + [f"spot_{m}" for m in lnp_marker_cols]
        + [f"bit_{m}" for m in lnp_marker_cols]
    ):
        if name in spot.columns and name not in spot_cols:
            spot_cols.append(name)
    spot_merge = spot[spot_cols].rename(columns={"region": "lnp_region"})

    overlap = [
        name
        for name in (marker_cols + lnp_marker_cols)
        if name in obs.columns and name in spot_merge.columns
    ]
    spot_merge = spot_merge.rename(columns={name: f"spot_mean_{name}" for name in overlap})

    merged_obs = obs.merge(
        spot_merge, on=["lnp_region", "cell"], how="left", validate="one_to_one", indicator=True
    )
    merge_counts = merged_obs["_merge"].value_counts(dropna=False).to_dict()

    reverse_merge = spot_merge.merge(
        obs[["lnp_region", "cell"]], on=["lnp_region", "cell"], how="left", indicator=True
    )
    extra_spot_rows = int((reverse_merge["_merge"] == "left_only").sum())

    if (merged_obs["_merge"] != "both").any():
        raise ValueError("Every annotated cell must find a matching spot row")
    merged_obs = merged_obs.drop(columns="_merge")

    merged_obs = merged_obs.set_index("_obs_name")
    merged_obs.index.name = None
    merged_obs = merged_obs.reindex(adata.obs_names)

    merged_obs["condition"] = np.where(
        merged_obs["lnp_region"].isin(reference_regions), "reference", "LNP"
    )
    merged_obs["region_group"] = pd.Categorical(
        merged_obs["lnp_region"], categories=analysis_regions, ordered=True
    )

    merged_obs["lnp_call"] = merged_obs["lnp_call"].fillna("no_barcode").astype(str)
    merged_obs["lnp_call_raw"] = merged_obs["lnp_call"]
    merged_obs["lnp_call_corrected"] = (
        merged_obs["lnp_call_raw"].map(raw_to_corrected).fillna(merged_obs["lnp_call_raw"])
    )
    merged_obs["lnp_call"] = merged_obs["lnp_call_corrected"]
    merged_obs["lnp_positive"] = merged_obs["lnp_positive"].fillna(False).astype(bool)
    merged_obs["lnp_call_positive_only"] = np.where(
        merged_obs["lnp_positive"], merged_obs["lnp_call_corrected"], "no_barcode"
    )

    detail_prefixed = lnp_detail_table.add_prefix("lnp_detail_")
    merged_obs = merged_obs.merge(
        detail_prefixed,
        left_on="lnp_call_corrected",
        right_on="lnp_detail_lnp_call",
        how="left",
    )
    merged_obs.index = adata.obs_names
    merged_obs = merged_obs.reindex(adata.obs_names)

    adata.obs = merged_obs
    adata.uns["spot_cell_annotation_merge"] = {
        "reference_regions": list(reference_regions),
        "lnp_regions": [r for r in analysis_regions if r not in reference_regions],
        # anndata's h5ad writer cannot serialise a list of heterogeneous
        # dicts under .uns; store it as a JSON string instead (matching the
        # porting spec's own description of this field as "JSON records").
        "lnp_detail_table_json": json.dumps(lnp_detail_table.to_dict(orient="records")),
        "lnp_swap_note": LNP_SWAP_NOTE,
        "extra_spot_rows_without_final_annotation": extra_spot_rows,
    }
    report = {
        "n_retained": int(keep_mask.sum()),
        "merge_counts": merge_counts,
        "extra_spot_rows": extra_spot_rows,
    }
    return adata, report


def compute_region_level_lnp_uptake(
    obs: pd.DataFrame,
    lnp_order: list[str] = LNP_ORDER,
    lnp_regions: list[str] = LNP_REGIONS,
    reference_regions: list[str] = REFERENCE_REGIONS,
    lnp_detail_table: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-region and pooled LNP-positive uptake percentages (Section 4).

    Returns
    -------
    ``(uptake_by_region, uptake_summary)``.
    """
    if lnp_detail_table is None:
        lnp_detail_table, _ = build_lnp_detail_table()

    obs = obs.loc[obs["lnp_region"].isin(lnp_regions + reference_regions)]
    region_totals = (
        obs.groupby(["condition", "lnp_region"], observed=False).size().rename("n_cells_total")
    )
    positive = obs.loc[obs["lnp_positive"] & obs["lnp_call"].isin(lnp_order)]
    positive_counts = (
        positive.groupby(["condition", "lnp_region", "lnp_call"], observed=False)
        .size()
        .rename("n_lnp_positive")
    )

    pairs = [("LNP", region) for region in lnp_regions] + [
        ("reference", region) for region in reference_regions
    ]
    index = pd.MultiIndex.from_tuples(
        [(condition, region, call) for condition, region in pairs for call in lnp_order],
        names=["condition", "lnp_region", "lnp_call"],
    )
    complete = pd.DataFrame(index=index).reset_index()
    complete = complete.merge(
        positive_counts, on=["condition", "lnp_region", "lnp_call"], how="left"
    )
    complete = complete.merge(region_totals, on=["condition", "lnp_region"], how="left")
    complete["n_lnp_positive"] = complete["n_lnp_positive"].fillna(0).astype(int)
    complete["pct_cells_lnp_positive"] = (
        100 * complete["n_lnp_positive"] / complete["n_cells_total"]
    )
    complete["lnp_call"] = pd.Categorical(complete["lnp_call"], categories=lnp_order, ordered=True)
    complete = complete.merge(lnp_detail_table, on="lnp_call", how="left")
    uptake_by_region = complete.sort_values(["condition", "lnp_call", "lnp_region"]).reset_index(
        drop=True
    )

    uptake_summary = (
        uptake_by_region.groupby(["condition", "lnp_call"], observed=False)
        .agg(
            n_regions=("lnp_region", "nunique"),
            mean_pct_cells_lnp_positive=("pct_cells_lnp_positive", "mean"),
            sem_pct_cells_lnp_positive=("pct_cells_lnp_positive", sem),
            total_positive_cells=("n_lnp_positive", "sum"),
            total_cells=("n_cells_total", "sum"),
        )
        .reset_index()
        .merge(lnp_detail_table, on="lnp_call", how="left")
    )
    uptake_summary["pooled_pct_cells_lnp_positive"] = (
        100 * uptake_summary["total_positive_cells"] / uptake_summary["total_cells"]
    )
    return uptake_by_region, uptake_summary


def compute_celltype_lnp_uptake(
    analysis_obs: pd.DataFrame,
    lnp_order: list[str] = LNP_ORDER,
    lnp_detail_table: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Index]:
    """Compute cell-type-level LNP uptake percentages (Section 4, part 2).

    Returns
    -------
    ``(celltype_lnp_uptake, celltype_lnp_summary, top_celltypes)``. Note that,
    matching the source notebook, ``celltype_lnp_uptake`` is built from the
    observed-positive rows only, not a complete zero-filled grid.
    """
    if lnp_detail_table is None:
        lnp_detail_table, _ = build_lnp_detail_table()

    lnp_obs = analysis_obs.loc[analysis_obs["condition"] == "LNP"]
    celltype_totals = (
        lnp_obs.groupby(["lnp_region", "cell_type"], observed=False)
        .size()
        .rename("n_cell_type_total")
    )
    celltype_positive = (
        lnp_obs.loc[lnp_obs["lnp_positive"] & lnp_obs["lnp_call"].isin(lnp_order)]
        .groupby(["lnp_region", "cell_type", "lnp_call"], observed=False)
        .size()
        .rename("n_lnp_positive")
        .reset_index()
    )
    celltype_lnp_uptake = celltype_positive.merge(
        celltype_totals, on=["lnp_region", "cell_type"], how="left"
    )
    celltype_lnp_uptake["pct_cell_type_lnp_positive"] = (
        100 * celltype_lnp_uptake["n_lnp_positive"] / celltype_lnp_uptake["n_cell_type_total"]
    )

    celltype_lnp_summary = (
        celltype_lnp_uptake.groupby(["cell_type", "lnp_call"], observed=False)
        .agg(
            n_regions=("lnp_region", "nunique"),
            mean_pct_cell_type_lnp_positive=("pct_cell_type_lnp_positive", "mean"),
            sem_pct_cell_type_lnp_positive=("pct_cell_type_lnp_positive", sem),
            total_positive_cells=("n_lnp_positive", "sum"),
            median_region_pct=("pct_cell_type_lnp_positive", "median"),
        )
        .reset_index()
        .merge(lnp_detail_table, on="lnp_call", how="left")
        .sort_values(["lnp_call", "mean_pct_cell_type_lnp_positive"], ascending=[True, False])
    )
    top_celltypes = (
        celltype_lnp_summary.groupby("cell_type")["total_positive_cells"]
        .sum()
        .sort_values(ascending=False)
        .head(12)
        .index
    )
    return celltype_lnp_uptake, celltype_lnp_summary, top_celltypes


def compute_lnp_positive_celltype_composition(
    analysis_obs: pd.DataFrame,
    lnp_order: list[str] = LNP_ORDER,
    lnp_detail_table: pd.DataFrame | None = None,
    tissue_average_label: str = TISSUE_AVERAGE_LABEL,
) -> pd.DataFrame:
    """Compute LNP-positive cell-type composition, plus a tissue-average baseline row (Section 5).

    Returns
    -------
    Long-form table with ``lnp_call, cell_type, n_lnp_positive,
    n_lnp_positive_total, pct_of_lnp_positive_cells, composition_basis`` (plus
    merged LNP-detail columns).
    """
    if lnp_detail_table is None:
        lnp_detail_table, _ = build_lnp_detail_table()

    lnp_obs = analysis_obs.loc[analysis_obs["condition"] == "LNP"]
    positive_cells = lnp_obs.loc[lnp_obs["lnp_positive"] & lnp_obs["lnp_call"].isin(lnp_order)]
    composition_counts = (
        positive_cells.groupby(["lnp_call", "cell_type"], observed=False)
        .size()
        .rename("n_lnp_positive")
    )
    composition_celltypes = (
        composition_counts.groupby("cell_type", observed=False)
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .index
    )

    tissue_average_counts = (
        lnp_obs["cell_type"]
        .value_counts()
        .reindex(composition_celltypes)
        .rename("n_lnp_positive")
        .reset_index()
        .rename(columns={"index": "cell_type"})
    )
    tissue_average_counts["lnp_call"] = tissue_average_label

    composition_counts = composition_counts.reset_index()
    combined = pd.concat([composition_counts, tissue_average_counts], ignore_index=True)

    plot_order = lnp_order + [tissue_average_label]
    grid = pd.MultiIndex.from_product(
        [plot_order, composition_celltypes], names=["lnp_call", "cell_type"]
    ).to_frame(index=False)
    result = grid.merge(combined, on=["lnp_call", "cell_type"], how="left")
    result["n_lnp_positive"] = result["n_lnp_positive"].fillna(0).astype(int)
    result = result.merge(lnp_detail_table, on="lnp_call", how="left")
    result["n_lnp_positive_total"] = result.groupby("lnp_call")["n_lnp_positive"].transform("sum")
    result["pct_of_lnp_positive_cells"] = np.where(
        result["n_lnp_positive_total"] > 0,
        100 * result["n_lnp_positive"] / result["n_lnp_positive_total"],
        np.nan,
    )
    is_tissue_average = result["lnp_call"] == tissue_average_label
    result["composition_basis"] = np.where(
        is_tissue_average,
        "All annotated cells in 16 LNP-treated tissue regions",
        "Codebook-matched LNP+ cells",
    )
    result.loc[is_tissue_average, "formulation"] = "All cells (tissue average)"
    return result


def compute_celltype_relative_enrichment(
    analysis_obs: pd.DataFrame,
    lnp_order: list[str] = LNP_ORDER,
    lnp_detail_table: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute LNP-positive cell-type enrichment relative to background composition (Section 5, part 2).

    Returns
    -------
    ``(relative_celltype_bias, celltype_lnp_relative, best_lnp_by_celltype)``.
    """
    if lnp_detail_table is None:
        lnp_detail_table, _ = build_lnp_detail_table()

    lnp_obs = analysis_obs.loc[analysis_obs["condition"] == "LNP"]
    baseline_celltype = (
        lnp_obs.groupby("cell_type", observed=False).size().rename("n_cells_background")
    )
    background_fraction = baseline_celltype / baseline_celltype.sum()

    positive_celltype = (
        lnp_obs.loc[lnp_obs["lnp_positive"] & lnp_obs["lnp_call"].isin(lnp_order)]
        .groupby(["lnp_call", "cell_type"], observed=False)
        .size()
        .rename("n_lnp_positive")
        .reset_index()
    )

    grid = pd.MultiIndex.from_product(
        [lnp_order, baseline_celltype.index], names=["lnp_call", "cell_type"]
    ).to_frame(index=False)
    result = grid.merge(positive_celltype, on=["lnp_call", "cell_type"], how="left")
    result["n_lnp_positive"] = result["n_lnp_positive"].fillna(0).astype(int)
    result = result.merge(
        baseline_celltype.rename("n_cells_background"), on="cell_type", how="left"
    )
    result = result.merge(
        background_fraction.rename("background_fraction"), on="cell_type", how="left"
    )
    result = result.merge(lnp_detail_table, on="lnp_call", how="left")

    result["n_lnp_positive_total"] = result.groupby("lnp_call")["n_lnp_positive"].transform("sum")
    result["positive_fraction_within_lnp"] = result["n_lnp_positive"] / result[
        "n_lnp_positive_total"
    ].replace(0, np.nan)
    result["enrichment_vs_background"] = result["positive_fraction_within_lnp"] / result[
        "background_fraction"
    ].replace(0, np.nan)
    result["log2_enrichment_vs_background"] = np.log2(
        result["enrichment_vs_background"].replace(0, np.nan)
    )
    relative_celltype_bias = result

    celltype_lnp_relative = relative_celltype_bias.copy()
    celltype_lnp_relative["max_enrichment_for_cell_type"] = celltype_lnp_relative.groupby(
        "cell_type"
    )["enrichment_vs_background"].transform("max")
    celltype_lnp_relative["relative_enrichment_within_cell_type"] = celltype_lnp_relative[
        "enrichment_vs_background"
    ] / celltype_lnp_relative["max_enrichment_for_cell_type"].replace(0, np.nan)

    best_lnp_by_celltype = (
        relative_celltype_bias.sort_values(
            ["cell_type", "enrichment_vs_background"], ascending=[True, False]
        )
        .groupby("cell_type", observed=False)
        .head(1)[
            [
                "cell_type",
                "lnp_call",
                "formulation",
                "helper_lipid",
                "ionizable_lipid",
                "n_lnp_positive",
                "positive_fraction_within_lnp",
                "background_fraction",
                "enrichment_vs_background",
                "log2_enrichment_vs_background",
            ]
        ]
        .sort_values("enrichment_vs_background", ascending=False)
        .reset_index(drop=True)
    )
    return relative_celltype_bias, celltype_lnp_relative, best_lnp_by_celltype


def compute_gene_expression_gates(
    obs: pd.DataFrame,
    gene_markers: dict[str, str] = GENE_MARKERS,
    reference_regions: list[str] = REFERENCE_REGIONS,
    quantile: float = REFERENCE_GATE_QUANTILE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Gate Luc/OVA positivity from a reference-region expression quantile (Section 6).

    Returns
    -------
    ``(expression_obs, gene_gate_table)``: ``obs`` with a ``<marker>_positive``
    boolean column per entry in ``gene_markers`` plus ``Luc_OVA_double_positive``,
    and the gate metadata table.
    """
    expression_obs = obs.copy()
    gate_rows = []
    for marker_name, column in gene_markers.items():
        reference_values = expression_obs.loc[
            expression_obs["lnp_region"].isin(reference_regions), column
        ]
        threshold = reference_quantile_gate(reference_values, quantile)
        expression_obs[f"{marker_name}_positive"] = expression_obs[column] > threshold
        numeric_reference = pd.to_numeric(reference_values, errors="coerce").dropna()
        gate_rows.append(
            {
                "marker": marker_name,
                "column": column,
                "gate_reference_regions": ",".join(reference_regions),
                "threshold": threshold,
                "reference_median": float(numeric_reference.median())
                if len(numeric_reference)
                else float("nan"),
                "reference_gate_quantile": quantile,
                "reference_gate_value": threshold,
                "reference_n_cells": int(len(numeric_reference)),
            }
        )
    expression_obs["Luc_OVA_double_positive"] = (
        expression_obs["Luc_positive"] & expression_obs["OVA_positive"]
    )
    return expression_obs, pd.DataFrame(gate_rows)


def summarize_gene_group(df: pd.DataFrame, group_cols: list[str], label: str) -> pd.DataFrame:
    """Aggregate Luc/OVA positivity counts and percentages over ``group_cols`` (Section 6)."""
    summary = df.groupby(group_cols, observed=False).agg(
        n_cells=("lnp_positive", "size"),
        n_lnp_positive=("lnp_positive", "sum"),
        n_luc_positive=("Luc_positive", "sum"),
        n_ova_positive=("OVA_positive", "sum"),
        n_luc_ova_double_positive=("Luc_OVA_double_positive", "sum"),
        mean_luc=("Fluc", "mean"),
        mean_ova=("OVA", "mean"),
        median_luc=("Fluc", "median"),
        median_ova=("OVA", "median"),
    )
    summary["analysis_population"] = label
    for flag_col, out_col in [
        ("n_luc_positive", "pct_luc_positive"),
        ("n_ova_positive", "pct_ova_positive"),
        ("n_luc_ova_double_positive", "pct_luc_ova_double_positive"),
    ]:
        summary[out_col] = 100 * summary[flag_col] / summary["n_cells"].replace(0, np.nan)
    return summary.reset_index()


def compute_gene_region_summaries(
    expression_obs: pd.DataFrame,
    lnp_order: list[str] = LNP_ORDER,
    lnp_detail_table: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize Luc/OVA positivity by region, for all cells and for LNP-positive cells only (Section 6)."""
    if lnp_detail_table is None:
        lnp_detail_table, _ = build_lnp_detail_table()

    group_cols = ["condition", "lnp_region", "lnp_call"]
    all_cells = summarize_gene_group(expression_obs, group_cols, "all_cells")
    lnp_positive_only = summarize_gene_group(
        expression_obs.loc[
            expression_obs["lnp_positive"] & expression_obs["lnp_call"].isin(lnp_order)
        ],
        group_cols,
        "lnp_positive_cells",
    )
    gene_region_summary = pd.concat([all_cells, lnp_positive_only], ignore_index=True).merge(
        lnp_detail_table, on="lnp_call", how="left"
    )

    lnp_condition_only = lnp_positive_only.loc[lnp_positive_only["condition"] == "LNP"]
    gene_lnp_positive_summary = (
        lnp_condition_only.groupby("lnp_call", observed=False)
        .agg(
            n_regions=("lnp_region", "nunique"),
            mean_pct_luc_positive=("pct_luc_positive", "mean"),
            sem_pct_luc_positive=("pct_luc_positive", sem),
            mean_pct_ova_positive=("pct_ova_positive", "mean"),
            sem_pct_ova_positive=("pct_ova_positive", sem),
            mean_pct_luc_ova_double_positive=("pct_luc_ova_double_positive", "mean"),
            sem_pct_luc_ova_double_positive=("pct_luc_ova_double_positive", sem),
            total_lnp_positive_cells=("n_cells", "sum"),
            total_luc_positive=("n_luc_positive", "sum"),
            total_ova_positive=("n_ova_positive", "sum"),
        )
        .reset_index()
        .merge(lnp_detail_table, on="lnp_call", how="left")
    )
    return gene_region_summary, gene_lnp_positive_summary


def compute_pooled_gene_lnp_label_composition(
    expression_obs: pd.DataFrame, lnp_order: list[str] = LNP_ORDER
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute all-cells Luc+/OVA+ composition by LNP label (Section 6)."""
    pooled = expression_obs.loc[expression_obs["condition"] == "LNP"].copy()
    pooled["lnp_label_for_gene_positive"] = np.where(
        pooled["lnp_positive"] & pooled["lnp_call"].isin(lnp_order),
        pooled["lnp_call"],
        "no_lnp_signal",
    )
    labels = [*lnp_order, "no_lnp_signal"]

    composition_rows = []
    summary_rows = []
    for marker, flag_col in [("Luc", "Luc_positive"), ("OVA", "OVA_positive")]:
        marker_positive = pooled.loc[pooled[flag_col]]
        counts = (
            marker_positive["lnp_label_for_gene_positive"]
            .value_counts()
            .reindex(labels, fill_value=0)
        )
        n_total = int(counts.sum())
        n_lnp_labeled = int(counts.reindex(lnp_order, fill_value=0).sum())
        pct_lnp_labeled = 100 * n_lnp_labeled / n_total if n_total else float("nan")
        for label, count in counts.items():
            composition_rows.append(
                {
                    "marker": marker,
                    "lnp_label": label,
                    "n_positive_cells": int(count),
                    "pct_of_marker_positive_cells": 100 * count / n_total
                    if n_total
                    else float("nan"),
                }
            )
        summary_rows.append(
            {
                "marker": marker,
                "n_positive_cells": n_total,
                "n_lnp_labeled": n_lnp_labeled,
                "pct_lnp_labeled": pct_lnp_labeled,
                "n_no_lnp_signal": int(counts["no_lnp_signal"]),
            }
        )
    return pd.DataFrame(composition_rows), pd.DataFrame(summary_rows)


def compute_gene_positive_celltype_and_lnp_label_composition(
    expression_obs: pd.DataFrame,
    lnp_order: list[str] = LNP_ORDER,
    selected_celltypes: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute Luc+/OVA+ cell-type composition and, for selected cell types, LNP-label composition (Section 6)."""
    if selected_celltypes is None:
        selected_celltypes = ["DC", "CD8+ T", "CD4+ T", "B", "Macrophage"]

    gene_expr = expression_obs.loc[expression_obs["condition"] == "LNP"].copy()
    gene_expr["lnp_label_for_gene_positive"] = np.where(
        gene_expr["lnp_positive"] & gene_expr["lnp_call"].isin(lnp_order),
        gene_expr["lnp_call"],
        "no_lnp_signal",
    )
    present = [ct for ct in selected_celltypes if ct in set(gene_expr["cell_type"])]
    labels = [*lnp_order, "no_lnp_signal"]

    celltype_rows = []
    label_rows = []
    for marker, flag_col in [("Luc", "Luc_positive"), ("OVA", "OVA_positive")]:
        marker_positive = gene_expr.loc[gene_expr[flag_col]]
        marker_total = max(len(marker_positive), 1) if len(marker_positive) else 0
        celltype_counts = (
            marker_positive.groupby("cell_type", observed=False).size().rename("n_positive_cells")
        )
        for cell_type, count in celltype_counts.items():
            celltype_rows.append(
                {
                    "cell_type": cell_type,
                    "n_positive_cells": int(count),
                    "marker": marker,
                    "total_positive_cells": len(marker_positive),
                    "pct_positive_cells": 100 * count / marker_total
                    if marker_total
                    else float("nan"),
                }
            )

        restricted = marker_positive.loc[marker_positive["cell_type"].isin(present)]
        lnp_counts = (
            restricted.groupby(["cell_type", "lnp_label_for_gene_positive"], observed=False)
            .size()
            .rename("n_positive_cells")
            .reset_index()
        )
        grid = pd.MultiIndex.from_product(
            [present, labels], names=["cell_type", "lnp_label_for_gene_positive"]
        ).to_frame(index=False)
        completed = grid.merge(
            lnp_counts, on=["cell_type", "lnp_label_for_gene_positive"], how="left"
        )
        completed["n_positive_cells"] = completed["n_positive_cells"].fillna(0).astype(int)
        completed["marker"] = marker
        completed["total_positive_cells_in_cell_type"] = completed.groupby("cell_type")[
            "n_positive_cells"
        ].transform("sum")
        completed["pct_positive_cells_in_cell_type"] = np.where(
            completed["total_positive_cells_in_cell_type"] > 0,
            100 * completed["n_positive_cells"] / completed["total_positive_cells_in_cell_type"],
            np.nan,
        )
        label_rows.append(completed)

    return pd.DataFrame(celltype_rows), pd.concat(label_rows, ignore_index=True)


def compute_siinfekl_gate_and_dc_summary(
    expression_obs: pd.DataFrame,
    lnp_order: list[str] = LNP_ORDER,
    lnp_detail_table: pd.DataFrame | None = None,
    siinfekl_column: str = SIINFEKL_COLUMN,
    cd86_column: str = CD86_COLUMN,
    reference_regions: list[str] = REFERENCE_REGIONS,
    quantile: float = SIINFEKL_GATE_QUANTILE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Gate SIINFEKL-H-2Kb positivity and summarize CD86 in SIINFEKL+ LNP-positive DCs (Section 7).

    Returns
    -------
    ``(gate_table, dc_siin_ova_by_region, dc_siin_ova_summary, siinfekl_lnp_composition)``.
    """
    if lnp_detail_table is None:
        lnp_detail_table, _ = build_lnp_detail_table()

    siin_values = pd.to_numeric(
        expression_obs.loc[expression_obs["lnp_region"].isin(reference_regions), siinfekl_column],
        errors="coerce",
    ).dropna()
    threshold = reference_quantile_gate(siin_values, quantile)
    gate_table = pd.DataFrame(
        [
            {
                "marker": "SIINFEKL-H-2Kb",
                "column": siinfekl_column,
                "threshold": threshold,
                "gate_quantile": quantile,
                "gate_reference_regions": ",".join(reference_regions),
                "reference_median": float(siin_values.median())
                if len(siin_values)
                else float("nan"),
                "reference_n_cells": int(len(siin_values)),
            }
        ]
    )

    obs = expression_obs.copy()
    obs[siinfekl_column] = pd.to_numeric(obs[siinfekl_column], errors="coerce")
    obs[cd86_column] = pd.to_numeric(obs[cd86_column], errors="coerce")
    obs["SIINFEKL_positive"] = obs[siinfekl_column] > threshold
    obs["CD86_in_SIINFEKL_positive"] = obs[cd86_column].where(obs["SIINFEKL_positive"])

    dc_lnp_positive = obs.loc[
        (obs["condition"] == "LNP")
        & (obs["cell_type"].astype(str) == "DC")
        & obs["lnp_positive"]
        & obs["lnp_call"].isin(lnp_order)
    ]
    dc_siin_ova_by_region = dc_lnp_positive.groupby(["lnp_region", "lnp_call"], observed=False).agg(
        n_lnp_positive_dc=("lnp_call", "size"),
        n_siinfekl_positive=("SIINFEKL_positive", "sum"),
        n_ova_positive=("OVA_positive", "sum"),
        mean_siinfekl_expression=(siinfekl_column, "mean"),
        median_siinfekl_expression=(siinfekl_column, "median"),
        mean_cd86_in_siinfekl_positive=("CD86_in_SIINFEKL_positive", "mean"),
        median_cd86_in_siinfekl_positive=("CD86_in_SIINFEKL_positive", "median"),
    )
    dc_siin_ova_by_region["pct_siinfekl_positive"] = (
        100
        * dc_siin_ova_by_region["n_siinfekl_positive"]
        / dc_siin_ova_by_region["n_lnp_positive_dc"]
    )
    dc_siin_ova_by_region["pct_ova_positive"] = (
        100 * dc_siin_ova_by_region["n_ova_positive"] / dc_siin_ova_by_region["n_lnp_positive_dc"]
    )
    dc_siin_ova_by_region["siinfekl_minus_ova_pct_points"] = (
        dc_siin_ova_by_region["pct_siinfekl_positive"] - dc_siin_ova_by_region["pct_ova_positive"]
    )
    dc_siin_ova_by_region = dc_siin_ova_by_region.reset_index()

    dc_siin_ova_summary = (
        dc_siin_ova_by_region.groupby("lnp_call", observed=False)
        .agg(
            n_regions=("lnp_region", "nunique"),
            total_lnp_positive_dc=("n_lnp_positive_dc", "sum"),
            total_siinfekl_positive=("n_siinfekl_positive", "sum"),
            total_ova_positive=("n_ova_positive", "sum"),
            mean_pct_siinfekl_positive=("pct_siinfekl_positive", "mean"),
            sem_pct_siinfekl_positive=("pct_siinfekl_positive", sem),
            mean_pct_ova_positive=("pct_ova_positive", "mean"),
            sem_pct_ova_positive=("pct_ova_positive", sem),
            mean_siinfekl_expression=("mean_siinfekl_expression", "mean"),
            sem_siinfekl_expression=("mean_siinfekl_expression", sem),
            mean_cd86_in_siinfekl_positive=("mean_cd86_in_siinfekl_positive", "mean"),
            sem_cd86_in_siinfekl_positive=("mean_cd86_in_siinfekl_positive", sem),
        )
        .reindex(lnp_order)
        .reset_index()
    )
    dc_siin_ova_summary["pooled_pct_siinfekl_positive"] = (
        100
        * dc_siin_ova_summary["total_siinfekl_positive"]
        / dc_siin_ova_summary["total_lnp_positive_dc"]
    )
    dc_siin_ova_summary["pooled_pct_ova_positive"] = (
        100
        * dc_siin_ova_summary["total_ova_positive"]
        / dc_siin_ova_summary["total_lnp_positive_dc"]
    )
    dc_siin_ova_summary["mean_siinfekl_minus_ova_pct_points"] = (
        dc_siin_ova_summary["mean_pct_siinfekl_positive"]
        - dc_siin_ova_summary["mean_pct_ova_positive"]
    )
    dc_siin_ova_summary = dc_siin_ova_summary.merge(lnp_detail_table, on="lnp_call", how="left")

    siinfekl_positive_lnp_dc = dc_lnp_positive.loc[
        dc_lnp_positive.index.isin(obs.loc[obs["SIINFEKL_positive"]].index)
    ]
    siinfekl_lnp_composition = (
        siinfekl_positive_lnp_dc.groupby("lnp_call", observed=False)
        .size()
        .reindex(lnp_order, fill_value=0)
        .rename("n_siinfekl_positive_lnp_dc")
        .reset_index()
    )
    total = max(int(siinfekl_lnp_composition["n_siinfekl_positive_lnp_dc"].sum()), 1)
    siinfekl_lnp_composition["pct_siinfekl_positive_lnp_dc"] = (
        100 * siinfekl_lnp_composition["n_siinfekl_positive_lnp_dc"] / total
    )
    siinfekl_lnp_composition = siinfekl_lnp_composition.merge(
        lnp_detail_table, on="lnp_call", how="left"
    )

    return gate_table, dc_siin_ova_by_region, dc_siin_ova_summary, siinfekl_lnp_composition


def compute_region_qc_tables(
    analysis_obs: pd.DataFrame, lnp_detail_table: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute region- and LNP-call-level QC tables (Section 8)."""
    if lnp_detail_table is None:
        lnp_detail_table, _ = build_lnp_detail_table()

    region_qc = analysis_obs.groupby(["condition", "lnp_region"], observed=False).agg(
        n_cells=("lnp_call", "size"), n_any_lnp_positive=("lnp_positive", "sum")
    )
    region_qc["pct_any_lnp_positive"] = 100 * region_qc["n_any_lnp_positive"] / region_qc["n_cells"]
    region_qc = region_qc.reset_index()

    call_qc = (
        analysis_obs.groupby(["condition", "lnp_call"], observed=False)
        .agg(n_cells=("lnp_call", "size"), n_positive=("lnp_positive", "sum"))
        .reset_index()
        .merge(lnp_detail_table, on="lnp_call", how="left")
        .sort_values(["condition", "n_cells"], ascending=[True, False])
    )
    return region_qc, call_qc


def compute_bcell_uptake_vs_ova_correlation(
    uptake_composition: pd.DataFrame,
    transfection_composition: pd.DataFrame,
    fix_denominator_bug: bool = True,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Correlate B-cell uptake composition against OVA-transfection composition across LNP calls (Section 10e).

    Parameters
    ----------
    uptake_composition
        Output of :func:`compute_lnp_positive_celltype_composition`.
    transfection_composition
        The ``OVA`` rows of the second output of
        :func:`compute_gene_positive_celltype_and_lnp_label_composition`.
    fix_denominator_bug
        The source notebook computes ``uptake_composition_pct`` over every row
        with ``cell_type == 'B'``, including the synthetic
        ``'Tissue_average'`` row, inflating its denominator (flagged in the
        porting spec as a numeric bug). When ``True`` (the default), the
        denominator excludes that row, matching the 10 real LNP calls the
        correlation is computed over; when ``False``, restores the upstream
        (buggy) denominator for exact reproduction.

    Returns
    -------
    ``(comparison_table, stats)`` where ``stats`` has ``spearman_rho,
    spearman_pvalue, pearson_r, pearson_pvalue``.
    """
    uptake_b = uptake_composition.loc[uptake_composition["cell_type"] == "B"][
        ["lnp_call", "n_lnp_positive", "lnp_number", "formulation"]
    ].copy()
    if fix_denominator_bug:
        denom = uptake_b.loc[uptake_b["lnp_call"] != TISSUE_AVERAGE_LABEL, "n_lnp_positive"].sum()
    else:
        denom = uptake_b["n_lnp_positive"].sum()
    uptake_b["uptake_composition_pct"] = 100 * uptake_b["n_lnp_positive"] / denom

    trans_b = transfection_composition.loc[
        (transfection_composition["cell_type"] == "B")
        & (transfection_composition["marker"] == "OVA")
        & transfection_composition["lnp_label_for_gene_positive"].str.startswith("LNP_", na=False)
    ].rename(
        columns={
            "lnp_label_for_gene_positive": "lnp_call",
            "n_positive_cells": "n_ova_positive_lnp_b",
        }
    )[["lnp_call", "n_ova_positive_lnp_b"]]
    trans_b["ova_transfection_composition_pct"] = (
        100 * trans_b["n_ova_positive_lnp_b"] / trans_b["n_ova_positive_lnp_b"].sum()
    )

    df = uptake_b.merge(trans_b, on="lnp_call", how="inner")
    df["composition_delta_transfection_minus_uptake"] = (
        df["ova_transfection_composition_pct"] - df["uptake_composition_pct"]
    )
    df["lnp_label"] = df["lnp_call"].str.replace("_", " ")
    df = df.sort_values("ova_transfection_composition_pct").reset_index(drop=True)

    rho, spearman_p = stats.spearmanr(
        df["uptake_composition_pct"], df["ova_transfection_composition_pct"]
    )
    pearson_r, pearson_p = stats.pearsonr(
        df["uptake_composition_pct"], df["ova_transfection_composition_pct"]
    )
    df["spearman_rho"] = rho
    df["spearman_pvalue"] = spearman_p
    df["pearson_r"] = pearson_r
    df["pearson_pvalue"] = pearson_p

    return df, {
        "spearman_rho": float(rho),
        "spearman_pvalue": float(spearman_p),
        "pearson_r": float(pearson_r),
        "pearson_pvalue": float(pearson_p),
    }
