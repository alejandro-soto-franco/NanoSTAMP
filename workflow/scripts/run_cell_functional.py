"""Snakemake script: Figure 2 / Supplementary 6-8 cell and functional analysis.

Also renders Section 9's selected-region spatial overlay and Section 10's
five final replot figures as real PDF/PNG rule outputs (the source notebook
computed their data but, per its own Section 10 "possible bugs", never
actually called ``savefig`` for any of them).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nanostamp.cell_functional import (
    LNP_MARKER_COLS,
    LNP_ORDER,
    MARKER_COLS,
    build_lnp_detail_table,
    compute_bcell_uptake_vs_ova_correlation,
    compute_celltype_lnp_uptake,
    compute_celltype_relative_enrichment,
    compute_gene_expression_gates,
    compute_gene_positive_celltype_and_lnp_label_composition,
    compute_gene_region_summaries,
    compute_lnp_positive_celltype_composition,
    compute_pooled_gene_lnp_label_composition,
    compute_region_level_lnp_uptake,
    compute_region_qc_tables,
    compute_siinfekl_gate_and_dc_summary,
    load_batches,
    merge_cell_annotations_with_lnp_calls,
)
from nanostamp.plotting import apply_base_style, save_figure

apply_base_style()

snakemake = globals()["snakemake"]
cfg = snakemake.config["figure_2_cell_functional"]
data_root = Path(snakemake.config["data_root"]) / cfg["data_dir"] / "Precomputed_Downstream_Input"
out_dir = Path(snakemake.output.tables)
out_dir.mkdir(parents=True, exist_ok=True)

batch_inputs = {}
for batch_key, subdir, annotation_glob in [
    ("S1_S2", "Round_1_S1_S2", "*annotated.h5ad"),
    ("S3_S4", "Round_2_S3_S4", "*annotated.h5ad"),
]:
    batch_dir = data_root / subdir
    annotation_candidates = list(batch_dir.glob(annotation_glob)) or list(batch_dir.glob("*.h5ad"))
    batch_inputs[batch_key] = {
        "annotation": annotation_candidates[0],
        "spot_cell": batch_dir / "cell_analysis_table_all_regions_v4.csv",
        "spot_summary": batch_dir / "cell_analysis_summary_by_lnp_call_v4.csv",
    }

lnp_detail_table, raw_to_corrected = build_lnp_detail_table()
lnp_detail_table.to_csv(out_dir / "lnp_detail_table.csv", index=False)

adata, spot, spot_summary = load_batches(batch_inputs)
print(f"Annotation h5ad: {adata.n_obs:,} cells x {adata.n_vars:,} markers")
print(f"Spot per-cell table: {len(spot):,} rows")
print(f"Spot summary table: {len(spot_summary):,} rows")

adata, report = merge_cell_annotations_with_lnp_calls(
    adata, spot, MARKER_COLS, LNP_MARKER_COLS, lnp_detail_table, raw_to_corrected
)
print(f"Retained {report['n_retained']:,} cells; merge counts: {report['merge_counts']}")
adata.write_h5ad(Path(snakemake.output.annotated_h5ad))

obs = adata.obs
uptake_by_region, uptake_summary = compute_region_level_lnp_uptake(
    # pyrefly: ignore [bad-argument-type]
    obs,
    lnp_detail_table=lnp_detail_table,
)
uptake_by_region.to_csv(out_dir / "lnp_uptake_percent_by_region.csv", index=False)
uptake_summary.to_csv(out_dir / "lnp_uptake_percent_summary.csv", index=False)

celltype_uptake, celltype_summary, top_celltypes = compute_celltype_lnp_uptake(
    # pyrefly: ignore [bad-argument-type]
    obs,
    lnp_detail_table=lnp_detail_table,
)
celltype_uptake.to_csv(out_dir / "lnp_uptake_by_cell_type_region.csv", index=False)
celltype_summary.to_csv(out_dir / "lnp_uptake_by_cell_type_summary.csv", index=False)

# pyrefly: ignore [bad-argument-type]
composition = compute_lnp_positive_celltype_composition(obs, lnp_detail_table=lnp_detail_table)
composition.to_csv(out_dir / "lnp_positive_cell_type_composition.csv", index=False)

# pyrefly: ignore [bad-argument-type]
bias, relative, best = compute_celltype_relative_enrichment(obs, lnp_detail_table=lnp_detail_table)
bias.to_csv(out_dir / "lnp_cell_type_relative_enrichment.csv", index=False)
relative.to_csv(out_dir / "lnp_cell_type_relative_enrichment_row_normalized.csv", index=False)
best.to_csv(out_dir / "best_lnp_by_cell_type_relative_enrichment.csv", index=False)

# pyrefly: ignore [bad-argument-type]
expression_obs, gate_table = compute_gene_expression_gates(obs)
gate_table.to_csv(out_dir / "luc_ova_reference_gate_thresholds.csv", index=False)

region_summary, lnp_positive_summary = compute_gene_region_summaries(
    expression_obs, lnp_detail_table=lnp_detail_table
)
region_summary.to_csv(out_dir / "luc_ova_expression_by_region.csv", index=False)
lnp_positive_summary.to_csv(out_dir / "luc_ova_expression_by_lnp_positive_cells.csv", index=False)

pooled_composition, pooled_summary = compute_pooled_gene_lnp_label_composition(expression_obs)
pooled_composition.to_csv(
    out_dir / "luc_ova_positive_all_cells_lnp_label_composition.csv", index=False
)
pooled_summary.to_csv(out_dir / "luc_ova_positive_all_cells_lnp_labeled_summary.csv", index=False)

celltype_composition, label_composition = compute_gene_positive_celltype_and_lnp_label_composition(
    expression_obs
)
celltype_composition.to_csv(out_dir / "luc_ova_positive_cell_type_composition.csv", index=False)
label_composition.to_csv(
    out_dir / "luc_ova_positive_selected_cell_type_lnp_label_composition.csv", index=False
)

siin_gate, dc_by_region, dc_summary, siin_composition = compute_siinfekl_gate_and_dc_summary(
    expression_obs, lnp_detail_table=lnp_detail_table
)
siin_gate.to_csv(out_dir / "siinfekl_h2kb_s2_reg004_gate_threshold.csv", index=False)
dc_by_region.to_csv(out_dir / "lnp_positive_dc_siinfekl_ova_by_region.csv", index=False)
dc_summary.to_csv(out_dir / "lnp_positive_dc_siinfekl_ova_summary.csv", index=False)
siin_composition.to_csv(out_dir / "siinfekl_positive_lnp_dc_lnp_composition.csv", index=False)

# pyrefly: ignore [bad-argument-type]
region_qc, call_qc = compute_region_qc_tables(obs, lnp_detail_table=lnp_detail_table)
region_qc.to_csv(out_dir / "region_lnp_positive_qc.csv", index=False)
call_qc.to_csv(out_dir / "lnp_call_qc.csv", index=False)

ova_label_composition = label_composition.loc[label_composition["marker"] == "OVA"]
correlation_df, correlation_stats = compute_bcell_uptake_vs_ova_correlation(
    composition, ova_label_composition, fix_denominator_bug=cfg["fix_bcell_uptake_denominator_bug"]
)
correlation_df.to_csv(
    out_dir / "b_cell_uptake_vs_ova_transfection_comparison_source_data.csv", index=False
)
print(
    f"Spearman rho={correlation_stats['spearman_rho']:.3f}, "
    f"p={correlation_stats['spearman_pvalue']:.4g}; "
    f"Pearson r={correlation_stats['pearson_r']:.3f}, p={correlation_stats['pearson_pvalue']:.4g}"
)

# ---------------------------------------------------------------------------
# Section 9: selected-region spatial overlay (B/T/DC/Macrophage vs LNP_01/08/10).
# ---------------------------------------------------------------------------
plot_dir = Path(snakemake.output.figures)
plot_dir.mkdir(parents=True, exist_ok=True)

selected_regions = cfg.get("selected_overlay_regions", [])
overlay_obs = (
    # pyrefly: ignore [missing-attribute]
    obs.loc[obs["lnp_region"].isin(selected_regions)] if selected_regions else obs.iloc[0:0]
)
Path(snakemake.output.figure_source_data).mkdir(parents=True, exist_ok=True)
# pyrefly: ignore [missing-attribute]
if not overlay_obs.empty:

    def _collapse_to_btdc_mac(cell_type: str) -> str | None:
        text = str(cell_type)
        if text == "B" or text.endswith(" B") or "B-cell" in text:
            return "B"
        if text in {"T", "CD4+ T", "CD8+ T"} or text.endswith(" T") or "T-cell" in text:
            return "T"
        if text == "DC" or text.endswith(" DC") or "dendritic" in text.lower():
            return "DC"
        if text == "Macrophage" or "macrophage" in text.lower():
            return "Macrophage"
        return None

    overlay_obs = overlay_obs.copy()
    overlay_obs["btdc_mac_class"] = overlay_obs["cell_type"].map(_collapse_to_btdc_mac)
    # pyrefly: ignore [missing-attribute]
    overlay_obs = overlay_obs.dropna(subset=["btdc_mac_class"])
    highlighted = ["LNP_01", "LNP_08", "LNP_10"]
    overlay_obs["overlay_label"] = "Other " + overlay_obs["btdc_mac_class"]
    for lnp_call in highlighted:
        is_call = overlay_obs["lnp_positive"] & (overlay_obs["lnp_call"] == lnp_call)
        overlay_obs.loc[is_call, "overlay_label"] = (
            f"{lnp_call}+ " + overlay_obs.loc[is_call, "btdc_mac_class"]
        )

    overlay_counts = (
        overlay_obs.groupby(["lnp_region", "btdc_mac_class", "overlay_label"], observed=False)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    Path(snakemake.output.figure_source_data).mkdir(parents=True, exist_ok=True)
    overlay_counts.to_csv(
        Path(snakemake.output.figure_source_data)
        / "selected_region_lnp01_lnp08_lnp10_btdc_mac_overlay_counts.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    palette = {"Other": "#D9D9D9"}
    for label, group in overlay_obs.groupby("overlay_label", observed=False):
        color = "#D9D9D9" if label.startswith("Other") else None
        ax.scatter(group["x"], group["y"], s=4, label=label, color=color, alpha=0.8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Selected-region LNP_01/08/10 overlay", loc="left", fontsize=8)
    ax.legend(fontsize=5, markerscale=2, loc="upper right", frameon=False)
    ax.set_aspect("equal")
    fig.tight_layout()
    save_figure(fig, plot_dir / "selected_region_overlay")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Section 10: final replot figures (real PDF/PNG, unlike the source notebook).
# ---------------------------------------------------------------------------
plot_dir.mkdir(parents=True, exist_ok=True)

# 10a: per-LNP cellular uptake, boxplot across regions.
lnp_only_uptake = uptake_by_region.loc[
    (uptake_by_region["condition"] == "LNP") & uptake_by_region["lnp_call"].isin(LNP_ORDER)
]
fig, ax = plt.subplots(figsize=(5.5, 3))
box_data = [
    lnp_only_uptake.loc[lnp_only_uptake["lnp_call"] == call, "pct_cells_lnp_positive"].dropna()
    for call in LNP_ORDER
]
ax.boxplot(box_data, tick_labels=LNP_ORDER, showfliers=False)
ax.set_ylabel("% cells LNP+")
ax.set_title("Per-LNP cellular uptake", loc="left", fontsize=8)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
fig.tight_layout()
save_figure(fig, plot_dir / "lnp_only_per_lnp_cellular_uptake_box")
plt.close(fig)

# 10b: LNP-positive cell-type composition, stacked bars.
plot_order = [*LNP_ORDER, "Tissue_average"]
wide_composition = (
    composition.pivot_table(
        # pyrefly: ignore [bad-argument-type]
        index="lnp_call",
        columns="cell_type",
        values="pct_of_lnp_positive_cells",
        # pyrefly: ignore [bad-argument-type]
        aggfunc="first",
    )
    .reindex(index=plot_order)
    .fillna(0)
)
fig, ax = plt.subplots(figsize=(6, 3.2))
bottom = np.zeros(len(wide_composition))
for cell_type in wide_composition.columns:
    ax.bar(wide_composition.index, wide_composition[cell_type], bottom=bottom, label=cell_type)
    bottom += wide_composition[cell_type].to_numpy()
ax.set_ylabel("% of LNP+ cells")
ax.set_title("LNP-positive cell-type composition", loc="left", fontsize=8)
ax.legend(fontsize=5, ncol=3, frameon=False, loc="upper right")
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
fig.tight_layout()
save_figure(fig, plot_dir / "lnp_positive_cell_type_composition_stacked")
plt.close(fig)

# 10c/10f: OVA+ cell-type-by-LNP-label dot matrix.
ova_dot = ova_label_composition.loc[
    (ova_label_composition["lnp_label_for_gene_positive"] != "no_lnp_signal")
    & ova_label_composition["pct_positive_cells_in_cell_type"].notna()
]
if not ova_dot.empty:
    cell_types_present = sorted(ova_dot["cell_type"].unique())
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for row_index, cell_type in enumerate(cell_types_present):
        subset = ova_dot.loc[ova_dot["cell_type"] == cell_type]
        for _, row in subset.iterrows():
            call_index = LNP_ORDER.index(row["lnp_label_for_gene_positive"])
            ax.scatter(
                call_index,
                row_index,
                s=max(row["pct_positive_cells_in_cell_type"], 1) * 8,
                color="#B44636",
                alpha=0.7,
            )
    ax.set_xticks(range(len(LNP_ORDER)), LNP_ORDER, rotation=45, ha="right")
    ax.set_yticks(range(len(cell_types_present)), cell_types_present)
    ax.set_title("OVA+ cell-type by LNP label", loc="left", fontsize=8)
    fig.tight_layout()
    save_figure(fig, plot_dir / "ova_positive_lnp_label_composition_dotmatrix")
    plt.close(fig)

# 10d: SIINFEKL+ LNP-DC composition, bar chart.
fig, ax = plt.subplots(figsize=(4.5, 3))
ax.bar(
    siin_composition["lnp_call"], siin_composition["pct_siinfekl_positive_lnp_dc"], color="#5DA9A6"
)
ax.set_ylabel("% of SIINFEKL+ LNP+ DCs")
ax.set_title("SIINFEKL+ LNP-positive DC composition", loc="left", fontsize=8)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
fig.tight_layout()
save_figure(fig, plot_dir / "siinfekl_positive_lnp_dc_lnp_composition_bar")
plt.close(fig)

# 10e: B-cell uptake vs OVA-transfection correlation scatter.
fig, ax = plt.subplots(figsize=(3.5, 3.5))
ax.scatter(
    correlation_df["uptake_composition_pct"],
    correlation_df["ova_transfection_composition_pct"],
    color="#3E6FA3",
)
for _, row in correlation_df.iterrows():
    ax.annotate(
        row["lnp_label"],
        (row["uptake_composition_pct"], row["ova_transfection_composition_pct"]),
        fontsize=5,
    )
ax.set_xlabel("B-cell uptake composition (%)")
ax.set_ylabel("OVA-transfection composition (%)")
ax.set_title(
    f"Spearman rho={correlation_stats['spearman_rho']:.2f}, p={correlation_stats['spearman_pvalue']:.3g}",
    loc="left",
    fontsize=7,
)
fig.tight_layout()
save_figure(fig, plot_dir / "b_cell_uptake_vs_ova_transfection_comparison")
plt.close(fig)

print(f"Wrote Section 9/10 figures to {plot_dir}")
