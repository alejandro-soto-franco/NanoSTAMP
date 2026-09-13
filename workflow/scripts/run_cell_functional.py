"""Snakemake script: Figure 2 / Supplementary 6-8 cell and functional analysis."""

from pathlib import Path

from nanostamp.cell_functional import (
    LNP_MARKER_COLS,
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
