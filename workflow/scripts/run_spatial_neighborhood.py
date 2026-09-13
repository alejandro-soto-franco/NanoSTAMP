"""Snakemake script: Figure 2 / Supplementary 9-11 spatial neighbourhood analysis.

Ports Steps 1-6 of the source notebook (gating, k-nearest-cell composition,
clustering, neighbourhood naming, LNP abundance, local enrichment); see
``nanostamp.spatial_neighborhood`` module docstring for the documented scope
limit on Sections 8-11's per-marker distance analyses.
"""

from pathlib import Path

import anndata as ad

from nanostamp.spatial_neighborhood import (
    build_neighbor_composition,
    compute_local_enrichment,
    fit_and_name_neighborhoods,
    load_and_gate_cells,
    pool_cell_types,
    run_elbow_analysis,
    summarize_lnp_neighborhood_distribution,
)

snakemake = globals()["snakemake"]
cfg = snakemake.config["figure_2_spatial_neighborhood"]
out_dir = Path(snakemake.output.tables)
out_dir.mkdir(parents=True, exist_ok=True)

adata = ad.read_h5ad(snakemake.input.annotated_h5ad)
print(f"Loaded {adata.n_obs:,} cells x {adata.n_vars:,} markers")

cells, gate_table = load_and_gate_cells(
    # pyrefly: ignore [bad-argument-type]
    adata.obs,
    gate_reference_regions=cfg["gate_reference_regions"],
    gate_quantile=cfg["gate_quantile"],
)
gate_table.to_csv(out_dir / "luc_ova_neighborhood_gate_thresholds.csv", index=False)

cells = cells.reset_index(drop=True)
cells["cell_type_pooled"] = pool_cell_types(cells["cell_type"])
work, composition = build_neighbor_composition(
    cells, k=cfg["k_neighbors"], robust_self_exclusion=cfg["robust_self_exclusion"]
)
print(f"Built {cfg['k_neighbors']}-nearest-neighbor windows for {len(cells):,} cells")

valid_mask = composition.sum(axis=1) > 0
elbow_results = run_elbow_analysis(
    composition,
    valid_mask,
    cluster_values=cfg["elbow_cluster_values"],
    max_cells=cfg["elbow_max_cells"],
    random_state=cfg["random_state"],
)
elbow_results.to_csv(out_dir / "neighborhood_kmeans_elbow_results.csv", index=False)

centroid_percent, name_table, neighbor_cells, ordered_names = fit_and_name_neighborhoods(
    composition,
    valid_mask,
    cells,
    n_clusters=cfg["n_neighborhoods"],
    random_state=cfg["random_state"],
)
centroid_percent.to_csv(out_dir / "neighborhood_cell_type_composition_percent.csv")
name_table.to_csv(out_dir / "neighborhood_name_key.csv", index=False)
neighbor_cells.to_csv(out_dir / "cell_neighborhood_assignments.csv", index=False)
# Raw per-cell neighbour-type fractions, row-aligned with the assignments
# CSV above; Sections 8-11 (run_spatial_neighborhood_distance.py) need this
# to compute direct neighbour-composition summaries for a target subset.
composition.reset_index(drop=True).to_csv(
    out_dir / "neighbor_composition_fractions.csv", index=False
)

lnp_abundance = summarize_lnp_neighborhood_distribution(neighbor_cells, ordered_names=ordered_names)
lnp_abundance.to_csv(out_dir / "corrected_lnp_neighborhood_abundance.csv")

region_composition, region_enrichment = compute_local_enrichment(neighbor_cells, composition)
region_composition.to_csv(
    out_dir / "mean_neighbor_composition_by_region_and_lnp_status.csv", index=False
)
region_enrichment.to_csv(
    out_dir / "neighbor_celltype_enrichment_around_lnp_positive_by_region.csv", index=False
)

print(f"Neighbourhoods detected: {len(centroid_percent)}")
print(lnp_abundance.head(20).to_string())
