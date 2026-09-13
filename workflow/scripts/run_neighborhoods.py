"""Snakemake script: Figure 1f/1g spleen neighbourhood analysis."""

from pathlib import Path

from nanostamp.neighborhoods import (
    build_region_windows,
    cluster_neighborhoods,
    collapse_neighborhood_rows,
    compute_neighborhood_abundance,
    compute_tissue_enrichment,
    load_neighborhood_cells,
    name_neighborhoods,
)
from nanostamp.plotting import apply_base_style

snakemake = globals()["snakemake"]
cfg = snakemake.config["figure_1f_1g_neighborhoods"]
data_root = Path(snakemake.config["data_root"]) / cfg["data_dir"]
out_dir = Path(snakemake.output.tables)
out_dir.mkdir(parents=True, exist_ok=True)

apply_base_style()

cells = load_neighborhood_cells(data_root)
cells_df, cell_types, windows = build_region_windows(cells, k=cfg["k_neighbors"])
clustered, percent, fold_change, summary = cluster_neighborhoods(
    cells_df,
    windows,
    cell_types,
    n_neighborhoods=cfg["n_neighborhoods"],
    random_state=cfg["random_state"],
)
label_map, weights = name_neighborhoods(percent, summary)
named_percent = collapse_neighborhood_rows(percent, label_map, weights)
clustered["neighborhood_name"] = clustered["neighborhood"].map(label_map)

cell_type_order = sorted(cell_types)
enrichment = compute_tissue_enrichment(cells, named_percent, cell_type_order)
abundance = compute_neighborhood_abundance(clustered)

named_percent.to_csv(out_dir / "figure_1f_neighborhood_composition_percent.csv")
enrichment.to_csv(out_dir / "figure_1f_neighborhood_enrichment.csv")
abundance.to_csv(out_dir / "figure_1g_neighborhood_abundance.csv")
summary.to_csv(out_dir / "neighborhood_size_summary.csv")

print(f"reg001 cells used: {len(cells):,}")
print(f"Neighbourhoods detected: {len(named_percent)}")
print(abundance.to_string())
