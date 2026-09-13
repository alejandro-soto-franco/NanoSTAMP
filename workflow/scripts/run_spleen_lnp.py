"""Snakemake script: Figure 1d/1e spleen LNP analysis."""

from pathlib import Path

from nanostamp.plotting import apply_base_style
from nanostamp.spleen_lnp import (
    assemble_obs,
    compute_cell_type_enrichment,
    compute_tile_summary,
    load_frozen_tables,
)

snakemake = globals()["snakemake"]
cfg = snakemake.config["figure_1d_1e_spleen_lnp"]
data_root = Path(snakemake.config["data_root"]) / cfg["data_dir"]
out_dir = Path(snakemake.output.tables)
out_dir.mkdir(parents=True, exist_ok=True)

apply_base_style()

codebook, bit_1, annotations = load_frozen_tables(data_root)
obs = assemble_obs(codebook, annotations)
tile_summary, whole_summary = compute_tile_summary(
    bit_1, codebook, tile_size_px=cfg["tile_size_px"], min_cells_per_tile=cfg["min_cells_per_tile"]
)
comparison = compute_cell_type_enrichment(
    obs, treated_region="reg001", min_percent=cfg["min_percent_for_named_cell_type"]
)

tile_summary.to_csv(out_dir / "figure_1d_tile_summary.csv", index=False)
whole_summary.to_csv(out_dir / "figure_1d_whole_region_summary.csv", index=False)
comparison.to_csv(out_dir / "figure_1e_cell_type_enrichment.csv", index=False)

print(f"Loaded {len(codebook):,} full-barcode detector cells")
print(f"Loaded {len(bit_1):,} Bit_1 detector cells")
print(f"Detector cells without a frozen cell-type label: {(obs['cell_type'] == 'Unknown').sum():,}")
print(comparison.to_string(index=False))
