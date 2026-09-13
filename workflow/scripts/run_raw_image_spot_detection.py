"""Snakemake script: raw-image barcode spot detection (one of full_barcode/bit_1/round_1/round_2).

Only runnable once a user has placed the family's raw/registered TIFF stacks
under the documented ``Raw_Spot_Detection_Input`` directory (see README);
this script fails with a clear message otherwise, and the Snakemake rule
never runs it automatically for a real (non-smoke) config.
"""

from pathlib import Path

import pandas as pd
import tifffile

from nanostamp.spot_detection import (
    BIT_1_CODEBOOK,
    FULL_BARCODE_CODEBOOK,
    ROUND_1_CODEBOOK,
    ROUND_2_CODEBOOK,
    aggregate_cell_bit_counts,
    assign_spots_to_cells,
    build_cell_analysis_table,
    decode_candidates,
    detect_log_candidates,
)

snakemake = globals()["snakemake"]
family = snakemake.wildcards.family
codebooks = {
    "full_barcode": FULL_BARCODE_CODEBOOK,
    "bit_1": BIT_1_CODEBOOK,
    "round_1": ROUND_1_CODEBOOK,
    "round_2": ROUND_2_CODEBOOK,
}
codebook = codebooks[family]
cfg = snakemake.config["raw_image_spot_detection"][family]
data_root = Path(snakemake.config["data_root"]) / cfg["data_dir"]
out_dir = Path(snakemake.output.tables)
out_dir.mkdir(parents=True, exist_ok=True)

samples = sorted(data_root.glob("*_integrated_registered_overlap_crop.tif"))
if not samples:
    raise FileNotFoundError(
        f"No registered TIFF stacks found under {data_root}. This rule needs the "
        "excluded raw/registered images described in the README; place them "
        "under this directory (per the notebook's documented layout) to run it."
    )

all_tables = []
for stack_path in samples:
    sample = stack_path.name.removesuffix("_integrated_registered_overlap_crop.tif")
    marker_list = (data_root / f"{sample}_integrated_MarkerList.txt").read_text().split()
    stack = tifffile.imread(stack_path)
    channel_images = dict(zip(marker_list, stack, strict=True))
    channel_images = {
        marker: channel_images[marker]
        for marker in codebook.marker_order
        if marker in channel_images
    }

    features_candidates = list(data_root.glob("*_features.csv"))
    features = pd.read_csv(features_candidates[0])

    per_marker_thresholds = cfg.get("per_marker_thresholds", {})
    thresholds = {
        marker: per_marker_thresholds.get(marker, cfg.get("threshold_peaks", 300.0))
        for marker in codebook.marker_order
    }

    candidates = detect_log_candidates(
        channel_images,
        thresholds,
        log_sigma=cfg["log_sigma"],
        peak_width=cfg["peak_width"],
        nms_min_distance=cfg["nms_min_distance"],
    )
    decoded = decode_candidates(
        channel_images,
        candidates,
        codebook,
        min_on_snr=cfg["min_on_snr"],
        min_snr_margin=cfg["min_snr_margin"],
        bright_snr_threshold=cfg.get("bright_snr_threshold", 1.75),
        max_bright_markers=cfg.get("max_bright_markers"),
    )
    accepted = decoded.loc[decoded["accepted"]].copy()
    assigned = assign_spots_to_cells(accepted, features, radius_px=cfg["cell_assignment_radius_px"])
    assigned.to_csv(out_dir / f"spots_{sample}.csv", index=False)

    bit_counts = aggregate_cell_bit_counts(assigned, codebook.marker_order)
    cell_table = build_cell_analysis_table(features, bit_counts, codebook)
    cell_table["sample"] = sample
    cell_table.to_csv(out_dir / f"cell_analysis_table_{sample}.csv", index=False)
    all_tables.append(cell_table)

    print(f"{sample}: {len(candidates):,} candidates, {len(accepted):,} accepted spots")

combined = pd.concat(all_tables, ignore_index=True)
combined.to_csv(out_dir / "cell_analysis_table_all_regions.csv", index=False)
print(f"{family}: {combined['lnp_positive'].sum():,} / {len(combined):,} cells LNP-positive")
