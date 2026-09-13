"""Snakemake script: Supplementary Figure 1c two-oligo RCA-FISH quantification.

Only runnable once a user has placed the raw per-FOV TIFFs under the
documented ``Raw_Images/<sample>/0/`` layout (see README); this script fails
with a clear message otherwise. Ports the spot-detection, segmentation and
per-cell classification steps (see ``nanostamp.spot_detection``); the Fiji
tile-stitching QC step is out of scope, since it produces a visual QC
preview only and is not required for the quantification.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage

from nanostamp.spot_detection import (
    classify_cells_by_barcode_presence,
    detect_log_candidates,
    segment_nuclei_cellpose,
    segment_nuclei_threshold,
)

snakemake = globals()["snakemake"]
cfg = snakemake.config["raw_image_spot_detection"]["supplementary_figure_1c"]
data_root = Path(snakemake.config["data_root"]) / cfg["data_dir"] / "Raw_Images"
out_dir = Path(snakemake.output.tables)
out_dir.mkdir(parents=True, exist_ok=True)

channel_suffix = {
    "DAPI": "Fluorescence_405_nm_Ex",
    "Cy3-F46": "Fluorescence_561_nm_Ex",
    "Cy5-F20": "Fluorescence_638_nm_Ex",
}
barcode_channels = {"barcode1": "Cy3-F46", "barcode2": "Cy5-F20"}

sample_dirs = sorted(p for p in data_root.iterdir() if p.is_dir())
if not sample_dirs:
    raise FileNotFoundError(
        f"No sample folders found under {data_root}. This rule needs the "
        "excluded raw per-FOV TIFFs described in the README; place them "
        "under this directory to run it."
    )

all_cells = []
for sample_dir in sample_dirs:
    sample = sample_dir.name
    image_dir = sample_dir / "0"
    dapi_files = sorted(
        p for p in image_dir.glob(f"*_{channel_suffix['DAPI']}.tif*") if not p.name.startswith("._")
    )
    for dapi_path in dapi_files:
        fov_prefix = dapi_path.name.removesuffix(f"_{channel_suffix['DAPI']}.tiff").removesuffix(
            f"_{channel_suffix['DAPI']}.tif"
        )
        dapi_image = tifffile.imread(dapi_path)

        if cfg.get("use_cellpose", False):
            mask = segment_nuclei_cellpose(
                dapi_image,
                diameter=cfg["cellpose_diameter"],
                min_area_px=cfg["cellpose_min_area_px"],
                max_area_px=cfg["cellpose_max_area_px"],
                gpu=cfg["cellpose_gpu"],
            )
        else:
            mask = segment_nuclei_threshold(
                dapi_image,
                min_area_px=cfg["cellpose_min_area_px"],
                max_area_px=cfg["cellpose_max_area_px"],
            )
        if mask.max() == 0:
            continue
        expanded = ndimage.distance_transform_edt(
            mask == 0, return_distances=False, return_indices=True
        )
        # pyrefly: ignore [bad-argument-type]
        dilated_mask = mask[tuple(expanded)]
        dilated_mask = np.where(
            ndimage.distance_transform_edt(mask == 0) <= cfg["expand_mask_px"], dilated_mask, 0
        )

        spot_counts = {
            label: {"barcode1_spot_count": 0, "barcode2_spot_count": 0}
            for label in range(1, mask.max() + 1)
        }
        for barcode, channel in barcode_channels.items():
            channel_path = image_dir / f"{fov_prefix}_{channel_suffix[channel]}.tiff"
            if not channel_path.exists():
                channel_path = image_dir / f"{fov_prefix}_{channel_suffix[channel]}.tif"
            channel_image = tifffile.imread(channel_path)
            candidates = detect_log_candidates(
                {barcode: channel_image},
                thresholds={barcode: cfg["threshold_peaks"].get(barcode, 500)},
                peak_width=cfg["peak_width"],
                nms_min_distance=cfg["nms_min_distance"],
            )
            for _, row in candidates.iterrows():
                label = int(dilated_mask[int(row["i"]), int(row["j"])])
                if label > 0:
                    spot_counts[label][f"{barcode}_spot_count"] += 1

        cell_rows = [
            {"sample": sample, "fov": fov_prefix, "cell_id": label, **counts}
            for label, counts in spot_counts.items()
        ]
        all_cells.extend(cell_rows)

per_cell = pd.DataFrame(all_cells)
classified = classify_cells_by_barcode_presence(
    per_cell, min_spots_per_barcode=cfg["min_spots_per_barcode"]
)
classified.to_csv(out_dir / "combined_per_cell_2oligo_barcode_assignment.csv", index=False)

summary = (
    classified.groupby("sample")
    .agg(
        total_cells=("classification", "size"),
        barcode1_only=("classification", lambda s: (s == "barcode1_only").sum()),
        barcode2_only=("classification", lambda s: (s == "barcode2_only").sum()),
        both_barcodes=("classification", lambda s: (s == "both_barcodes").sum()),
        no_barcode=("classification", lambda s: (s == "no_barcode").sum()),
    )
    .reset_index()
)
summary.to_csv(out_dir / "summary_2oligo_barcode_assignment_by_sample.csv", index=False)
print(f"Segmented {len(classified):,} cells across {len(sample_dirs)} sample(s)")
print(summary.to_string(index=False))
