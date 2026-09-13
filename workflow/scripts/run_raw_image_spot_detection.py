"""Snakemake script: raw-image barcode spot detection (one of full_barcode/bit_1/round_1/round_2).

Only runnable once a user has placed the family's raw/registered TIFF stacks
under the documented ``Raw_Spot_Detection_Input`` directory (see README);
this script fails with a clear message otherwise, and the Snakemake rule
never runs it automatically for a real (non-smoke) config.

Two families calibrate their detection parameters from matched negative
(``reg000``) and positive (``reg001``) regions, exactly as
``Figure_1d_1e_Full_Barcode_Spot_Detection.ipynb`` and
``Figure_1d_Bit_1_Spot_Detection.ipynb`` do (Section 4); the other two use
fixed manual parameters plus tiled, checkpointed, rescue-assisted detection,
as ``..._Spot_Detection_Round_{1,2}.ipynb`` do.
"""

import json
from pathlib import Path

import numpy as np
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
    calibrate_decode_threshold,
    calibrate_marker_threshold,
    decode_candidates,
    detect_log_candidates,
    detect_log_candidates_tiled,
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

stack_paths = sorted(data_root.glob("*_integrated_registered_overlap_crop.tif"))
if not stack_paths:
    raise FileNotFoundError(
        f"No registered TIFF stacks found under {data_root}. This rule needs the "
        "excluded raw/registered images described in the README; place them "
        "under this directory (per the notebook's documented layout) to run it."
    )


def _load_sample(stack_path: Path) -> tuple[str, str, dict[str, np.ndarray], pd.DataFrame]:
    """Return ``(sample, region, channel_images, features)`` for one registered stack."""
    sample = stack_path.name.removesuffix("_integrated_registered_overlap_crop.tif")
    marker_list = (data_root / f"{sample}_integrated_MarkerList.txt").read_text().split()
    summary = json.loads((data_root / f"{sample}_registration_summary.json").read_text())
    region = summary["region"]
    stack = tifffile.imread(stack_path)
    channel_images = dict(zip(marker_list, stack, strict=True))
    channel_images = {
        marker: channel_images[marker]
        for marker in codebook.marker_order
        if marker in channel_images
    }
    features = pd.read_csv(data_root / f"{region}_features.csv")
    return sample, region, channel_images, features


samples = [_load_sample(path) for path in stack_paths]

if family in {"full_barcode", "bit_1"}:
    # Section 4: calibrate per-marker LoG thresholds and the global decode
    # threshold from the (only) negative/positive region pair.
    negative = next((s for s in samples if s[1] == "reg000"), samples[0])
    positive = next((s for s in samples if s[1] == "reg001"), samples[-1])

    marker_thresholds: dict[str, float] = {}
    marker_calibration_rows = []
    for marker in codebook.marker_order:
        best, table = calibrate_marker_threshold(
            [negative[2][marker]],
            [positive[2][marker]],
            candidate_thresholds=cfg["calibration_candidate_thresholds"],
            log_sigma=cfg["log_sigma"],
            peak_width=cfg["peak_width"],
            max_neg_matches_per_mpx=cfg["max_neg_matches_per_mpx"],
            min_pos_matches_per_mpx=cfg["min_pos_matches_per_mpx"],
        )
        marker_thresholds[marker] = best
        table["marker"] = marker
        marker_calibration_rows.append(table)
    pd.concat(marker_calibration_rows, ignore_index=True).to_csv(
        out_dir / "calibrated_marker_log_thresholds.csv", index=False
    )

    # Decode both calibration regions at a permissive min_on_snr to gather
    # the pool of candidate SNRs the threshold grid search selects from.
    neg_candidates = detect_log_candidates(
        negative[2],
        marker_thresholds,
        log_sigma=cfg["log_sigma"],
        peak_width=cfg["peak_width"],
        nms_min_distance=cfg["nms_min_distance"],
    )
    pos_candidates = detect_log_candidates(
        positive[2],
        marker_thresholds,
        log_sigma=cfg["log_sigma"],
        peak_width=cfg["peak_width"],
        nms_min_distance=cfg["nms_min_distance"],
    )
    neg_decoded = decode_candidates(
        negative[2], neg_candidates, codebook, min_on_snr=0.0, min_snr_margin=-1e9
    )
    pos_decoded = decode_candidates(
        positive[2], pos_candidates, codebook, min_on_snr=0.0, min_snr_margin=-1e9
    )
    negative_mpx = next(iter(negative[2].values())).size / 1_000_000
    positive_mpx = next(iter(positive[2].values())).size / 1_000_000
    min_on_snr, decode_calibration = calibrate_decode_threshold(
        neg_decoded,
        pos_decoded,
        candidate_min_on_snr=cfg["calibration_candidate_min_on_snr"],
        negative_megapixels=negative_mpx,
        positive_megapixels=positive_mpx,
        max_neg_matches_per_mpx=cfg["max_neg_matches_per_mpx"],
        min_pos_matches_per_mpx=cfg["min_pos_matches_per_mpx"],
    )
    decode_calibration.to_csv(out_dir / "calibrated_decode_thresholds.csv", index=False)
    (out_dir / "selected_detection_parameters.json").write_text(
        json.dumps(
            {
                "log_sigma": cfg["log_sigma"],
                "peak_width": cfg["peak_width"],
                "nms_min_distance": cfg["nms_min_distance"],
                "min_on_snr": min_on_snr,
                "min_snr_margin": cfg["min_snr_margin"],
                "per_marker_log_thresholds": marker_thresholds,
            },
            indent=2,
        )
    )
    print(
        f"{family}: calibrated min_on_snr={min_on_snr:.3f}, per-marker thresholds={marker_thresholds}"
    )

    detection_thresholds: dict[str, float] = marker_thresholds
    decode_min_on_snr: float = min_on_snr
    use_rescue = False
else:
    # Round 1/2: fixed manual parameters (never recalibrated), tiled and
    # rescue-assisted detection to match the real hundred-gigabyte stacks.
    per_marker_thresholds = cfg.get("per_marker_thresholds", {})
    detection_thresholds = {
        marker: per_marker_thresholds.get(marker, cfg["threshold_peaks"])
        for marker in codebook.marker_order
    }
    decode_min_on_snr = cfg["min_on_snr"]
    use_rescue = True
    (out_dir / "selected_detection_parameters_manual.json").write_text(
        json.dumps(
            {
                "thresholds": detection_thresholds,
                "min_on_snr": decode_min_on_snr,
                "use_rescue": use_rescue,
            },
            indent=2,
        )
    )

all_tables = []
for sample, region, channel_images, features in samples:
    if use_rescue:
        checkpoint_dir = out_dir / "region_tile_checkpoints" / region
        candidates = detect_log_candidates_tiled(
            channel_images,
            detection_thresholds,
            n_tiles=cfg["n_tiles"],
            tile_overlap=cfg["tile_overlap"],
            checkpoint_dir=checkpoint_dir,
            region_name=region,
            log_sigma=cfg["log_sigma"],
            peak_width=cfg["peak_width"],
            nms_min_distance=cfg["nms_min_distance"],
            use_raw_rescue=True,
            rescue_kwargs={
                "bright_percentile": cfg["rescue_bright_percentile"],
                "object_percentile": cfg["rescue_object_percentile"],
                "object_min_area": cfg["rescue_object_min_area"],
                "object_max_area": cfg["rescue_object_max_area"],
            },
        )
    else:
        candidates = detect_log_candidates(
            channel_images,
            detection_thresholds,
            log_sigma=cfg["log_sigma"],
            peak_width=cfg["peak_width"],
            nms_min_distance=cfg["nms_min_distance"],
        )

    decoded = decode_candidates(
        channel_images,
        candidates,
        codebook,
        min_on_snr=decode_min_on_snr,
        min_snr_margin=cfg["min_snr_margin"],
        bright_snr_threshold=cfg.get("bright_snr_threshold", 1.75),
        max_bright_markers=cfg.get("max_bright_markers"),
    )
    accepted = decoded.loc[decoded["accepted"]].copy() if not decoded.empty else decoded
    assigned = assign_spots_to_cells(accepted, features, radius_px=cfg["cell_assignment_radius_px"])
    assigned.to_csv(out_dir / f"spots_{sample}.csv", index=False)

    bit_counts = aggregate_cell_bit_counts(assigned, codebook.marker_order)
    cell_table = build_cell_analysis_table(features, bit_counts, codebook)
    cell_table["region"] = region
    cell_table.to_csv(out_dir / f"cell_analysis_table_{region}.csv", index=False)
    all_tables.append(cell_table)

    print(f"{sample} ({region}): {len(candidates):,} candidates, {len(accepted):,} accepted spots")

combined = pd.concat(all_tables, ignore_index=True)
combined.to_csv(out_dir / "cell_analysis_table_all_regions.csv", index=False)
print(f"{family}: {combined['lnp_positive'].sum():,} / {len(combined):,} cells LNP-positive")
