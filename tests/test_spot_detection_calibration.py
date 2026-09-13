"""Tests for the calibration, tiling and raw-intensity-rescue additions to nanostamp.spot_detection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nanostamp.spot_detection import (
    BIT_1_CODEBOOK,
    calibrate_decode_threshold,
    calibrate_marker_threshold,
    decode_candidates,
    detect_log_candidates,
    detect_log_candidates_tiled,
    raw_intensity_rescue_candidates,
)


def _field(shape: tuple[int, int], n_spots: int, amplitude: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = rng.normal(20, 2, size=shape).clip(0)
    ys = rng.integers(5, shape[0] - 5, size=n_spots)
    xs = rng.integers(5, shape[1] - 5, size=n_spots)
    for y, x in zip(ys, xs, strict=True):
        image[y, x] = amplitude
    return image


def test_calibrate_marker_threshold_prefers_policy_satisfying_candidate() -> None:
    # Negative fields: no real spots. Positive fields: several bright spots.
    negative_images = [_field((80, 80), 0, 20, seed) for seed in range(3)]
    positive_images = [_field((80, 80), 8, 500, seed + 10) for seed in range(3)]
    best, results = calibrate_marker_threshold(
        negative_images,
        positive_images,
        candidate_thresholds=[1.0, 5.0, 20.0, 100.0],
        max_neg_matches_per_mpx=3.0,
        min_pos_matches_per_mpx=0.5,
    )
    chosen_row = results.loc[results["threshold"] == best].iloc[0]
    assert chosen_row["passes_policy"]
    assert chosen_row["pos_matches_per_mpx"] > 0


def test_calibrate_decode_threshold_separates_populations() -> None:
    negative_decoded = pd.DataFrame(
        {"min_on_snr": [0.5, 0.6, 0.4], "barcode_match_status": ["exact", "exact", "exact"]}
    )
    positive_decoded = pd.DataFrame(
        {"min_on_snr": [5.0, 6.0, 4.5, 0.3], "barcode_match_status": ["exact"] * 4}
    )
    best, results = calibrate_decode_threshold(
        negative_decoded,
        positive_decoded,
        candidate_min_on_snr=[0.2, 1.0, 3.0, 10.0],
        negative_megapixels=1.0,
        positive_megapixels=1.0,
        max_neg_matches_per_mpx=0.5,
        min_pos_matches_per_mpx=1.0,
    )
    # Thresholds 1.0 and 3.0 both keep 0 negative matches and 3 positive
    # matches (all real positives sit at snr >= 4.5); ties are broken by the
    # lowest such threshold, so 1.0 is selected over the too-strict 10.0
    # (which drops every positive match and so fails the policy floor).
    assert best == pytest.approx(1.0)
    chosen_row = results.loc[results["min_on_snr"] == best].iloc[0]
    assert chosen_row["passes_policy"]
    assert chosen_row["pos_matches_per_mpx"] == pytest.approx(3.0)
    too_strict_row = results.loc[results["min_on_snr"] == 10.0].iloc[0]
    assert not too_strict_row["passes_policy"]


def test_raw_intensity_rescue_finds_bright_and_blob_candidates() -> None:
    image = np.full((60, 60), 20.0)
    image[10, 10] = 500.0  # isolated bright point
    image[30:35, 30:35] = 300.0  # 5x5 = 25px blob
    candidates = raw_intensity_rescue_candidates(
        {"A": image},
        bright_percentile=99.0,
        object_percentile=98.0,
        object_min_area=4,
        object_max_area=100,
    )
    assert not candidates.empty
    assert set(candidates["seed_marker"]) <= {"rescue_bright", "rescue_object"}


def test_detect_log_candidates_tiled_matches_untiled_count(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    image = rng.normal(20, 2, size=(120, 60)).clip(0)
    # Five well-separated spots down the image, away from the tile seam.
    for y in [10, 35, 60, 85, 110]:
        image[y, 30] = 500.0

    untiled = detect_log_candidates({"A20": image}, thresholds={"A20": 5.0}, peak_width=9)

    tiled = detect_log_candidates_tiled(
        {"A20": image},
        thresholds={"A20": 5.0},
        n_tiles=4,
        tile_overlap=10,
        checkpoint_dir=tmp_path,
        region_name="test_region",
        peak_width=9,
    )
    assert len(tiled) == len(untiled)
    assert sorted(tiled["i"]) == sorted(untiled["i"])
    # Checkpoint files were written, one per tile.
    assert len(list(tmp_path.glob("spots_all_candidates_test_region_tile*.csv"))) == 4


def test_detect_log_candidates_tiled_resumes_from_checkpoint(tmp_path: Path) -> None:
    image = np.full((40, 40), 20.0)
    image[20, 20] = 500.0
    first = detect_log_candidates_tiled(
        {"A20": image},
        thresholds={"A20": 5.0},
        n_tiles=2,
        tile_overlap=5,
        checkpoint_dir=tmp_path,
        region_name="r",
        peak_width=9,
    )
    # Corrupt the underlying image; a resumed run must still return the
    # checkpointed result rather than recomputing from the (now different) image.
    checkpoint_files = list(tmp_path.glob("*.csv"))
    assert checkpoint_files
    second = detect_log_candidates_tiled(
        {"A20": np.zeros((40, 40))},
        thresholds={"A20": 5.0},
        n_tiles=2,
        tile_overlap=5,
        checkpoint_dir=tmp_path,
        region_name="r",
        peak_width=9,
    )
    assert len(second) == len(first)


def test_tiled_detection_feeds_decode_pipeline(tmp_path: Path) -> None:
    image = np.full((60, 60), 20.0)
    image[30, 30] = 500.0
    candidates = detect_log_candidates_tiled(
        {"A20": image},
        thresholds={"A20": 5.0},
        n_tiles=3,
        tile_overlap=5,
        checkpoint_dir=tmp_path,
        region_name="r2",
        peak_width=9,
    )
    decoded = decode_candidates(
        {"A20": image}, candidates, BIT_1_CODEBOOK, min_on_snr=0.5, min_snr_margin=0.0
    )
    assert decoded["accepted"].any()
