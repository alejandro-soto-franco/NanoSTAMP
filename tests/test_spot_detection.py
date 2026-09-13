"""Tests for nanostamp.spot_detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nanostamp.spot_detection import (
    BIT_1_CODEBOOK,
    FULL_BARCODE_CODEBOOK,
    ROUND_1_CODEBOOK,
    ROUND_2_CODEBOOK,
    BarcodeCodebook,
    aggregate_cell_bit_counts,
    assign_spots_to_cells,
    build_cell_analysis_table,
    classify_cells_by_barcode_presence,
    decode_candidates,
    detect_log_candidates,
)


def test_codebooks_are_internally_consistent() -> None:
    for codebook in (FULL_BARCODE_CODEBOOK, BIT_1_CODEBOOK, ROUND_1_CODEBOOK, ROUND_2_CODEBOOK):
        for bits in codebook.library:
            assert len(bits) == len(codebook.marker_order)
            assert bits.count("1") == codebook.expected_on_bits


def test_barcode_codebook_rejects_inconsistent_weight() -> None:
    with pytest.raises(ValueError, match="constant weight"):
        BarcodeCodebook(
            marker_order=["A", "B"],
            library={"10": "x", "11": "y"},
            expected_on_bits=1,
            hamming_tolerance=0,
        )


def _synthetic_spot_image(
    shape: tuple[int, int], spots: list[tuple[int, int, float]]
) -> np.ndarray:
    image = np.full(shape, 10.0)
    for i, j, amplitude in spots:
        image[i, j] = amplitude
    return image


def test_detect_log_candidates_finds_bright_spot() -> None:
    image = _synthetic_spot_image((50, 50), [(25, 25, 500.0)])
    candidates = detect_log_candidates(
        {"A": image}, thresholds={"A": 1.0}, log_sigma=1.0, peak_width=9
    )
    assert len(candidates) >= 1
    # pyrefly: ignore [bad-index]
    closest = candidates.iloc[
        (candidates["i"] - 25).abs().add((candidates["j"] - 25).abs()).idxmin()
    ]
    assert abs(closest["i"] - 25) <= 2
    assert abs(closest["j"] - 25) <= 2


def test_detect_log_candidates_nms_merges_close_peaks() -> None:
    image = _synthetic_spot_image((50, 50), [(25, 25, 500.0), (26, 26, 480.0)])
    candidates = detect_log_candidates(
        {"A": image}, thresholds={"A": 1.0}, log_sigma=1.0, peak_width=9, nms_min_distance=9.0
    )
    # Two peaks 1.4px apart should be merged into a single candidate by NMS.
    assert len(candidates) == 1


def test_decode_candidates_exact_match_for_bit1_codebook() -> None:
    image = _synthetic_spot_image((50, 50), [(25, 25, 500.0)])
    candidates = pd.DataFrame([{"i": 25, "j": 25, "score": 1.0, "seed_marker": "A20"}])
    decoded = decode_candidates(
        {"A20": image}, candidates, BIT_1_CODEBOOK, min_on_snr=0.5, min_snr_margin=0.0
    )
    row = decoded.iloc[0]
    assert row["called_code"] == "1"
    assert row["lnp_call"] == "LNP_A"
    assert row["barcode_match_status"] == "exact"
    assert row["accepted"]


def test_decode_candidates_rejects_low_snr() -> None:
    image = _synthetic_spot_image((50, 50), [(25, 25, 10.5)])  # barely above background
    candidates = pd.DataFrame([{"i": 25, "j": 25, "score": 1.0, "seed_marker": "A20"}])
    decoded = decode_candidates(
        {"A20": image}, candidates, BIT_1_CODEBOOK, min_on_snr=5.0, min_snr_margin=0.0
    )
    assert not decoded.iloc[0]["accepted"]


def test_assign_spots_to_cells_within_radius() -> None:
    spots = pd.DataFrame({"i": [10.0, 100.0], "j": [10.0, 100.0]})
    centroids = pd.DataFrame({"label": [1, 2], "y": [11.0, 500.0], "x": [11.0, 500.0]})
    assigned = assign_spots_to_cells(spots, centroids, radius_px=5.0)
    assert assigned.iloc[0]["cell"] == 1
    assert pd.isna(assigned.iloc[1]["cell"])


def test_aggregate_cell_bit_counts_sums_per_cell() -> None:
    spots = pd.DataFrame(
        {
            "cell": pd.array([1, 1, 2], dtype="Int64"),
            "called_code": ["1", "1", "0"],
            "barcode_match_status": ["exact", "exact", "no_barcode"],
        }
    )
    result = aggregate_cell_bit_counts(spots, marker_order=["A20"])
    cell1 = result.loc[result["cell"] == 1].iloc[0]
    assert cell1["bit_A20"] == 2
    assert cell1["decoded_spots"] == 2


def test_build_cell_analysis_table_calls_lnp_positive() -> None:
    features = pd.DataFrame({"label": [1, 2], "x": [1.0, 2.0], "y": [1.0, 2.0]})
    cell_bit_counts = pd.DataFrame(
        {"cell": [1, 2], "bit_A20": [1, 0], "decoded_spots": [1, 0], "total_spots": [1, 0]}
    )
    table = build_cell_analysis_table(features, cell_bit_counts, BIT_1_CODEBOOK)
    row1 = table.loc[table["cell"] == 1].iloc[0]
    assert row1["lnp_call"] == "LNP_A"
    assert row1["lnp_positive"]
    row2 = table.loc[table["cell"] == 2].iloc[0]
    assert row2["lnp_call"] == "no_barcode"
    assert not row2["lnp_positive"]


def test_classify_cells_by_barcode_presence() -> None:
    spot_counts = pd.DataFrame(
        {
            "barcode1_spot_count": [0, 3, 0, 2],
            "barcode2_spot_count": [0, 0, 4, 1],
        }
    )
    result = classify_cells_by_barcode_presence(spot_counts, min_spots_per_barcode=1)
    assert list(result["classification"]) == [
        "no_barcode",
        "barcode1_only",
        "barcode2_only",
        "both_barcodes",
    ]


def test_segment_nuclei_cellpose_is_cpu_only_by_signature() -> None:
    pytest.importorskip("cellpose")
    from nanostamp.spot_detection import segment_nuclei_cellpose

    # Only confirm the function accepts gpu=False without raising at import
    # time; a real segmentation run needs a downloaded model and is exercised
    # by the smoke workflow, not by this unit test.
    # pyrefly: ignore [unsupported-operation]
    assert segment_nuclei_cellpose.__defaults__[3] is False


def test_segment_nuclei_threshold_finds_bright_blob() -> None:
    from nanostamp.spot_detection import segment_nuclei_threshold

    image = np.full((60, 60), 10.0)
    image[20:35, 20:35] = 200.0  # 15x15 = 225 px blob
    labels = segment_nuclei_threshold(image, min_area_px=100, max_area_px=400)
    assert labels.max() == 1
    assert (labels == 1).sum() == pytest.approx(225, abs=5)
