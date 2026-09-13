"""Raw-image LNP-barcode spot detection (Figures 1d/1e, 2, and Supplementary Figure 1c).

Ported from ``Code/Figure_1d_1e_Full_Barcode_Spot_Detection.ipynb``,
``Code/Figure_1d_Bit_1_Spot_Detection.ipynb``,
``Code/Figure_2_and_Supplementary_6_11_Multiplex_LNP_Spot_Detection_Round_{1,2}.ipynb``
and ``Code/Supplementary_Figure_1c_Image_Processing_and_Quantification.ipynb``.

None of these five notebooks can run in this environment: each needs raw or
registered multichannel TIFF stacks that are excluded from the Duke Research
Data Repository deposit (see the README). This module ports their algorithm
as one generic, configurable pipeline rather than five near-duplicated
per-notebook implementations:

1. :func:`detect_log_candidates` - per-marker Laplacian-of-Gaussian candidate
   detection with greedy non-max suppression.
2. :func:`decode_candidates` - per-candidate local core/annulus SNR
   measurement, top-``expected_on_bits`` bit calling, and Hamming-distance
   barcode-library matching.
3. :func:`assign_spots_to_cells` - nearest-centroid assignment against a
   frozen segmentation's cell table.
4. :func:`aggregate_cell_bit_counts` and :func:`build_cell_analysis_table` -
   per-cell bit aggregation and the final codebook-call table, matching the
   schema documented in the porting spec (``region, cell, x, y, barcode,
   lnp_call, lnp_positive, barcode_match_status, ...``).

5. :func:`calibrate_marker_threshold` and :func:`calibrate_decode_threshold` -
   the per-marker LoG-threshold and global decode-threshold grid searches
   (Section 4 of the raw-image notebooks), against matched negative/positive
   calibration fields.
6. :func:`detect_log_candidates_tiled` - y-axis tiling with resumable
   per-tile CSV checkpoints, needed because the real registered stacks are
   hundreds of gigabytes.
7. :func:`raw_intensity_rescue_candidates` - the raw-intensity "rescue" pass
   (bright-point and blob-shaped candidates from a channel-max projection).

Scope note: the source notebooks' optional CuPy GPU fast path for the
Laplacian-of-Gaussian filter is not ported; every function here runs on CPU
only (the target environment has no GPU access for this repository), which
the calibration and tiling functions above do not change the numerical
behaviour of, only the runtime. Supplementary Figure 1c's Cellpose nuclear segmentation
(:func:`segment_nuclei_cellpose`) and barcode-presence classification
(:func:`classify_cells_by_barcode_presence`) are ported in full, since they
run identically whether the input is real or synthetic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

# pyrefly: ignore [missing-module-attribute]
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class BarcodeCodebook:
    """A marker order and library of valid barcode words.

    Attributes
    ----------
    marker_order
        Channel names in bit-string index order.
    library
        Barcode bit-string -> LNP call name.
    expected_on_bits
        Number of bits called ON per candidate; derived from the library's
        (constant) Hamming weight.
    hamming_tolerance
        Maximum Hamming distance for a ``"tolerant"`` (non-exact) match.
    """

    marker_order: list[str]
    library: dict[str, str]
    expected_on_bits: int
    hamming_tolerance: int

    def __post_init__(self) -> None:
        weights = {bits.count("1") for bits in self.library}
        if len(weights) != 1:
            raise ValueError(f"Barcode library must have constant weight, got {weights}")
        if weights.pop() != self.expected_on_bits:
            raise ValueError("expected_on_bits does not match the library's bit weight")
        if any(len(bits) != len(self.marker_order) for bits in self.library):
            raise ValueError("Every library entry must have one bit per marker")


#: Figure 1d/1e full-barcode codebook (12 channels, single library entry).
#: The porting spec's prose claimed a bit weight of 5, but the literal
#: library string '100110100011' has 6 ones; the codebook validates its own
#: weight against expected_on_bits at construction time, so this mismatch
#: was caught immediately rather than silently miscounting bits at runtime.
FULL_BARCODE_CODEBOOK = BarcodeCodebook(
    marker_order=["A6", "A17", "A55", "A56", "A76", "A79", "A20", "A46", "A28", "A72", "A2", "A63"],
    library={"100110100011": "LNP_A"},
    expected_on_bits=6,
    hamming_tolerance=4,
)

#: Figure 1d Bit_1 (A20-only) single-channel codebook.
BIT_1_CODEBOOK = BarcodeCodebook(
    marker_order=["A20"], library={"1": "LNP_A"}, expected_on_bits=1, hamming_tolerance=0
)

#: Figure 2 Round 1 (S1/S2) 12-channel, 10-entry codebook.
ROUND_1_CODEBOOK = BarcodeCodebook(
    marker_order=["A6", "A17", "A55", "A56", "A76", "A79", "A20", "A46", "A28", "A72", "A2", "A63"],
    library={
        "101001100011": "LNP_01",
        "100110100011": "LNP_02",
        "101010001011": "LNP_03",
        "000101101110": "LNP_04",
        "011010100101": "LNP_05",
        "010000110111": "LNP_06",
        "011110010100": "LNP_07",
        "111101010000": "LNP_08",
        "010100011011": "LNP_09",
        "011011000110": "LNP_10",
    },
    expected_on_bits=6,
    hamming_tolerance=2,
)

#: Figure 2 Round 2 (S3/S4) 15-channel codebook (Round 1 codes, zero-extended by 3 bits).
ROUND_2_CODEBOOK = BarcodeCodebook(
    marker_order=[*ROUND_1_CODEBOOK.marker_order, "A126", "A127", "A128"],
    library={bits + "000": call for bits, call in ROUND_1_CODEBOOK.library.items()},
    expected_on_bits=6,
    hamming_tolerance=2,
)


def detect_log_candidates(
    channel_images: dict[str, np.ndarray],
    thresholds: dict[str, float],
    log_sigma: float = 1.0,
    peak_width: int = 21,
    nms_min_distance: float = 9.0,
    tissue_mask: np.ndarray | None = None,
) -> pd.DataFrame:
    """Detect candidate puncta per marker channel via Laplacian-of-Gaussian filtering.

    Parameters
    ----------
    channel_images
        ``{marker_name: 2D image}``, all the same shape.
    thresholds
        Per-marker LoG-score threshold.
    log_sigma
        Gaussian sigma for the LoG filter.
    peak_width
        Local-maximum-filter footprint size (pixels).
    nms_min_distance
        Minimum pixel distance between accepted candidates from different
        marker channels (greedy, seed-score order).
    tissue_mask
        Optional boolean mask (``True`` = tissue); candidates outside it are
        dropped.

    Returns
    -------
    One row per accepted candidate: ``i, j, score, seed_marker``, ordered by
    descending score before NMS.
    """
    candidates = []
    for marker, image in channel_images.items():
        log_image = _log_score_image(image, log_sigma)
        mask = _local_maxima_mask(log_image, peak_width, thresholds.get(marker, 0.0), tissue_mask)
        ys, xs = np.nonzero(mask)
        for y, x in zip(ys, xs, strict=True):
            candidates.append(
                {"i": int(y), "j": int(x), "score": float(log_image[y, x]), "seed_marker": marker}
            )

    if not candidates:
        return pd.DataFrame(columns=["i", "j", "score", "seed_marker"])

    table = pd.DataFrame(candidates)
    return _greedy_nms(table, nms_min_distance)


def _log_score_image(image: np.ndarray, log_sigma: float) -> np.ndarray:
    """Laplacian-of-Gaussian score image (higher = brighter puncta)."""
    return -ndimage.gaussian_laplace(image.astype(np.float64), sigma=log_sigma)


def _local_maxima_mask(
    score_image: np.ndarray,
    peak_width: int,
    threshold: float,
    tissue_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Boolean mask of local maxima above ``threshold`` (optionally restricted to ``tissue_mask``)."""
    local_max = ndimage.maximum_filter(score_image, size=peak_width) == score_image
    mask = local_max & (score_image > threshold)
    if tissue_mask is not None:
        mask &= tissue_mask
    return mask


def _greedy_nms(table: pd.DataFrame, nms_min_distance: float) -> pd.DataFrame:
    """Greedy non-max suppression: keep candidates in descending score order, dropping any
    within ``nms_min_distance`` pixels of an already-kept candidate."""
    if table.empty:
        return table.reset_index(drop=True)
    table = table.sort_values("score", ascending=False).reset_index(drop=True)
    kept_positions: list[tuple[int, int]] = []
    keep_mask = np.zeros(len(table), dtype=bool)
    for idx, row in table.iterrows():
        # pyrefly: ignore [bad-argument-type]
        idx = int(idx)
        position = (row["i"], row["j"])
        if any(
            (position[0] - ky) ** 2 + (position[1] - kx) ** 2 < nms_min_distance**2
            for ky, kx in kept_positions
        ):
            continue
        keep_mask[idx] = True
        kept_positions.append(position)
    return table.loc[keep_mask].reset_index(drop=True)


def _matches_per_megapixel(
    images: list[np.ndarray], log_sigma: float, peak_width: int, threshold: float
) -> float:
    """Local maxima above ``threshold``, summed over ``images`` and normalised per megapixel."""
    total_matches = 0
    total_pixels = 0
    for image in images:
        score_image = _log_score_image(image, log_sigma)
        mask = _local_maxima_mask(score_image, peak_width, threshold)
        total_matches += int(mask.sum())
        total_pixels += image.size
    megapixels = total_pixels / 1_000_000
    return total_matches / megapixels if megapixels > 0 else 0.0


def calibrate_marker_threshold(
    negative_images: list[np.ndarray],
    positive_images: list[np.ndarray],
    candidate_thresholds: list[float],
    log_sigma: float = 1.0,
    peak_width: int = 21,
    max_neg_matches_per_mpx: float = 3.0,
    min_pos_matches_per_mpx: float = 0.5,
    policy_bonus: float = 1000.0,
) -> tuple[float, pd.DataFrame]:
    """Grid-search a per-marker LoG threshold from matched negative/positive calibration fields.

    Ports the per-marker threshold calibration performed in Section 4 of the
    raw-image notebooks (a per-candidate-threshold grid search maximising an
    enrichment/pass-rate score), generalised to any marker rather than
    hardcoded per notebook.

    Parameters
    ----------
    negative_images, positive_images
        Small calibration fields (e.g. random tissue-containing crops) from
        a negative-control region and a positive region, for one marker
        channel.
    candidate_thresholds
        LoG-score thresholds to evaluate.
    log_sigma, peak_width
        LoG filter parameters (must match the parameters used downstream).
    max_neg_matches_per_mpx
        Policy ceiling on negative-field match density.
    min_pos_matches_per_mpx
        Policy floor on positive-field match density.
    policy_bonus
        Added to a candidate's score when it satisfies both policy bounds,
        so a policy-satisfying threshold is always preferred over one that
        merely has a higher raw enrichment.

    Returns
    -------
    ``(selected_threshold, results)``: the threshold with the highest score
    (ties broken by the lowest threshold), and the full per-candidate table
    with columns ``threshold, neg_matches_per_mpx, pos_matches_per_mpx,
    enrichment, passes_policy, selection_score``.
    """
    rows = []
    for threshold in candidate_thresholds:
        neg_rate = _matches_per_megapixel(negative_images, log_sigma, peak_width, threshold)
        pos_rate = _matches_per_megapixel(positive_images, log_sigma, peak_width, threshold)
        enrichment = (
            pos_rate / neg_rate if neg_rate > 0 else (float("inf") if pos_rate > 0 else 0.0)
        )
        passes_policy = neg_rate <= max_neg_matches_per_mpx and pos_rate >= min_pos_matches_per_mpx
        finite_enrichment = enrichment if np.isfinite(enrichment) else 0.0
        selection_score = finite_enrichment + (policy_bonus if passes_policy else 0.0)
        rows.append(
            {
                "threshold": threshold,
                "neg_matches_per_mpx": neg_rate,
                "pos_matches_per_mpx": pos_rate,
                "enrichment": enrichment,
                "passes_policy": passes_policy,
                "selection_score": selection_score,
            }
        )
    results = pd.DataFrame(rows)
    best = results.sort_values(["selection_score", "threshold"], ascending=[False, True]).iloc[0]
    return float(best["threshold"]), results


def calibrate_decode_threshold(
    negative_decoded: pd.DataFrame,
    positive_decoded: pd.DataFrame,
    candidate_min_on_snr: list[float],
    negative_megapixels: float,
    positive_megapixels: float,
    max_neg_matches_per_mpx: float = 3.0,
    min_pos_matches_per_mpx: float = 0.5,
    policy_bonus: float = 1000.0,
) -> tuple[float, pd.DataFrame]:
    """Grid-search the global ``min_on_snr`` decode threshold from calibration-field decodes.

    Ports Section 4's global decode-threshold calibration: candidates from
    :func:`decode_candidates` (run once at a very low ``min_on_snr`` over the
    calibration fields) are re-filtered at each candidate threshold, scored
    the same way as :func:`calibrate_marker_threshold`.

    Parameters
    ----------
    negative_decoded, positive_decoded
        Output of :func:`decode_candidates` for the negative and positive
        calibration fields, with a ``min_on_snr`` column per candidate.
    candidate_min_on_snr
        Threshold values to evaluate.
    negative_megapixels, positive_megapixels
        Total imaged area of each field set, for density normalisation.
    max_neg_matches_per_mpx, min_pos_matches_per_mpx, policy_bonus
        As in :func:`calibrate_marker_threshold`.

    Returns
    -------
    ``(selected_min_on_snr, results)``.
    """
    rows = []
    for threshold in candidate_min_on_snr:
        neg_count = int(
            (
                (negative_decoded["min_on_snr"] >= threshold)
                & negative_decoded["barcode_match_status"].isin(["exact", "tolerant"])
            ).sum()
        )
        pos_count = int(
            (
                (positive_decoded["min_on_snr"] >= threshold)
                & positive_decoded["barcode_match_status"].isin(["exact", "tolerant"])
            ).sum()
        )
        neg_rate = neg_count / negative_megapixels if negative_megapixels > 0 else 0.0
        pos_rate = pos_count / positive_megapixels if positive_megapixels > 0 else 0.0
        enrichment = (
            pos_rate / neg_rate if neg_rate > 0 else (float("inf") if pos_rate > 0 else 0.0)
        )
        passes_policy = neg_rate <= max_neg_matches_per_mpx and pos_rate >= min_pos_matches_per_mpx
        finite_enrichment = enrichment if np.isfinite(enrichment) else 0.0
        selection_score = finite_enrichment + (policy_bonus if passes_policy else 0.0)
        rows.append(
            {
                "min_on_snr": threshold,
                "neg_matches_per_mpx": neg_rate,
                "pos_matches_per_mpx": pos_rate,
                "enrichment": enrichment,
                "passes_policy": passes_policy,
                "selection_score": selection_score,
            }
        )
    results = pd.DataFrame(rows)
    best = results.sort_values(["selection_score", "min_on_snr"], ascending=[False, True]).iloc[0]
    return float(best["min_on_snr"]), results


def raw_intensity_rescue_candidates(
    channel_images: dict[str, np.ndarray],
    bright_percentile: float = 98.0,
    object_percentile: float = 99.0,
    object_min_area: int = 4,
    object_max_area: int = 400,
    peak_width: int = 3,
) -> pd.DataFrame:
    """Recover bright-point and blob-shaped candidates from a raw channel-max projection.

    Ports the raw-intensity "rescue" pass (Round 1/2 notebooks): a
    percentile-threshold local-maximum pass over the per-pixel maximum
    across channels, and a percentile-threshold connected-component pass
    within an area window, catching bright or blob-shaped candidates the
    LoG bank might miss. Merge the result with LoG candidates via
    :func:`detect_log_candidates` and :func:`_greedy_nms` (see
    ``workflow/scripts/run_raw_image_spot_detection.py``).

    Parameters
    ----------
    channel_images
        ``{marker_name: 2D image}``; the per-pixel maximum across all of
        them is the "raw rescue" image.
    bright_percentile
        Percentile threshold for the bright-point pass.
    object_percentile
        Percentile threshold for the connected-component pass.
    object_min_area, object_max_area
        Connected-component area window (pixels).
    peak_width
        Local-maximum-filter footprint for the bright-point pass.

    Returns
    -------
    One row per rescued candidate: ``i, j, score, seed_marker``
    (``"rescue_bright"`` or ``"rescue_object"``).
    """
    stacked = np.stack(list(channel_images.values()), axis=0)
    raw_max = stacked.max(axis=0).astype(np.float64)

    bright_threshold = float(np.percentile(raw_max, bright_percentile))
    bright_mask = _local_maxima_mask(raw_max, peak_width, bright_threshold)
    ys, xs = np.nonzero(bright_mask)
    bright_rows = [
        {"i": int(y), "j": int(x), "score": float(raw_max[y, x]), "seed_marker": "rescue_bright"}
        for y, x in zip(ys, xs, strict=True)
    ]

    object_threshold = float(np.percentile(raw_max, object_percentile))
    labels, n_labels = ndimage.label(raw_max > object_threshold)
    object_rows = []
    if n_labels:
        areas = ndimage.sum(np.ones_like(labels), labels, index=np.arange(1, n_labels + 1))
        centroids = ndimage.center_of_mass(raw_max, labels, index=np.arange(1, n_labels + 1))
        for area, centroid in zip(areas, centroids, strict=True):
            if not (object_min_area <= area <= object_max_area):
                continue
            y, x = centroid
            object_rows.append(
                {
                    "i": int(round(y)),
                    "j": int(round(x)),
                    "score": float(raw_max[int(round(y)), int(round(x))]),
                    "seed_marker": "rescue_object",
                }
            )

    rows = bright_rows + object_rows
    if not rows:
        return pd.DataFrame(columns=["i", "j", "score", "seed_marker"])
    return pd.DataFrame(rows)


def detect_log_candidates_tiled(
    channel_images: dict[str, np.ndarray],
    thresholds: dict[str, float],
    n_tiles: int,
    tile_overlap: int,
    checkpoint_dir: Path,
    region_name: str,
    log_sigma: float = 1.0,
    peak_width: int = 21,
    nms_min_distance: float = 9.0,
    use_raw_rescue: bool = False,
    rescue_kwargs: dict | None = None,
    force_rerun: bool = False,
) -> pd.DataFrame:
    """Tiled, checkpointed candidate detection over a large image, resumable per tile.

    Ports the y-axis tiling and per-tile CSV checkpointing the Round 1/2
    notebooks use because their registered stacks are hundreds of gigabytes:
    splits ``channel_images`` into ``n_tiles`` overlapping horizontal bands,
    runs :func:`detect_log_candidates` (optionally merged with
    :func:`raw_intensity_rescue_candidates`) on each tile independently,
    keeps only candidates in the tile's non-overlapping "core" rows (so
    overlap regions are not double-counted), and writes/reads one checkpoint
    CSV per tile so a killed run resumes without recomputation.

    Parameters
    ----------
    channel_images
        ``{marker_name: full-size 2D image}``.
    thresholds
        Per-marker LoG-score threshold.
    n_tiles
        Number of y-axis tiles to split the image into.
    tile_overlap
        Rows of overlap on each side of a tile's core band.
    checkpoint_dir
        Directory for per-tile checkpoint CSVs (created if absent).
    region_name
        Used to name checkpoint files (``spots_all_candidates_<region>_tile<NNN>.csv``).
    log_sigma, peak_width, nms_min_distance
        As in :func:`detect_log_candidates`.
    use_raw_rescue
        Also merge in :func:`raw_intensity_rescue_candidates` per tile.
    rescue_kwargs
        Extra keyword arguments forwarded to
        :func:`raw_intensity_rescue_candidates`.
    force_rerun
        Recompute every tile even if a checkpoint exists.

    Returns
    -------
    All tiles' kept candidates, with coordinates already in full-image space.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    height = next(iter(channel_images.values())).shape[0]
    tile_height = height // n_tiles
    tile_frames = []

    for tile_index in range(n_tiles):
        checkpoint_path = (
            checkpoint_dir / f"spots_all_candidates_{region_name}_tile{tile_index:03d}.csv"
        )
        if checkpoint_path.exists() and not force_rerun:
            tile_frames.append(pd.read_csv(checkpoint_path))
            continue

        core_start = tile_index * tile_height
        core_end = height if tile_index == n_tiles - 1 else core_start + tile_height
        read_start = max(0, core_start - tile_overlap)
        read_end = min(height, core_end + tile_overlap)

        tile_images = {
            marker: image[read_start:read_end, :] for marker, image in channel_images.items()
        }
        candidates = detect_log_candidates(
            tile_images,
            thresholds,
            log_sigma=log_sigma,
            peak_width=peak_width,
            nms_min_distance=nms_min_distance,
        )
        if use_raw_rescue:
            rescue = raw_intensity_rescue_candidates(tile_images, **(rescue_kwargs or {}))
            candidates = _greedy_nms(
                pd.concat([candidates, rescue], ignore_index=True), nms_min_distance
            )

        candidates["i"] = candidates["i"] + read_start
        core_mask = (candidates["i"] >= core_start) & (candidates["i"] < core_end)
        tile_result = candidates.loc[core_mask].reset_index(drop=True)
        tile_result.to_csv(checkpoint_path, index=False)
        tile_frames.append(tile_result)

    non_empty = [frame for frame in tile_frames if not frame.empty]
    if not non_empty:
        return pd.DataFrame(columns=["i", "j", "score", "seed_marker"])
    return pd.concat(non_empty, ignore_index=True)


def _local_snr(
    image: np.ndarray, i: int, j: int, core_radius: int, inner: int, outer: int
) -> float:
    """Local signal-to-background ratio: max in a core disk over median in a surrounding annulus."""
    y0, y1 = max(0, i - outer), min(image.shape[0], i + outer + 1)
    x0, x1 = max(0, j - outer), min(image.shape[1], j + outer + 1)
    patch = image[y0:y1, x0:x1]
    yy, xx = np.ogrid[y0 - i : y1 - i, x0 - j : x1 - j]
    distance = np.sqrt(yy**2 + xx**2)
    core = patch[distance <= core_radius]
    annulus = patch[(distance > inner) & (distance <= outer)]
    signal = float(core.max()) if core.size else 0.0
    background = float(np.median(annulus)) if annulus.size else 0.0
    return signal / background if background > 0 else float("inf") if signal > 0 else 0.0


def decode_candidates(
    channel_images: dict[str, np.ndarray],
    candidates: pd.DataFrame,
    codebook: BarcodeCodebook,
    min_on_snr: float = 1.2,
    min_snr_margin: float = 0.05,
    core_radius: int = 1,
    annulus_inner: int = 3,
    annulus_outer: int = 5,
    bright_snr_threshold: float = 1.75,
    max_bright_markers: int | None = None,
) -> pd.DataFrame:
    """Decode each candidate's per-marker SNR into a barcode word and match it against the codebook.

    Parameters
    ----------
    channel_images
        ``{marker_name: 2D image}`` covering every ``codebook.marker_order`` entry.
    candidates
        Output of :func:`detect_log_candidates`.
    codebook
        Marker order, library and decode tolerances.
    min_on_snr
        Minimum SNR required of the weakest called-ON bit.
    min_snr_margin
        Minimum gap between the weakest ON-bit SNR and the strongest OFF-bit SNR.
    core_radius, annulus_inner, annulus_outer
        Local SNR ring geometry (pixels).
    bright_snr_threshold, max_bright_markers
        A candidate is flagged ``promiscuous`` when more than
        ``max_bright_markers`` channels exceed ``bright_snr_threshold``
        (``max_bright_markers`` defaults to ``len(marker_order)``, i.e. the
        filter is disabled unless a caller sets a stricter cap, matching how
        the source notebooks set it close to or equal to the total marker
        count for their multi-channel codebooks).

    Returns
    -------
    ``candidates`` with added ``called_code, matched_barcode, lnp_call,
    barcode_match_distance, barcode_match_status, min_on_snr, snr_margin,
    promiscuous, pass_quality, accepted`` columns.
    """
    if max_bright_markers is None:
        max_bright_markers = len(codebook.marker_order)

    decoded_columns = [
        *candidates.columns,
        "called_code",
        "matched_barcode",
        "lnp_call",
        "barcode_match_distance",
        "barcode_match_status",
        "min_on_snr",
        "snr_margin",
        "promiscuous",
        "pass_quality",
        "accepted",
    ]
    if candidates.empty:
        return pd.DataFrame(columns=decoded_columns)

    rows = []
    for _, candidate in candidates.iterrows():
        i, j = int(candidate["i"]), int(candidate["j"])
        snrs = {
            marker: _local_snr(
                channel_images[marker], i, j, core_radius, annulus_inner, annulus_outer
            )
            for marker in codebook.marker_order
        }
        ranked = sorted(snrs.items(), key=lambda item: item[1], reverse=True)
        on_markers = {marker for marker, _ in ranked[: codebook.expected_on_bits]}
        called_code = "".join(
            "1" if marker in on_markers else "0" for marker in codebook.marker_order
        )

        on_snrs = [snrs[m] for m in on_markers]
        off_snrs = [snrs[m] for m in codebook.marker_order if m not in on_markers]
        weakest_on = min(on_snrs) if on_snrs else 0.0
        strongest_off = max(off_snrs) if off_snrs else 0.0
        n_bright = sum(1 for snr in snrs.values() if snr >= bright_snr_threshold)
        promiscuous = n_bright > max_bright_markers

        best_distance = min(
            (
                sum(a != b for a, b in zip(called_code, lib_code, strict=True))
                for lib_code in codebook.library
            ),
            default=len(codebook.marker_order),
        )
        if best_distance == 0:
            status, lnp_call = "exact", codebook.library[called_code]
        elif best_distance <= codebook.hamming_tolerance:
            closest = min(
                codebook.library,
                key=lambda lib_code: sum(
                    a != b for a, b in zip(called_code, lib_code, strict=True)
                ),
            )
            status, lnp_call = "tolerant", codebook.library[closest]
        else:
            status, lnp_call = "unmapped", "no_barcode"

        pass_quality = (
            weakest_on >= min_on_snr
            and (weakest_on - strongest_off) >= min_snr_margin
            and not promiscuous
            and status in {"exact", "tolerant"}
        )
        rows.append(
            {
                **candidate.to_dict(),
                "called_code": called_code,
                "matched_barcode": lnp_call if status != "unmapped" else None,
                "lnp_call": lnp_call,
                "barcode_match_distance": best_distance,
                "barcode_match_status": status,
                "min_on_snr": weakest_on,
                "snr_margin": weakest_on - strongest_off,
                "promiscuous": promiscuous,
                "pass_quality": pass_quality,
                "accepted": pass_quality,
            }
        )
    return pd.DataFrame(rows)


def assign_spots_to_cells(
    spots: pd.DataFrame, cell_centroids: pd.DataFrame, radius_px: float = 25.0
) -> pd.DataFrame:
    """Assign accepted spots to their nearest cell centroid, within ``radius_px``.

    Parameters
    ----------
    spots
        Accepted spots with ``i`` (row), ``j`` (column) pixel coordinates.
    cell_centroids
        Frozen segmentation table with ``label``, ``y``, ``x`` columns.
    radius_px
        Maximum assignment distance.

    Returns
    -------
    ``spots`` with added ``cell`` (nearest centroid's ``label``, or ``NA`` if
    none within range) and ``cell_distance_px`` columns.
    """
    if spots.empty or cell_centroids.empty:
        result = spots.copy()
        result["cell"] = pd.array([pd.NA] * len(spots), dtype="Int64")
        result["cell_distance_px"] = np.nan
        return result

    tree = cKDTree(cell_centroids[["y", "x"]].to_numpy())
    distances, indices = tree.query(spots[["i", "j"]].to_numpy(), distance_upper_bound=radius_px)
    labels = cell_centroids["label"].to_numpy()
    assigned = np.where(
        np.isfinite(distances), labels[np.clip(indices, 0, len(labels) - 1)], np.nan
    )

    result = spots.copy()
    result["cell"] = pd.array(assigned, dtype="Int64")
    result["cell_distance_px"] = np.where(np.isfinite(distances), distances, np.nan)
    return result


def aggregate_cell_bit_counts(spots: pd.DataFrame, marker_order: list[str]) -> pd.DataFrame:
    """Sum per-marker bit counts per assigned cell, from accepted, cell-assigned spots.

    Parameters
    ----------
    spots
        Output of :func:`assign_spots_to_cells`, restricted to
        ``accepted & cell.notna()``.
    marker_order
        Codebook marker order; produces one ``bit_<marker>`` column per entry.

    Returns
    -------
    One row per cell: ``cell``, one ``bit_<marker>`` column per marker
    (spot count with that marker ON), ``decoded_spots``, ``total_spots``.
    """
    assigned = spots.dropna(subset=["cell"])
    if assigned.empty:
        columns = ["cell", *[f"bit_{m}" for m in marker_order], "decoded_spots", "total_spots"]
        return pd.DataFrame(columns=columns)

    rows = []
    for cell, group in assigned.groupby("cell"):
        row: dict[str, object] = {"cell": cell}
        for marker in marker_order:
            row[f"bit_{marker}"] = int(
                group["called_code"].str[marker_order.index(marker)].eq("1").sum()
            )
        row["decoded_spots"] = int((group["barcode_match_status"] != "unmapped").sum())
        row["total_spots"] = int(len(group))
        rows.append(row)
    return pd.DataFrame(rows)


def build_cell_analysis_table(
    features: pd.DataFrame, cell_bit_counts: pd.DataFrame, codebook: BarcodeCodebook
) -> pd.DataFrame:
    """Build the final per-cell codebook-call table from frozen features and aggregated bit counts.

    Parameters
    ----------
    features
        Frozen segmentation table (``label`` renamed ``cell``, plus ``x``,
        ``y`` and any other per-cell measurements).
    cell_bit_counts
        Output of :func:`aggregate_cell_bit_counts`.
    codebook
        Marker order, library and decode tolerance, for the consensus
        barcode call.

    Returns
    -------
    One row per cell with ``cell, x, y, barcode, lnp_call, lnp_positive,
    barcode_match_status, barcode_match_distance, total_barcode_spots``, plus
    every column carried over from ``features``.
    """
    merged = features.rename(columns={"label": "cell"}).merge(
        cell_bit_counts, on="cell", how="left"
    )
    bit_cols = [f"bit_{m}" for m in codebook.marker_order]
    for col in bit_cols:
        if col not in merged.columns:
            merged[col] = 0
    merged[bit_cols] = merged[bit_cols].fillna(0).astype(int)
    merged["total_barcode_spots"] = merged[bit_cols].sum(axis=1)

    barcodes = []
    calls = []
    statuses = []
    distances = []
    for _, row in merged.iterrows():
        bit_counts = [row[f"bit_{m}"] for m in codebook.marker_order]
        called_code = "".join("1" if count >= 1 else "0" for count in bit_counts)
        best_distance = min(
            (
                sum(a != b for a, b in zip(called_code, lib_code, strict=True))
                for lib_code in codebook.library
            ),
            default=len(codebook.marker_order),
        )
        if row["total_barcode_spots"] == 0:
            status, call = "no_barcode", "no_barcode"
        elif best_distance == 0:
            status, call = "exact", codebook.library[called_code]
        elif best_distance <= codebook.hamming_tolerance:
            closest = min(
                codebook.library,
                key=lambda lib_code: sum(
                    a != b for a, b in zip(called_code, lib_code, strict=True)
                ),
            )
            status, call = "tolerant", codebook.library[closest]
        else:
            status, call = "ambiguous_mixed", "no_barcode"
        barcodes.append(called_code)
        calls.append(call)
        statuses.append(status)
        distances.append(best_distance)

    merged["barcode"] = barcodes
    merged["lnp_call"] = calls
    merged["barcode_match_status"] = statuses
    merged["barcode_match_distance"] = distances
    merged["lnp_positive"] = merged["barcode_match_status"].isin(["exact", "tolerant"])
    return merged


def segment_nuclei_cellpose(
    image: np.ndarray,
    diameter: float = 25.0,
    min_area_px: int = 100,
    max_area_px: int = 5000,
    gpu: bool = False,
    model_type: str = "cpsam",
) -> np.ndarray:
    """Segment nuclei with Cellpose (CPU by default), size-filtered and relabelled from 1.

    Parameters
    ----------
    image
        Single-channel 2D nuclear-stain image.
    diameter
        Expected nucleus diameter, pixels.
    min_area_px, max_area_px
        Connected-component area window; labels outside it are dropped.
    gpu
        Whether to request Cellpose's GPU path. This repository has no GPU
        access, so callers must pass ``False`` (the default); the workflow
        never sets this to ``True``.
    model_type
        Cellpose pretrained model name. Cellpose 4.x (``pyproject.toml``
        pins ``cellpose>=4,<5``) always uses the single bundled ``cpsam``
        ("Cellpose-SAM") model and its ``models.CellposeModel`` class
        (``models.Cellpose`` and per-image ``channels=`` no longer exist);
        this parameter is accepted for documentation and forward
        compatibility but currently ignored by Cellpose itself.

    Returns
    -------
    An ``int32`` label mask, relabelled consecutively from 1 (0 = background).
    """
    from cellpose import models

    model = models.CellposeModel(gpu=gpu, model_type=model_type)
    masks, _, _ = model.eval(image, diameter=diameter)

    # pyrefly: ignore [unsupported-operation]
    areas = ndimage.sum(np.ones_like(masks), masks, index=np.unique(masks[masks > 0]))
    # pyrefly: ignore [unsupported-operation]
    labels = np.unique(masks[masks > 0])
    keep = labels[(areas >= min_area_px) & (areas <= max_area_px)]

    relabelled = np.zeros_like(masks, dtype=np.int32)
    for new_label, old_label in enumerate(keep, start=1):
        relabelled[masks == old_label] = new_label
    return relabelled


def segment_nuclei_threshold(
    image: np.ndarray, min_area_px: int = 100, max_area_px: int = 5000
) -> np.ndarray:
    """Segment nuclei by Otsu thresholding and connected components (Cellpose-free fallback).

    Not part of the source notebook: :func:`segment_nuclei_cellpose` needs a
    downloaded pretrained model, which the smoke workflow should not depend
    on fetching over the network. This threshold-based segmentation is used
    only when ``config["raw_image_spot_detection"]["supplementary_figure_1c"]
    ["use_cellpose"]`` is ``false`` (the smoke default), and is documented in
    the README as an intentional smoke-only substitution, never used for a
    real-data run.

    Parameters
    ----------
    image
        Single-channel 2D nuclear-stain image.
    min_area_px, max_area_px
        Connected-component area window; labels outside it are dropped.

    Returns
    -------
    An ``int32`` label mask, relabelled consecutively from 1 (0 = background).
    """
    from skimage.filters import threshold_otsu

    threshold = threshold_otsu(image)
    labels, _ = ndimage.label(image > threshold)
    areas = ndimage.sum(np.ones_like(labels), labels, index=np.unique(labels[labels > 0]))
    keep = np.unique(labels[labels > 0])[(areas >= min_area_px) & (areas <= max_area_px)]

    relabelled = np.zeros_like(labels, dtype=np.int32)
    for new_label, old_label in enumerate(keep, start=1):
        relabelled[labels == old_label] = new_label
    return relabelled


def classify_cells_by_barcode_presence(
    spot_counts: pd.DataFrame, min_spots_per_barcode: int = 1
) -> pd.DataFrame:
    """Classify cells by which of two barcodes they express (Supplementary Figure 1c).

    Parameters
    ----------
    spot_counts
        One row per cell with ``barcode1_spot_count``, ``barcode2_spot_count``.
    min_spots_per_barcode
        Minimum spot count for a barcode to be called present.

    Returns
    -------
    ``spot_counts`` with added ``barcode1_present``, ``barcode2_present`` and
    ``classification`` (``barcode1_only``, ``barcode2_only``,
    ``both_barcodes`` or ``no_barcode``) columns.
    """
    if spot_counts.empty:
        return pd.DataFrame(
            columns=[*spot_counts.columns, "barcode1_present", "barcode2_present", "classification"]
        )
    result = spot_counts.copy()
    result["barcode1_present"] = result["barcode1_spot_count"] >= min_spots_per_barcode
    result["barcode2_present"] = result["barcode2_spot_count"] >= min_spots_per_barcode

    def classify(row: pd.Series) -> str:
        if row["barcode1_present"] and row["barcode2_present"]:
            return "both_barcodes"
        if row["barcode1_present"]:
            return "barcode1_only"
        if row["barcode2_present"]:
            return "barcode2_only"
        return "no_barcode"

    result["classification"] = result.apply(classify, axis=1)
    return result
