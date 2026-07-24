#!/usr/bin/env python3
"""Read-only preflight check for the NanoSTAMP publication package."""

from __future__ import annotations

import ast
import csv
import gzip
import json
import sys
from pathlib import Path


EXPECTED_NOTEBOOKS = [
    "Supplementary_Figure_1c_Image_Processing_and_Quantification.ipynb",
    "Figure_1d_Bit_1_Spot_Detection.ipynb",
    "Figure_1d_1e_Full_Barcode_Spot_Detection.ipynb",
    "Figure_1d_1e_Spleen_LNP_Analysis.ipynb",
    "Figure_1f_1g_Spleen_Neighborhood_Analysis.ipynb",
    "Figure_2_and_Supplementary_6_11_Multiplex_LNP_Spot_Detection_Round_1.ipynb",
    "Figure_2_and_Supplementary_6_11_Multiplex_LNP_Spot_Detection_Round_2.ipynb",
    "Figure_2_and_Supplementary_6_8_Multiplex_LNP_Cell_and_Functional_Analysis.ipynb",
    "Figure_2_and_Supplementary_9_11_Multiplex_LNP_Spatial_Neighborhood_Analysis.ipynb",
]

SUP1C_SAMPLES = [
    "B10_2026-02-27_17-04-09.302384",
    "B8_2026-02-27_16-56-27.406809",
    "C11_2026-02-27_17-02-43.916718",
    "C3_1_2026-02-26_17-11-36.990927",
    "C5_2026-02-26_17-04-44.335246",
    "C7_2026-02-27_16-55-00.641006",
    "D10_2026-02-27_16-57-48.341383",
    "D11_2026-02-27_17-01-18.287206",
    "E10_2026-02-27_16-59-01.446545",
    "E11_2026-02-27_17-00-05.688559",
    "PBS_2026-02-26_16-59-28.287747",
    "PBS_2_2026-02-26_17-01-06.874313",
]


def find_nanostamp_root() -> Path:
    here = Path(__file__).resolve()
    for base in [here.parent, *here.parents]:
        if base.name == "NanoSTAMP" and (base / "Code").is_dir() and (base / "Data").is_dir():
            return base
    raise FileNotFoundError("Could not locate the NanoSTAMP publication root.")


def csv_header(path: Path) -> set[str]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as handle:
        return set(next(csv.reader(handle)))


def validate_notebooks(code_dir: Path, errors: list[str]) -> None:
    for name in EXPECTED_NOTEBOOKS:
        path = code_dir / name
        if not path.is_file():
            errors.append(f"Missing notebook: {path}")
            continue
        try:
            notebook = json.loads(path.read_text())
        except Exception as exc:
            errors.append(f"Unreadable notebook JSON: {path}: {exc}")
            continue
        for index, cell in enumerate(notebook.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            text = "".join(cell.get("source", []))
            try:
                ast.parse(text)
            except SyntaxError as exc:
                errors.append(f"Syntax error: {path.name}, cell {index}, line {exc.lineno}: {exc.msg}")
            if cell.get("execution_count") is not None:
                errors.append(f"Execution count retained: {path.name}, cell {index}")
            if cell.get("outputs"):
                errors.append(f"Stored output retained: {path.name}, cell {index}")
            if "/mnt/" in text or "/Volumes/" in text:
                errors.append(f"Machine-specific absolute path: {path.name}, cell {index}")


def validate_manifest(base: Path, manifest: Path, errors: list[str]) -> int:
    count = 0
    with manifest.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if "destination" in row:
                path = base.parents[2] / row["destination"]
                expected = int(row["bytes"])
            else:
                path = manifest.parent / row["folder"] / row["filename"]
                expected = int(row["size_bytes"])
            count += 1
            if not path.is_file():
                errors.append(f"Missing manifest file: {path}")
            elif path.stat().st_size != expected:
                errors.append(
                    f"Size mismatch: {path}: found {path.stat().st_size}, expected {expected}"
                )
    return count


def validate_compact_schemas(data_dir: Path, errors: list[str]) -> None:
    checks = [
        (
            data_dir / "Figure_1d_1e_Spleen_LNP/Precomputed_Analysis_Input/bit_1_cell_calls_spleen.csv.gz",
            {"region", "cell", "x", "y", "bit_1_positive"},
        ),
        (
            data_dir / "Figure_1d_1e_Spleen_LNP/Precomputed_Analysis_Input/codebook_cell_calls_spleen.csv.gz",
            {"region", "cell", "x", "y", "lnp_positive"},
        ),
        (
            data_dir / "Figure_1d_1e_Spleen_LNP/Precomputed_Analysis_Input/frozen_cell_annotations_spleen.csv.gz",
            {"region", "cell", "cell_type"},
        ),
        (
            data_dir / "Figure_1f_1g_Spleen_Neighborhoods/Precomputed_Analysis_Input/reg001_neighborhood_cells.csv.gz",
            {"region", "cell", "x", "y", "lnp_positive", "cell_type"},
        ),
    ]
    for path, required in checks:
        if not path.is_file():
            errors.append(f"Missing compact analysis input: {path}")
            continue
        missing = required - csv_header(path)
        if missing:
            errors.append(f"Missing columns in {path}: {sorted(missing)}")


def validate_sup1c(data_dir: Path, errors: list[str]) -> None:
    root = data_dir / "Supplementary_Figure_1c_2Oligo_RCA"
    raw = root / "Raw_Images"
    total_tiffs = 0
    total_bytes = 0
    channel_tokens = ["405_nm_Ex", "561_nm_Ex", "638_nm_Ex"]
    for sample in SUP1C_SAMPLES:
        sample_root = raw / sample
        image_dir = sample_root / "0"
        if not image_dir.is_dir():
            errors.append(f"Missing Supplementary Figure 1c image folder: {image_dir}")
            continue
        files = [
            path
            for path in image_dir.glob("*.tif*")
            if not path.name.startswith("._") and path.stat().st_size > 100_000
        ]
        total_tiffs += len(files)
        total_bytes += sum(path.stat().st_size for path in files)
        if len(files) != 75:
            errors.append(f"{sample}: found {len(files)} analysis TIFFs; expected 75")
        for token in channel_tokens:
            count = sum(token in path.name for path in files)
            if count != 25:
                errors.append(f"{sample}: found {count} {token} TIFFs; expected 25")
        coordinate_file = image_dir / "coordinates.csv"
        if not coordinate_file.is_file():
            errors.append(f"Missing coordinates: {coordinate_file}")
        else:
            missing = {"x (mm)", "y (mm)"} - csv_header(coordinate_file)
            if missing:
                errors.append(f"Missing coordinate columns in {coordinate_file}: {sorted(missing)}")
        for metadata_name in ["acquisition parameters.json", "configurations.xml"]:
            if not (sample_root / metadata_name).is_file():
                errors.append(f"Missing acquisition metadata: {sample_root / metadata_name}")
    if total_tiffs != 900:
        errors.append(f"Supplementary Figure 1c has {total_tiffs} analysis TIFFs; expected 900")
    if total_bytes != 7_817_731_200:
        errors.append(
            f"Supplementary Figure 1c TIFF bytes are {total_bytes}; expected 7,817,731,200"
        )
    processed = root / "Processed_Data"
    processed_csvs = [p for p in processed.rglob("*.csv") if not p.name.startswith("._")]
    if len(processed_csvs) != 46:
        errors.append(f"Processed Supplementary Figure 1c archive has {len(processed_csvs)} CSVs; expected 46")
    for path in processed_csvs:
        try:
            csv_header(path)
        except Exception as exc:
            errors.append(f"Unreadable processed CSV: {path}: {exc}")


def validate_downstream_headers(data_dir: Path, errors: list[str]) -> None:
    root = data_dir / "Figure_2_and_Supplementary_Figures_6_11_Multiplex_LNP/Precomputed_Downstream_Input"
    configs = [
        ("Round_1_S1_S2", "20260628_10_LNP_subcluster_labeled_v3.h5ad"),
        ("Round_2_S3_S4", "20260711_10_LNP_subcluster_labeled_v3.h5ad"),
    ]
    hdf5_signature = b"\x89HDF\r\n\x1a\n"
    required_cell_columns = {
        "region", "cell", "x", "y", "lnp_call", "lnp_positive",
        "barcode", "barcode_match_status", "Fluc", "OVA",
        "SIINFEKL_H-2Kb", "CD86",
    }
    for batch, filename in configs:
        h5ad_path = root / batch / filename
        try:
            with h5ad_path.open("rb") as handle:
                if handle.read(8) != hdf5_signature:
                    errors.append(f"Invalid HDF5/H5AD signature: {h5ad_path}")
        except Exception as exc:
            errors.append(f"Unreadable H5AD: {h5ad_path}: {exc}")
        cell_path = root / batch / "cell_analysis_table_all_regions_v4.csv"
        try:
            missing = required_cell_columns - csv_header(cell_path)
            if missing:
                errors.append(f"Missing full-barcode columns in {cell_path}: {sorted(missing)}")
        except Exception as exc:
            errors.append(f"Unreadable full-barcode cell table: {cell_path}: {exc}")


def main() -> int:
    nanostamp = find_nanostamp_root()
    code_dir = nanostamp / "Code"
    data_dir = nanostamp / "Data"
    errors: list[str] = []
    warnings: list[str] = []

    validate_notebooks(code_dir, errors)
    manifest_count = 0
    for manifest in [
        data_dir / "Figure_1d_1e_Spleen_LNP/FILE_MANIFEST.csv",
        data_dir / "Figure_1f_1g_Spleen_Neighborhoods/FILE_MANIFEST.csv",
        data_dir / "Figure_2_and_Supplementary_Figures_6_11_Multiplex_LNP/spot_detection_input_manifest.csv",
        data_dir / "Figure_2_and_Supplementary_Figures_6_11_Multiplex_LNP/downstream_analysis_input_manifest.csv",
    ]:
        if not manifest.is_file():
            errors.append(f"Missing manifest: {manifest}")
        else:
            manifest_count += validate_manifest(data_dir, manifest, errors)
    validate_compact_schemas(data_dir, errors)
    validate_sup1c(data_dir, errors)
    validate_downstream_headers(data_dir, errors)

    apple_double = list(nanostamp.rglob("._*"))
    if apple_double:
        warnings.append(
            f"Found {len(apple_double)} AppleDouble metadata files; exclude `._*` when uploading"
        )

    print(f"Checked {len(EXPECTED_NOTEBOOKS)} notebooks and {manifest_count} manifest-tracked files.")
    for warning in warnings:
        print(f"WARNING: {warning}")
    if errors:
        print(f"FAILED with {len(errors)} problem(s):")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("PASS: publication package structure, file sizes, notebook syntax, and required schemas are complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
