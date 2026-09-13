"""Build a tiny synthetic fixture, shaped like the real deposited/raw data.

Used only by ``config/smoke.yaml`` (via the ``make_smoke_fixture`` Snakemake
rule) so the whole DAG can run end to end in this repository without the
excluded raw TIFFs or the Duke-deposited processed data. Every table here
matches the column schema documented in the porting spec and enforced by
``nanostamp.manifest``/the analysis modules; the values themselves carry no
scientific meaning.
"""

import gzip
import json
import shutil
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import tifffile

RNG = np.random.default_rng(0)


def _write_csv_gz(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="") as handle:
        df.to_csv(handle, index=False)


def make_figure_1d_1e_fixture(data_root: Path, n_per_region: int = 30) -> None:
    figure_dir = data_root / "Figure_1d_1e_Spleen_LNP" / "Precomputed_Analysis_Input"
    rows_codebook, rows_bit1, rows_annotations = [], [], []
    cell_types = ["B", "T", "DC", "Macrophage", "Lymphatic endothelial"]
    for region in ["reg000", "reg001"]:
        for cell in range(1, n_per_region + 1):
            x, y = RNG.uniform(0, 3000), RNG.uniform(0, 3000)
            rows_codebook.append(
                {
                    "region": region,
                    "cell": cell,
                    "x": x,
                    "y": y,
                    "lnp_positive": bool(RNG.random() < 0.2),
                }
            )
            rows_bit1.append(
                {
                    "region": region,
                    "cell": cell,
                    "x": x,
                    "y": y,
                    "bit_1_positive": bool(RNG.random() < 0.2),
                }
            )
            rows_annotations.append(
                {"region": region, "cell": cell, "cell_type": RNG.choice(cell_types)}
            )
    _write_csv_gz(pd.DataFrame(rows_codebook), figure_dir / "codebook_cell_calls_spleen.csv.gz")
    _write_csv_gz(pd.DataFrame(rows_bit1), figure_dir / "bit_1_cell_calls_spleen.csv.gz")
    _write_csv_gz(
        pd.DataFrame(rows_annotations), figure_dir / "frozen_cell_annotations_spleen.csv.gz"
    )


def make_figure_1f_1g_fixture(data_root: Path, n_cells: int = 60) -> None:
    figure_dir = data_root / "Figure_1f_1g_Spleen_Neighborhoods" / "Precomputed_Analysis_Input"
    cell_types = ["B", "T", "DC", "Macrophage"]
    rows = []
    for cell in range(1, n_cells + 1):
        rows.append(
            {
                "region": "reg001",
                "cell": cell,
                "x": RNG.uniform(0, 3000),
                "y": RNG.uniform(0, 3000),
                "lnp_positive": bool(RNG.random() < 0.2),
                "cell_type": RNG.choice(cell_types),
            }
        )
    _write_csv_gz(pd.DataFrame(rows), figure_dir / "reg001_neighborhood_cells.csv.gz")


_MARKER_COLS = [
    "DAPI",
    "Podoplanin",
    "aSMA",
    "CD4",
    "CD31",
    "SCA1",
    "CD152",
    "CD45.2",
    "Ly6C",
    "PD1",
    "CD3",
    "CD11c",
    "CD19",
    "F480",
    "CD138",
    "CD169",
    "CD62L",
    "TCRb",
    "GZMB",
    "FOXP3",
    "CD27",
    "CD8a",
    "CD11b",
    "Fluc",
    "MHCII",
    "CCR7",
    "CD86",
    "CD28",
    "B220",
    "CD103",
    "CD25",
    "MPO",
    "CD90",
    "SIINFEKL_H-2Kb",
    "CD44",
    "OVA",
    "NKp46",
]
_LNP_MARKER_COLS = [
    "A6",
    "A17",
    "A55",
    "A56",
    "A76",
    "A79",
    "A20",
    "A46",
    "A28",
    "A72",
    "A2",
    "A63",
    "A126",
    "A127",
    "A128",
]


def make_figure_2_fixture(data_root: Path, n_per_region: int = 25) -> None:
    figure_root = data_root / "Figure_2_and_Supplementary_Figures_6_11_Multiplex_LNP"
    cell_types = ["B", "CD4+ T", "CD8+ T", "Macrophage", "DC", "Endothelial"]
    lnp_calls_by_batch = {
        "S1_S2": {
            "regions": [f"S{s}_reg{r:03d}" for s in (1, 2) for r in range(4)] + ["S2_reg004"]
        },
        "S3_S4": {"regions": [f"S{s}_reg{r:03d}" for s in (3, 4) for r in range(4)]},
    }
    for batch, batch_dir, info in [
        ("S1_S2", "Round_1_S1_S2", lnp_calls_by_batch["S1_S2"]),
        ("S3_S4", "Round_2_S3_S4", lnp_calls_by_batch["S3_S4"]),
    ]:
        out_dir = figure_root / "Precomputed_Downstream_Input" / batch_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        obs_rows = []
        spot_rows = []
        label_counter = 0
        for region in info["regions"]:
            slide = region.split("_")[0]
            for _ in range(n_per_region):
                label_counter += 1
                label = label_counter
                cell_type = RNG.choice(cell_types)
                obs_row = {
                    "label": label,
                    "slide_name": f"sample_registered_{slide}",
                    "region": region.split("_")[1],
                    "cell_type": cell_type,
                    "cell_type_pooled": cell_type,
                    "x": RNG.uniform(0, 3000),
                    "y": RNG.uniform(0, 3000),
                }
                is_reference = region == "S2_reg004"
                for marker in _MARKER_COLS:
                    obs_row[marker] = float(
                        RNG.normal(15, 3) if is_reference else RNG.normal(10, 3)
                    )
                obs_rows.append(obs_row)

                lnp_call = (
                    RNG.choice([f"LNP_{i:02d}" for i in range(1, 11)]) if not is_reference else None
                )
                spot_row = {
                    "region": region,
                    "cell": label,
                    "barcode": "".join(RNG.choice(["0", "1"], size=12)),
                    "lnp_call": lnp_call,
                    "lnp_positive": bool(lnp_call is not None and RNG.random() < 0.3),
                    "barcode_in_library": True,
                    "barcode_match_distance": 0,
                    "barcode_match_status": "exact" if lnp_call else "no_barcode",
                    "barcode_excluded": False,
                    "total_barcode_spots": int(RNG.integers(0, 5)),
                    "decoded_exact_spots": int(RNG.integers(0, 3)),
                    "decoded_tolerant_spots": 0,
                    "n_positive_bits": int(RNG.integers(0, 6)),
                    "dominant_decoded_barcode": "",
                    "dominant_decoded_barcode_count": 0,
                    "dominant_barcode_marker": "",
                    "dominant_barcode_count": 0,
                    "dominant_barcode_fraction": 0.0,
                    "barcode_confidence": float(RNG.uniform(0, 1)),
                    "decoded_spots": int(RNG.integers(0, 3)),
                }
                for marker in _LNP_MARKER_COLS:
                    spot_row[f"spot_{marker}"] = int(RNG.integers(0, 2))
                    spot_row[f"bit_{marker}"] = int(RNG.integers(0, 2))
                spot_rows.append(spot_row)

        obs = pd.DataFrame(obs_rows)
        adata = ad.AnnData(X=np.zeros((len(obs), 1), dtype=np.float32), obs=obs)
        adata.write_h5ad(out_dir / f"smoke_{batch}_annotated.h5ad")

        spot = pd.DataFrame(spot_rows)
        spot.to_csv(out_dir / "cell_analysis_table_all_regions_v4.csv", index=False)
        summary = (
            spot.groupby(["region", "lnp_call"], dropna=False)
            .size()
            .rename("n_cells")
            .reset_index()
        )
        summary.to_csv(out_dir / "cell_analysis_summary_by_lnp_call_v4.csv", index=False)


def make_raw_image_fixture(data_root: Path, shape: tuple[int, int] = (96, 96)) -> None:
    """One tiny registered TIFF stack + features CSV + marker list per raw-image family."""
    families = {
        "Figure_1d_1e_Spleen_LNP/Raw_Spot_Detection_Input": {
            "markers": [
                "A6",
                "A17",
                "A55",
                "A56",
                "A76",
                "A79",
                "A20",
                "A46",
                "A28",
                "A72",
                "A2",
                "A63",
            ],
            "samples": ["registered1"],
            "region": "reg000",
        },
        "Figure_2_and_Supplementary_Figures_6_11_Multiplex_LNP/Raw_Spot_Detection_Input/Round_1_S1_S2": {
            "markers": [
                "A6",
                "A17",
                "A55",
                "A56",
                "A76",
                "A79",
                "A20",
                "A46",
                "A28",
                "A72",
                "A2",
                "A63",
            ],
            "samples": ["registered_S1_reg000"],
            "region": "S1_reg000",
        },
        "Figure_2_and_Supplementary_Figures_6_11_Multiplex_LNP/Raw_Spot_Detection_Input/Round_2_S3_S4": {
            "markers": [
                "A6",
                "A17",
                "A55",
                "A56",
                "A76",
                "A79",
                "A20",
                "A46",
                "A28",
                "A72",
                "A2",
                "A63",
                "A126",
                "A127",
                "A128",
            ],
            "samples": ["registered_S3_reg000"],
            "region": "S3_reg000",
        },
    }
    for relative_dir, info in families.items():
        out_dir = data_root / relative_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        n_channels = len(info["markers"])
        for sample in info["samples"]:
            stack = RNG.normal(20, 3, size=(n_channels, *shape)).clip(0).astype(np.uint16)
            # Bright the pixel in every channel, not just the first few, so
            # every codebook's marker subset (bit_1 needs only A20, at index
            # 6) has a detectable candidate.
            stack[:, shape[0] // 2, shape[1] // 2] = 800
            tifffile.imwrite(out_dir / f"{sample}_integrated_registered_overlap_crop.tif", stack)
            (out_dir / f"{sample}_integrated_MarkerList.txt").write_text(
                "\n".join(info["markers"]) + "\n"
            )
            (out_dir / f"{sample}_registration_summary.json").write_text(
                json.dumps({"integrated_shape": [n_channels, *shape]})
            )
        features = pd.DataFrame(
            {
                "label": range(1, 6),
                "y": RNG.uniform(0, shape[0], 5),
                "x": RNG.uniform(0, shape[1], 5),
            }
        )
        features.to_csv(out_dir / f"{info['region']}_features.csv", index=False)


def make_supplementary_1c_fixture(data_root: Path, shape: tuple[int, int] = (64, 64)) -> None:
    sample = "B10_smoke_sample"
    out_dir = data_root / "Supplementary_Figure_1c_2Oligo_RCA" / "Raw_Images" / sample / "0"
    out_dir.mkdir(parents=True, exist_ok=True)
    channel_suffix = {
        "DAPI": "Fluorescence_405_nm_Ex",
        "Cy3-F46": "Fluorescence_561_nm_Ex",
        "Cy5-F20": "Fluorescence_638_nm_Ex",
    }
    center = shape[0] // 2, shape[1] // 2
    for fov in range(1, 3):
        for channel, suffix in channel_suffix.items():
            image = RNG.normal(20, 3, size=shape).clip(0).astype(np.uint16)
            if channel == "DAPI":
                # A segmentable nucleus needs an area blob, not one pixel.
                image[center[0] - 5 : center[0] + 5, center[1] - 5 : center[1] + 5] = 3000
            else:
                # A few bright puncta for spot detection.
                for dy, dx in [(0, 0), (10, 10), (-10, 8)]:
                    image[center[0] + dy, center[1] + dx] = 3000
            tifffile.imwrite(out_dir / f"smoke_{fov}_0_{suffix}.tiff", image)
    pd.DataFrame({"x (mm)": [0.0, 1.0], "y (mm)": [0.0, 0.0]}).to_csv(
        out_dir / "coordinates.csv", index=False
    )


def main(data_root: Path) -> None:
    if data_root.exists():
        shutil.rmtree(data_root)
    make_figure_1d_1e_fixture(data_root)
    make_figure_1f_1g_fixture(data_root)
    make_figure_2_fixture(data_root)
    make_raw_image_fixture(data_root)
    make_supplementary_1c_fixture(data_root)
    (data_root / ".smoke_fixture").write_text(
        "Synthetic fixture generated by make_smoke_fixture.py\n"
    )


if __name__ == "__main__":
    if "snakemake" in globals():
        main(Path(snakemake.output[0]).parent)  # type: ignore[name-defined]  # noqa: F821
    else:
        main(Path(sys.argv[1]))
