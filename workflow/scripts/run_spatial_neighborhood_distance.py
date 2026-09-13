"""Snakemake script: Figure 2 / Supplementary Figures 9-11, Sections 8-12.

Ports the second half of
``Code/Figure_2_and_Supplementary_9_11_Multiplex_LNP_Spatial_Neighborhood_Analysis.ipynb``:
SIINFEKL+ DC vs CD8 T-cell spatial/state analysis (Section 8), OVA+ DC vs
CD4 T-cell activation (Section 9), OVA+ B-cell state and surrounding CD4
T cells (Section 10), LNP+OVA+ neighbourhood abundance (Section 11), and the
eight final paired-panel re-derivations (Section 12), each with a real
paired-boxplot PDF/PNG figure. See ``nanostamp.spatial_distance`` for the
shared, parameterised primitives this script calls repeatedly.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from nanostamp.plotting import apply_base_style, save_figure
from nanostamp.spatial_distance import (
    dedup_nearest_pairs,
    focused_neighbor_composition,
    format_pvalue,
    hybrid_neighbor_metrics,
    multiscale_neighbor_metrics,
    neighbor_state_summary,
    own_state_summary,
    paired_region_ttest,
    paired_ttest_table,
    positive_subset_neighborhood_distribution,
    radius_neighbor_metrics,
    summarize_neighbor_metrics_by_region,
)
from nanostamp.spatial_neighborhood import collect_neighbor_pairs

snakemake = globals()["snakemake"]
cfg = snakemake.config["figure_2_spatial_neighborhood"]
distance_cfg = snakemake.config["figure_2_spatial_neighborhood_distance"]
out_dir = Path(snakemake.output.tables)
out_dir.mkdir(parents=True, exist_ok=True)
apply_base_style()

DC_LNPS = ("LNP_08", "LNP_10")
UM_PER_UNIT = distance_cfg["um_per_coordinate_unit"]
K10_RADIUS25 = {"k": 10, "radius": 25.0}

spatial_tables_dir = Path(snakemake.input.spatial_tables)
cell_functional_tables_dir = Path(snakemake.input.cell_functional_tables)
cells = pd.read_csv(spatial_tables_dir / "cell_neighborhood_assignments.csv")
neighbor_composition = pd.read_csv(spatial_tables_dir / "neighbor_composition_fractions.csv")
cells["lnp_positive"] = cells["lnp_positive"].astype(bool)
if "ova_positive" not in cells.columns or "luc_positive" not in cells.columns:
    raise KeyError("Expected luc_positive/ova_positive columns from run_spatial_neighborhood.py")

# Section 8 preamble: a second, independent SIINFEKL-H-2Kb gate at a lower
# quantile than the Luc/OVA gate.
reference_values = cells.loc[
    cells["lnp_region"].isin(cfg["gate_reference_regions"]), "SIINFEKL_H-2Kb"
].dropna()
siinfekl_threshold = float(reference_values.quantile(distance_cfg["siinfekl_gate_quantile"]))
cells["siinfekl_positive"] = cells["SIINFEKL_H-2Kb"] > siinfekl_threshold
pd.DataFrame(
    [
        {
            "marker": "SIINFEKL-H-2Kb",
            "column": "SIINFEKL_H-2Kb",
            "threshold": siinfekl_threshold,
            "gate_quantile": distance_cfg["siinfekl_gate_quantile"],
            "gate_reference_regions": ",".join(cfg["gate_reference_regions"]),
            "reference_n_cells": int(len(reference_values)),
        }
    ]
).to_csv(out_dir / "siinfekl_gate_threshold.csv", index=False)


def _paired_boxplot_figure(
    wide: pd.DataFrame, groups: tuple[str, str], ylabel: str, title: str, stem: Path, pvalue: float
) -> None:
    """Section 12: a paired boxplot with connecting lines and a bracketed p-value."""
    fig, ax = plt.subplots(figsize=(3.2, 3.4))
    positions = [0, 1]
    ax.boxplot(
        [wide[groups[0]], wide[groups[1]]], positions=positions, widths=0.5, showfliers=False
    )
    for _, row in wide.iterrows():
        ax.plot(
            positions, [row[groups[0]], row[groups[1]]], color="#B0B0B0", linewidth=0.8, zorder=1
        )
        ax.scatter(positions, [row[groups[0]], row[groups[1]]], color="#444444", s=14, zorder=2)
    y_max = wide[[groups[0], groups[1]]].to_numpy().max()
    ax.plot(
        [0, 0, 1, 1],
        [y_max * 1.05, y_max * 1.1, y_max * 1.1, y_max * 1.05],
        color="black",
        linewidth=0.8,
    )
    ax.text(0.5, y_max * 1.12, format_pvalue(pvalue), ha="center", va="bottom", fontsize=7)
    ax.set_xticks(positions, groups)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_figure(fig, stem)
    plt.close(fig)


def _section12_block(
    input_csv: Path,
    metric: str,
    filter_query: str | None,
    stem_name: str,
    ylabel: str,
    alternative: str = "two-sided",
    also_wilcoxon: bool = False,
) -> None:
    """One of Section 12's eight final paired LNP_08-vs-LNP_10 re-derivations."""
    table = pd.read_csv(input_csv)
    if filter_query:
        table = table.query(filter_query)
    wide = (
        table.pivot(index="lnp_region", columns="lnp_call", values=metric)
        .reindex(columns=list(DC_LNPS))
        .dropna()
    )
    if wide.empty:
        return
    wide["region_id"] = wide.index
    wide.to_csv(out_dir / f"{stem_name}_source_data.csv", index=False)

    test_rows = [paired_region_ttest(table, "lnp_call", DC_LNPS, metric, alternative=alternative)]
    display_pvalue = test_rows[0]["pvalue"]
    if also_wilcoxon and len(wide) >= 1:
        try:
            wilcoxon_stat, wilcoxon_p = stats.wilcoxon(
                wide[DC_LNPS[0]],
                wide[DC_LNPS[1]],
                zero_method="wilcox",
                alternative="two-sided",
                method="auto",
            )
        except ValueError:
            wilcoxon_stat, wilcoxon_p = np.nan, np.nan
        test_rows.append(
            {"test": "wilcoxon", "metric": metric, "statistic": wilcoxon_stat, "pvalue": wilcoxon_p}
        )
        display_pvalue = wilcoxon_p
    pd.DataFrame(test_rows).to_csv(out_dir / f"{stem_name}_test.csv", index=False)

    _paired_boxplot_figure(
        # pyrefly: ignore [bad-argument-type]
        wide,
        DC_LNPS,
        ylabel,
        stem_name.replace("_", " "),
        out_dir / stem_name,
        # pyrefly: ignore [bad-argument-type]
        display_pvalue,
    )
    print(
        f"{stem_name}: n={len(wide)}, p={display_pvalue:.4g}"
        if pd.notna(display_pvalue)  # pyrefly: ignore [no-matching-overload]
        else f"{stem_name}: n={len(wide)}"
    )


# ---------------------------------------------------------------------------
# Section 8: SIINFEKL+ LNP-positive DCs vs nearby CD8+ T cells.
# ---------------------------------------------------------------------------
dc_targets = cells.loc[
    (cells["cell_type"].astype(str) == "DC")
    & cells["lnp_positive"]
    & cells["lnp_call"].isin(DC_LNPS)
    & cells["siinfekl_positive"]
].copy()

multiscale = multiscale_neighbor_metrics(
    dc_targets,
    "CD8+ T",
    cells,
    ks=distance_cfg["cd8_neighbor_k_values"],
    metric_name="cd8",
    um_per_coordinate_unit=UM_PER_UNIT,
)
multiscale.to_csv(
    out_dir / "lnp08_vs_lnp10_siinfekl_positive_dc_cd8_multiscale_per_cell.csv", index=False
)
multiscale_region = summarize_neighbor_metrics_by_region(
    multiscale,
    "lnp_call",
    count_cols=[f"n_cd8_neighbors_k{k}" for k in distance_cfg["cd8_neighbor_k_values"]],
    has_cols=[f"has_cd8_neighbor_k{k}" for k in distance_cfg["cd8_neighbor_k_values"]],
    distance_cols=[f"mean_cd8_distance_k{k}" for k in distance_cfg["cd8_neighbor_k_values"]],
)
multiscale_region.to_csv(
    out_dir / "lnp08_vs_lnp10_siinfekl_positive_dc_cd8_multiscale_by_region.csv", index=False
)

radius_metrics = radius_neighbor_metrics(
    dc_targets,
    "CD8+ T",
    cells,
    radii=distance_cfg["cd8_radius_values"],
    metric_name="cd8",
    um_per_coordinate_unit=UM_PER_UNIT,
)
radius_region = summarize_neighbor_metrics_by_region(
    radius_metrics,
    "lnp_call",
    count_cols=[f"n_cd8_within_r{r:g}" for r in distance_cfg["cd8_radius_values"]],
    has_cols=[f"has_cd8_within_r{r:g}" for r in distance_cfg["cd8_radius_values"]],
    distance_cols=[f"mean_cd8_distance_within_r{r:g}" for r in distance_cfg["cd8_radius_values"]],
)
radius_region.to_csv(
    out_dir / "lnp08_vs_lnp10_siinfekl_positive_dc_cd8_radius_by_region.csv", index=False
)

hybrid = hybrid_neighbor_metrics(
    dc_targets,
    "CD8+ T",
    cells,
    metric_name="cd8",
    um_per_coordinate_unit=UM_PER_UNIT,
    **K10_RADIUS25,
)
hybrid_region = summarize_neighbor_metrics_by_region(
    hybrid,
    "lnp_call",
    count_cols=["hybrid_n_cd8_neighbors"],
    has_cols=["hybrid_has_cd8_neighbor"],
    distance_cols=["hybrid_mean_cd8_distance", "hybrid_median_cd8_distance"],
)
hybrid_region["radius"], hybrid_region["k_neighbors"] = K10_RADIUS25["radius"], K10_RADIUS25["k"]
hybrid_region.to_csv(
    out_dir / "lnp08_vs_lnp10_siinfekl_positive_dc_cd8_hybrid_settings_by_region.csv", index=False
)
paired_ttest_table(
    hybrid_region,
    "lnp_call",
    DC_LNPS,
    [
        "pct_hybrid_has_cd8_neighbor",
        "mean_hybrid_n_cd8_neighbors",
        "median_hybrid_median_cd8_distance",
    ],
).to_csv(
    out_dir / "lnp08_vs_lnp10_siinfekl_positive_dc_cd8_hybrid_settings_paired_ttests.csv",
    index=False,
)

ordered_names = sorted(cells["neighborhood"].dropna().unique())
pooled, by_region, direct_pooled, direct_by_region = focused_neighbor_composition(
    dc_targets,
    neighbor_composition,
    "lnp_call",
    list(DC_LNPS),
    ordered_names,
)
pooled.to_csv(out_dir / "lnp08_vs_lnp10_siinfekl_positive_dc_merged_neighborhood_abundance.csv")
by_region.to_csv(
    out_dir / "lnp08_vs_lnp10_siinfekl_positive_dc_merged_neighborhood_abundance_by_region.csv"
)
direct_by_region.to_csv(
    out_dir / "lnp08_vs_lnp10_siinfekl_positive_dc_direct_neighbor_composition_by_region.csv"
)
direct_pooled.to_csv(
    out_dir / "lnp08_vs_lnp10_siinfekl_positive_dc_direct_neighbor_composition_pooled_percent.csv"
)

dc_region_metrics = (
    dc_targets.groupby(["lnp_region", "lnp_call"], observed=False)
    .agg(n_target_dc=("cell_type", "size"), mean_cd86=("CD86", "mean"))
    .reset_index()
)
dc_region_metrics = dc_region_metrics.merge(
    hybrid.groupby(["lnp_region", "lnp_call"], observed=False)
    .agg(
        mean_cd8_neighbor_fraction=("hybrid_n_cd8_neighbors", "mean"),
        mean_n_cd8_neighbors=("hybrid_n_cd8_neighbors", "mean"),
        n_dcs_with_cd8_neighbor=("hybrid_has_cd8_neighbor", "sum"),
        mean_nearest_cd8_distance=("hybrid_mean_cd8_distance", "mean"),
        median_nearest_cd8_distance=("hybrid_median_cd8_distance", "median"),
    )
    .reset_index(),
    on=["lnp_region", "lnp_call"],
    how="left",
)
dc_region_metrics["pct_with_cd8_neighbor"] = (
    100 * dc_region_metrics["n_dcs_with_cd8_neighbor"] / dc_region_metrics["n_target_dc"]
)
dc_region_metrics.to_csv(
    out_dir / "lnp08_vs_lnp10_siinfekl_positive_dc_cd8_metrics_by_region.csv", index=False
)
hybrid[
    [
        "lnp_region",
        "lnp_call",
        "neighborhood",
        "hybrid_n_cd8_neighbors",
        "hybrid_has_cd8_neighbor",
        "hybrid_mean_cd8_distance",
    ]
].rename(
    columns={
        "hybrid_n_cd8_neighbors": "n_cd8_neighbors",
        "hybrid_has_cd8_neighbor": "has_cd8_neighbor",
        "hybrid_mean_cd8_distance": "nearest_cd8_distance",
    }
).to_csv(out_dir / "lnp08_vs_lnp10_siinfekl_positive_dc_cd8_metrics_per_cell.csv", index=False)
paired_ttest_table(
    dc_region_metrics,
    "lnp_call",
    DC_LNPS,
    ["pct_with_cd8_neighbor", "n_dcs_with_cd8_neighbor", "mean_nearest_cd8_distance", "mean_cd86"],
).to_csv(out_dir / "lnp08_vs_lnp10_siinfekl_positive_dc_paired_ttests.csv", index=False)

cd8_pairs = collect_neighbor_pairs(
    dc_targets,
    "CD8+ T",
    cells,
    group_col="lnp_call",
    um_per_coordinate_unit=UM_PER_UNIT,
    **K10_RADIUS25,
)
cd8_pairs.to_csv(
    out_dir / "lnp08_vs_lnp10_siinfekl_positive_dc_cd8_k10_radius25_neighbor_pairs.csv", index=False
)
cd8_state_markers = distance_cfg["cd8_state_markers"]
if not cd8_pairs.empty:
    deduped_cd8 = dedup_nearest_pairs(cd8_pairs)
    unique_cd8_state = pd.concat(
        [
            deduped_cd8.reset_index(drop=True),
            cells.loc[deduped_cd8["neighbor_index"], cd8_state_markers].reset_index(drop=True),
        ],
        axis=1,
    )
    unique_cd8_state.to_csv(
        out_dir / "lnp08_vs_lnp10_siinfekl_positive_dc_cd8_k10_radius25_unique_cd8_state.csv",
        index=False,
    )
    cd8_state_region = neighbor_state_summary(
        cd8_pairs, cells, cd8_state_markers, ["lnp_region", "lnp_call"]
    )
    cd8_state_region.to_csv(
        out_dir / "lnp08_vs_lnp10_siinfekl_positive_dc_cd8_k10_radius25_state_by_region.csv",
        index=False,
    )
    cd8_state_tests = pd.concat(
        [
            paired_ttest_table(
                cd8_state_region.loc[cd8_state_region["marker"] == marker],
                "lnp_call",
                DC_LNPS,
                ["mean_expression", "median_expression"],
            ).assign(marker=marker)
            for marker in cd8_state_markers
        ],
        ignore_index=True,
    )
    cd8_state_tests.to_csv(
        out_dir / "lnp08_vs_lnp10_siinfekl_positive_dc_cd8_k10_radius25_state_paired_ttests.csv",
        index=False,
    )

dc_state_markers = distance_cfg["dc_state_markers"]
dc_targets[["lnp_region", "lnp_call", *dc_state_markers]].to_csv(
    out_dir / "lnp08_vs_lnp10_siinfekl_positive_dc_state_per_cell.csv", index=False
)
dc_state_region = own_state_summary(dc_targets, dc_state_markers, ["lnp_region", "lnp_call"])
dc_state_region.to_csv(
    out_dir / "lnp08_vs_lnp10_siinfekl_positive_dc_state_by_region.csv", index=False
)
pd.concat(
    [
        paired_ttest_table(
            dc_state_region.loc[dc_state_region["marker"] == marker],
            "lnp_call",
            DC_LNPS,
            ["mean_expression", "median_expression"],
        ).assign(marker=marker)
        for marker in dc_state_markers
    ],
    ignore_index=True,
).to_csv(out_dir / "lnp08_vs_lnp10_siinfekl_positive_dc_state_paired_ttests.csv", index=False)
print(f"Section 8: {len(dc_targets)} SIINFEKL+ LNP-positive DC targets")


# ---------------------------------------------------------------------------
# Section 9: OVA+ LNP-positive DCs vs nearby CD4+ T-cell activation.
# ---------------------------------------------------------------------------
ova_dc_targets = cells.loc[
    (cells["cell_type"].astype(str) == "DC")
    & cells["lnp_positive"]
    & cells["lnp_call"].isin(DC_LNPS)
    & cells["ova_positive"]
].copy()
cd4_activation_markers = distance_cfg["cd4_activation_markers"]

cd4_pairs = collect_neighbor_pairs(
    ova_dc_targets,
    "CD4+ T",
    cells,
    group_col="lnp_call",
    um_per_coordinate_unit=UM_PER_UNIT,
    **K10_RADIUS25,
)
cd4_pairs.to_csv(
    out_dir / "lnp08_vs_lnp10_ova_positive_dc_cd4_k10_radius25_neighbor_pairs.csv", index=False
)

target_metrics = ova_dc_targets.copy()
target_metrics["n_cd4_neighbors"] = 0
target_metrics["median_cd4_distance_pixels"] = np.nan
if not cd4_pairs.empty:
    per_target = cd4_pairs.groupby("target_index").agg(
        n_cd4_neighbors=("neighbor_index", "nunique"),
        median_cd4_distance_pixels=("distance", "median"),
    )
    target_metrics.loc[per_target.index, "n_cd4_neighbors"] = per_target["n_cd4_neighbors"]
    target_metrics.loc[per_target.index, "median_cd4_distance_pixels"] = per_target[
        "median_cd4_distance_pixels"
    ]
target_metrics["has_cd4_neighbor"] = target_metrics["n_cd4_neighbors"] > 0
target_metrics["median_cd4_distance_um"] = (
    target_metrics["median_cd4_distance_pixels"] * UM_PER_UNIT
)
target_metrics.to_csv(
    out_dir / "lnp08_vs_lnp10_ova_positive_dc_cd4_k10_radius25_target_metrics.csv", index=False
)

spatial_region = (
    target_metrics.groupby(["lnp_region", "lnp_call"], observed=False)
    .agg(
        n_target_dcs=("cell_type", "size"),
        pct_dcs_with_cd4=("has_cd4_neighbor", "mean"),
        mean_n_cd4_neighbors=("n_cd4_neighbors", "mean"),
        median_n_cd4_neighbors=("n_cd4_neighbors", "median"),
        median_cd4_distance_um=("median_cd4_distance_um", "median"),
    )
    .reset_index()
)
spatial_region["pct_dcs_with_cd4"] *= 100
spatial_region.to_csv(
    out_dir / "lnp08_vs_lnp10_ova_positive_dc_cd4_k10_radius25_spatial_by_region.csv", index=False
)
paired_ttest_table(
    spatial_region,
    "lnp_call",
    DC_LNPS,
    [
        "pct_dcs_with_cd4",
        "mean_n_cd4_neighbors",
        "median_n_cd4_neighbors",
        "median_cd4_distance_um",
    ],
).to_csv(
    out_dir / "lnp08_vs_lnp10_ova_positive_dc_cd4_k10_radius25_spatial_paired_ttests.csv",
    index=False,
)

if not cd4_pairs.empty:
    deduped_cd4 = dedup_nearest_pairs(cd4_pairs)
    unique_cd4_activation = pd.concat(
        [
            deduped_cd4.reset_index(drop=True),
            cells.loc[deduped_cd4["neighbor_index"], cd4_activation_markers].reset_index(drop=True),
        ],
        axis=1,
    )
    unique_cd4_activation.to_csv(
        out_dir / "lnp08_vs_lnp10_ova_positive_dc_cd4_k10_radius25_unique_cd4_activation.csv",
        index=False,
    )
    activation_region = neighbor_state_summary(
        cd4_pairs, cells, cd4_activation_markers, ["lnp_region", "lnp_call"]
    )
    activation_region.to_csv(
        out_dir / "lnp08_vs_lnp10_ova_positive_dc_cd4_k10_radius25_activation_by_region.csv",
        index=False,
    )
    pd.concat(
        [
            paired_ttest_table(
                activation_region.loc[activation_region["marker"] == marker],
                "lnp_call",
                DC_LNPS,
                ["mean_expression", "median_expression"],
            ).assign(marker=marker)
            for marker in cd4_activation_markers
        ],
        ignore_index=True,
    ).to_csv(
        out_dir / "lnp08_vs_lnp10_ova_positive_dc_nearby_cd4_activation_paired_ttests.csv",
        index=False,
    )
print(f"Section 9: {len(ova_dc_targets)} OVA+ LNP-positive DC targets")


# ---------------------------------------------------------------------------
# Section 10: OVA+ LNP-positive B cells - own state and surrounding CD4 T cells.
# ---------------------------------------------------------------------------
ova_b_targets = cells.loc[
    (cells["cell_type"].astype(str) == "B")
    & cells["lnp_positive"]
    & cells["lnp_call"].isin(DC_LNPS)
    & cells["ova_positive"]
].copy()
ova_b_state_markers = distance_cfg["ova_b_state_markers"]

ova_b_targets[["lnp_region", "lnp_call", *ova_b_state_markers]].to_csv(
    out_dir / "section11_lnp08_vs_lnp10_ova_positive_b_state_per_cell.csv", index=False
)
b_state_region = own_state_summary(ova_b_targets, ova_b_state_markers, ["lnp_region", "lnp_call"])
b_state_region.to_csv(
    out_dir / "section11_lnp08_vs_lnp10_ova_positive_b_state_by_region.csv", index=False
)
pd.concat(
    [
        paired_ttest_table(
            b_state_region.loc[b_state_region["marker"] == marker],
            "lnp_call",
            DC_LNPS,
            ["mean_expression", "median_expression"],
        ).assign(marker=marker)
        for marker in ova_b_state_markers
    ],
    ignore_index=True,
).to_csv(out_dir / "section11_lnp08_vs_lnp10_ova_positive_b_state_paired_ttests.csv", index=False)

b_cd4_pairs = collect_neighbor_pairs(
    ova_b_targets,
    "CD4+ T",
    cells,
    group_col="lnp_call",
    um_per_coordinate_unit=UM_PER_UNIT,
    **K10_RADIUS25,
)
b_cd4_pairs.to_csv(
    out_dir / "section11_lnp08_vs_lnp10_ova_positive_b_cd4_k10_radius25_neighbor_pairs.csv",
    index=False,
)

b_target_metrics = ova_b_targets.copy()
b_target_metrics["n_cd4_neighbors"] = 0
b_target_metrics["median_cd4_distance_pixels"] = np.nan
if not b_cd4_pairs.empty:
    per_b_target = b_cd4_pairs.groupby("target_index").agg(
        n_cd4_neighbors=("neighbor_index", "nunique"),
        median_cd4_distance_pixels=("distance", "median"),
    )
    b_target_metrics.loc[per_b_target.index, "n_cd4_neighbors"] = per_b_target["n_cd4_neighbors"]
    b_target_metrics.loc[per_b_target.index, "median_cd4_distance_pixels"] = per_b_target[
        "median_cd4_distance_pixels"
    ]
b_target_metrics["has_cd4_neighbor"] = b_target_metrics["n_cd4_neighbors"] > 0
b_target_metrics["median_cd4_distance_um"] = (
    b_target_metrics["median_cd4_distance_pixels"] * UM_PER_UNIT
)
b_target_metrics.to_csv(
    out_dir / "section11_lnp08_vs_lnp10_ova_positive_b_cd4_k10_radius25_target_metrics.csv",
    index=False,
)

b_spatial_region = (
    b_target_metrics.groupby(["lnp_region", "lnp_call"], observed=False)
    .agg(
        n_ova_positive_b=("cell_type", "size"),
        pct_with_cd4_neighbor=("has_cd4_neighbor", "mean"),
        mean_n_cd4_neighbors=("n_cd4_neighbors", "mean"),
        median_n_cd4_neighbors=("n_cd4_neighbors", "median"),
        median_cd4_distance_um=("median_cd4_distance_um", "median"),
    )
    .reset_index()
)
b_spatial_region["pct_with_cd4_neighbor"] *= 100
b_spatial_region.to_csv(
    out_dir / "section11_lnp08_vs_lnp10_ova_positive_b_cd4_k10_radius25_spatial_by_region.csv",
    index=False,
)
paired_ttest_table(
    b_spatial_region,
    "lnp_call",
    DC_LNPS,
    [
        "pct_with_cd4_neighbor",
        "mean_n_cd4_neighbors",
        "median_n_cd4_neighbors",
        "median_cd4_distance_um",
    ],
).to_csv(
    out_dir / "section11_lnp08_vs_lnp10_ova_positive_b_cd4_k10_radius25_spatial_paired_ttests.csv",
    index=False,
)

section11_cd4_state_markers = distance_cfg["cd4_activation_markers"]
if not b_cd4_pairs.empty:
    deduped_b_cd4 = dedup_nearest_pairs(b_cd4_pairs)
    surrounding_unique_cd4 = pd.concat(
        [
            deduped_b_cd4.reset_index(drop=True),
            cells.loc[deduped_b_cd4["neighbor_index"], section11_cd4_state_markers].reset_index(
                drop=True
            ),
        ],
        axis=1,
    )
    surrounding_unique_cd4.to_csv(
        out_dir / "section11_lnp08_vs_lnp10_ova_positive_b_surrounding_unique_cd4.csv", index=False
    )
    surrounding_cd4_state_region = neighbor_state_summary(
        b_cd4_pairs, cells, section11_cd4_state_markers, ["lnp_region", "lnp_call"]
    )
    surrounding_cd4_state_region.to_csv(
        out_dir / "section11_lnp08_vs_lnp10_ova_positive_b_surrounding_cd4_state_by_region.csv",
        index=False,
    )
    pd.concat(
        [
            paired_ttest_table(
                surrounding_cd4_state_region.loc[surrounding_cd4_state_region["marker"] == marker],
                "lnp_call",
                DC_LNPS,
                ["mean_expression", "median_expression"],
            ).assign(marker=marker)
            for marker in section11_cd4_state_markers
        ],
        ignore_index=True,
    ).to_csv(
        out_dir / "section11_lnp08_vs_lnp10_ova_positive_b_surrounding_cd4_state_paired_ttests.csv",
        index=False,
    )
print(f"Section 10: {len(ova_b_targets)} OVA+ LNP-positive B-cell targets")


# ---------------------------------------------------------------------------
# Section 11 (Supplementary Figure 11): neighbourhoods of LNP+OVA+ cells.
# ---------------------------------------------------------------------------
lnp_order = [f"LNP_{i:02d}" for i in range(1, 11)]
ova_lnp_mask = cells["lnp_positive"] & cells["ova_positive"] & cells["lnp_call"].isin(lnp_order)
ova_lnp_neighborhood_counts = positive_subset_neighborhood_distribution(
    cells, ova_lnp_mask, group_col="lnp_call", group_order=lnp_order, ordered_names=ordered_names
)
ova_lnp_neighborhood_counts.to_csv(
    out_dir / "section13_corrected_lnp_ova_positive_neighborhood_abundance.csv"
)

ova_lnp_subset = cells.loc[ova_lnp_mask]
ova_region_counts = (
    ova_lnp_subset.groupby(["lnp_region", "lnp_call", "neighborhood"], observed=False)
    .size()
    .rename("n_ova_lnp_cells")
    .reset_index()
)
ova_region_counts["n_ova_lnp_cells_total"] = ova_region_counts.groupby(["lnp_region", "lnp_call"])[
    "n_ova_lnp_cells"
].transform("sum")
ova_region_counts["percent"] = (
    100 * ova_region_counts["n_ova_lnp_cells"] / ova_region_counts["n_ova_lnp_cells_total"]
)
ova_region_counts.to_csv(
    out_dir / "section13_corrected_lnp_ova_positive_neighborhood_abundance_by_region.csv",
    index=False,
)
print(f"Section 11: {int(ova_lnp_mask.sum())} LNP+OVA+ cells")


# ---------------------------------------------------------------------------
# Section 12: eight final Figure 2 panel re-derivations.
# ---------------------------------------------------------------------------
_section12_block(
    cell_functional_tables_dir / "lnp_positive_dc_siinfekl_ova_by_region.csv",
    "mean_cd86_in_siinfekl_positive",
    f"lnp_call in {list(DC_LNPS)!r}",
    "figure2_mean_cd86_siinfekl_positive_dc",
    "Mean CD86 (SIINFEKL+ DC)",
    also_wilcoxon=True,
)
_section12_block(
    out_dir / "lnp08_vs_lnp10_siinfekl_positive_dc_cd8_hybrid_settings_by_region.csv",
    "median_hybrid_median_cd8_distance",
    "radius == 25 and k_neighbors == 10" if False else None,
    "figure2_median_cd8_distance",
    "Median CD8+T distance (µm)",
    alternative="less",
)
_section12_block(
    out_dir / "lnp08_vs_lnp10_siinfekl_positive_dc_cd8_hybrid_settings_by_region.csv",
    "pct_hybrid_has_cd8_neighbor",
    None,
    "figure2_pct_cd8_presence",
    "% DCs with nearby CD8+T",
    alternative="greater",
)
if not cd4_pairs.empty:
    _section12_block(
        out_dir / "lnp08_vs_lnp10_ova_positive_dc_cd4_k10_radius25_activation_by_region.csv",
        "median_expression",
        "marker == 'FOXP3'",
        "figure2_cd4_foxp3_median",
        "Median FOXP3 (nearby CD4+T)",
    )
_section12_block(
    out_dir / "section11_lnp08_vs_lnp10_ova_positive_b_state_by_region.csv",
    "median_expression",
    "marker == 'CD19'",
    "figure2_ova_b_cd19_median",
    "Median CD19 (OVA+ B)",
)
_section12_block(
    out_dir / "section11_lnp08_vs_lnp10_ova_positive_b_state_by_region.csv",
    "median_expression",
    "marker == 'CD138'",
    "figure2_ova_b_cd138_median",
    "Median CD138 (OVA+ B)",
)
_section12_block(
    out_dir / "section11_lnp08_vs_lnp10_ova_positive_b_state_by_region.csv",
    "median_expression",
    "marker == 'CD25'",
    "figure2_ova_b_cd25_median",
    "Median CD25 (OVA+ B)",
)
if not b_cd4_pairs.empty:
    _section12_block(
        out_dir / "section11_lnp08_vs_lnp10_ova_positive_b_surrounding_cd4_state_by_region.csv",
        "median_expression",
        "marker == 'FOXP3'",
        "figure2_surrounding_cd4_foxp3_median",
        "Median FOXP3 (surrounding CD4+T)",
    )
