"""Shared matplotlib style for NanoSTAMP figures.

The upstream notebooks each set ``plt.rcParams`` inline with the same intent
(a small, print-ready sans-serif figure) but with slightly different literal
values per notebook. This module holds one canonical style so every figure
rule in the workflow renders consistently; per-figure code may still override
individual entries where the source notebook did.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

BASE_RCPARAMS: dict[str, object] = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "figure.dpi": 160,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def apply_base_style(overrides: dict[str, object] | None = None) -> None:
    """Apply :data:`BASE_RCPARAMS`, then any figure-specific overrides.

    Parameters
    ----------
    overrides
        rcParams to set after the base style, matching a specific source
        notebook's literal values where they differ from the shared base.
    """
    plt.rcParams.update(BASE_RCPARAMS)
    if overrides:
        plt.rcParams.update(overrides)


def save_figure(fig: plt.Figure, stem: str | Path) -> list[str]:
    """Save a figure as both PDF and PNG next to ``stem``.

    Parameters
    ----------
    fig
        Figure to save.
    stem
        Output path without extension; ``.pdf`` and ``.png`` are appended.

    Returns
    -------
    The two written paths, as strings.
    """
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in (".pdf", ".png"):
        out = stem.with_suffix(suffix)
        fig.savefig(out, bbox_inches="tight")
        written.append(str(out))
    return written
