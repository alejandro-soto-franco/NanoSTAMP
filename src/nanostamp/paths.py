"""Data-root resolution.

The upstream notebooks each searched their working directory and its parents
for a sibling ``Data/`` folder. The workflow instead passes an explicit data
root from ``config.yaml``, but this helper stays available for anything that
still needs to resolve a figure's data directory relative to a base path.
"""

from __future__ import annotations

from pathlib import Path


def resolve_figure_data_root(data_root: Path, figure_dir_name: str) -> Path:
    """Resolve and validate a figure's data directory.

    Parameters
    ----------
    data_root
        The workflow's configured data root (``config["data_root"]``).
    figure_dir_name
        The figure-specific subdirectory name, e.g.
        ``"Figure_1d_1e_Spleen_LNP"``.

    Returns
    -------
    The resolved, existing directory.

    Raises
    ------
    FileNotFoundError
        If the directory does not exist under ``data_root``.
    """
    candidate = Path(data_root) / figure_dir_name
    if not candidate.is_dir():
        raise FileNotFoundError(
            f"Could not locate {figure_dir_name!r} under data root {data_root!s}. "
            "Download the processed data from the Duke Research Data Repository "
            "record referenced in the README and place it under this data root."
        )
    return candidate
