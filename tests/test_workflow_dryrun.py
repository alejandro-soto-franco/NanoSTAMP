"""Confirm the Snakemake workflow's DAG resolves, for both configs, without executing rules."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _snakemake_available() -> bool:
    return shutil.which("snakemake") is not None


@pytest.mark.skipif(not _snakemake_available(), reason="snakemake not on PATH")
@pytest.mark.parametrize("config_path", ["config/config.yaml", "config/smoke.yaml"])
def test_snakemake_dry_run(config_path: str) -> None:
    target = "all_smoke" if config_path.endswith("smoke.yaml") else "all"
    result = subprocess.run(
        ["snakemake", "-s", "workflow/Snakefile", "--configfile", config_path, "-n", target],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
