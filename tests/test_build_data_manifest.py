"""Tests for workflow/scripts/build_data_manifest.py."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workflow" / "scripts"))

# pyrefly: ignore [missing-import]
from build_data_manifest import build_manifest  # noqa: E402


def test_build_manifest_matches_hashlib(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.csv").write_bytes(b"hello")
    (tmp_path / "b.csv").write_bytes(b"world")

    manifest = build_manifest(tmp_path)
    by_path = {entry["path"]: entry for entry in manifest["files"]}

    assert set(by_path) == {"sub/a.csv", "b.csv"}
    assert by_path["sub/a.csv"]["size_bytes"] == 5
    assert by_path["sub/a.csv"]["sha256"] == hashlib.sha256(b"hello").hexdigest()


def test_build_manifest_empty_directory(tmp_path: Path) -> None:
    assert build_manifest(tmp_path) == {"files": []}
