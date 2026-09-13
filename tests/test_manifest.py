"""Tests for nanostamp.manifest."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from nanostamp.manifest import (
    ManifestEntry,
    find_local_paths,
    load_manifest,
    sha256_of,
    verify_manifest,
)


def test_manifest_entry_rejects_short_sha256() -> None:
    with pytest.raises(ValueError, match="64 lowercase hex"):
        ManifestEntry(path="a.csv", size_bytes=1, sha256="deadbeef")


def test_manifest_entry_rejects_uppercase_sha256() -> None:
    with pytest.raises(ValueError, match="64 lowercase hex"):
        ManifestEntry(path="a.csv", size_bytes=1, sha256="A" * 64)


def test_sha256_of_matches_hashlib(tmp_path: Path) -> None:
    target = tmp_path / "data.bin"
    payload = b"nanostamp" * 10_000
    target.write_bytes(payload)
    assert sha256_of(target) == hashlib.sha256(payload).hexdigest()


def test_load_manifest_round_trip() -> None:
    manifest = {
        "files": [
            {"path": "a/b.csv.gz", "size_bytes": 42, "sha256": "0" * 64},
        ]
    }
    entries = load_manifest(manifest)
    assert entries == [ManifestEntry(path="a/b.csv.gz", size_bytes=42, sha256="0" * 64)]


def test_verify_manifest_passes_for_matching_file(tmp_path: Path) -> None:
    payload = b"expected content"
    (tmp_path / "file.csv").write_bytes(payload)
    entry = ManifestEntry(
        path="file.csv", size_bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()
    )
    assert verify_manifest([entry], tmp_path) == []


def test_verify_manifest_reports_missing_file(tmp_path: Path) -> None:
    entry = ManifestEntry(path="missing.csv", size_bytes=1, sha256="0" * 64)
    problems = verify_manifest([entry], tmp_path)
    assert len(problems) == 1
    assert "missing file" in problems[0]


def test_verify_manifest_reports_size_mismatch(tmp_path: Path) -> None:
    (tmp_path / "file.csv").write_bytes(b"short")
    entry = ManifestEntry(path="file.csv", size_bytes=999, sha256="0" * 64)
    problems = verify_manifest([entry], tmp_path)
    assert len(problems) == 1
    assert "size mismatch" in problems[0]


def test_verify_manifest_reports_sha256_mismatch(tmp_path: Path) -> None:
    payload = b"actual content"
    (tmp_path / "file.csv").write_bytes(payload)
    entry = ManifestEntry(path="file.csv", size_bytes=len(payload), sha256="f" * 64)
    problems = verify_manifest([entry], tmp_path)
    assert len(problems) == 1
    assert "sha256 mismatch" in problems[0]


def test_find_local_paths_detects_home_directory(tmp_path: Path) -> None:
    (tmp_path / "script.py").write_text("DATA = '/home/alice/data'\n")
    findings = find_local_paths(tmp_path)
    assert len(findings) == 1
    assert "/home/alice/data" in findings[0]


def test_find_local_paths_ignores_relative_paths(tmp_path: Path) -> None:
    (tmp_path / "script.py").write_text("DATA = Path('data') / 'file.csv'\n")
    assert find_local_paths(tmp_path) == []


def test_find_local_paths_skips_data_and_results_directories(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "note.md").write_text("/home/alice/leftover\n")
    assert find_local_paths(tmp_path) == []
