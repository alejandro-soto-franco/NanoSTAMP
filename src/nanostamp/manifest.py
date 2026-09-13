"""Data-manifest verification and local-path scanning.

Replaces the notebook-era ``Code/verify_publication_package.py`` preflight
check. The workflow downloads processed data from the Duke Research Data
Repository into ``data/`` (see ``workflow/rules/fetch.smk``); this module
confirms that every downloaded file matches a committed size and SHA-256
digest, and that the repository contains no machine-specific absolute path.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

_HASH_CHUNK_BYTES = 1 << 20  # 1 MiB

# Absolute-path prefixes that indicate a hardcoded local machine path rather
# than a path relative to the repository or a documented data root.
_LOCAL_PATH_PATTERN = re.compile(
    r"(?:/home/[\w.-]+(?:/[\w.-]+)*"
    r"|/Users/[\w.-]+(?:/[\w.-]+)*"
    r"|/mnt/[\w.-]+(?:/[\w.-]+)*"
    r"|/Volumes/[\w.-]+(?:/[\w.-]+)*"
    r"|[A-Za-z]:\\\\Users\\\\[\w.-]+)"
)


@dataclass(frozen=True)
class ManifestEntry:
    """One expected file in the downloaded-data manifest.

    Attributes
    ----------
    path
        Path relative to the manifest's data root.
    size_bytes
        Expected file size in bytes.
    sha256
        Expected lowercase hex SHA-256 digest (64 characters).
    """

    path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if len(self.sha256) != 64 or not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError(
                f"{self.path}: sha256 must be 64 lowercase hex characters, got {self.sha256!r}"
            )
        if self.size_bytes < 0:
            raise ValueError(f"{self.path}: size_bytes must be non-negative")


def sha256_of(path: Path) -> str:
    """Compute the lowercase hex SHA-256 digest of a file.

    Parameters
    ----------
    path
        File to hash.

    Returns
    -------
    64-character lowercase hex digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(manifest: dict) -> list[ManifestEntry]:
    """Parse a manifest mapping (as loaded from YAML) into entries.

    Parameters
    ----------
    manifest
        Mapping with a top-level ``files`` list, each item a mapping with
        ``path``, ``size_bytes`` and ``sha256`` keys.

    Returns
    -------
    List of :class:`ManifestEntry`.
    """
    entries = []
    for item in manifest.get("files", []):
        entries.append(
            ManifestEntry(
                path=item["path"],
                size_bytes=int(item["size_bytes"]),
                sha256=str(item["sha256"]).lower(),
            )
        )
    return entries


def verify_manifest(entries: Iterable[ManifestEntry], data_root: Path) -> list[str]:
    """Verify every manifest entry exists under ``data_root`` with a matching
    size and SHA-256 digest.

    Parameters
    ----------
    entries
        Expected files, as returned by :func:`load_manifest`.
    data_root
        Directory the entries' paths are relative to.

    Returns
    -------
    List of human-readable problem descriptions; empty when everything
    matches.
    """
    problems: list[str] = []
    for entry in entries:
        full_path = data_root / entry.path
        if not full_path.is_file():
            problems.append(f"missing file: {entry.path}")
            continue
        actual_size = full_path.stat().st_size
        if actual_size != entry.size_bytes:
            problems.append(
                f"size mismatch: {entry.path}: expected {entry.size_bytes} bytes, "
                f"found {actual_size}"
            )
            continue
        actual_sha256 = sha256_of(full_path)
        if actual_sha256 != entry.sha256:
            problems.append(
                f"sha256 mismatch: {entry.path}: expected {entry.sha256}, found {actual_sha256}"
            )
    return problems


def find_local_paths(
    root: Path, extensions: tuple[str, ...] = (".py", ".smk", ".md", ".yaml", ".yml")
) -> list[str]:
    """Scan text files under ``root`` for hardcoded local machine paths.

    ``tests/`` is excluded: its fixtures legitimately construct example
    local-looking path strings to exercise this very detector.

    Parameters
    ----------
    root
        Repository root to scan.
    extensions
        File suffixes to inspect.

    Returns
    -------
    List of ``"<file>:<line>: <matched text>"`` findings; empty when clean.
    """
    findings: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in extensions:
            continue
        if any(
            part in {".git", ".pixi", ".snakemake", "data", "results", "tests"}
            for part in path.parts
        ):
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            match = _LOCAL_PATH_PATTERN.search(line)
            if match:
                findings.append(f"{path.relative_to(root)}:{lineno}: {match.group(0)}")
    return findings
