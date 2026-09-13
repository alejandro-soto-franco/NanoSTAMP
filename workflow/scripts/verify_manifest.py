"""Snakemake script: verify the downloaded data manifest and scan for local paths.

Replaces ``Code/verify_publication_package.py``. Checks every file listed in
``config/data_manifest.yaml`` against its committed size and SHA-256 digest,
and scans the repository for hardcoded local-machine paths.
"""

import sys
from pathlib import Path

import yaml

from nanostamp.manifest import find_local_paths, load_manifest, verify_manifest

snakemake = globals()["snakemake"]
repo_root = Path(snakemake.config.get("_repo_root", "."))
data_root = Path(snakemake.config["data_root"])
manifest_path = Path(snakemake.input.manifest)

manifest = yaml.safe_load(manifest_path.read_text()) or {"files": []}
entries = load_manifest(manifest)

problems = []
if not entries:
    print(
        "data_manifest.yaml lists no files. This is expected until a copy of "
        "the Duke Research Data Repository deposit has been downloaded and "
        "its manifest populated (see README); nothing to verify yet."
    )
else:
    problems.extend(verify_manifest(entries, data_root))

local_path_findings = find_local_paths(repo_root)
if local_path_findings:
    problems.append(f"Found {len(local_path_findings)} hardcoded local path(s):")
    problems.extend(f"  {finding}" for finding in local_path_findings)

report_path = Path(snakemake.output.report)
report_path.parent.mkdir(parents=True, exist_ok=True)
if problems:
    report_path.write_text("FAILED\n" + "\n".join(problems) + "\n")
    print("FAILED:")
    for problem in problems:
        print(f"  {problem}")
    sys.exit(1)

report_path.write_text(f"PASS: {len(entries)} manifest file(s) verified, no local paths found.\n")
print(f"PASS: {len(entries)} manifest file(s) verified, no local paths found.")
