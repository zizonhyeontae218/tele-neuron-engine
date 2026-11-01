"""Read-only verification of the 2026-11-01 backup consolidation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "docs/integration/backup-consolidation-2026-11-01.json"


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", default="HEAD", help="Commit to inspect (default: HEAD)")
    args = parser.parse_args()
    audit = json.loads(AUDIT.read_text())
    entries = {}
    for line in git("ls-tree", "-r", args.ref).splitlines():
        meta, path = line.split("\t", 1)
        mode, _, blob = meta.split()
        entries[path] = {"mode": mode, "git_blob": blob}
    for path, expected in audit["required_master_files"].items():
        if entries.get(path) != expected:
            raise SystemExit(f"Reproduction file changed or missing: {path}")
    for branch in audit["branches"]:
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", branch["commit"], args.ref],
            cwd=ROOT, check=True,
        )
    print(
        f"PASS: {len(audit['required_master_files'])} byte-identical reproduction files, "
        f"{audit['model_count']} models, {len(audit['branches'])} original heads reachable."
    )
    print(f"Document archive: {audit['document_archive']['url']}")
    print("Drive archive SHA-256 was verified by download during consolidation; "
          "this offline check does not recheck Drive or remote tags.")


if __name__ == "__main__":
    main()
