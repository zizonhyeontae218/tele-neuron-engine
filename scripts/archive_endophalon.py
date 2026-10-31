"""Back up and retire only the six audited Endophalon model branches."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

MODELS = (
    "endophalon", "endophalon-plus", "endophalon-pro",
    "endophalon2", "endophalon2-pro", "endophalon2-promax",
)
PREFIX = "backup/2026-10-31/"


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True,
    ).stdout.strip()


def remote_refs(root: Path) -> dict[str, str]:
    return {ref: sha for sha, ref in (
        line.split() for line in git(root, "ls-remote", "--refs", "origin").splitlines()
    )}


def tag_matches(root: Path, value: str | None, expected: str) -> bool:
    return value is not None and git(root, "rev-parse", f"{value}^{{commit}}") == expected


def validate(root: Path, expected: dict[str, str], refs: dict[str, str]) -> None:
    for branch, sha in expected.items():
        if refs.get(f"refs/heads/{PREFIX}{branch}") != sha:
            raise RuntimeError(f"Safety branch missing or changed: {branch}")
        current = refs.get(f"refs/heads/{branch}")
        if current is not None and current != sha:
            raise RuntimeError(f"Original branch changed: {branch}; nothing may be deleted")
        tag = refs.get(f"refs/tags/{PREFIX}{branch}")
        if tag is not None and not tag_matches(root, tag, sha):
            raise RuntimeError(f"Conflicting preservation tag: {branch}")
        archive = "archive/endophalon-v1" if branch in MODELS[:3] else "archive/endophalon-v2"
        for destination in (archive, "master"):
            target = refs.get(f"refs/heads/{destination}")
            if target is None:
                raise RuntimeError(f"Missing integrated branch: {destination}")
            # The original history must be reachable before any branch is retired.
            git(root, "merge-base", "--is-ancestor", sha, target)


def finalize(root: Path, *, apply: bool = False) -> None:
    audit = json.loads((root / "docs/integration/original-heads.json").read_text())
    expected = {branch: audit[branch]["commit"] for branch in MODELS}
    if any(re.fullmatch(r"[0-9a-f]{40}", sha) is None for sha in expected.values()):
        raise RuntimeError("Invalid original commit SHA in audit")
    refs = remote_refs(root)
    validate(root, expected, refs)
    for branch, sha in expected.items():
        print(f"{branch}: tag {PREFIX}{branch} -> {sha}", flush=True)
    if not apply:
        print("Preview only: no remote references changed")
        return

    missing_tags = [f"{sha}:refs/tags/{PREFIX}{branch}" for branch, sha in expected.items()
                    if f"refs/tags/{PREFIX}{branch}" not in refs]
    if missing_tags:
        git(root, "push", "--atomic", "origin", *missing_tags)
    # Deletion is a separate transaction, strictly after remote tag verification.
    refs = remote_refs(root)
    validate(root, expected, refs)
    for branch, sha in expected.items():
        if not tag_matches(root, refs.get(f"refs/tags/{PREFIX}{branch}"), sha):
            raise RuntimeError(f"Remote tag verification failed: {branch}")
    deletions = {f"refs/heads/{branch}": sha for branch, sha in expected.items()
                 if f"refs/heads/{branch}" in refs}
    if deletions:
        # Check the server-advertised old OID, closing the gap between our last
        # read and Git's push negotiation. No force or force-with-lease is used.
        with tempfile.TemporaryDirectory(prefix="endophalon-push-guard-") as directory:
            hook = Path(directory) / "pre-push"
            hook.write_text(
                "#!/usr/bin/env python3\nimport sys\n"
                f"expected = {deletions!r}\n"
                "for line in sys.stdin:\n"
                "    local_ref, local_oid, remote_ref, remote_oid = line.split()\n"
                "    if (local_ref != '(delete)' or set(local_oid) != {'0'}\n"
                "            or expected.get(remote_ref) != remote_oid):\n"
                "        sys.exit('Deletion refused: unexpected or changed remote ref: ' + remote_ref)\n"
            )
            hook.chmod(0o755)
            git(root, "-c", f"core.hooksPath={directory}", "push", "--atomic", "origin",
                *(f":{ref}" for ref in deletions))
    refs = remote_refs(root)
    for branch, sha in expected.items():
        if f"refs/heads/{branch}" in refs:
            raise RuntimeError(f"Branch still exists: {branch}")
        if not tag_matches(root, refs.get(f"refs/tags/{PREFIX}{branch}"), sha):
            raise RuntimeError(f"Backup tag no longer matches: {branch}")
    message = "Six preservation tags verified; six superseded model branches absent.\n"
    message += "Master, archives, working branches and nine safety branches retained.\n"
    print(message)
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(summary).open("a") as output:
            output.write("## Endophalon archive cleanup\n\n" + message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Create tags, verify them, then delete the six old model branches.")
    args = parser.parse_args()
    finalize(Path(__file__).resolve().parents[1], apply=args.apply)


if __name__ == "__main__":
    main()
