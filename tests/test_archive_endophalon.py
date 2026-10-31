"""Exercise cleanup against a disposable bare remote, never against GitHub."""
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

SPEC = importlib.util.spec_from_file_location("archive_endophalon", Path(__file__).resolve().parents[1] / "scripts/archive_endophalon.py")
archive = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(archive)


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.PIPE).strip()


@pytest.fixture
def sandbox(tmp_path):
    remote, root = tmp_path / "remote.git", tmp_path / "work"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    subprocess.run(["git", "init", str(root)], check=True, capture_output=True)
    git(root, "config", "user.name", "Cleanup test")
    git(root, "config", "user.email", "cleanup@example.invalid")
    (root / "fixture").write_text("original")
    git(root, "add", ".")
    git(root, "commit", "-m", "original fixture")
    sha = git(root, "rev-parse", "HEAD")
    git(root, "remote", "add", "origin", str(remote))
    refs = ["master", "archive/endophalon-v1", "archive/endophalon-v2", "agent/keep"]
    refs += list(archive.MODELS) + [archive.PREFIX + b for b in archive.MODELS]
    git(root, "push", "origin", *(f"{sha}:refs/heads/{b}" for b in refs))
    path = root / "docs/integration/original-heads.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({b: {"commit": sha} for b in archive.MODELS}))
    return root, remote, sha


def test_preview_apply_and_idempotent_rerun_preserve_unrelated_refs(sandbox):
    root, _, sha = sandbox
    before = archive.remote_refs(root)
    archive.finalize(root)
    assert archive.remote_refs(root) == before
    archive.finalize(root, apply=True)
    after = archive.remote_refs(root)
    assert after["refs/heads/agent/keep"] == sha
    for branch in archive.MODELS:
        assert f"refs/heads/{branch}" not in after
        assert after[f"refs/tags/{archive.PREFIX}{branch}"] == sha
        assert after[f"refs/heads/{archive.PREFIX}{branch}"] == sha
    archive.finalize(root, apply=True)
    assert archive.remote_refs(root) == after


def test_changed_branch_stops_before_any_remote_mutation(sandbox):
    root, _, _ = sandbox
    git(root, "commit", "--allow-empty", "-m", "new work")
    git(root, "push", "origin", "HEAD:refs/heads/endophalon")
    before = archive.remote_refs(root)
    with pytest.raises(RuntimeError, match="Original branch changed"):
        archive.finalize(root, apply=True)
    assert archive.remote_refs(root) == before


def test_conflicting_tag_stops_before_deleting_branches(sandbox):
    root, _, _ = sandbox
    git(root, "commit", "--allow-empty", "-m", "different tag")
    git(root, "push", "origin", f"HEAD:refs/tags/{archive.PREFIX}endophalon")
    before = archive.remote_refs(root)
    with pytest.raises(RuntimeError, match="Conflicting preservation tag"):
        archive.finalize(root, apply=True)
    assert archive.remote_refs(root) == before


def test_push_guard_blocks_a_branch_changed_after_validation(sandbox, monkeypatch):
    root, remote, original = sandbox
    git(root, "commit", "--allow-empty", "-m", "concurrent work")
    newer = git(root, "rev-parse", "HEAD")
    git(root, "push", "origin", "HEAD:refs/heads/agent/keep")
    real_git = archive.git

    def racing_git(path, *args):
        if args[:1] == ("-c",) and "push" in args:
            # Move the remote AFTER validation, BEFORE push advertisement.
            git(remote, "update-ref", "refs/heads/endophalon", newer, original)
        return real_git(path, *args)

    monkeypatch.setattr(archive, "git", racing_git)
    with pytest.raises(subprocess.CalledProcessError):
        archive.finalize(root, apply=True)
    after = archive.remote_refs(root)
    assert after["refs/heads/endophalon"] == newer
    for branch in archive.MODELS:
        assert f"refs/heads/{branch}" in after  # Atomic transaction deleted none.
        assert after[f"refs/tags/{archive.PREFIX}{branch}"] == original
