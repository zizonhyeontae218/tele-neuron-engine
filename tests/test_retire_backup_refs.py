"""Exercise the nine-backup transaction against a disposable bare remote."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
SPEC = importlib.util.spec_from_file_location('retire_backup_refs', SCRIPTS / 'retire_backup_refs.py')
retire = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(SCRIPTS))
SPEC.loader.exec_module(retire)


def git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root, text=True, stderr=subprocess.PIPE).strip()


@pytest.fixture
def sandbox(tmp_path):
    remote, root = tmp_path / 'remote.git', tmp_path / 'work'
    subprocess.run(['git', 'init', '--bare', str(remote)], check=True, capture_output=True)
    subprocess.run(['git', 'init', str(root)], check=True, capture_output=True)
    git(root, 'config', 'user.name', 'Cleanup test')
    git(root, 'config', 'user.email', 'cleanup@example.invalid')
    (root / 'model.json').write_text('{"original":true}')
    git(root, 'add', '.')
    git(root, 'commit', '-m', 'original fixture')
    sha = git(root, 'rev-parse', 'HEAD')
    blob = git(root, 'rev-parse', 'HEAD:model.json')
    git(root, 'remote', 'add', 'origin', str(remote))
    names = ['master', 'claude/keep', 'endophalon3', 'experiment/keep']
    names += [retire.PREFIX + name for name in retire.NAMES]
    git(root, 'push', 'origin', *(f'{sha}:refs/heads/{name}' for name in names))
    audit = {'branches': [{'branch': retire.PREFIX + name, 'commit': sha,
                          'preservation_tag': retire.PREFIX + name} for name in retire.NAMES],
             'document_archive': {'round_trip_sha256_verified': True},
             'required_master_files': {'model.json': {'mode': '100644', 'git_blob': blob}}}
    path = root / retire.AUDIT
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(audit))
    return root, remote, sha


def test_preview_apply_and_idempotent_rerun(sandbox):
    root, _, sha = sandbox
    before = retire.remote_refs(root)
    retire.finalize(root)
    assert retire.remote_refs(root) == before
    retire.finalize(root, apply=True)
    after = retire.remote_refs(root)
    for name in retire.NAMES:
        assert f'refs/heads/{retire.PREFIX}{name}' not in after
        assert after[f'refs/tags/{retire.PREFIX}{name}'] == sha
    for name in ('master', 'claude/keep', 'endophalon3', 'experiment/keep'):
        assert after[f'refs/heads/{name}'] == sha
    retire.finalize(root, apply=True)
    assert retire.remote_refs(root) == after


def test_changed_backup_refuses_before_tags_or_deletions(sandbox):
    root, _, _ = sandbox
    git(root, 'commit', '--allow-empty', '-m', 'new work')
    git(root, 'push', 'origin', 'HEAD:refs/heads/' + retire.PREFIX + retire.NAMES[0])
    # Keep HEAD at published master so this specifically checks the backup lease.
    git(root, 'checkout', '--detach', 'HEAD~1')
    before = retire.remote_refs(root)
    with pytest.raises(RuntimeError, match='Backup changed'):
        retire.finalize(root, apply=True)
    assert retire.remote_refs(root) == before
