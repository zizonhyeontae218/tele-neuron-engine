"""Retire only the nine owner-authorized 2026-10-31 backup branches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import tempfile

from archive_endophalon import git, remote_refs, tag_matches

NAMES = (
    'agent/add-ipad-lab', 'claude/eloquent-volta-ylkb47', 'endophalon',
    'endophalon-plus', 'endophalon-pro', 'endophalon2', 'endophalon2-pro',
    'endophalon2-promax', 'master',
)
PREFIX = 'backup/2026-10-31/'
AUDIT = 'docs/integration/backup-consolidation-2026-11-01.json'


def validate(root, audit, refs):
    expected = {PREFIX + name for name in NAMES}
    if {b['branch'] for b in audit['branches']} != expected:
        raise RuntimeError('Audit must name exactly the nine authorized backup branches')
    if not audit['document_archive']['round_trip_sha256_verified']:
        raise RuntimeError('Document archive upload has not been verified')
    if refs.get('refs/heads/master') != git(root, 'rev-parse', 'HEAD'):
        raise RuntimeError('Main changed; inspect the current head before retrying')
    for branch in audit['branches']:
        name, sha = branch['branch'], branch['commit']
        if re.fullmatch(r'[0-9a-f]{40}', sha) is None or branch['preservation_tag'] != name:
            raise RuntimeError('Invalid original-head record')
        current = refs.get(f'refs/heads/{name}')
        if current is not None and current != sha:
            raise RuntimeError(f'Backup changed: {name}')
        tag = refs.get(f'refs/tags/{name}')
        if tag is not None and not tag_matches(root, tag, sha):
            raise RuntimeError(f'Conflicting preservation tag: {name}')
        if current is None and not tag_matches(root, tag, sha):
            raise RuntimeError(f'Backup and verified tag both absent: {name}')
        git(root, 'merge-base', '--is-ancestor', sha, 'HEAD')
    entries = {}
    for line in git(root, 'ls-tree', '-r', 'HEAD').splitlines():
        meta, path = line.split('\t', 1)
        mode, _, blob = meta.split()
        entries[path] = {'mode': mode, 'git_blob': blob}
    for path, value in audit['required_master_files'].items():
        if entries.get(path) != value:
            raise RuntimeError(f'Reproduction file changed or missing: {path}')


def finalize(root: Path, *, apply: bool = False):
    audit = json.loads((root / AUDIT).read_text())
    refs = remote_refs(root)
    validate(root, audit, refs)
    targets = {f"refs/heads/{b['branch']}": b['commit'] for b in audit['branches']}
    kept = {ref: sha for ref, sha in refs.items()
            if ref.startswith('refs/heads/') and ref not in targets}
    print('Verified original heads, reproduction files and Drive preservation record.', flush=True)
    if not apply:
        print('Preview only: nine authorized backup refs; no remote changes.')
        return
    missing = [f"{b['commit']}:refs/tags/{b['preservation_tag']}"
               for b in audit['branches']
               if f"refs/tags/{b['preservation_tag']}" not in refs]
    if missing:
        git(root, 'push', '--atomic', 'origin', *missing)
    refs = remote_refs(root)
    validate(root, audit, refs)
    for branch in audit['branches']:
        if not tag_matches(root, refs.get(f"refs/tags/{branch['preservation_tag']}"), branch['commit']):
            raise RuntimeError('Original-head tag verification failed')
    deletions = {ref: sha for ref, sha in targets.items() if ref in refs}
    if deletions:
        # Validate server-advertised old object IDs, closing the read/push race.
        with tempfile.TemporaryDirectory(prefix='backup-retirement-guard-') as tmp:
            hook = Path(tmp) / 'pre-push'
            hook.write_text(
                '#!/usr/bin/env python3\nimport sys\n'
                f'expected={deletions!r}\nseen=set()\n'
                'for line in sys.stdin:\n'
                '    local_ref,local_oid,remote_ref,remote_oid=line.split()\n'
                '    if local_ref!="(delete)" or set(local_oid)!={"0"} or expected.get(remote_ref)!=remote_oid:\n'
                '        sys.exit("Ref changed or outside backup allowlist: "+remote_ref)\n'
                '    seen.add(remote_ref)\n'
                'if seen!=set(expected):\n'
                '    sys.exit("Incomplete deletion transaction")\n'
            )
            hook.chmod(0o755)
            git(root, '-c', f'core.hooksPath={tmp}', 'push', '--atomic', 'origin',
                *(f':{ref}' for ref in deletions))
    refs = remote_refs(root)
    for ref in targets:
        if ref in refs:
            raise RuntimeError(f'Backup still present: {ref}')
    for ref, sha in kept.items():
        if refs.get(ref) != sha:
            raise RuntimeError(f'Retained branch changed concurrently: {ref}')
    for branch in audit['branches']:
        if not tag_matches(root, refs.get(f"refs/tags/{branch['preservation_tag']}"), branch['commit']):
            raise RuntimeError('Final preservation tag verification failed')
    print('COMPLETE: nine backup branches absent; nine tags verified; other branches unchanged.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    finalize(Path(__file__).resolve().parents[1], apply=args.apply)


if __name__ == '__main__':
    main()
