"""Verify every pre-existing artifact except the explicitly scoped source edits."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

BASE = '15d897b111ced59fd881c01201c5a530fa36e9c1'
ALLOWED = {'.github/workflows/tests.yml', '.gitignore', 'AGENTS.md', 'README.md',
           'README.ko.md', 'FileTree.md', 'FILETREE.hash.json',
           'src/tele_neuron/dynamath/AGENTS.md', 'src/tele_neuron/dynamath/runtime.py',
           'tests/test_dynamath.py'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refs-before')
    parser.add_argument('--output', default='docs/dynamath/experiment-001a/preservation.json')
    args = parser.parse_args()
    names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', BASE]).decode().splitlines()
    checked, changed = {}, []
    for name in names:
        if name in ALLOWED:
            continue
        old = subprocess.check_output(['git', 'show', f'{BASE}:{name}'])
        path = Path(name)
        if not path.is_file() or path.read_bytes() != old:
            changed.append(name)
        else:
            checked[name] = hashlib.sha256(old).hexdigest()
    report = {'base_commit': BASE, 'unchanged_preexisting_files': len(checked),
              'sha256': checked, 'unexpected_changed_or_missing_files': changed,
              'explicitly_allowed_original_edits': sorted(ALLOWED)}
    if args.refs_before:
        old = Path(args.refs_before).read_text().splitlines()
        new = subprocess.check_output(['git', 'ls-remote', '--heads', '--tags', 'origin']).decode().splitlines()
        def mapping(lines):
            return {r: h for h, r in (line.split() for line in lines) if r != 'refs/heads/feature/dynamath-001a'}
        report['other_remote_refs_unchanged'] = mapping(old) == mapping(new)
        report['other_remote_ref_count'] = len(mapping(old))
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    assert not changed, changed
    assert report.get('other_remote_refs_unchanged', True)
    print(f'PASS: {len(checked)} pre-existing files byte-identical; other remote refs preserved')


if __name__ == '__main__':
    main()
