"""Check frozen DynaMath 001a artifacts without requiring old Git history."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', help='optional audit JSON, preferably in scratch')
    parser.add_argument('--refs-before', help='optional ls-remote snapshot; performs a remote read')
    args = parser.parse_args()
    manifest = json.loads(Path('models/dynamath/001a/artifact-integrity.json').read_text())
    changed = []
    for name, expected in manifest['files'].items():
        path = Path(name)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            changed.append(name)
    report = {'baseline_commit': manifest['baseline_commit'],
              'frozen_artifacts_checked': len(manifest['files']),
              'changed_or_missing': changed}
    if args.refs_before:
        def mapping(lines):
            return {ref: sha for sha, ref in (line.split() for line in lines)
                    if ref != 'refs/heads/feature/dynamath-001a'}
        before = Path(args.refs_before).read_text().splitlines()
        after = subprocess.check_output(['git', 'ls-remote', '--heads', '--tags', 'origin']).decode().splitlines()
        report['other_remote_refs_unchanged'] = mapping(before) == mapping(after)
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    if changed or report.get('other_remote_refs_unchanged') is False:
        raise SystemExit(json.dumps(report, indent=2))
    print(f"PASS: {len(manifest['files'])} frozen DynaMath 001a artifacts byte-identical")


if __name__ == '__main__':
    main()
