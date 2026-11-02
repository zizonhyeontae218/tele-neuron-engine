"""Repository-local index fallback; update actual paths/hashes without the external skill CLI."""
import argparse
from collections import defaultdict
import fnmatch
import hashlib
import json
from pathlib import Path
import re
import subprocess


def scan(root, config):
    names=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=root).decode().split('\0')
    generated={'.filetree.json',config['manifest_path'],config['hash_path']}
    def selected(name):
        return name and name not in generated and (root/name).is_file() and not any(fnmatch.fnmatch(name,g) for g in config['exclude']) and (not config['include'] or any(fnmatch.fnmatch(name,g) for g in config['include']))
    return sorted({name for name in names if selected(name)})


def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['update','lint']);ap.add_argument('--root',type=Path,default=Path('.'));args=ap.parse_args();root=args.root
    cfg=json.loads((root/'.filetree.json').read_text());manifest=root/cfg['manifest_path'];ledger=root/cfg['hash_path'];paths=scan(root,cfg)
    hashes={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}
    descriptions={};parent=''
    for line in manifest.read_text().splitlines():
        group=re.match(r'^## `(.*)`$',line)
        if group:parent=group[1]
        entry=re.match(r'^- `([^`]+)` — (.*)$',line)
        if entry:descriptions[(parent+'/' if parent else '')+entry[1]]=entry[2]
    if args.command=='lint':
        previous=json.loads(ledger.read_text())['files']
        drift=sorted({p for p in set(previous)|set(hashes) if previous.get(p)!=hashes.get(p)})
        if set(descriptions)!=set(paths):drift.append('manifest path set')
        if drift:raise SystemExit('Index drift: '+', '.join(drift))
        print(f'PASS: {len(paths)} indexed paths and SHA-256 hashes');return
    groups=defaultdict(list)
    for p in paths:
        path=Path(p);parent='' if path.parent==Path('.') else str(path.parent)
        desc=descriptions.get(p)
        if path.suffix=='.md':
            first=next((s[2:].strip() for s in (root/p).read_text(errors='replace').splitlines() if s.startswith('# ')),None)
            if first:desc=first.rstrip('.')+'.'
        if not desc:
            desc='Contains '+path.stem.replace('_',' ')+' data.'
        groups[parent].append((path.name,desc))
    text='# FILETREE\n\n> Generated repository index. Open the real files before editing.\n'
    for parent,entries in sorted(groups.items()):
        text+=f'\n## `{parent}`\n\n'+''.join(f'- `{name}` — {desc}\n' for name,desc in entries)
    manifest.write_text(text)
    ledger.write_text(json.dumps({'version':1,'manifest_path':cfg['manifest_path'],'files':hashes},indent=2,ensure_ascii=False)+'\n')
    print(f'Updated {len(paths)} indexed paths and actual SHA-256 hashes')

if __name__=='__main__':main()
