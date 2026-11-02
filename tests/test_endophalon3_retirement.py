"""A changed/new obsolete-tree file blocks the entire authorized deletion."""
import importlib.util
from pathlib import Path
import subprocess
import pytest

spec=importlib.util.spec_from_file_location('retirement',Path(__file__).resolve().parents[1]/'scripts/retire_endophalon3_experiments.py')
retirement=importlib.util.module_from_spec(spec);spec.loader.exec_module(retirement)

@pytest.fixture
def repo(tmp_path):
    subprocess.run(['git','init','-q',str(tmp_path)],check=True)
    for name in ['models/endophalon3/old.json','docs/endophalon3_runs/log.txt','models/endophalon2/keep.json','ipad_lab.html']:
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(name)
    subprocess.run(['git','add','.'],cwd=tmp_path,check=True)
    return tmp_path

def test_only_approved_unchanged_tree_is_deleted(repo):
    plan=retirement.plan(repo);assert len(plan)==2
    retirement.retire(repo,plan)
    assert not (repo/'models/endophalon3/old.json').exists()
    assert (repo/'models/endophalon2/keep.json').read_text()=='models/endophalon2/keep.json'
    assert (repo/'ipad_lab.html').is_file()

@pytest.mark.parametrize('change',['modify','new'])
def test_concurrent_work_rejects_all_deletions(repo,change):
    plan=retirement.plan(repo)
    name='old.json' if change=='modify' else 'new.json'
    (repo/'models/endophalon3'/name).write_text('new work')
    with pytest.raises(ValueError):retirement.retire(repo,plan)
    assert (repo/'models/endophalon3/old.json').is_file()
    assert (repo/'docs/endophalon3_runs/log.txt').is_file()
