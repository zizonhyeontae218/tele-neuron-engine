"""Replay frozen outputs with corpus loading disabled; export event traces."""
import json,time,platform,os,tempfile
from pathlib import Path
import torch
from dynamath.v002 import Model,Session
from dynamath.v002.__main__ import generate
from dynamath import korean_data

def unavailable(*args,**kwargs):raise RuntimeError('dataset intentionally disabled')
korean_data.load=unavailable;korean_data.fetch=unavailable
root=Path('docs/dynamath/experiment-002').resolve();repo=Path.cwd();torch.set_num_threads(2);results={}
for name in json.loads(Path('configs/dynamath/experiment-002.json').read_text())['variants']:
    path=root/(name+'.json')
    if not path.exists():continue
    saved=json.loads(path.read_text());model,_=Model.load('models/dynamath/002/'+name+'.pt')
    identical=0;started=time.process_time();tokens=0;ticks=0
    for sample in saved['generations']:
        with tempfile.TemporaryDirectory() as empty:
            os.chdir(empty)
            try:actual=generate(model,sample['question'],360)
            finally:os.chdir(repo)
        assert actual['token_ids']==sample['token_ids'],name
        assert actual['question_events']==sample['question_events'],name
        identical+=1;tokens+=len(actual['token_ids']);ticks+=actual['ticks']
    cpu=time.process_time()-started
    session=Session(model);tok=korean_data.Tokenizer(model.config.vocabulary)
    session.enqueue([0]+tok.encode('안녕하세요')+[1])
    for _ in range(90):session.tick()
    snapshot=session.snapshot();other=Session(model);other.restore(snapshot)
    for _ in range(30):
        assert session.tick()==other.tick()
        assert session.snapshot()==other.snapshot()
    (root/(name+'-trace.json')).write_text(json.dumps(session.log,ensure_ascii=False,indent=2)+'\n')
    results[name]=dict(identical_samples=identical,dataset_disabled=True,snapshot_resume_exact=True,
                       cpu_seconds=cpu,ticks=ticks,ticks_per_cpu_second=ticks/cpu,
                       emitted_tokens=tokens,emitted_tokens_per_cpu_second=tokens/cpu,
                       scalar_bytes=4*model.config.memories,state_bytes=4*model.n)
(root/'saved-checkpoint-verification.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
