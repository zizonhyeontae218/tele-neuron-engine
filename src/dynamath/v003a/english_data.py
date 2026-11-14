"""Pinned DailyDialog public splits; corpus is cached outside the repository."""
import collections, hashlib, json, urllib.request, zipfile
from pathlib import Path
from dynamath.korean_data import Tokenizer
REVISION='745c1796cfe209b469394567f496815d2bc495d2'
SHA256='b1f38f3bf5431f9384107011a943f1a3446027721292e3a248a894b7766fa9d3'
URL=f'https://huggingface.co/datasets/ConvLab/dailydialog/resolve/{REVISION}/data.zip'
def load(path='work/dailydialog/data.zip', raw=False):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    if not p.exists():
        with urllib.request.urlopen(URL,timeout=120) as f: p.write_bytes(f.read())
    if hashlib.sha256(p.read_bytes()).hexdigest()!=SHA256: raise ValueError('source hash mismatch')
    with zipfile.ZipFile(p) as z: data=json.loads(z.read('data/dialogues.json'))
    splits={k:[] for k in ('train','validation','test')}
    for d in data:
        turns=[t['utterance'] for t in d['turns']]
        splits[d['data_split']].append({'id':d['dialogue_id'],'turns':turns,'speakers':[t['speaker'] for t in d['turns']]})
    return splits if raw else remove_cross_split_dialogues(splits)[0]

def remove_cross_split_dialogues(splits):
    # Preserve public split membership; hold out test first, then validation.
    seen=set(); clean={}; removed={}
    for name in ('test','validation','train'):
        keep=[]; excluded=[]
        for d in splits[name]:
            (excluded if tuple(d['turns']) in seen else keep).append(d)
        clean[name]=keep; removed[name]=[d['id'] for d in excluded]
        seen.update(tuple(d['turns']) for d in keep)
    return clean,removed

def pairs(d):
    # Preserve speaker order and avoid duplicating utterances across adjacent training pairs.
    return [{'Q':d['turns'][i],'A':d['turns'][i+1]} for i in range(0,len(d['turns'])-1,2)]

def audit(splits):
    docs={k:{tuple(d['turns']) for d in v} for k,v in splits.items()}
    qa={k:{(p['Q'],p['A']) for d in v for p in pairs(d)} for k,v in splits.items()}
    tok=Tokenizer.build([p for d in splits['train'] for p in pairs(d)])
    out={'source':URL,'revision':REVISION,'archive_sha256':SHA256,'license':'CC BY-NC-SA 4.0','splits':{},'overlap':{},'vocabulary_size':len(tok.vocabulary)}
    for k,ds in splits.items():
        ps=[p for d in ds for p in pairs(d)]; lengths=[len(tok.encode(p[t])) for p in ps for t in ('Q','A')]
        ids=[tok.encode(p[t]) for p in ps for t in ('Q','A')]; n=sum(map(len,ids))
        out['splits'][k]={'dialogues':len(ds),'pairs':len(ps),'duplicate_qa_excess':len(ps)-len(qa[k]),'unk_rate':sum(x.count(4) for x in ids)/n,'roundtrip_failures':sum(tok.decode(seq)!=p[t] for p in ps for t,seq in [(t,tok.encode(p[t])) for t in ('Q','A')]),'length_mean':sum(lengths)/len(lengths),'length_max':max(lengths)}
    for a,b in [('train','validation'),('train','test'),('validation','test')]: out['overlap'][a+':'+b]={'exact_dialogues':len(docs[a]&docs[b]),'exact_qa':len(qa[a]&qa[b])}
    return tok,out
