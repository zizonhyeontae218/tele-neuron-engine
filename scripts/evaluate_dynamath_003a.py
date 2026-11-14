"""Validation-only F1 restoration on the trained pilot contenders."""
import copy,json,torch
from pathlib import Path
from dynamath.v003a import Model,BASE
from dynamath.v003a.ast import expression,validate
from dynamath.v003a.english_data import load,audit
from dynamath.v003a.training import streams,evaluate
from dynamath.korean_data import load as ko_load
from dynamath.v002.__main__ import evaluate as ko_evaluate

def main():
    torch.set_num_threads(2);results={}
    for lang in ('english','korean'):
        if lang=='english':
            data=load();tok,_=audit(data);val=streams(data['validation'][:16],tok,8);score=lambda m:evaluate(m,val)
        else:
            data,tok=ko_load('data/dynamath/001a');val=data['validation'][:32];score=lambda m:ko_evaluate(m,val,tok)
        results[lang]={}
        for name in ('leader','offspring'):
            model,_=Model.load(f'models/dynamath/003a/{lang}/{name}-final.pt');original=score(model);restore=copy.deepcopy(model);restore.asts[0]=BASE;restored=score(restore)
            results[lang][name]={'expression':expression(model.asts[0]),'nodes':validate(model.asts[0]),'alpha':float(model.alpha[0].detach()),'validation':original,'restored_validation':restored,'restored_minus_changed_CE':restored['response_CE']-original['response_CE']}
            print(lang,name,results[lang][name]['restored_minus_changed_CE'],flush=True)
    Path('docs/dynamath/experiment-003a/ablation.json').write_text(json.dumps(results,indent=2,ensure_ascii=False)+'\n')
if __name__=='__main__':main()
