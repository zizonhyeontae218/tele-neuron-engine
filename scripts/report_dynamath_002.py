"""Consolidate fixed-run evidence without training or selecting models."""
import json,hashlib
from pathlib import Path
from dynamath.korean_data import load
root=Path('docs/dynamath/experiment-002')
cfg=json.loads(Path('configs/dynamath/experiment-002.json').read_text())
results={name:json.loads((root/(name+'.json')).read_text()) for name in cfg['variants']}
splits,tok=load('data/dynamath/001a')
summary={};reviews={}
for name,r in results.items():
    gs=r['generations'];n=len(gs)
    summary[name]=dict(parameters=r['parameters'],response_CE=r['teacher_forced']['response_CE'],
        response_PPL=r['teacher_forced']['response_perplexity'],EOS_rate=sum(g['eos'] for g in gs)/n,
        empty_visible_rate=sum(not g['text'] for g in gs)/n,
        question_emissions=sum(len(g['question_events']) for g in gs),
        max_queue=max(g['peak_queue'] for g in gs),uncomposed_jamo=sum(g['uncomposed_jamo'] for g in gs),
        unexpected_controls=sum(g['unexpected_controls'] for g in gs),
        exact_answer_matches=sum(g['text']==row['A'] for g,row in zip(gs,splits['test'])),
        total_generation_ticks=sum(g['ticks'] for g in gs),total_output_tokens=sum(len(g['token_ids']) for g in gs),
        generation_writes_by_law=[sum(g['write_counts'][i] for g in gs) for i in range(9)],
        training_seconds=r['training']['seconds'],steps=r['training']['steps'],
        recall50={arm:v['accuracy'] for arm,v in r['recall']['results']['50'].items()})
    reviews[name]=[dict(question=g['question'],output=g['text'],expected=row['A'],
                        automatic_exact_match=g['text']==row['A']) for g,row in zip(gs,splits['test'])]
(root/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
(root/'sample-review.json').write_text(json.dumps(reviews,ensure_ascii=False,indent=2)+'\n')
rows=['| Condition | Coefficients | Response CE ↓ | PPL ↓ | EOS /12 | Train seconds |',
      '|---|---:|---:|---:|---:|---:|']
for name,s in summary.items():
    rows.append(f"| {name} | {s['parameters']} | {s['response_CE']:.4f} | {s['response_PPL']:.3f} | {round(s['EOS_rate']*12)} | {s['training_seconds']:.2f} |")
(root/'comparison.md').write_text('\n'.join(rows)+'\n')
# Checkpoint digest inventory includes every new checkpoint, never modifies legacy.
inventory={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path('models/dynamath/002').glob('*.pt'))}
(root/'artifact-integrity.json').write_text(json.dumps(inventory,indent=2)+'\n')
print('\n'.join(rows))
