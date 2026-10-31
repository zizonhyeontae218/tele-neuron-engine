"""Build evidence tables and a Korean research report from completed audits."""
from __future__ import annotations

import csv
from datetime import datetime
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'docs/endophalon3_runs/20261031-training'
EXPECTED=['a_binary_s1','a_binary_s2','a_no_regions_s1','a_no_regions_s2','a_onehot_s1','b_binary_s1','b_binary_s2','b_onehot_s1','b_onehot_s2']
OPS=['xor','recall1','recall2','xor_mem','and','or','parity3w','and_mem']


def pct(x):return f'{100*x:.2f}'
def f(x):return f'{x:.5f}'
def table(header,rows):
    return '\n'.join(['| '+' | '.join(header)+' |','| '+' | '.join(['---']*len(header))+' |']+['| '+' | '.join(str(v) for v in r)+' |' for r in rows])


def ablation(audits):
    results={};paired=[]
    for seed in [1,2]:
        names=[f'a_binary_s{seed}',f'a_no_regions_s{seed}']
        if any(n not in audits for n in names):continue
        by=[]
        for name in names:
            with (RUN/(name+'.predictions.csv')).open() as fp:
                rows=[r for r in csv.DictReader(fp) if r['op']=='recall2']
            streams=sorted({int(r['stream']) for r in rows})
            by.append(np.array([[sum(int(r['correct']) for r in rows if int(r['stream'])==s),sum(int(r['stream'])==s for r in rows)] for s in streams]))
        assert np.array_equal(by[0][:,1],by[1][:,1])
        rng=np.random.default_rng(20261031+seed);indices=rng.integers(0,len(by[0]),size=(10000,len(by[0])))
        rates=[b[indices,0].sum(axis=1)/b[indices,1].sum(axis=1) for b in by]
        delta=by[0][:,0].sum()/by[0][:,1].sum()-by[1][:,0].sum()/by[1][:,1].sum()
        ci=np.quantile(rates[0]-rates[1],[.025,.975]).tolist()
        results[str(seed)]={'recall2_delta':float(delta),'conditional_paired_stream_bootstrap_95ci':ci,'test_streams':len(by[0]),'bootstrap_replicates':10000,'bootstrap_seed':20261031+seed}
        paired.append((by[0],by[1]))
    if len(paired)==2:
        # Same stream resampling for both optimizer seeds; no event iid assumption.
        rng=np.random.default_rng(20261031);idx=rng.integers(0,len(paired[0][0]),size=(10000,len(paired[0][0])))
        samples=np.mean([p[idx,0].sum(axis=1)/p[idx,1].sum(axis=1)-q[idx,0].sum(axis=1)/q[idx,1].sum(axis=1) for p,q in paired],axis=0)
        ci=np.quantile(samples,[.025,.975]).tolist()
        results['mean']={'recall2_delta':float(np.mean([results[str(s)]['recall2_delta'] for s in [1,2]])),'conditional_paired_stream_bootstrap_95ci':ci,'both_seeds_positive':all(results[str(s)]['recall2_delta']>0 for s in [1,2]),'interpretation':'CI conditions on these two optimizer seeds and fixed data/physics generator; does not establish general seed-level significance.'}
    return results


def main():
    audits={p.name.removesuffix('.audit.json'):json.loads(p.read_text()) for p in RUN.glob('*.audit.json')}
    if not audits:raise RuntimeError('No audited models')
    names=[n for n in EXPECTED if n in audits];missing=[n for n in EXPECTED if n not in audits]
    abl=ablation(audits);(RUN/'slow_region_ablation.json').write_text(json.dumps(abl,indent=2)+'\n')
    summary=[]
    for name in names:
        a=audits[name];s=a['split_metrics'];test=s['test'];passes=test['accuracy']>=.9 and all(v['accuracy']>=.85 for v in test['per_op'].values())
        summary.append({'run':name,'level':'A' if name.startswith('a_') else 'B','encoding':'onehot' if 'onehot' in name else 'binary','slow_region':'no_regions' not in name,'search_seed':a['invariants']['search_seed'],'train_accuracy':s['train']['accuracy'],'validation_accuracy':s['validation']['accuracy'],'test_accuracy':test['accuracy'],'per_op':{k:v['accuracy'] for k,v in test['per_op'].items()},'a_accuracy_gate':passes if name.startswith('a_') else None,'seconds':a['timing']['seconds'],'failed_candidates':a['failed_training_candidates']})
    (RUN/'summary.json').write_text(json.dumps({'completed':summary,'missing':missing,'ablation':abl},indent=2)+'\n')
    with (RUN/'accuracy.csv').open('w',newline='') as fp:
        fields=['run','train','validation','test',*OPS,'seconds','failed_candidates']
        w=csv.DictWriter(fp,fieldnames=fields);w.writeheader()
        for r in summary:w.writerow({'run':r['run'],'train':r['train_accuracy'],'validation':r['validation_accuracy'],'test':r['test_accuracy'],**r['per_op'],'seconds':r['seconds'],'failed_candidates':r['failed_candidates']})
    gate=[r for r in summary if r['level']=='A' and r['slow_region']]
    verdict=f"기본 Slow Region을 유지한 A 모델 {len(gate)}개 중 정확도 조건 통과 {sum(bool(r['a_accuracy_gate']) for r in gate)}개."
    if gate and not any(r['a_accuracy_gate'] for r in gate):verdict+=' 고정 예산에서 요구 성능을 입증하지 못했다.'
    lines=['# Endophalon 3 — 학습 실험 및 성능 검증 보고서','',verdict,'',
        '기준 구현: `master@1425d63fafee8a3ab5127b59a51324600fed00e3`. 실험일: 2026-10-31.',
        '기존 모델·로그·구현·설정은 보존했다. 학습에는 기존 `python -m tele_neuron.endophalon3` CLI를 사용했다.','',
        f'완료한 학습 모델: **{len(names)}/9개**. 미완료: '+(', '.join(missing) if missing else '없음')+'.',
        '모든 정확도는 백분율이다. A의 정확도 합격 기준은 전체 ≥90%, 각 연산 ≥85%이다. Slow Region의 recall2 개선은 별도 조건이다.','',
        '## 1. 사전 검증과 실행 조건','',
        '- 전체 pytest: 85 passed. CLI/config smoke: 9개 성공. 4개 미학습 probe JSON이 기존 결과와 정확히 일치했다.',
        '- ProMax8/16 각각 2개, 총 4개 기존 모델의 저장된 theta/readout 재실행이 원래 테스트 정확도와 정확히 일치했다.',
        '- 144노드, 506 springs, 2 substeps, 16 readout zones, delay=12, 64 events, washout=4, regular gap=16 유지.',
        '- 40 generations ×24 population: 초기 theta 1개 포함 **961 theta 평가/학습**. Seed-1 Binary Random Baseline은 별도로 961개 평가.',
        '- A: train 0–15 / validation 16–23 / test 24–39. B: train 0–23 / validation 24–31 / test 32–47. 스트림 단위로 분리.',
        '- 모든 E3 설정은 기존 ProMax16 Seed-1 물리를 warm start로 사용한다. A/B를 따로 학습하며, B를 이번 A 결과에서 시작하지 않는다.',
        '- Seed 1/2는 **outer search seed**다. 물리, network jitter, 데이터 생성 seed는 모두 기존 config의 1로 고정된다. 독립 데이터 생성 반복은 아니다.',
        '- Python 3.12.14, NumPy 2.3.5, Linux, CPU quota 8 cores. CLI당 workers=4, 최대 2개 독립 탐색 동시 실행. BLAS/OMP/MKL 스레드는 1.',
        '- 환경에 pytest가 없어 최초 명령은 실패했다. 패키지 설치 후 전체 테스트가 통과했으며 최초 실패 로그도 보존했다.','',
        '## 2. 연산별 정확도','',
        table(['미학습 기준','Single zones','Single nodes','Per-op nodes'],[[p.stem,pct(json.loads(p.read_text())['zones_single']['accuracy']),pct(json.loads(p.read_text())['nodes_single']['accuracy']),', '.join(op+' '+pct(v) for op,v in json.loads(p.read_text())['nodes_separate_per_op'].items())] for p in sorted((RUN/'probes').glob('*.json'))]),'',
        table(['실험','Train','Validation','Test',*OPS,'A 정확도 조건'],[[r['run'],pct(r['train_accuracy']),pct(r['validation_accuracy']),pct(r['test_accuracy']),*[pct(r['per_op'][op]) if op in r['per_op'] else '—' for op in OPS],('PASS' if r['a_accuracy_gate'] else 'FAIL') if r['level']=='A' else '해당 없음'] for r in summary]),'',
        '아래 모델은 테스트 정확도가 아니라 **validation 기반 목적함수 J**로 선택되었다. 유리한 시드를 제외하거나 테스트 결과로 theta를 고르지 않았다.','',
        '## 3. Readout와 Baseline 비교','',
        table(['실험','Single','+Instruction','Separate zones','Linear nodes (432)','History'],[[n,*[pct(audits[n]['controls'][k]['accuracy']) for k in ['single_readout','single_readout_plus_instruction','separate_readouts','linear_nodes']],pct(audits[n]['history_baseline']['accuracy'])] for n in names]),'',
        'Separate Readouts는 operation별 16-zone readout이다. Linear Nodes는 432개 변위값의 단일 readout이며 실제 모델이 아니라 진단용이다. 기존 probe의 per-op nodes readout과 혼동하지 않는다.','']
    if 'a_binary_s1' in audits and 'random_search' in audits['a_binary_s1']:
        a=audits['a_binary_s1'];r=a['random_search'];lines += ['### 동일 예산 Random Baseline','',table(['선택 방식','평가 수','Validation J','Test',*list(a['split_metrics']['test']['per_op'])],[['Cross-entropy search',961,f(a['fitness']['j']),pct(a['split_metrics']['test']['accuracy']),*[pct(v['accuracy']) for v in a['split_metrics']['test']['per_op'].values()]],['Random search',961,f(r['selected_j']),pct(r['split_metrics']['test']['accuracy']),*[pct(v['accuracy']) for v in r['split_metrics']['test']['per_op'].values()]]]),'']
    theta_rows=[]
    for n in names:
        if not n.startswith('a_'):continue
        theta=json.loads((ROOT/audits[n]['card']).read_text())['theta']
        theta_rows.append([n,f(theta['damping']),f(theta['damping_slow']),f(theta['stiffness_slow'])])
    lines+=['## 4. Slow Region Ablation','',table(['실험','Fast damping','Slow damping','Slow stiffness scale'],theta_rows),'','Slow는 고정 공간 영역의 이름이며, 실제 느린 시간 척도를 학습했다는 해석은 수치만으로 단정하지 않는다. No-regions에서 slow 파라미터 값은 물리에 영향을 주지 않는다.','',table(['Seed','recall2 차이 (region−homogeneous, pp)','조건부 95% CI (pp)'],[[k,pct(v['recall2_delta']),'['+', '.join(pct(x) for x in v['conditional_paired_stream_bootstrap_95ci'])+']'] for k,v in abl.items()]),'',
        '16개 테스트 **전체 스트림**을 쌍으로 bootstrap 10,000회 재표집했다. 두 시드 평균은 동일한 스트림 인덱스를 두 시드에 적용한다. 이벤트를 독립 표본으로 취급하지 않았다. 이 구간은 관측한 두 탐색 시드에 조건부이며, 새 탐색 시드나 새 데이터 생성 seed에 대한 유의성을 증명하지 않는다.',
        'Homogeneous ablation도 기존 CLI대로 77차원 탐색을 유지한다. Slow Region 관련 2개 파라미터는 이때 물리에 사용되지 않는다.','',
        '## 5. Binary / One-hot 및 A / B','']
    comparison=[]
    for level in ['a','b']:
        for seed in [1,2]:
            ns=[f'{level}_binary_s{seed}',f'{level}_onehot_s{seed}']
            if all(n in audits for n in ns):
                aa,bb=[audits[n] for n in ns]
                comparison.append([level.upper(),seed,pct(aa['split_metrics']['test']['accuracy']),pct(bb['split_metrics']['test']['accuracy']),pct(bb['split_metrics']['test']['accuracy']-aa['split_metrics']['test']['accuracy']),aa['invariants']['theta_dimensions'],bb['invariants']['theta_dimensions']])
    if 'mean' in abl:
        v=abl['mean']; consistent=v['both_seeds_positive'] and v['conditional_paired_stream_bootstrap_95ci'][0]>0
        lines += [('두 관측 시드에서 recall2 개선이 관측되었다. 이 판정은 해당 시드·스트림 조건부다.' if consistent else 'Slow Region이 recall2를 일관되게 개선한다는 조건은 입증되지 않았다.'),'']
    lines += [table(['Level','Seed','Binary','One-hot','One-hot−Binary (pp)','Binary dim','One-hot dim'],comparison),'',
        'A One-hot는 소유자의 필수 순서대로 Seed 1만 수행했다. Binary 대비 단일 시드 비교로 일반적인 우열을 단정하지 않는다. B는 입력 채널 수뿐 아니라 연산 수 4→8, 학습 스트림 수 16→24도 달라진다.','',
        'A Binary에서는 level과 op_bit2 두 채널이 항상 0이며, A One-hot에서는 level 및 사용하지 않는 4개 opcode 채널이 항상 0이다. 해당 stimulus-point 차원도 기존 77/92차원 탐색에 그대로 포함했다. B의 level=1은 추가 자극도 가하므로 A/B 차이를 연산 개수 하나의 효과로 해석하지 않는다.','',
        '## 6. 정규화와 수치 안정성','',
        'readout의 mean/scale/weights는 train에서만 적합했다. 저장된 readout의 replay와 train 재적합 결과, validation fitness가 모델 카드와 일치하는지 검사했다. Split별 입력 스트림 SHA-256 중복도 검사했다.','',
        table(['실험 / Split','MSE','R_noise','R_drift','R_complexity','J_split'],[[n+' / '+label,*[f(audits[n]['split_metrics'][label][k]) for k in ['mse','r_noise','r_drift','r_complexity','j_split']]] for n in names for label in ['train','validation','test']]),'',
        'R_drift는 입력 직전 평균 운동에너지의 log(E+1e−9)에 대한 이벤트별 선형 추세 절댓값을 스트림별로 구하고 중앙값×60을 취한다. 강제 입력과 감쇠가 있는 계의 추세 지표이며, 전체 역학적 에너지 보존 오차는 아니다.','',
        '**기존 목적함수의 실제 범위:** MSE와 R_noise는 validation, R_complexity는 train으로 적합한 readout의 가중치, R_drift는 **train+validation 합집합**에서 계산한다. 위 J_split은 비교용으로 각 split의 drift를 사용하므로 선택 목적함수 J와 완전히 같은 값은 아니다. 복잡도는 같은 readout의 속성이므로 모든 split에서 같다.','',
        table(['실험','발산 후보 수 / 961','선택 모델 finite','Test energy max','Test displacement max abs','Early','Late'],[[n,audits[n]['failed_training_candidates'],audits[n]['finite_flag'],f(audits[n]['numerical']['test']['energy']['max']),f(audits[n]['numerical']['test']['nodes']['max_abs']),pct(audits[n]['controls']['single_readout']['early']),pct(audits[n]['controls']['single_readout']['late'])] for n in names]),'',
        'finite와 drift 검사는 64-event 입력 범위의 수치 관측이다. 무한 시간 안정성을 보증하지 않는다. 엔진의 failed flag는 기록 zone/최종 위치의 비유한값을 검사한다. 별도 audit는 energy/stretch/node/features도 검사했다. 기존 Random Baseline은 후보별 실패 수를 저장하지 않으므로 그 수는 미확인이다.','',
        '## 7. 학습 비용','',
        table(['실험','CLI wall time (s)','Theta 수','학습+검증 stream-theta 수'],[[n,f(audits[n]['timing']['seconds']),1922 if n=='a_binary_s1' else 961,(1922 if n=='a_binary_s1' else 961)*(24 if n.startswith('a_') else 32)] for n in names]),'',
        'Theta 평가 수는 목적함수 호출 횟수이며 반복 평가된 mean도 포함한다. Seed-1 A Binary 시간은 학습과 Random Baseline 및 최종 control 평가를 합한 시간이다. 나머지도 최종 평가와 카드 저장을 포함한다. 이 시간을 순수 물리학습 시간으로 해석하거나 Seed-1 시간을 임의로 절반으로 나누지 않는다. 두 시드 동시 실행은 같은 8-core quota를 공유한다.','',
        '## 8. 기존 Endophalon 2 ProMax 비교','',
        '기존 ProMax8: Seed 1 **91.04%**, Seed 2 **95.00%**. ProMax16: Seed 1 **96.88%**, Seed 2 **91.88%**. 이번 replay에서 네 결과를 모두 재현했다.',
        'ProMax는 xor/recall1/xor_mem 3개 연산, A는 recall2를 추가한 4개, B는 8개다. 데이터 생성 규칙과 opcode, 연산별 표본 분포도 다르므로 이 정확도 차이는 통제된 아키텍처 우열 비교가 아니다. 기존 ProMax를 재학습하거나 새 E3 task를 실행할 수 있도록 개조하지 않았다.','',
        table(['실험','E3의 공통 3연산 정확도 (기존 ProMax task와는 다른 스트림)'],[[n,pct(sum(v['accuracy']*v['n'] for op,v in audits[n]['split_metrics']['test']['per_op'].items() if op in ['xor','recall1','xor_mem'])/sum(v['n'] for op,v in audits[n]['split_metrics']['test']['per_op'].items() if op in ['xor','recall1','xor_mem']))] for n in names]),'',
        '## 9. 실패 양상과 한계','']
    paired_cost=[]
    for encoding in ['binary','onehot']:
        for seed in [1,2]:
            an,bn=f'a_{encoding}_s{seed}',f'b_{encoding}_s{seed}'
            if an not in audits or bn not in audits:continue
            aa,bb=audits[an],audits[bn]
            ratio='A에 Random Baseline 포함: 비교 불가' if an=='a_binary_s1' else f"{bb['timing']['seconds']/aa['timing']['seconds']:.2f}×"
            paired_cost.append([encoding,seed,pct(aa['split_metrics']['test']['accuracy']),pct(bb['split_metrics']['test']['accuracy']),pct(bb['split_metrics']['test']['accuracy']-aa['split_metrics']['test']['accuracy']),ratio])
    cost_notes = ['### A 대비 B의 관측 성능·비용','',table(['인코딩','Seed','A Test','B Test','B−A (pp)','B/A CLI wall time'],paired_cost),'',
        '고정 탐색 예산에서 B의 학습+검증 물리 스트림 수는 32/24=1.33배다. 위 wall time 비율은 최종 평가를 포함한 관측값이며 opcode 수·자극 부하·동시 실행 조건의 영향을 모두 포함한다. 아키텍처 하나의 순수 비용 효과로 분리할 수 없다.','']
    finite_count=sum(bool(audits[n]['finite_flag']) for n in names)
    failures=sum(audits[n]['failed_training_candidates'] for n in names)
    largest=max(names,key=lambda n:abs(audits[n]['split_metrics']['test']['r_drift']-audits[n]['split_metrics']['train']['r_drift']))
    aa=audits[largest]['split_metrics']
    stability_notes = [f"선택 모델 {finite_count}/{len(names)}개가 측정 범위에서 finite였으며, 학습 탐색 {961*len(names)}회 중 발산 flag는 {failures}회였다(Random Search 후보별 집계는 미저장).",
        f"Train–Test R_drift 차이가 가장 큰 관측 실행은 {largest}: train {aa['train']['r_drift']:.5f}, test {aa['test']['r_drift']:.5f}다. 유한한 결과와 동일한 분포의 drift 값을 구분해 해석한다.",'']
    b_means=[]
    for op in OPS:
        vals=[]
        for encoding in ['binary','onehot']:
            group=[audits[f'b_{encoding}_s{seed}']['split_metrics']['test']['per_op'][op]['accuracy'] for seed in [1,2] if f'b_{encoding}_s{seed}' in audits]
            vals.append(pct(float(np.mean(group)))+f' (n={len(group)} seeds)' if group else '미완료')
        b_means.append([op,*vals])
    lines += ['### B 연산별 시드 평균','',table(['연산','Binary','One-hot'],b_means),'',
        '각 평균은 완료된 모든 해당 시드를 포함한다. 주 결과 표에는 각 시드를 따로 기록했으며, 평균으로 불리한 시드를 숨기지 않는다.','']
    patterns=[]
    for n in names:
        if not n.startswith('b_'):continue
        for op in OPS:
            groups=audits[n]['pattern_errors_abaa1a2'][op]
            # Describe the largest observed contribution, including sample size.
            key,g=max(groups.items(),key=lambda item:(item[1]['errors'],item[1]['n']))
            total=audits[n]['split_metrics']['test']['per_op'][op]
            patterns.append([n,op,total['n'],pct(total['accuracy']),pct(total['majority_accuracy']),key,f"{g['errors']}/{g['n']}"])
    lines += [table(['B 실험','연산','표본','정확도','다수 클래스 기준','최다 오류 입력 (a,b,a1,a2)','오류/표본'],patterns),'',
        '표의 다수 클래스 값은 test 라벨 비율 max(p, 1−p)라는 설명용 통계이며, train에서 학습한 기준 모델의 성능은 아니다.',
        '입력 패턴의 오류 수는 기술통계이며 인과 설명이나 추가로 검정한 가설이 아니다. 낮은 표본 수 패턴의 오류율을 전체 성능으로 일반화하지 않는다.',
        'readout routing gap, parity3w, recall2는 표의 실제 값으로 평가한다. 별도 readout가 높은 정확도를 내더라도 단일 readout 모델의 합격으로 간주하지 않는다.','',
        '## 10. 증거와 재현','',
        '- 원본 CLI 로그·실행 argv·시각·wall time: `docs/endophalon3_runs/20261031-training/*.log`, `*.run.json`.',
        '- 저장 모델과 readout: `models/endophalon3/20261031_*.json`.',
        '- Split·정규화·안정성 검증: `*.audit.json`. 이벤트별 test score/prediction: `*.predictions.csv`.',
        '- 기계 판독 요약: `summary.json`, `accuracy.csv`, `slow_region_ablation.json`, `probe_comparison.json`, `promax_replay.jsonl`.',
        '- 실행: `python scripts/run_endophalon3_training.py --stage <a_binary|a_ablation|a_onehot|b_binary|b_onehot>`.',
        '- 감사: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/audit_endophalon3_training.py`.',
        '- 표 재생성: `python scripts/summarize_endophalon3_training.py`. 기존 학습 산출물이 있으면 runner는 덮어쓰지 않는다.',
        '- 모든 설정·기존 모델·기존 로그·src/tests가 기준 commit과 동일한지는 `preservation.json`으로 검증한다.','',
        '자동 병합하지 않는다. 결과가 기준 미달이어도 그대로 보고하고, 기존 설계를 바꾸지 않는다.','']
    body='\n'.join(lines)
    body=body.replace('## 7. 학습 비용', '\n'.join(stability_notes)+'\n\n## 7. 학습 비용')
    body=body.replace('## 8. 기존 Endophalon 2 ProMax 비교', '\n'.join(cost_notes)+'\n\n## 8. 기존 Endophalon 2 ProMax 비교')
    (ROOT/'docs/endophalon3_training_report.ko.md').write_text(body)
    print('Summarized',len(names),'audited models; missing',missing)

if __name__=='__main__':main()
