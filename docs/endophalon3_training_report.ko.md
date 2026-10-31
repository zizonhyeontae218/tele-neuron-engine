# Endophalon 3 — 학습 실험 및 성능 검증 보고서

기본 Slow Region을 유지한 A 모델 3개 중 정확도 조건 통과 0개. 고정 예산에서 요구 성능을 입증하지 못했다.

기준 구현: `master@1425d63fafee8a3ab5127b59a51324600fed00e3`. 실험일: 2026-10-31.
기존 모델·로그·구현·설정은 보존했다. 학습에는 기존 `python -m tele_neuron.endophalon3` CLI를 사용했다.

완료한 학습 모델: **5/9개**. 미완료: b_binary_s1, b_binary_s2, b_onehot_s1, b_onehot_s2.
모든 정확도는 백분율이다. A의 정확도 합격 기준은 전체 ≥90%, 각 연산 ≥85%이다. Slow Region의 recall2 개선은 별도 조건이다.

## 1. 사전 검증과 실행 조건

- 전체 pytest: 85 passed. CLI/config smoke: 9개 성공. 4개 미학습 probe JSON이 기존 결과와 정확히 일치했다.
- ProMax8/16 각각 2개, 총 4개 기존 모델의 저장된 theta/readout 재실행이 원래 테스트 정확도와 정확히 일치했다.
- 144노드, 506 springs, 2 substeps, 16 readout zones, delay=12, 64 events, washout=4, regular gap=16 유지.
- 40 generations ×24 population: 초기 theta 1개 포함 **961 theta 평가/학습**. Seed-1 Binary Random Baseline은 별도로 961개 평가.
- A: train 0–15 / validation 16–23 / test 24–39. B: train 0–23 / validation 24–31 / test 32–47. 스트림 단위로 분리.
- 모든 E3 설정은 기존 ProMax16 Seed-1 물리를 warm start로 사용한다. A/B를 따로 학습하며, B를 이번 A 결과에서 시작하지 않는다.
- Seed 1/2는 **outer search seed**다. 물리, network jitter, 데이터 생성 seed는 모두 기존 config의 1로 고정된다. 독립 데이터 생성 반복은 아니다.
- Python 3.12.14, NumPy 2.3.5, Linux, CPU quota 8 cores. CLI당 workers=4, 최대 2개 독립 탐색 동시 실행. BLAS/OMP/MKL 스레드는 1.
- 환경에 pytest가 없어 최초 명령은 실패했다. 패키지 설치 후 전체 테스트가 통과했으며 최초 실패 로그도 보존했다.

## 2. 연산별 정확도

| 미학습 기준 | Single zones | Single nodes | Per-op nodes |
| --- | --- | --- | --- |
| level_a | 71.46 | 87.92 | xor 100.00, recall1 100.00, recall2 100.00, xor_mem 100.00 |
| level_a_onehot | 72.92 | 89.90 | xor 100.00, recall1 100.00, recall2 100.00, xor_mem 100.00 |
| level_b | 66.46 | 70.73 | xor 100.00, recall1 100.00, recall2 100.00, xor_mem 100.00, and 100.00, or 100.00, parity3w 71.67, and_mem 100.00 |
| level_b_onehot | 65.62 | 72.40 | xor 100.00, recall1 100.00, recall2 100.00, xor_mem 100.00, and 100.00, or 100.00, parity3w 79.17, and_mem 100.00 |

| 실험 | Train | Validation | Test | xor | recall1 | recall2 | xor_mem | and | or | parity3w | and_mem | A 정확도 조건 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a_binary_s1 | 79.17 | 79.38 | 78.85 | 73.22 | 81.50 | 73.50 | 87.12 | — | — | — | — | FAIL |
| a_binary_s2 | 88.02 | 88.75 | 88.33 | 97.07 | 94.49 | 67.09 | 93.99 | — | — | — | — | FAIL |
| a_no_regions_s1 | 81.35 | 85.21 | 80.73 | 91.63 | 77.56 | 68.38 | 85.41 | — | — | — | — | FAIL |
| a_no_regions_s2 | 84.79 | 88.75 | 85.42 | 93.31 | 90.55 | 73.93 | 83.26 | — | — | — | — | FAIL |
| a_onehot_s1 | 86.56 | 89.58 | 87.71 | 92.05 | 93.31 | 77.78 | 87.12 | — | — | — | — | FAIL |

아래 모델은 테스트 정확도가 아니라 **validation 기반 목적함수 J**로 선택되었다. 유리한 시드를 제외하거나 테스트 결과로 theta를 고르지 않았다.

## 3. Readout와 Baseline 비교

| 실험 | Single | +Instruction | Separate zones | Linear nodes (432) | History |
| --- | --- | --- | --- | --- | --- |
| a_binary_s1 | 78.85 | 79.48 | 97.81 | 79.79 | 61.98 |
| a_binary_s2 | 88.33 | 87.81 | 94.48 | 86.67 | 61.98 |
| a_no_regions_s1 | 80.73 | 81.35 | 95.42 | 84.79 | 61.98 |
| a_no_regions_s2 | 85.42 | 85.42 | 92.71 | 82.08 | 61.98 |
| a_onehot_s1 | 87.71 | 85.52 | 96.46 | 91.35 | 61.77 |

Separate Readouts는 operation별 16-zone readout이다. Linear Nodes는 432개 변위값의 단일 readout이며 실제 모델이 아니라 진단용이다. 기존 probe의 per-op nodes readout과 혼동하지 않는다.

### 동일 예산 Random Baseline

| 선택 방식 | 평가 수 | Validation J | Test | xor | recall1 | recall2 | xor_mem |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Cross-entropy search | 961 | 0.18064 | 78.85 | 73.22 | 81.50 | 73.50 | 87.12 |
| Random search | 961 | 0.24025 | 73.75 | 86.19 | 62.99 | 72.65 | 73.82 |

## 4. Slow Region Ablation

| 실험 | Fast damping | Slow damping | Slow stiffness scale |
| --- | --- | --- | --- |
| a_binary_s1 | 0.88769 | 0.88535 | 0.30301 |
| a_binary_s2 | 0.87704 | 0.83407 | 0.75162 |
| a_no_regions_s1 | 0.86083 | 0.86873 | 0.94545 |
| a_no_regions_s2 | 0.86292 | 0.84835 | 0.71947 |
| a_onehot_s1 | 0.83953 | 0.86107 | 0.66834 |

Slow는 고정 공간 영역의 이름이며, 실제 느린 시간 척도를 학습했다는 해석은 수치만으로 단정하지 않는다. No-regions에서 slow 파라미터 값은 물리에 영향을 주지 않는다.

| Seed | recall2 차이 (region−homogeneous, pp) | 조건부 95% CI (pp) |
| --- | --- | --- |
| 1 | 5.13 | [0.83, 9.25] |
| 2 | -6.84 | [-11.30, -2.15] |
| mean | -0.85 | [-3.66, 2.01] |

16개 테스트 **전체 스트림**을 쌍으로 bootstrap 10,000회 재표집했다. 두 시드 평균은 동일한 스트림 인덱스를 두 시드에 적용한다. 이벤트를 독립 표본으로 취급하지 않았다. 이 구간은 관측한 두 탐색 시드에 조건부이며, 새 탐색 시드나 새 데이터 생성 seed에 대한 유의성을 증명하지 않는다.
Homogeneous ablation도 기존 CLI대로 77차원 탐색을 유지한다. Slow Region 관련 2개 파라미터는 이때 물리에 사용되지 않는다.

## 5. Binary / One-hot 및 A / B

Slow Region이 recall2를 일관되게 개선한다는 조건은 입증되지 않았다.

| Level | Seed | Binary | One-hot | One-hot−Binary (pp) | Binary dim | One-hot dim |
| --- | --- | --- | --- | --- | --- | --- |
| A | 1 | 78.85 | 87.71 | 8.85 | 77 | 92 |

A One-hot는 소유자의 필수 순서대로 Seed 1만 수행했다. Binary 대비 단일 시드 비교로 일반적인 우열을 단정하지 않는다. B는 입력 채널 수뿐 아니라 연산 수 4→8, 학습 스트림 수 16→24도 달라진다.

A Binary에서는 level과 op_bit2 두 채널이 항상 0이며, A One-hot에서는 level 및 사용하지 않는 4개 opcode 채널이 항상 0이다. 해당 stimulus-point 차원도 기존 77/92차원 탐색에 그대로 포함했다. B의 level=1은 추가 자극도 가하므로 A/B 차이를 연산 개수 하나의 효과로 해석하지 않는다.

## 6. 정규화와 수치 안정성

readout의 mean/scale/weights는 train에서만 적합했다. 저장된 readout의 replay와 train 재적합 결과, validation fitness가 모델 카드와 일치하는지 검사했다. Split별 입력 스트림 SHA-256 중복도 검사했다.

| 실험 / Split | MSE | R_noise | R_drift | R_complexity | J_split |
| --- | --- | --- | --- | --- | --- |
| a_binary_s1 / train | 0.14009 | 0.56352 | 0.10469 | 0.18448 | 0.18058 |
| a_binary_s1 / validation | 0.13674 | 0.54724 | 0.23143 | 0.18448 | 0.18909 |
| a_binary_s1 / test | 0.14407 | 0.58285 | 0.19261 | 0.18448 | 0.19432 |
| a_binary_s2 / train | 0.11119 | 0.44724 | 0.14692 | 0.13902 | 0.14964 |
| a_binary_s2 / validation | 0.09912 | 0.40213 | 0.26690 | 0.13902 | 0.14731 |
| a_binary_s2 / test | 0.10593 | 0.42959 | 0.29701 | 0.13902 | 0.15850 |
| a_no_regions_s1 / train | 0.13353 | 0.53333 | 0.18818 | 0.28770 | 0.18189 |
| a_no_regions_s1 / validation | 0.12173 | 0.48345 | 0.23044 | 0.28770 | 0.17183 |
| a_no_regions_s1 / test | 0.13911 | 0.56029 | 0.24439 | 0.28770 | 0.19444 |
| a_no_regions_s2 / train | 0.12263 | 0.48940 | 0.16528 | 0.30501 | 0.16668 |
| a_no_regions_s2 / validation | 0.11413 | 0.44915 | 0.22714 | 0.30501 | 0.16235 |
| a_no_regions_s2 / test | 0.12284 | 0.48930 | 0.30221 | 0.30501 | 0.18057 |
| a_onehot_s1 / train | 0.10464 | 0.42272 | 0.16000 | 0.24619 | 0.14424 |
| a_onehot_s1 / validation | 0.09693 | 0.39428 | 0.17315 | 0.24619 | 0.13642 |
| a_onehot_s1 / test | 0.10949 | 0.43886 | 0.06867 | 0.24619 | 0.14076 |

R_drift는 입력 직전 평균 운동에너지의 log(E+1e−9)에 대한 이벤트별 선형 추세 절댓값을 스트림별로 구하고 중앙값×60을 취한다. 강제 입력과 감쇠가 있는 계의 추세 지표이며, 전체 역학적 에너지 보존 오차는 아니다.

**기존 목적함수의 실제 범위:** MSE와 R_noise는 validation, R_complexity는 train으로 적합한 readout의 가중치, R_drift는 **train+validation 합집합**에서 계산한다. 위 J_split은 비교용으로 각 split의 drift를 사용하므로 선택 목적함수 J와 완전히 같은 값은 아니다. 복잡도는 같은 readout의 속성이므로 모든 split에서 같다.

| 실험 | 발산 후보 수 / 961 | 선택 모델 finite | Test energy max | Test displacement max abs | Early | Late |
| --- | --- | --- | --- | --- | --- | --- |
| a_binary_s1 | 0 | True | 0.00073 | 1.07721 | 78.96 | 78.75 |
| a_binary_s2 | 0 | True | 0.00609 | 1.98907 | 88.33 | 88.33 |
| a_no_regions_s1 | 0 | True | 0.00157 | 1.25851 | 81.67 | 79.79 |
| a_no_regions_s2 | 0 | True | 0.00500 | 2.04802 | 85.62 | 85.21 |
| a_onehot_s1 | 0 | True | 0.00163 | 1.02344 | 87.08 | 88.33 |

finite와 drift 검사는 64-event 입력 범위의 수치 관측이다. 무한 시간 안정성을 보증하지 않는다. 엔진의 failed flag는 기록 zone/최종 위치의 비유한값을 검사한다. 별도 audit는 energy/stretch/node/features도 검사했다. 기존 Random Baseline은 후보별 실패 수를 저장하지 않으므로 그 수는 미확인이다.

## 7. 학습 비용

| 실험 | CLI wall time (s) | Theta 수 | 학습+검증 stream-theta 수 |
| --- | --- | --- | --- |
| a_binary_s1 | 1875.92931 | 1922 | 46128 |
| a_binary_s2 | 929.35372 | 961 | 23064 |
| a_no_regions_s1 | 934.14141 | 961 | 23064 |
| a_no_regions_s2 | 963.48450 | 961 | 23064 |
| a_onehot_s1 | 857.96180 | 961 | 23064 |

Theta 평가 수는 목적함수 호출 횟수이며 반복 평가된 mean도 포함한다. Seed-1 A Binary 시간은 학습과 Random Baseline 및 최종 control 평가를 합한 시간이다. 나머지도 최종 평가와 카드 저장을 포함한다. 이 시간을 순수 물리학습 시간으로 해석하거나 Seed-1 시간을 임의로 절반으로 나누지 않는다. 두 시드 동시 실행은 같은 8-core quota를 공유한다.

## 8. 기존 Endophalon 2 ProMax 비교

기존 ProMax8: Seed 1 **91.04%**, Seed 2 **95.00%**. ProMax16: Seed 1 **96.88%**, Seed 2 **91.88%**. 이번 replay에서 네 결과를 모두 재현했다.
ProMax는 xor/recall1/xor_mem 3개 연산, A는 recall2를 추가한 4개, B는 8개다. 데이터 생성 규칙과 opcode, 연산별 표본 분포도 다르므로 이 정확도 차이는 통제된 아키텍처 우열 비교가 아니다. 기존 ProMax를 재학습하거나 새 E3 task를 실행할 수 있도록 개조하지 않았다.

| 실험 | E3의 공통 3연산 정확도 (기존 ProMax task와는 다른 스트림) |
| --- | --- |
| a_binary_s1 | 80.58 |
| a_binary_s2 | 95.18 |
| a_no_regions_s1 | 84.71 |
| a_no_regions_s2 | 89.12 |
| a_onehot_s1 | 90.91 |

## 9. 실패 양상과 한계

| B 실험 | 연산 | 표본 | 정확도 | 다수 클래스 기준 | 최다 오류 입력 (a,b,a1,a2) | 오류/표본 |
| --- | --- | --- | --- | --- | --- | --- |

표의 다수 클래스 값은 test 라벨 비율 max(p, 1−p)라는 설명용 통계이며, train에서 학습한 기준 모델의 성능은 아니다.
입력 패턴의 오류 수는 기술통계이며 인과 설명이나 추가로 검정한 가설이 아니다. 낮은 표본 수 패턴의 오류율을 전체 성능으로 일반화하지 않는다.
readout routing gap, parity3w, recall2는 표의 실제 값으로 평가한다. 별도 readout가 높은 정확도를 내더라도 단일 readout 모델의 합격으로 간주하지 않는다.

## 10. 증거와 재현

- 원본 CLI 로그·실행 argv·시각·wall time: `docs/endophalon3_runs/20261031-training/*.log`, `*.run.json`.
- 저장 모델과 readout: `models/endophalon3/20261031_*.json`.
- Split·정규화·안정성 검증: `*.audit.json`. 이벤트별 test score/prediction: `*.predictions.csv`.
- 기계 판독 요약: `summary.json`, `accuracy.csv`, `slow_region_ablation.json`, `probe_comparison.json`, `promax_replay.jsonl`.
- 실행: `python scripts/run_endophalon3_training.py --stage <a_binary|a_ablation|a_onehot|b_binary|b_onehot>`.
- 감사: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/audit_endophalon3_training.py`.
- 표 재생성: `python scripts/summarize_endophalon3_training.py`. 기존 학습 산출물이 있으면 runner는 덮어쓰지 않는다.
- 모든 설정·기존 모델·기존 로그·src/tests가 기준 commit과 동일한지는 `preservation.json`으로 검증한다.

자동 병합하지 않는다. 결과가 기준 미달이어도 그대로 보고하고, 기존 설계를 바꾸지 않는다.
