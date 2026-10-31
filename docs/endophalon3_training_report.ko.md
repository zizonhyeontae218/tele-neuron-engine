# Endophalon 3 — 학습 실험 및 성능 검증 보고서

기본 Slow Region을 유지한 A 모델 3개 중 정확도 조건 통과 0개. 고정 예산에서 요구 성능을 입증하지 못했다.

기준 구현: `master@1425d63fafee8a3ab5127b59a51324600fed00e3`. 실험일: 2026-10-31.
기존 모델·로그·구현·설정은 보존했다. 학습에는 기존 `python -m tele_neuron.endophalon3` CLI를 사용했다.

완료한 학습 모델: **9/9개**. 미완료: 없음.
모든 정확도는 백분율이다. A의 정확도 합격 기준은 전체 ≥90%, 각 연산 ≥85%이다. Slow Region의 recall2 개선은 별도 조건이다.

## 1. 사전 검증과 실행 조건

- 전체 pytest: 85 passed. CLI/config smoke: 9개 성공. 4개 미학습 probe JSON이 기존 결과와 정확히 일치했다.
- ProMax8/16 각각 2개, 총 4개 기존 모델의 저장된 theta/readout 재실행이 원래 테스트 정확도와 정확히 일치했다.
- 144노드, 506 springs, 2 substeps, 16 readout zones, delay=12, 64 events, washout=4, regular gap=16 유지.
- 40 generations ×24 population: 초기 theta 1개 포함 **961 theta 평가/학습**. Seed-1 Binary Random Baseline은 별도로 961개 평가.
- A: train 0–15 / validation 16–23 / test 24–39. B: train 0–23 / validation 24–31 / test 32–47. 스트림 단위로 분리.
- 모든 E3 설정은 기존 ProMax16 Seed-1 물리를 warm start로 사용한다. A/B를 따로 학습하며, B를 이번 A 결과에서 시작하지 않는다.
- Seed 1/2는 **outer search seed**다. 물리, network jitter, 데이터 생성 seed는 모두 기존 config의 1로 고정된다. 독립 데이터 생성 반복은 아니다.
- 최초 7개 모델은 Python 3.12.14, NumPy 2.3.5, Linux, CPU quota 8 cores에서 실행했다. 복구된 B One-hot 두 모델의 runner·Python·CPU·wall time은 각 .run.json에 별도 기록했다. 모든 CLI는 workers=4, BLAS/OMP/MKL=1을 사용한다. 하드웨어가 달라 wall time의 직접적인 구조 비교에는 한계가 있다.
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
| b_binary_s1 | 75.07 | 76.88 | 74.48 | 80.87 | 80.17 | 61.19 | 62.18 | 89.92 | 94.59 | 55.00 | 75.21 | 해당 없음 |
| b_binary_s2 | 72.71 | 75.00 | 73.65 | 83.48 | 83.47 | 59.70 | 60.50 | 87.39 | 89.19 | 50.83 | 77.69 | 해당 없음 |
| b_onehot_s1 | 74.93 | 81.88 | 74.17 | 63.48 | 88.43 | 58.96 | 57.98 | 97.48 | 99.10 | 52.50 | 78.51 | 해당 없음 |
| b_onehot_s2 | 72.99 | 81.04 | 73.96 | 72.17 | 85.95 | 56.72 | 57.98 | 80.67 | 96.40 | 57.50 | 87.60 | 해당 없음 |

아래 모델은 테스트 정확도가 아니라 **validation 기반 목적함수 J**로 선택되었다. 유리한 시드를 제외하거나 테스트 결과로 theta를 고르지 않았다.

## 3. Readout와 Baseline 비교

| 실험 | Single | +Instruction | Separate zones | Linear nodes (432) | History |
| --- | --- | --- | --- | --- | --- |
| a_binary_s1 | 78.85 | 79.48 | 97.81 | 79.79 | 61.98 |
| a_binary_s2 | 88.33 | 87.81 | 94.48 | 86.67 | 61.98 |
| a_no_regions_s1 | 80.73 | 81.35 | 95.42 | 84.79 | 61.98 |
| a_no_regions_s2 | 85.42 | 85.42 | 92.71 | 82.08 | 61.98 |
| a_onehot_s1 | 87.71 | 85.52 | 96.46 | 91.35 | 61.77 |
| b_binary_s1 | 74.48 | 74.06 | 87.50 | 72.71 | 66.98 |
| b_binary_s2 | 73.65 | 73.65 | 85.31 | 69.06 | 66.98 |
| b_onehot_s1 | 74.17 | 73.54 | 88.65 | 77.40 | 62.40 |
| b_onehot_s2 | 73.96 | 74.27 | 87.60 | 77.92 | 62.40 |

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
| B | 1 | 74.48 | 74.17 | -0.31 | 77 | 92 |
| B | 2 | 73.65 | 73.96 | 0.31 | 77 | 92 |

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
| b_binary_s1 / train | 0.17669 | 0.77656 | 0.12009 | 0.10976 | 0.22862 |
| b_binary_s1 / validation | 0.15611 | 0.63368 | 0.09242 | 0.10976 | 0.19813 |
| b_binary_s1 / test | 0.17405 | 0.74981 | 0.15928 | 0.10976 | 0.22857 |
| b_binary_s2 / train | 0.18092 | 0.80365 | 0.14225 | 0.05165 | 0.23584 |
| b_binary_s2 / validation | 0.16553 | 0.68989 | 0.09017 | 0.05165 | 0.20956 |
| b_binary_s2 / test | 0.17552 | 0.75368 | 0.16569 | 0.05165 | 0.23029 |
| b_onehot_s1 / train | 0.16901 | 0.72460 | 0.07857 | 0.08142 | 0.21391 |
| b_onehot_s1 / validation | 0.14672 | 0.58847 | 0.03993 | 0.08142 | 0.18095 |
| b_onehot_s1 / test | 0.17287 | 0.73783 | 0.12410 | 0.08142 | 0.22299 |
| b_onehot_s2 / train | 0.17627 | 0.77562 | 0.09355 | 0.07044 | 0.22511 |
| b_onehot_s2 / validation | 0.15206 | 0.61262 | 0.04612 | 0.07044 | 0.18801 |
| b_onehot_s2 / test | 0.18122 | 0.80261 | 0.13033 | 0.07044 | 0.23509 |

R_drift는 입력 직전 평균 운동에너지의 log(E+1e−9)에 대한 이벤트별 선형 추세 절댓값을 스트림별로 구하고 중앙값×60을 취한다. 강제 입력과 감쇠가 있는 계의 추세 지표이며, 전체 역학적 에너지 보존 오차는 아니다.

**기존 목적함수의 실제 범위:** MSE와 R_noise는 validation, R_complexity는 train으로 적합한 readout의 가중치, R_drift는 **train+validation 합집합**에서 계산한다. 위 J_split은 비교용으로 각 split의 drift를 사용하므로 선택 목적함수 J와 완전히 같은 값은 아니다. 복잡도는 같은 readout의 속성이므로 모든 split에서 같다.

| 실험 | 발산 후보 수 / 961 | 선택 모델 finite | Test energy max | Test displacement max abs | Early | Late |
| --- | --- | --- | --- | --- | --- | --- |
| a_binary_s1 | 0 | True | 0.00073 | 1.07721 | 78.96 | 78.75 |
| a_binary_s2 | 0 | True | 0.00609 | 1.98907 | 88.33 | 88.33 |
| a_no_regions_s1 | 0 | True | 0.00157 | 1.25851 | 81.67 | 79.79 |
| a_no_regions_s2 | 0 | True | 0.00500 | 2.04802 | 85.62 | 85.21 |
| a_onehot_s1 | 0 | True | 0.00163 | 1.02344 | 87.08 | 88.33 |
| b_binary_s1 | 0 | True | 0.01121 | 2.88496 | 73.75 | 75.21 |
| b_binary_s2 | 0 | True | 0.02224 | 3.47793 | 73.33 | 73.96 |
| b_onehot_s1 | 0 | True | 0.00108 | 1.40835 | 74.79 | 73.54 |
| b_onehot_s2 | 0 | True | 0.00151 | 1.63023 | 73.12 | 74.79 |

finite와 drift 검사는 64-event 입력 범위의 수치 관측이다. 무한 시간 안정성을 보증하지 않는다. 엔진의 failed flag는 기록 zone/최종 위치의 비유한값을 검사한다. 별도 audit는 energy/stretch/node/features도 검사했다. 기존 Random Baseline은 후보별 실패 수를 저장하지 않으므로 그 수는 미확인이다.

선택 모델 9/9개가 측정 범위에서 finite였으며, 학습 탐색 8649회 중 발산 flag는 0회였다(Random Search 후보별 집계는 미저장).
Train–Test R_drift 차이가 가장 큰 관측 실행은 a_binary_s2: train 0.14692, test 0.29701다. 유한한 결과와 동일한 분포의 drift 값을 구분해 해석한다.


## 7. 학습 비용

| 실험 | CLI wall time (s) | Theta 수 | 학습+검증 stream-theta 수 |
| --- | --- | --- | --- |
| a_binary_s1 | 1875.92931 | 1922 | 46128 |
| a_binary_s2 | 929.35372 | 961 | 23064 |
| a_no_regions_s1 | 934.14141 | 961 | 23064 |
| a_no_regions_s2 | 963.48450 | 961 | 23064 |
| a_onehot_s1 | 857.96180 | 961 | 23064 |
| b_binary_s1 | 911.40090 | 961 | 30752 |
| b_binary_s2 | 1024.78075 | 961 | 30752 |
| b_onehot_s1 | 1313.66039 | 961 | 30752 |
| b_onehot_s2 | 946.70296 | 961 | 30752 |

Theta 평가 수는 목적함수 호출 횟수이며 반복 평가된 mean도 포함한다. Seed-1 A Binary 시간은 학습과 Random Baseline 및 최종 control 평가를 합한 시간이다. 나머지도 최종 평가와 카드 저장을 포함한다. 이 시간을 순수 물리학습 시간으로 해석하거나 Seed-1 시간을 임의로 절반으로 나누지 않는다. 두 시드 동시 실행은 같은 8-core quota를 공유한다.

### A 대비 B의 관측 성능·비용

| 인코딩 | Seed | A Test | B Test | B−A (pp) | B/A CLI wall time |
| --- | --- | --- | --- | --- | --- |
| binary | 1 | 78.85 | 74.48 | -4.38 | A에 Random Baseline 포함: 비교 불가 |
| binary | 2 | 88.33 | 73.65 | -14.69 | 1.10× |
| onehot | 1 | 87.71 | 74.17 | -13.54 | 1.53× |

고정 탐색 예산에서 B의 학습+검증 물리 스트림 수는 32/24=1.33배다. 위 wall time 비율은 최종 평가를 포함한 관측값이며 opcode 수·자극 부하·동시 실행 조건의 영향을 모두 포함한다. 아키텍처 하나의 순수 비용 효과로 분리할 수 없다.


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
| b_binary_s1 | 74.37 |
| b_binary_s2 | 75.77 |
| b_onehot_s1 | 70.14 |
| b_onehot_s2 | 72.11 |

## 9. 실패 양상과 한계

### B 연산별 시드 평균

| 연산 | Binary | One-hot |
| --- | --- | --- |
| xor | 82.17 (n=2 seeds) | 67.83 (n=2 seeds) |
| recall1 | 81.82 (n=2 seeds) | 87.19 (n=2 seeds) |
| recall2 | 60.45 (n=2 seeds) | 57.84 (n=2 seeds) |
| xor_mem | 61.34 (n=2 seeds) | 57.98 (n=2 seeds) |
| and | 88.66 (n=2 seeds) | 89.08 (n=2 seeds) |
| or | 91.89 (n=2 seeds) | 97.75 (n=2 seeds) |
| parity3w | 52.92 (n=2 seeds) | 55.00 (n=2 seeds) |
| and_mem | 76.45 (n=2 seeds) | 83.06 (n=2 seeds) |

각 평균은 완료된 모든 해당 시드를 포함한다. 주 결과 표에는 각 시드를 따로 기록했으며, 평균으로 불리한 시드를 숨기지 않는다.

| B 실험 | 연산 | 표본 | 정확도 | 다수 클래스 기준 | 최다 오류 입력 (a,b,a1,a2) | 오류/표본 |
| --- | --- | --- | --- | --- | --- | --- |
| b_binary_s1 | xor | 115 | 80.87 | 56.52 | 1000 | 4/7 |
| b_binary_s1 | recall1 | 121 | 80.17 | 53.72 | 0011 | 6/12 |
| b_binary_s1 | recall2 | 134 | 61.19 | 53.73 | 1101 | 8/11 |
| b_binary_s1 | xor_mem | 119 | 62.18 | 52.94 | 0110 | 7/10 |
| b_binary_s1 | and | 119 | 89.92 | 76.47 | 1110 | 4/8 |
| b_binary_s1 | or | 111 | 94.59 | 72.07 | 0011 | 2/11 |
| b_binary_s1 | parity3w | 120 | 55.00 | 51.67 | 1000 | 9/13 |
| b_binary_s1 | and_mem | 121 | 75.21 | 76.03 | 1100 | 8/9 |
| b_binary_s2 | xor | 115 | 83.48 | 56.52 | 1110 | 5/7 |
| b_binary_s2 | recall1 | 121 | 83.47 | 53.72 | 1000 | 5/8 |
| b_binary_s2 | recall2 | 134 | 59.70 | 53.73 | 0001 | 8/9 |
| b_binary_s2 | xor_mem | 119 | 60.50 | 52.94 | 0010 | 9/13 |
| b_binary_s2 | and | 119 | 87.39 | 76.47 | 1110 | 2/8 |
| b_binary_s2 | or | 111 | 89.19 | 72.07 | 1000 | 5/8 |
| b_binary_s2 | parity3w | 120 | 50.83 | 51.67 | 1000 | 10/13 |
| b_binary_s2 | and_mem | 121 | 77.69 | 76.03 | 1100 | 5/9 |
| b_onehot_s1 | xor | 115 | 63.48 | 56.52 | 0101 | 8/11 |
| b_onehot_s1 | recall1 | 121 | 88.43 | 53.72 | 1110 | 4/9 |
| b_onehot_s1 | recall2 | 134 | 58.96 | 53.73 | 0001 | 9/9 |
| b_onehot_s1 | xor_mem | 119 | 57.98 | 52.94 | 0010 | 9/13 |
| b_onehot_s1 | and | 119 | 97.48 | 76.47 | 0110 | 2/10 |
| b_onehot_s1 | or | 111 | 99.10 | 72.07 | 1010 | 1/5 |
| b_onehot_s1 | parity3w | 120 | 52.50 | 51.67 | 0101 | 9/10 |
| b_onehot_s1 | and_mem | 121 | 78.51 | 76.03 | 1100 | 9/9 |
| b_onehot_s2 | xor | 115 | 72.17 | 56.52 | 1110 | 7/7 |
| b_onehot_s2 | recall1 | 121 | 85.95 | 53.72 | 0010 | 5/6 |
| b_onehot_s2 | recall2 | 134 | 56.72 | 53.73 | 1101 | 11/11 |
| b_onehot_s2 | xor_mem | 119 | 57.98 | 52.94 | 0010 | 12/13 |
| b_onehot_s2 | and | 119 | 80.67 | 76.47 | 1101 | 6/8 |
| b_onehot_s2 | or | 111 | 96.40 | 72.07 | 1001 | 3/8 |
| b_onehot_s2 | parity3w | 120 | 57.50 | 51.67 | 0101 | 6/10 |
| b_onehot_s2 | and_mem | 121 | 87.60 | 76.03 | 1100 | 9/9 |

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

## 실행 환경 중단과 복구

원래 Work 실행기가 연결 해제되어 B One-hot Seed 1·2의 최종 모델을 회수하지 못했다. 마지막 직접 확인은 양쪽 25/40세대다. 부분 실행을 성공으로 계산하지 않았다.
각 중단 실행은 최소 601 theta 평가를 사용했다(초기 1 + 25×24). 정확한 최종 진행률·전체 원본 로그·wall time은 회수하지 못했으며, 관측한 tail 발췌만 interrupted_runs.json에 보존했다.
복구 실행은 GitHub Actions에서 기존 theta_init과 같은 탐색 시드 1·2로 처음부터 수행했다. 중간 theta를 넘기거나 시드를 바꾸지 않았고, 각 완료 실행은 기존 CLI의 40×24 예산을 유지했다. 복구 환경·실제 argv·시간은 각 .run.json에 기록했다.
원래 실행기와 복구 runner의 하드웨어가 다르므로 마지막 두 실행의 wall time을 모델 구조 차이 하나로 해석할 수 없다. 폐기된 부분 실행 비용은 완료 모델 학습 비용과 별개로 존재한다.
Filetree 스킬의 updater/linter는 실행기 손실 뒤 사용할 수 없어 마지막 재생성·lint를 수행하지 못했다. 기존 생성 인덱스와 ledger는 보존했으며, 검증된 신규/기존 증거의 실제 SHA-256은 artifact_manifest.json과 preservation.json에 기록했다.
최초 복구 감사는 mean/scale의 bitwise equality에서 실패했다(recovery_audit.log). 새 감사는 실제 최대 절대오차를 각 audit JSON에 기록하고 mean/scale rtol=atol=1e-12로 수치 동등성을 검사한다. 모델 카드·원본 학습 로그·정확도 기준은 바꾸지 않았으며, aggregate/per-op test 정확도는 여전히 정확히 재현되어야 한다.
복구 audit return code: 0. 완료 9/9; 미완료: 없음.

## 후속 보존 정리 — 2026-10-31

실행 환경이 복구되어 중단된 cloud 원본 부분 로그와 당시 메타데이터를 추가 회수했다. `cloud-attempts-recovered/`에 그대로 복사했으며 stale running 상태는 성공으로 해석하지 않는다. 마지막 기록은 Seed 1의 28세대, Seed 2의 29세대다. 중단 실행의 확인 가능한 최소 평가 수는 각각 673/697, 합계 1,370회로 갱신된다. 최종 theta/readout은 회수하지 못했다. 이전 절의 25세대·로그 미회수 설명은 당시 확인 범위이며, 원래 보고서와 성공 모델 수·정확도를 수정하지 않았다. 기존 모델·로그·감사 증거 45개 파일의 SHA-256은 모두 유지된다.

Filetree 생성기와 lint는 이번 보존 정리에서 다시 실행하고 결과를 `closeout_recovery.json`에 기록한다.
