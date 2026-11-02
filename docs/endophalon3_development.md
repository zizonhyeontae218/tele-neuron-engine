# Endophalon 3 버전 개발기록

소유자 지시에 따라 최종 독립 A/B 모델과 현재 실행·재현 코드만 유지한다. 이전 3T·공유 상태·초기 연결 실험의 모델 파일과 원시 산출물은 현재 트리에서 삭제하며, 아래는 개발 과정의 기록이다. 별도 보존 브랜치·태그·백업 폴더는 만들지 않는다. 과거 커밋은 정상 Git 이력에 남고 이력을 재작성하지 않는다.

## 실험 흐름

| 단계 | 실제 결과 | 판정 |
| --- | --- | --- |
| 초기 3T 독립 A/B, 40세대 | A 최고 88.33%, B 최고 74.48% | 정확도 목표 미달; 시드별 차이/slow-region 효과 불일치 |
| 연산 균형 라우팅 수정 | A 75.31% / 81.77%, 새 확인 73.33% / 78.54% | 개선 실패; B 확장 사전 게이트 종료 |
| 단일-head joint AB, 추가120세대 | 확인 72.32% / 73.02% / 76.77% | 3회 모두 목표 미달 |
| 공유 상태·분리 head alpha 초기 | 확인 67.19% | 기억 유지 효과 일부 관측, 목표 미달 |
| 공유 상태 A/B 파라미터 공동120세대 | 확인 65.44%→75.36%, A77.77%, B73.07% | 목표 미달; 감사 통과 후 경로 오류 복구 |
| 기존 독립 A/B 소프트웨어 연결 | 원래 88.33/74.48 정확재현, 새 로컬87.71/74.74 | 새 학습 없음, 모델별 이력·출력 레지스터 연결 |
| 현재 독립 A/B 각각120세대 | 완료 후 아래에 기록 | 실행 완료와 성능 달성을 구분 |

과거 보고서의 상대경로는 당시 트리를 가리킨다. 원본 커밋: 학습24ebc6b, 라우팅2506a9d, joint c70f262/5861aad, 공유alpha d400466, 공동120 91e4fc0/13ce85e, 소프트웨어연결378bcbd. 다음 인용은 수치·실패·한계 기록이며, 제거된 모델을 현재 유지한다는 지시가 아니다.

<details><summary>docs/endophalon3_training_report.ko.md — 당시 기록</summary>

````markdown
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

````

</details>

<details><summary>docs/endophalon3_routing_report.ko.md — 당시 기록</summary>

````markdown
# Endophalon 3 — 공유 판독 병목 실험

## 진단과 수정 결정 (학습 전 고정)

진단은 기존 9개 학습 상태와 4개 초기 상태의 train/validation만 사용했다. 새 theta 탐색과 원래 test의 재평가는 하지 않았다. 각 readout은 train에서만 적합했다. 자세한 수치는 `endophalon3_routing_runs/diagnosis/`에 있다.

| A Binary | 공유 16 zones | 연산별 16 zones | 연산별 전체 노드 | 명시적 게이팅 진단 | opcode 복원 zones / nodes |
| --- | --- | --- | --- | --- | --- |
| Seed 1 | 79.38% | 99.17% | 98.75% | 98.33% | 98.54% / 100% |
| Seed 2 | 88.75% | 94.79% | 99.38% | 94.58% | 85.42% / 100% |

recall2는 전체 노드의 연산별 readout에서 두 시드 모두 validation 100%였다. Seed 2의 16-zone 연산별 recall2는 81.82%, 공유 readout에서는 66.67%였다. 따라서 물리 상태에서 기억이 소실됐다고 단정할 수 없으며, zone 투영과 공유 판독의 연산 선택에서 손실이 나타난다. 게이팅 진단은 외부에서 opcode×zone 곱과 추가 특징을 공급하므로 실제 모델 성능이나 개선안으로 제시하지 않는다.

단일 수정은 **선택 목적함수의 validation 손실**이다.

`L = 0.5 × 전체 MSE + 0.5 × 연산별 MSE의 최댓값`

평균 MSE가 낮아지는 동안 특정 연산을 희생하는 해를 억제하는 가설이다. A Seed 2의 validation 전체 MSE는 0.09912, recall2 MSE는 0.19162였다. 계수 0.5는 대칭 혼합으로 학습 전에 고정했으며 탐색하지 않는다. 기존 noise/drift/complexity 계수, inner Ridge α, CEM, 초기 모델, binary 입력, seed, split, 144노드·2 substeps·16 zones·delay 12·40×24 예산을 유지한다. 생산 readout은 여전히 16개 특징의 단일 선형 readout이다.

기본 설정의 weight=0 동작은 기존 pooled MSE를 그대로 사용한다. 새로운 `l_val_mean`/`l_val_worst`와 목적함수를 카드에 기록한다. 정의가 다른 기존 J와 새 J의 크기를 성능 우열로 비교하지 않는다.

## 검증 계획

A Seed 1·2를 모두 수행한다. 원래 test는 이미 알려진 벤치마크이므로 theta 선택에 사용하지 않고, 미사용 스트림 10000–10015를 별도 확인 세트로 유지한다. 두 시드가 두 평가 세트에서 모두 전체 ≥90%, 각 연산 ≥85%를 충족해야 B로 확장한다. 실패한 시드도 동일하게 보고하며, 이번 실험에서 두 번째 수정이나 예산 확대를 하지 않는다.

## 구현 검증

전체 pytest 89개 통과(기존 85개 + 새 목적함수/설정 검증 4개). A Seed 1·2 학습과 평가를 완료했다. 아래 결과를 따른다.

## 완료 결과

두 실행은 원래 CLI에서 각각 40세대×24 population(초기 theta 포함 961회)을 완료했다. 새 test/확인 정확도는 카드의 저장된 readout으로 재실행했으며, 원래 test의 전체·연산별 정확도가 카드와 정확히 일치했다. 확인 세트는 theta 선택 후 처음 사용했다.

| Seed | 기존 Test | 수정 Test | 차이 (pp) | 기존 새 확인 | 수정 새 확인 | 차이 (pp) | 두 세트 합격 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 78.85 | 75.31 | -3.54 | 77.29 | 73.33 | -3.96 | FAIL |
| 2 | 88.33 | 81.77 | -6.56 | 86.56 | 78.54 | -8.02 | FAIL |

**이번 0.5 혼합 수정은 개선안으로 채택하지 않는다.** 두 시드 모두 원래 test와 새 확인 세트에서 기준 미달이며 전체 정확도가 하락했다. 사전에 고정한 gate에 따라 B 학습은 실행하지 않았다. 추가 계수 탐색, seed 교체, 예산 확대 또는 두 번째 수정은 하지 않았다.

## 연산별 비교

| 평가 | xor | recall1 | recall2 | xor_mem |
| --- | --- | --- | --- | --- |
| S1 기존 test | 73.22 | 81.50 | 73.50 | 87.12 |
| S1 수정 test | 79.92 | 74.80 | 75.21 | 71.24 |
| S1 기존 확인 | 71.91 | 75.20 | 72.88 | 88.89 |
| S1 수정 확인 | 73.62 | 72.36 | 74.15 | 73.25 |
| S2 기존 test | 97.07 | 94.49 | 67.09 | 93.99 |
| S2 수정 test | 84.10 | 83.46 | 77.78 | 81.55 |
| S2 기존 확인 | 95.74 | 91.87 | 63.98 | 94.24 |
| S2 수정 확인 | 82.13 | 79.67 | 75.85 | 76.54 |

recall2 test 정확도는 Seed 1 +1.71 pp, Seed 2 +10.68 pp 상승했지만 다른 연산 손실을 함께 회복하지 못했다. Seed 2의 xor/recall1/xor_mem test는 97.07/94.49/93.99%에서 84.10/83.46/81.55%로 하락했다. 이는 이번 계수·탐색 예산의 결과이며 모든 손실 균형 방식이나 물리계 전체의 불가능성을 증명하지 않는다.

## 목적함수의 실제 변화

| Seed / 선택 모델 | validation pooled MSE | validation worst-op MSE | 새 혼합 L |
| --- | --- | --- | --- |
| S1 기존 | 0.13674 | 0.16547 | 적용 안 함 |
| S2 기존 | 0.09912 | 0.19162 | 적용 안 함 |
| S1 수정 | 0.14644 | 0.14934 | 0.14789 |
| S2 수정 | 0.12024 | 0.12698 | 0.12361 |

새 validation 최악 연산 MSE는 두 시드 모두 감소했지만 pooled MSE는 증가했다. 최악 MSE를 줄이는 선택 압력이 일부 연산 간 절충을 바꿨으며, 전체 정확도 목표 달성으로 이어지지 않았다. Inner Ridge는 기존 pooled fit을 유지했다. 이 실험은 외부 theta 선택 손실만 바꾼 실험이다.

## 수정 후의 상태 진단 (validation)

| Seed | 공유 zones | 분리 zones | 분리 nodes | gated 진단 | opcode zones / nodes |
| --- | --- | --- | --- | --- | --- |
| 1 | 80.83 | 95.83 | 98.12 | 86.46 | 100.00 / 100.00 |
| 2 | 84.58 | 96.25 | 99.58 | 95.83 | 98.75 / 100.00 |

전체 노드의 연산별 readout에는 여전히 높은 정확도의 정보가 남지만 공유 readout과의 격차가 남았다. 진단용 gated 특징과 분리 head는 정규화 방식·특징 수·head 수가 다르므로 그 차이를 순수한 단일 인과 효과로 해석하지 않는다. 특정 readout의 높은 정확도가 물리계의 일반적 정보량이나 용량 상한을 입증하는 것도 아니다.

## 정규화·안정성

| Seed / Split | 평균 MSE | 최악 MSE | 혼합 L | R_noise | R_drift | R_complexity |
| --- | --- | --- | --- | --- | --- | --- |
| S1 / train | 0.15401 | 0.17285 | 0.16343 | 0.63018 | 0.14265 | 0.36942 |
| S1 / validation | 0.14644 | 0.14934 | 0.14789 | 0.58500 | 0.22466 | 0.36942 |
| S1 / test | 0.15989 | 0.18507 | 0.17248 | 0.66406 | 0.29638 | 0.36942 |
| S1 / confirmation | 0.16559 | 0.17710 | 0.17135 | 0.69604 | 0.20865 | 0.36942 |
| S2 / train | 0.12936 | 0.15527 | 0.14231 | 0.51709 | 0.14527 | 0.24776 |
| S2 / validation | 0.12024 | 0.12698 | 0.12361 | 0.47671 | 0.24047 | 0.24776 |
| S2 / test | 0.12990 | 0.14680 | 0.13835 | 0.52033 | 0.32360 | 0.24776 |
| S2 / confirmation | 0.14113 | 0.15326 | 0.14720 | 0.56927 | 0.22243 | 0.24776 |

선택 목적함수의 drift는 원래대로 train+validation 합집합에서 계산한다. 위 split별 drift는 개별 split의 비교값이다. 복잡도는 train에서 적합한 같은 readout 가중치의 속성이므로 split 간 동일하다. 에너지는 입력 직전 운동에너지의 추세이며 보존 오차가 아니다.

| Seed | 발산 후보 / 961 | Test displacement max abs | 확인 displacement max abs | CLI wall time (s) |
| --- | --- | --- | --- | --- |
| 1 | 0 | 0.75560 | 0.73645 | 994.58 |
| 2 | 0 | 1.58984 | 1.50737 | 1012.59 |

두 선택 모델은 64-event 범위에서 finite였으며 초기/선택 fitness의 모든 항을 카드와 수치적으로 재현했다. 무한 시간 안정성을 주장하지 않는다. 이전 Seed 1 CLI 시간에는 Random Baseline이 포함되므로 새 학습 시간과 직접 비교하거나 임의로 절반으로 나누지 않는다.

## 검증과 보존

- 전체 pytest: **89 passed**. 수정 코드의 기본 weight=0에서 기존 두 카드의 J·MSE·noise·drift·complexity·validation 정확도 차이는 모두 **0**이며 연산별 정확도도 정확히 일치했다.
- 새 모델 두 개의 평가 수 961, history 41개, 원래 test 전체/연산별 정확도, validation fitness와 저장 head 재적합을 검증했다.
- 원래 모델·설정·로그와 물리 engine/ISA/theta/Ridge는 `baseline_preservation.json`의 SHA-256으로 유지 여부를 확인한다. 원래 default loss가 보존됐고 수정은 opt-in config에만 적용된다.
- 원래 9개 모델의 결과는 고정했다. 새 모델은 `models/endophalon3_routing/`, 새 로그와 지표는 `docs/endophalon3_routing_runs/`에 분리했다.
- 기존 test를 이미 확인한 적응적 후속 연구다. 새 확인 세트의 ID 10000–10015는 학습 전 고정했으며 모델·계수 선택에 사용하지 않았다. 두 탐색 seed, 하나의 물리/데이터 seed만 평가했다.
- PR은 baseline 실험 브랜치를 대상으로 하며 master는 자동 병합하지 않는다. 성능 기준을 충족한 개선이라고 표현하지 않는다.

````

</details>

<details><summary>docs/endophalon3_joint_report.md — 당시 기록</summary>

````markdown
# Endophalon 3 — joint A/B training closeout

All three predeclared joint trials completed the 120-generation limit (2,881 theta evaluations each; 8,643 total). **0/3 passed the A/B accuracy gates.** Training completion does not establish the requested performance. No further seed, coefficient or test-driven restart was performed.

Each saved card uses one physics system and one train-fitted linear readout for interleaved A/B instructions, including memory across level switches. The 144-node, 2-substep, 16-zone invariants, original objective, physical/data seed 1, regular timing and readout delay 12 were retained. Each trial warm-started the corresponding original B-card theta; it did not resume the old optimizer distribution. Binary search seeds 1/2 and one-hot search seed 1 were fixed before training. All old cards/configurations/logs remain immutable.

| Joint trial | Validation pooled | Original AB test pooled | Fresh pooled | Fresh A | Fresh B | Both-level gate |
| --- | --- | --- | --- | --- | --- | --- |
| ab_binary_s1 | 81.25% | 71.98% | 72.32% | 73.89% | 70.67% | FAIL |
| ab_binary_s2 | 81.67% | 73.85% | 73.02% | 73.59% | 72.43% | FAIL |
| ab_onehot_s1 | 86.04% | 75.42% | 76.77% | 80.87% | 72.48% | FAIL |

Thresholds are >=90% overall for **each** level and >=85% for **each** operation within each level. Full operation-level values are retained in each card's `metrics`; pooled metrics do not substitute for these gates. Fresh streams 20000–20063 were declared in the plan before training and evaluated once after theta selection. Original joint AB splits are train 0–23 / validation 24–31 / test 32–47. Selected mean validation accuracy is 81–86%, while fresh accuracy is 72–77%; this gap and the weak recall/parity results limit the claims. Confirmation has now been observed and cannot be treated as untouched for a later adaptive experiment.

No candidate passed the validation gates. All trials exhausted the finite 120-generation budget, without triggering the 40-generation plateau rule. This does not prove that more compute or a different method cannot succeed. It also does not demonstrate slow-region recall2 benefit or long-horizon stability. Seeds vary the optimizer only, not physical/data generation; no independent-data or broad generalization claim is made. The earlier negative routing follow-up and its closed B expansion gate remain historical results; this joint study was separately authorized by the owner.

## Evidence and reproduction

- Training source commit: `c70f262`; [Actions run 41287301037](https://github.com/zizonhyeontae218/tele-neuron-engine/actions/runs/41287301037).
- Original output, optimizer/RNG checkpoints and logs: `docs/endophalon3_joint_runs/20261101/ab_*/`.
- Canonical byte-identical model copies: `models/endophalon3_joint/ab_*.json`.
- SHA-256 manifest: `docs/endophalon3_joint_runs/20261101/artifact_manifest.json`.
- Frozen procedure and stop rules: [joint plan](endophalon3_joint_plan.md).
- Read-only replay: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/audit_endophalon3_joint.py` after installing the project. It checks serialized heads, training-only refits, validation fitness, exact level/operation accuracies, finite states, checkpoint consistency and artifact hashes. It never retrains or rewrites published output.

GitHub CLI artifact download encountered a redirected-storage 403. The GitHub connector downloaded all three artifacts successfully; their original model/checkpoint/log bytes were recovered. The local audit initially needed the project source path; the audit script now resolves it directly. Training outputs were not altered to repair the audit environment.

The first checkpoint audit compared `encode(theta)` with the raw optimizer vector and failed for binary seed 2 (one scalar, max difference 0.0443435). The existing encoder clips probabilities to [1e-4, 1-1e-4], so it is not an exact inverse near a bound. The correct audit decodes the checkpoint vector and requires exact equality with the saved theta. Models, checkpoint vectors, physics and training results were not changed.

## Closeout verification

Full pytest: **95 passed**. All three saved heads exactly reproduce train/validation/original-test/fresh accuracies by level and operation; train-only refits, validation fitness, finite states and checkpoint theta checks pass. All three downloaded ZIP SHA-256 values match GitHub artifact digests, and every extracted model/checkpoint/log matches the original ZIP bytes. Existing 166 backup reproduction files and nine reachable original heads still pass integrity verification. Detailed check record: `docs/endophalon3_joint_runs/20261101/closeout_verification.json`.

The external Filetree skill generator is unavailable in this runtime. A separate local generator updates the index and actual SHA-256 ledger; an independent tracked-path/hash check verifies it. This is not a claim that the unavailable skill CLI lint ran.

````

</details>

<details><summary>docs/endophalon3_alpha.md — 당시 기록</summary>

````markdown
# Endophalon 3 (AB alpha) — implementation and first measurements

2026-11-02: the owner named the previous Endophalon 3 experiments **3T** and the new separate-bank AB alpha **3**. [Name mapping](model_names.md) preserves historical paths and raw card family strings. The new format and CLI are separate from the previous single-head family.

## Implemented behavior

Two complete independently configured A/B banks share one continuous positions/velocities/clock/RNG state. The command dispatcher selects the bank's input points, physical parameters, active 16-zone observation and its own train-fitted linear head. There is no per-operation head, per-bank saved physical state, input-history buffer or ground-truth memory supplied to the readout. Global data-event history spans mode switches; `level A/B` alone changes no physical state, time, RNG or event count. Initial jitter/warmup run only once; gap 16, delay 12, 144 nodes and 2 substeps remain.

Parameter switching holds positions/velocities fixed. Stored energy may jump because mass, stiffness or anchor settings change; that external-control energy change is recorded per event. No energy/momentum conservation claim is made. Each bank stores 16 observation centers and a head, so there are two stored observation/readout parameter sets, with only 16 zones evaluated per event.

Commands accept full `A OP a b`/`B OP a b` or `level A/B` followed by `OP a b`. Invalid commands are rejected before physical mutation. A saved session carries state, clock, active/selected modules and complete RNG states, tied to an exact model fingerprint. Restoring with another bundle is rejected. Non-finite integration fails explicitly. Output and checkpoint paths never overwrite existing files.

## Frozen first experiment

This is a bounded MVP, with **no outer physics search**. Exactly one original binary seed-1 A/B pair is used. Existing physics is frozen; A/B heads are separately refitted once on that level's events within the same training streams. Original heads and a pooled-head control are measured on the same physical observations. This pooled-head control is not the old joint-theta 3T model. [Plan](endophalon3_alpha_experiment_plan.md).

New AB stream IDs: train 30000–30023, validation 30024–30031, test 30032–30063, confirmation 30064–30127. Physical/data seed 1; 64 events, washout 4, split by whole stream. No seed/configuration/head-coefficient selection from test or confirmation.

| Fresh confirmation variant | Overall | A | B |
| --- | --- | --- | --- |
| original_heads | 65.60% | 60.80% | 70.49% |
| two_fitted_heads | 67.19% | 64.57% | 69.86% |
| pooled_head_control | 62.81% | 61.63% | 64.02% |
| same_heads_reset_at_switch | 63.72% | 58.48% | 69.07% |

The fitted two-head model remains below >=90% per level / >=85% per operation. Simply switching previously trained complete profiles does not preserve their fixed-level accuracy. Per-level/op and first/second-after-switch metrics, counts and negative outcomes remain in `docs/endophalon3_alpha_runs/20261102/seed1_binary/evaluation.json`. This implementation establishes a runnable separate-bank system; the high-accuracy model remains an unmet research goal.

## Shared-memory control

The reset control changes only positions/velocities to the original jitter state at each module transition. It keeps the same commands, fitted heads, elapsed clock and RNG progression; no extra warmup. Confirmation point estimates:

| Operation | Continuous state | Reset at switch | Difference |
| --- | --- | --- | --- |
| A recall1 | 80.53% | 66.80% | +13.73 pp |
| A recall2 | 62.37% | 57.42% | +4.95 pp |
| B recall1 | 64.03% | 56.92% | +7.11 pp |
| B recall2 | 58.16% | 50.63% | +7.53 pp |

These point estimates support a useful state-history contribution in this setting. They are not confidence intervals or proof that all profile switches preserve useful memory. Other operations still fail the target. This one-pair/one-data-seed experiment does not establish performance rankings or generalization. Observation is limited to 64-event streams; no infinite-horizon stability claim. The largest measured parameter-switch mechanical-energy jump was 0.40250055 in simulation units. Warmup/noise and switches are controlled simulation operations, not a closed conservative system.

## Verification and artifacts

- Fixed-level A and B observations and pre-event kinetic energies exactly match the preserved engine for tested streams; head scores match to 1e-13 and thresholded predictions exactly. Initial bitwise score checks exposed a maximum 4.44e-16 difference from one-row versus batched dot products; physical observations remain exact.
- State/time/RNG unchanged by selection-only or invalid commands; checkpoint continuation, model mismatch guards, explicit head selection and cross-level dependence tests pass.
- Full suite: **107 passed**, including CLI build/program/restore, explicit 3T serialization and mixed-bank batch-versus-individual equivalence.
- CLI smoke: 10 model/config combinations, including new 3 and the explicit 3T alias.
- Saved bundle audit: artifact hashes, disjoint splits, train-only head mean/scale/weights, all split metrics and exact physical replay of three fresh streams pass.
- Historical scientific source/configs/cards/logs remain byte-identical; current naming/documentation is updated without rewriting the originals.

Model: `models/endophalon3_alpha/20261102_seed1_binary.json`, an exact copy of the experiment bundle. Original observations, reset observations, model, evaluation and SHA-256 manifest: `docs/endophalon3_alpha_runs/20261102/seed1_binary/`.

Read-only audit after installation:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/audit_endophalon3_alpha.py docs/endophalon3_alpha_runs/20261102/seed1_binary
```

For an interactive program, use `python -m tele_neuron.endophalon3_alpha --model models/endophalon3_alpha/20261102_seed1_binary.json --program configs/endophalon3_alpha/example_program.txt`. [More commands](model_commands.md). Observed confirmation IDs are now used data and cannot be relabeled as untouched for an adaptive follow-up.

The next research step is adapting the bank physics to transitions while retaining separate heads, after inspecting validation transition/op failures and freezing another finite training budget. No further physical retraining or successful accuracy gate is implied by the current bundle. The old 3T results remain available as baselines.

Repository file index: the external skill generator was unavailable; `python scripts/filetree_index.py update` and `lint` provide a reproducible repository-local fallback using real tracked/unignored paths and actual SHA-256 values. This is separate from the unavailable external skill CLI.

## Subsequent 120-generation physics training
The new-3 paired-bank run completed; confirmation accuracy improved to 75.36%
(A 77.77%, B 73.07%), below target. See [original result report](endophalon3_alpha_120_report.md)
and [recovery/verification](endophalon3_alpha_120_closeout.md). The MVP results above are unchanged.

````

</details>

<details><summary>docs/endophalon3_alpha_120_report.md — 당시 기록</summary>

````markdown
# Endophalon 3 — 120-generation physics training result

The owner-authorized new-3 paired-bank run completed **120 generations / 2,881 candidate pairs**. A/B parameters and heads remain separate; one production Session carries state through commands. This is not the historical 3T single-head experiment. The selection used training/validation only. [Frozen plan](endophalon3_alpha_120_plan.md).

| Fresh streams 40000–40063 | Overall | A | B |
| --- | --- | --- | --- |
| Frozen baseline | 65.44% | 58.90% | 71.69% |
| Trained paired banks | 75.36% | 77.77% | 73.07% |
| Trained heads, reset-at-switch | 70.18% | 71.00% | 69.40% |

Fresh confirmation gate (>=90% each level and >=85% every operation): **FAIL**. A completed budget is distinct from a performance success. All per-operation and first/second-after-switch results, regularizers, failed candidates, switching energy and timing are in `docs/endophalon3_alpha_runs/20261102/physics120/evaluation.json`.

Both 77-dimensional bank theta vectors were optimized together, while each candidate's two heads were separately fitted on only the corresponding training events. Search seed 1, physical/data seed 1; one binary bank pair, no seed/coefficient search. The balanced-level MSE and existing noise/drift/weight penalties were frozen before training. Drift covers train+validation; switching energy is an external-control change, not a conservation residual. Fresh confirmation was first used after theta selection. The old test was previously observed and is descriptive. Confirmation is now observed and must not be reused as untouched evidence in adaptive follow-ups.

Reset controls change positions/velocities at bank switches with no extra warmup, preserving time/RNG and the same heads. They are state ablations, not separately trained reset models. Finite observations cover 64-event streams only; no infinite-horizon or multi-data-seed generalization claim. This one run cannot establish that a different seed/method/budget could never improve the result.

Artifacts include 121 archived optimizer states, full latest mean/sigma/RNG/best state and candidate metrics, raw log, environment, settings/source/start hashes, selected two-head bundle and held-out observations. The original model/configs/MVP/3T evidence is preserved. The standalone trained bundle is `models/endophalon3_alpha/20261102_physics120_seed1_binary.json`.

Read-only audit: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/audit_endophalon3_alpha_training.py docs/endophalon3_alpha_runs/20261102/physics120`. It verifies the budget, checkpoints, source/start fingerprints, train-only head refits, objective and all held-out metrics, plus exact physical replay on six held-out streams. Tests and model smoke are separate execution checks, not accuracy claims.

````

</details>

<details><summary>docs/endophalon3_alpha_120_closeout.md — 당시 기록</summary>

````markdown
# Endophalon 3 physics120 — closeout recovery

The 120-generation training budget completed (2,881 candidate pairs), with zero failed candidates. The complete saved-model audit **passed in the original Actions job**, including train-only head refits, objective, all held-out metrics and exact six-stream physical replay. Original job 41435945048 is still recorded as **failure** because the subsequent manifest-writing step mixed a relative output path with an absolute repository root. Auto-publication was skipped. This is a closeout error, not incomplete training or a failed accuracy reproduction.

The closeout path is now normalized before repository-relative hashing. A regression test runs the actual closeout with workflow-style relative paths, verifies output hashes, and confirms that a completed model cannot be overwritten. No training code, settings, original starting bundle, selected theta or trained head was changed. No retraining, seed change or additional candidate search occurred.

## Recovery and preservation

All 132 original artifact files were recovered, including the originally generated report and standalone model; they remain byte-identical. Artifact ZIP: 1,945,980 bytes; SHA-256 `eb56d6165d46c319edfb67e196b38bf35aa3ceb24800ce716bc81eceeb6ffc8d`. Original artifact ID 13142542905. Per-file original hashes: `docs/endophalon3_alpha_runs/20261102/physics120/recovery_original_files.json`. The full original Actions job log, including the successful audit and later traceback, is preserved as `actions_job.log` in the same directory. The original report is [retained unchanged](endophalon3_alpha_120_report.md).

Recovered budget, 121 archived checkpoints, start/source hashes and serialized held-out metrics were independently checked. The original successful expensive physical/head-refit audit was not repeated. Original 114 tests passed before training; the new closeout regression passed separately. CLI smoke (skipped by the failed job) and preservation/index checks were completed during recovery. Prior model/config/source/experiment files remain unchanged.

## Measured outcome

On the same fresh confirmation streams 40000–40063, overall accuracy rose from **65.44% to 75.36% (+9.92 pp)**. A rose from **58.90% to 77.77% (+18.87 pp)**; B rose from **71.69% to 73.07% (+1.37 pp)**. Both level and per-operation acceptance gates remain unmet. Validation-selected objective fell from 0.3286299508 to 0.1820288241; validation accuracy was 80.00%, with A 84.06% / B 75.55%. The run took about 74.6 minutes of measured process time; this is one seed/pair and one physical/data seed, not a cross-machine efficiency comparison.

| Level | Confirmation operation | Accuracy | Events |
| --- | --- | --- | --- |
| A | xor | 91.16% | 475 |
| A | recall1 | 76.69% | 429 |
| A | recall2 | 60.24% | 508 |
| A | xor_mem | 84.27% | 464 |
| B | xor | 69.17% | 253 |
| B | recall1 | 68.67% | 233 |
| B | recall2 | 50.65% | 231 |
| B | xor_mem | 73.98% | 246 |
| B | and | 93.96% | 265 |
| B | or | 92.75% | 262 |
| B | parity3w | 51.34% | 261 |
| B | and_mem | 82.16% | 213 |

The main remaining weaknesses are A recall2, B recall2 and B parity3w. A xor and B and/or exceed 90%, but other operations prevent acceptance. Reset-at-switch control was 70.18% overall versus 75.36% with continuous state; this is the same-head state-ablation comparison, not evidence that all memory tasks meet the target. Only 64-event streams were measured. Fresh confirmation has now been observed and cannot be reused as untouched validation for a follow-up.

The original GitHub workflow is not rerun: a clean rerun could repeat the completed training budget. The recovered artifacts and focused code fix are integrated through ordinary commits, preserving the failure record. Master and endophalon3 are verified after normal fast-forward publication; unrelated branches are not changed. No new remote branch is created. A separate verification record and complete hash manifest record this recovery.

````

</details>

<details><summary>docs/endophalon3_software.md — 당시 기록</summary>

````markdown
# Endophalon 3 — independent A/B models with software connections

Implemented and verified, 2026-11-02 owner follow-up. A and B run as independent saved models. Software selects a model and explicitly connects predicted outputs to later inputs. Previous 3T and shared-state alpha experiments, including the completed 120-generation run, remain intact.

## Contracts

- Two independent physical states, local clocks, RNGs and histories. Only the selected model advances; the other pauses, without reset or idle physics steps. Each retains its own 144 nodes, 2 substeps and 16-zone readout. This stores two reservoirs, not one shared reservoir.
- `A OP a b` / `B OP a b` dispatch a data event. `level A/B` selects the default for `OP a b` without advancing either model.
- `A xor 1 0 -> r0` stores the actual thresholded model prediction as register r0. `B and @r0 1 -> r1` resolves that stored bit before input encoding. Registers hold outputs only; no software task oracle or hidden physical-history transfer.
- Recall1/recall2 refer to the selected model's own previous data inputs. Cross-model memory is explicit register dataflow. Merely switching A/B does not make one model read the other's input history.
- Invalid commands or unknown registers fail before advancing either session or changing registers. Register identifiers are explicit. Partial physics failures remain failed sessions; they are not turned into fabricated output bits.
- Checkpoints bind both local states, selected level and output registers to the exact model bundle. New model/result/checkpoint paths never overwrite originals.

## Existing models and accuracy

The default preset imports A binary seed 2 (`20261031_a_binary_s2.json`) and B binary seed 1 (`20261031_b_binary_s1.json`), unchanged theta/readout/config/seed. Historical held-out scores: A 88.33%, B 74.48%. These are the highest observed separate-level baseline scores, chosen for deployment from already viewed results. They are not an unbiased architecture ranking. A has 4 ops; B has 8. ProMax's 91–97% figures describe a different 3-op task and cannot stand in for B's eight-op accuracy. Wiring in software does not create a new 90/80 success result.

## Implementation/verification plan

New package `src/tele_neuron/endophalon3_software/` contains a thin router around two production Sessions. The existing engine remains unchanged. Bundle schema `endophalon3-software/v1` identifies independent sessions; checkpoints use their own schema and preserve each model fingerprint. Example commands/preset belong in `configs/endophalon3_software/`.

Tests cover identical interleaved-versus-standalone results, paused state/time/RNG, actual-output register transfer (including wrong predictions), malformed commands, and checkpoint restoration. Saved-card checks replay each original native test independently. Fresh local streams 50000–50031 per model are frozen here before evaluation; no theta/readout refit, seed search, or test-driven training occurs. They evaluate each imported model's own local task, not a previously shared global-history task. Chained accuracy depends on upstream prediction errors and input distribution; do not multiply marginal accuracies as if errors were independent.

## Measured implementation outcome

No new physics or head fitting was performed. Both native saved-card tests replay **exactly**, overall and per operation: A 88.33%, B 74.48%. These are not a newly trained A>=90% / B>=80% pair. Fresh model-local streams 50000–50031, with calls alternating A[k],B[k], gave **A 87.71% and B 74.74%**. Each task uses local input history, so this is not the previous global-history shared-state task and cannot be used as a direct improvement claim over its 75.36% result.

The remaining fresh recall2 scores were A 62.36%, B 62.93%; B xor_mem was 57.69% and parity3w 57.32%. High current-task scores coexist with these limits. No post-hoc retraining, alternate seed or operation-set substitution was used. Choosing cards from previously observed native results is a deployment decision; the new check is conditional on that choice.

Tests: **120 passed**. CLI smoke: **11** model/config combinations. Tests verify that interleaving equals each standalone local input stream, inactive positions/velocity/time/RNG never change, actual predicted bits (including deliberately wrong predictions) are wired to the other model, invalid commands do not mutate state, and checkpoints resume both contexts/registers. An initial test caught only stale B selected-level metadata; dispatch now updates that model's selection after successful execution. Physical outputs/state were unaffected by that correction.

Serialized model audit verifies exact imported theta/readouts against the original cards, bundle/observation hashes and native/fresh per-op accuracies. Canonical bundle: `models/endophalon3_software/20261102_existing_cards.json`. Evidence: `docs/endophalon3_software_runs/20261102/existing_cards/` (observations, evaluation, model, hashes, verification). Prior 3T and alpha source/cards/evidence are preserved.

```bash
python -m tele_neuron.endophalon3_software --model models/endophalon3_software/20261102_existing_cards.json --program configs/endophalon3_software/example_program.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/audit_endophalon3_software.py docs/endophalon3_software_runs/20261102/existing_cards
```

The short example demonstrates wiring, not a score benchmark; each model initially has insufficient recall2 history and its own warmup. End-to-end chain accuracy must be measured on a defined program distribution before claiming 90/80 performance for composed programs.

````

</details>
