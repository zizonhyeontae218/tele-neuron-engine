# DynaMath 002 실행·비교 보고서

실행일: 2026-11-13 KST. 브랜치: `feature/dynamath-002`.
시작 커밋: `f304c5a87787a468f708044fe44320b2b6847f63`.
기준 계약은 [CODEX_HANDOFF.md](CODEX_HANDOFF.md), 추가 승인 범위는 Model 9/3/1과
Model 3의 교차 접근 비교다. 기존 실험, 체크포인트, 기본 브랜치는 보존했다.

## 판정

**구현과 실제 학습·생성·검증은 완료했지만, 유용한 한국어 대화와 가변 스칼라를
활용한 학습 기억은 실패했다. 이슈의 과학적 성공 조건을 전부 달성했다고 주장하지 않는다.**

9개 독립 법칙이 하나의 S를 한 틱에 하나씩 갱신한다. 입력은 블록 시작마다 하나,
출력 검사는 나머지 두 틱마다 실제로 수행한다. 별도 인코더/출력 헤드, 답변 검색,
질문 모드 출력 차단, 출력 상태 덮어쓰기는 없다. 런타임 스칼라는 optimizer의
Parameter가 아니며, 현재 소유자 값만 hard retain/replace 한다. 초기값은 float32의
0/1/e, 쓰기 후보는 [-e,e]다. 자세한 수학, FIFO 정책, 명령은 [README](README.md).

Model 3 local은 한 블록의 세 법칙이 그 블록의 한 값을 공유하고 다른 블록 값은
읽지 못한다. cross는 세 값을 모두 읽을 수 있다. 둘 다 다른 블록 값에 직접 쓰지는
못한다. Model 1도 법칙 수는 9개이며, 모든 법칙이 하나의 값을 읽고 쓴다.

## 고정 예산과 데이터

Song Ys / songys, Chatbot_data, MIT. 원본 저작자와 모델 개발자를 구분한다.
원본 commit `4cf20d13fc46f5037fd1c531cd566e2dd9f72974`, CSV SHA-256
`287eb129695b577321c80ad397bb3c2279164d4ca577874d129fd3db5b30afe2`.
기존 001a split/tokenizer/라이선스를 그대로 사용했다. train 9,460쌍, 고정 112주소
Unicode NFD tokenizer, 입력/출력 주소는 분리되어 있다.

[config](../../../configs/dynamath/experiment-002.json)와
[evaluation-plan.json](evaluation-plan.json)에 seed43, batch12, rank12, latent64,
Adam .002, 1,200 updates 또는 300초 상한, TBPTT 24 입력 슬롯=72틱을 고정했다.
각 조건은 1,200 updates를 완료했다. 28,800 input slots/lane, 86,400 absolute ticks,
batch를 합치면 1,036,800 상태 전이다. lane stream 길이 47,496의 약 0.606회만
학습했다. 전체 코퍼스를 수렴시킨 실험이 아니다. lane 길이를 최솟값으로 잘라
긴 lane 끝부분은 제외된다. 전체 test 대신 ID 오름차순 96쌍의 lane-prefix 평가,
첫 12쌍 자유 생성을 사용했다. 응답 CE 분모는 2,250 tokens다. 표본 대표성은 제한된다.

초기 implementation pilot은 계산 후 저장 단계의 NumPy 초기화 오류로 실패했다.
[pilot-failure.log](pilot-failure.log)에 보존했으며 checkpoint/선택 점수는 없다.
의존 해시를 JSON으로 바꾼 뒤 각 조건을 초기 seed에서 한 번씩 본 실행했다.
테스트 점수에 따른 재학습·모델 선택은 하지 않았다.

## 한국어 결과

응답 CE는 첫 readout의 답변 문자와 EOS를 대상으로 한다. 두 번째 readout은
SILENCE를 학습하며 실제 실행에서는 두 슬롯 모두 자유롭게 emit할 수 있다.
같은 002 schedule의 frozen 대조군이 스칼라 쓰기 효과를 분리한다.

| Condition | Coefficients | Response CE ↓ | PPL ↓ | EOS /12 | Train seconds |
|---|---:|---:|---:|---:|---:|
| model9 | 99594 | 2.1520 | 8.602 | 7 | 61.79 |
| model9-frozen | 99594 | 2.1469 | 8.558 | 12 | 45.73 |
| model3-local | 83394 | 2.1992 | 9.018 | 0 | 50.94 |
| model3-cross | 83394 | 2.2106 | 9.122 | 12 | 63.09 |
| model1 | 77994 | 2.1829 | 8.872 | 12 | 65.03 |

Model 9 writable은 frozen보다 CE가 0.00515 높다. Model 3 cross는 local보다
0.01144 높다. 이 단일 seed/예산에서 쓰기나 교차 읽기의 언어 이점은 확인되지 않았다.
Model 3 cross는 EOS가 개선됐지만 구성되지 않은 자모가 늘었다.

001a의 기존 전체-test CE는 five-1tick 1.9702, five-5tick 1.6283,
one-1tick 1.7912다. 002의 더 작은 학습량, 다른 스케줄/상태 크기와 부분 test
때문에 공정한 우열·효율 비교가 아니다. 기존 숫자는 별도 역사 기준이다.

| 조건 | 질문 중 실제 방출 수 /12 질문 | 미완성 자모 합계 | EOS 성공 | 정답 문자열 일치 |
|---|---:|---:|---:|---:|
| Model 9 | 0 | 1 | 7/12 | 0/12 |
| Model 9 frozen | 0 | 0 | 12/12 | 0/12 |
| Model 3 local | 0 | 0 | 0/12 | 0/12 |
| Model 3 cross | 12 | 54 | 12/12 | 0/12 |
| Model 1 | 0 | 0 | 12/12 | 0/12 |

가시 답변 누락은 0/12씩이지만, 이것이 의미 있는 답변을 뜻하지 않는다. 예를 들어
같은 질문 “SNS 시간낭비인 거 아는데 매일 하는 중”에 Model 9는 “징인 강이 있아요.”,
Model 1은 “망인 강이요.”를 생성했다. local은 반복하다 360틱 예산에 도달한다.
[sample-review.json](sample-review.json)에 질문·실제 출력·참조 답변을 모두 남겼다.
질문 관련성과 문장 일관성에 대한 수동 검토에서는 유용한 응답을 확인하지 못했다.
문자열 불일치만으로 의미 실패를 판정한 것이 아니며, 별도의 정량 의미 점수는 없다.

어떤 조건도 외부 출력 마스크나 문장 교정을 적용하지 않았다. NFC 합성만 하며
미완성 자모도 그대로 센다. FIFO peak 45~46은 주로 초기 질문 burst다. 답변 종료의
EOS도 feedback queue에 남긴다. 반복/누락/두 번 방출은 자연스러운 queue 지연을
일으킨다. 무제한 FIFO라 지속적인 최대 이중 방출은 장기적으로 backlog를 늘릴 수
있으며, 강제 이중 방출 테스트로 실제 적재를 확인했다. 이 표본의 peak를 장기
안정성 증거로 해석하지 않는다.

## 학습된 기억 검증

한국어 모델과 별도로 동일 topology의 synthetic bit/query 모델을 seed143으로
600 updates 학습했다. 고정 test seed2026, 128 cases, delay 0/5/50 idle ticks.
query 입력은 다음 합법적 block-start까지 정렬한다(최종 tick 6/9/54).
인코딩 직후 네 개 reset arm을 분기하고 절대 tick은 모두 유지했다.

모든 조건에서 0/5/50틱의 preserve, reset-only-m 정확도는 **100%**였다.
reset-only-S와 reset-both는 Model 9/frozen/3-cross **44.53%**,
Model 3-local/1 **55.47%**였다. 클래스 수는 2개뿐이고 held-out은 비트 순서다.
새로운 이름/한국어 사실 일반화로 해석하면 안 된다.

recall 모델의 scalar들은 목표 bit에 따라 분리되지 않고 초기값을 유지했다.
따라서 성공한 것은 **S를 이용한 작은 이진 기억**이며, 스칼라에 쓰고 보존·재조회하는
학습된 기억은 입증하지 못했다. 한국어 모델에서는 scalar 변경을 관찰했지만 이것이
유용한 정보 저장이라는 증거는 아니다. generation에서 Model 9의 법칙별 실제 변경
횟수는 `[0,0,0,312,256,0,312,312,312]`다. frozen은 모두 0이다.
나머지 조건과 bit별 m 값, loss 이력, reset 결과는 각 모델 JSON에 있다.
다음 연구는 S 제거를 포함한 사전 고정 학습 과제로 scalar 사용의 식별 가능성을
높이는 것이지만, 이번 결과를 보고 추가 튜닝한 실험은 수행하지 않았다.

## 검증과 재현

- 전체 테스트: **34 passed**. 9법칙 순환/공유상태, FIFO/덧셈, 양쪽 readout,
  silence/threshold/tie/EOS, 강제 double event, owner-only write, cross-read mask,
  정확한 retain, scalar 영향, 상태 복원, optimizer/RNG/data identity 재시작을 검증했다.
- 기존 001a 자산 24개 byte-identical audit 통과. 기존 샘플 64개 재생 동일,
  상태 checkpoint resume 동일, dataset 접근 차단. legacy 코드는 수정하지 않았다.
- 새 체크포인트는 데이터 로딩 함수를 차단하고 빈 cwd에서 12개씩 재생하며 원본
  token IDs/질문 방출을 비교한다. runtime snapshot도 연속 실행과 동일해야 한다.
  [saved-checkpoint-verification.json](saved-checkpoint-verification.json) 참조.
- 학습 restart는 작은 fixture에서 uninterrupted와 비트 단위로 일치한다. primary의
  누락된 update/Python RNG 메타데이터는 tick/72와 소비되지 않은 seed43 상태로
  복구한다. 실제 수치 state/optimizer/torch RNG/data hash는 저장되어 있다.
- primary의 부동소수점 식 순서를 보존한 `scripts/reproduce_dynamath_002.py` 제공.
  최종 runtime은 hard replacement를 정확한 `(1-gate)*old+gate*value`로 정리했다.
  원본 결과를 덮어 재학습하지 않았으며 저장 출력 재생으로 호환성을 검증한다.

## 실행 비용·산출물·한계

CPU only, AMD EPYC 9V74, 노출 3 vCPU, PyTorch CPU, torch threads2.
조건별 한국어 학습 wall time은 표에 기재했다. 별도 recall 학습은 약 4.9~7.4초다.
프로세스 peak RSS는 각 raw 결과에 있으며, 공유 프로세스의 누적 최고값이므로
조건별 독립 메모리 비교가 아니다. 각 state는 288 float32=1,152 bytes이며
scalar는 Model 9/3/1 각각 36/12/4 bytes다. 고정 계수·optimizer·PyTorch 자체
메모리는 이 숫자에 포함되지 않는다. 실제 inference CPU 시간과 tokens/ticks per
CPU second는 저장 모델 검증 JSON에서 재현 가능하게 별도 측정했다.
001a와 다른 스레드/실행 조건이므로 효율 우위를 주장하지 않는다.

체크포인트 10개(언어 5 + recall 5), seed/config, source pin, raw loss histories,
샘플·snapshot·120틱 진단 trace·검증 로그·SHA-256 inventory를 저장했다.
단일 seed, 제한된 노출량과 test subset, 72틱 TBPTT, 정형 binary 기억,
낮은 생성 품질이 핵심 한계다. 9개 scalar 모두가 유용하게 학습됐다는 증거는 없다.
master 병합이나 기존 실험 교체는 하지 않았다.
