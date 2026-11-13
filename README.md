# DynaMath

하나의 지속되는 정보 상태에 다섯 개의 학습된 수학 법칙을 순환 적용하는 실험 프로젝트입니다. 매 틱 입력을 더하고, 법칙 하나를 적용한 직후 출력 조건을 검사합니다.

이 브랜치는 **DynaMath 시리즈의 메인 페이지이자 후속 개발 기준점**입니다. 현재 버전은 **DynaMath 001a**입니다.

> 한국어 데이터 출처: **Song Ys (`songys`)**의 [원본 Chatbot_data](https://github.com/songys/Chatbot_data), MIT License. 데이터 제공자와 DynaMath 모델 개발 주체는 별개입니다.

## 현재 버전

| 버전 | 실험 | 상태 | 기록 |
|---|---|---|---|
| **001a** | 매 1틱 출력 검사 · 한국어 Q/A 학습 · 지속 상태 | 구현·제한된 학습 완료, 대화 품질 미달 | [보고서](docs/dynamath/experiment-001a/report.md) · [실행 안내](docs/dynamath/experiment-001a/README.md) · [모델 카드](models/dynamath/001a/model-card.json) |

- **구조:** 하나의 공유 상태, 다섯 법칙의 순환, 가산 입력·피드백, 출력 후 상태 유지.
- **출력:** 매 틱 검사하며 조건을 충족할 때만 출력합니다. 침묵도 고정 주소에서 학습합니다.
- **학습:** 원본 11,823쌍을 분할해 학습 9,460쌍을 사용했습니다. 주 모델과 비교군 3개의 체크포인트를 포함합니다.
- **한계:** 한국어 생성은 재현되지만 의미 평가 0/6, 유효한 다중 턴 기억은 확인되지 않았습니다. 5법칙의 우위도 입증되지 않았습니다.

## 바로 실행

Python 3.11 이상, CPU PyTorch를 사용합니다. 저장소 루트에서 실행하세요.

```bash
python -m venv .venv
. .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e '.[dev]'
python -m dynamath generate --question '안녕하세요'
python -m dynamath chat --state-out /tmp/dynamath-session.pt
python -m dynamath tick --state-in /tmp/dynamath-session.pt --count 1
```

`dynamath` 명령도 `python -m dynamath`와 같습니다. 저장 모델의 생성에는 데이터셋 다운로드가 필요하지 않습니다. `chat`은 턴 사이의 상태와 법칙 위상을 유지합니다. `tick`은 실제 법칙을 실행하고 출력을 검사합니다.

데이터 취득과 학습 재현 방법, 계산 예산은 [001a 실행 안내](docs/dynamath/experiment-001a/README.md)를 참고하세요. 실패·반복·무출력 예문도 [원시 결과](docs/dynamath/experiment-001a/results.json)와 버전별 JSON에 남겨 두었습니다.

## 개발과 검증

```bash
python -m pytest -q
python scripts/audit_dynamath_001a.py
python scripts/verify_dynamath_001a.py --output /tmp/dynamath-replay.json
python scripts/filetree_index.py lint
```

- [현재 작업 목표](GOAL.md) · [개발 지침](AGENTS.md) · [기여 안내](CONTRIBUTING.md)
- [공통 구조와 수식](docs/dynamath/architecture.md) · [파일 인덱스](FileTree.md)
- [001a 고정 목표·예산](docs/dynamath/experiment-001a/GOAL.md) · [평가 계획](docs/dynamath/experiment-001a/evaluation-plan.json)
- [브랜치 정리 기록](docs/dynamath/branch-cleanup.md)

후속 버전은 별도 목표·예산·평가 계획과 버전별 설정/모델/실험 기록을 추가합니다. 001a의 동결된 가중치와 평가 자료는 기준선으로 유지합니다. 초기 DynaMath 001 자료는 [기존 시리즈 브랜치](https://github.com/zizonhyeontae218/tele-neuron-engine/tree/feature/dynamath-1/docs/dynamath)에서 확인할 수 있습니다.

## 파일 구성

```text
src/dynamath/                  법칙 · 상태/출력 · 한국어 학습/CLI · 데이터 도구
tests/                         DynaMath 불변조건 · 학습 재개 · CLI 검사
configs/dynamath/               버전별 실험 설정
data/dynamath/001a/             출처 · 라이선스 · 고정 분할 · 토크나이저
models/dynamath/001a/           체크포인트 · 모델 카드 · 무결성 해시
docs/dynamath/                  공통 구조 · 버전별 실험 기록
scripts/                       재현 · 무결성 검사 · 파일 인덱스
.github/workflows/tests.yml    DynaMath 전용 CI
```
