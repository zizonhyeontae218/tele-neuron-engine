# Tele-Neuron Engine

물리 리저버 컴퓨팅 실험을 위한 Phase 1 단일 PC 프로토타입입니다. 입력 비트는 3차원 공간의 가상 공에 힘을 주입하고, 공들은 K 단계 동안 이동·충돌하며, 리드아웃 영역은 마지막 분포를 출력 비트로 변환합니다.

이것은 실용적인 LLM 대체재가 아니라 물리 리저버 컴퓨팅 프로토타입입니다.

## 이것이 하는 일

- 결정론적인 Python + NumPy 시뮬레이션 골격
- AND/XOR 같은 작은 이진 분류 실험 대상
- 공간 해싱, 충돌 반응, 리드아웃, 시각화, 시드 탐색을 위한 기반

## 이것이 아닌 것

- 아직 분산형 휴대폰 노드 시스템이 아닙니다.
- GPU 전용 코드가 아닙니다.
- 물리 충돌 시뮬레이션이 최신 LLM을 대체한다는 주장이 아닙니다.

## 설정

```bash
python -m pip install -e .
python -m pip install -r requirements.txt
```

## 현재 상태

Phase 1은 단일 PC 프로토타입으로 구현되어 있습니다. 결정론적 공 초기화, 입력 힘 주입, K 단계 물리, 반사 경계, 공간 해싱 기반 충돌 후보, 단순 충돌 반응, 개수 기반 리드아웃, 시드 탐색, 점수 기록, Matplotlib 시각화를 포함합니다.

## 실행

```bash
pytest
python -m tele_neuron.experiment --config configs/and_tiny.json
python -m tele_neuron.experiment --config configs/or_tiny.json
python -m tele_neuron.experiment --config configs/xor_tiny.json
```

## 시뮬레이션 보기

x/y를 위치로, z를 색으로 표시하는 Matplotlib 실시간 화면을 엽니다.

```bash
python -m tele_neuron.experiment --config configs/and_tiny.json --display --bits 11
```

같은 화면을 GIF로 저장합니다.

```bash
python -m tele_neuron.experiment --config configs/xor_tiny.json --display --bits 01 --save outputs/xor_01.gif
```

마지막 K단계의 위치를 PNG로 저장합니다.

```bash
python -m tele_neuron.experiment --config configs/xor_tiny.json --bits 01 --snapshot outputs/xor_01_final.png
```

## Config Lab

좌표, 이미터, 리드아웃, 케이스를 편집하고 스냅샷을 미리 볼 수 있는 가벼운 브라우저 편집기를 엽니다.

```bash
python -m tele_neuron.config_lab
```

Lab은 편집한 설정을 `configs/lab/`에, 미리보기 PNG를 `outputs/lab/`에 저장합니다. 원본 설정은 덮어쓰지 않습니다.

## Learner Lab

설정 기반 모델 생성과 이어 학습을 위한 브라우저 트레이너를 엽니다.

```bash
python -m tele_neuron.learner_lab
```

알고리즘은 `Vanila_learner/Algorithm/*.py`에서 불러옵니다. Lab은 공별 질량을 무작위로 둔 새 모델을 만들거나 기존 모델 JSON에서 계속 학습할 수 있습니다. 학습된 모델은 `models/lab/`에 저장됩니다.

## 첫 번째 모델

`TeleNeuron-001`은 시드 탐색으로 찾은 최초의 재현 가능한 작은 XOR 모델입니다.

```bash
python -m tele_neuron.experiment --config configs/xor_first_model.json
python -m tele_neuron.experiment --config configs/xor_first_model.json --display --bits 01
```

모델 카드:

```txt
models/tele_neuron_001.json
```

출력 예시:

```txt
tele_neuron_001_xor seed=1 threshold=3 score=4/4 accuracy=100.00%
bits=00 target=0 output=0 count=2 collisions=8
bits=01 target=1 output=1 count=7 collisions=33
bits=10 target=1 output=1 count=8 collisions=34
bits=11 target=0 output=0 count=2 collisions=28
```

## 알려진 한계

- 물리 모델은 의도적으로 단순하며, 작은 이진 과제에만 맞춰져 있습니다.
- 충돌 반응은 완전한 강체 엔진이 아니라 안정적인 근사입니다.
- Phase 1에서는 시드 탐색과 임계값 선택으로 충분하지만, 일반 학습 알고리즘은 아닙니다.
- 분산 휴대폰 노드, 네트워킹, GPU 전용 경로, LLM 대체 주장은 범위 밖입니다.

## 라이선스

MIT. [LICENSE](LICENSE)를 참고하세요.

## 기여

기여를 환영합니다. [CONTRIBUTING.md](CONTRIBUTING.md)를 읽고 Phase 1 범위를 분명히 지켜 주세요. 이 프로젝트는 실용적인 LLM 대체재가 아닌 물리 리저버 컴퓨팅 프로토타입입니다.
