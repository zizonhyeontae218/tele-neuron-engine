# Tele-Neuron Engine

물리 리저버 컴퓨팅을 실험하기 위한 Phase 1 단일 PC 프로토타입입니다.

입력 비트에 따라 3차원 공간의 가상 입자에 힘을 가합니다. 입자들은 K개의 시간 단계 동안 이동하고 충돌하며, 마지막 분포는 판독 영역(readout zone)을 거쳐 출력 비트로 변환됩니다.

이 프로젝트는 물리 리저버 컴퓨팅의 가능성을 살펴보기 위한 프로토타입이며, 실용적인 LLM 대체재를 목표로 하지 않습니다.

## 프로젝트 개요

- 결정론적으로 동작하는 Python + NumPy 기반 시뮬레이션
- AND, XOR 같은 작은 이진 분류 과제를 위한 실험 환경
- 공간 해싱, 충돌 처리, 출력 판독, 시각화, 시드 탐색을 확장해 나갈 기반

## 이 프로젝트가 아닌 것

- 여러 휴대폰 노드에서 분산 실행하는 시스템은 아직 아닙니다.
- GPU 전용 코드가 아닙니다.
- 물리 충돌 시뮬레이션이 최신 LLM을 대체한다는 주장을 담고 있지 않습니다.

## 설치

```bash
python -m pip install -e .
python -m pip install -r requirements.txt
```

## 현재 구현 상태

Phase 1은 단일 PC에서 동작하는 프로토타입으로 구현되어 있습니다. 결정론적 입자 초기화, 입력 힘 주입, K단계 물리 시뮬레이션, 반사 경계, 공간 해싱을 이용한 충돌 후보 탐색, 간단한 충돌 처리, 입자 수 기반 판독, 시드 탐색, 점수 기록, Matplotlib 시각화를 포함합니다.

## 실행

```bash
pytest
python -m tele_neuron.experiment --config configs/and_tiny.json
python -m tele_neuron.experiment --config configs/or_tiny.json
python -m tele_neuron.experiment --config configs/xor_tiny.json
```

## 시뮬레이션 시각화

x/y 값은 위치로, z 값은 색으로 표현하는 Matplotlib 실시간 화면을 엽니다.

```bash
python -m tele_neuron.experiment --config configs/and_tiny.json --display --bits 11
```

같은 장면을 GIF로 저장할 수 있습니다.

```bash
python -m tele_neuron.experiment --config configs/xor_tiny.json --display --bits 01 --save outputs/xor_01.gif
```

K단계가 끝난 뒤의 최종 위치를 PNG로 저장할 수도 있습니다.

```bash
python -m tele_neuron.experiment --config configs/xor_tiny.json --bits 01 --snapshot outputs/xor_01_final.png
```

## Config Lab

좌표, 이미터(emitter), 판독 영역, 테스트 케이스를 편집하고 스냅샷을 미리 볼 수 있는 가벼운 브라우저 기반 편집기입니다.

```bash
python -m tele_neuron.config_lab
```

편집한 설정은 `configs/lab/`에, 미리보기 PNG는 `outputs/lab/`에 저장됩니다. 원본 설정 파일은 덮어쓰지 않습니다.

## Learner Lab

설정 파일을 바탕으로 모델을 만들고 이어 학습할 수 있는 브라우저 기반 트레이너입니다.

```bash
python -m tele_neuron.learner_lab
```

학습 알고리즘은 `Vanila_learner/Algorithm/*.py`에서 불러옵니다. 각 입자에 무작위 질량을 부여한 새 모델을 만들거나, 기존 모델 JSON을 불러와 학습을 이어갈 수 있습니다. 학습된 모델은 `models/lab/`에 저장됩니다.

## 첫 번째 모델

`TeleNeuron-001`은 시드 탐색으로 찾은, 재현 가능한 최초의 소형 XOR 모델입니다.

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

- 물리 모델은 의도적으로 단순화되어 있으며, 작은 이진 과제에 맞춰 조정되어 있습니다.
- 충돌 처리는 완전한 강체 물리 엔진이 아니라 안정성을 우선한 근사 방식입니다.
- Phase 1에서는 시드 탐색과 임계값 선택으로 충분하지만, 일반적인 학습 알고리즘은 아닙니다.
- 분산 휴대폰 노드, 네트워킹, GPU 전용 실행 경로, LLM 대체는 현재 범위에 포함되지 않습니다.

## 라이선스

MIT. 자세한 내용은 [LICENSE](LICENSE)를 참고하세요.

## 기여

기여를 환영합니다. [CONTRIBUTING.md](CONTRIBUTING.md)를 읽고, Phase 1이 물리 리저버 컴퓨팅 프로토타입이라는 범위가 유지되도록 해 주세요. 이 프로젝트는 실용적인 LLM 대체재가 아닙니다.
