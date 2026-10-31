from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tele_neuron.endophalon.config import EndophalonConfig
from tele_neuron.endophalon.observer import VoxelObserver
from tele_neuron.endophalon.readout import RidgeReadout
from tele_neuron.endophalon.reservoir import BallReservoir
from tele_neuron.endophalon.tasks import output_names


@dataclass(frozen=True, slots=True)
class Dataset:
    features: np.ndarray
    targets: np.ndarray
    bits: np.ndarray
    realizations: np.ndarray


@dataclass(frozen=True, slots=True)
class SplitMetrics:
    samples: int
    accuracy: float
    output_accuracy: dict[str, float]
    case_accuracy: dict[str, float]


@dataclass(frozen=True, slots=True)
class TrainingReport:
    readout: RidgeReadout
    feature_count: int
    train: SplitMetrics
    test: SplitMetrics | None
    baseline_train: SplitMetrics
    baseline_test: SplitMetrics | None


def build_pipeline(config: EndophalonConfig) -> tuple[BallReservoir, VoxelObserver]:
    reservoir = BallReservoir(
        config.reservoir,
        config.observer.taps,
        position_jitter=config.noise.position_jitter,
        velocity_jitter=config.noise.velocity_jitter,
    )
    observer = VoxelObserver(config.reservoir.space_size, config.observer.grid)
    return reservoir, observer


def collect(
    config: EndophalonConfig,
    realizations: range,
    pipeline: tuple[BallReservoir, VoxelObserver] | None = None,
) -> Dataset:
    reservoir, observer = pipeline or build_pipeline(config)
    features, targets, bits, owners = [], [], [], []
    for realization in realizations:
        for case_bits, target in config.cases:
            features.append(observer.features(reservoir.run(case_bits, realization)))
            targets.append(target)
            bits.append(case_bits)
            owners.append(realization)
    return Dataset(
        features=np.array(features, dtype=np.float64),
        targets=np.array(targets, dtype=np.int64),
        bits=np.array(bits, dtype=np.int64),
        realizations=np.array(owners, dtype=np.int64),
    )


def train(config: EndophalonConfig) -> TrainingReport:
    pipeline = build_pipeline(config)
    train_count = config.noise.train_realizations
    test_count = config.noise.test_realizations
    train_set = collect(config, range(train_count), pipeline)
    test_set = collect(config, range(train_count, train_count + test_count), pipeline) if test_count else None

    readout = RidgeReadout(alpha=config.readout.ridge_alpha).fit(train_set.features, train_set.targets)
    # Baseline: the same readout fed the raw input bits. A linear map of the bits
    # cannot express XOR/parity, so any gap is nonlinearity the physics supplied.
    baseline = RidgeReadout(alpha=config.readout.ridge_alpha).fit(train_set.bits, train_set.targets)

    return TrainingReport(
        readout=readout,
        feature_count=train_set.features.shape[1],
        train=evaluate(config, readout.predict(train_set.features), train_set),
        test=evaluate(config, readout.predict(test_set.features), test_set) if test_set else None,
        baseline_train=evaluate(config, baseline.predict(train_set.bits), train_set),
        baseline_test=evaluate(config, baseline.predict(test_set.bits), test_set) if test_set else None,
    )


def evaluate(config: EndophalonConfig, predictions: np.ndarray, dataset: Dataset) -> SplitMetrics:
    hits = predictions == dataset.targets
    whole = np.all(hits, axis=1)
    names = output_names(config.task, config.output_width)
    case_accuracy = {}
    for case_bits, _ in config.cases:
        rows = np.all(dataset.bits == np.array(case_bits), axis=1)
        case_accuracy["".join(map(str, case_bits))] = float(whole[rows].mean())
    return SplitMetrics(
        samples=len(whole),
        accuracy=float(whole.mean()),
        output_accuracy={name: float(hits[:, index].mean()) for index, name in enumerate(names)},
        case_accuracy=case_accuracy,
    )
