from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tele_neuron.endophalon.tasks import Case
from tele_neuron.endophalon_plus.readout import WideRidgeReadout as RidgeReadout
from tele_neuron.endophalon_plus.tasks import output_names
from tele_neuron.endophalon_plus.config import PlusConfig
from tele_neuron.endophalon_plus.gate import LinearGate
from tele_neuron.plus.observe import MultiScaleObserver
from tele_neuron.plus.runner import run_features
from tele_neuron.plus.world import Job


@dataclass(frozen=True, slots=True)
class SplitMetrics:
    samples: int
    accuracy: float
    output_accuracy: dict[str, float]
    case_accuracy: dict[str, float]


@dataclass(frozen=True, slots=True)
class PlusReport:
    gate: LinearGate
    assignment: tuple[int, ...]
    readout: RidgeReadout
    feature_count: int
    gate_validation_accuracy: float
    gate_evaluations: int
    train: SplitMetrics
    test: SplitMetrics | None
    controls: dict[str, dict[str, SplitMetrics | None]] = field(default_factory=dict)


class FeatureCache:
    """Observed features for every (preset, realization, case), simulated once."""

    def __init__(self, config: PlusConfig) -> None:
        self.config = config
        realizations = config.train_realizations + config.test_realizations
        cases = len(config.cases)
        presets = range(len(config.presets)) if config.gate.mode == "linear" else (config.gate.fixed_preset,)
        jobs = [
            Job(preset=preset, bits=bits, realization=realization)
            for preset in presets
            for realization in range(realizations)
            for bits, _ in config.cases
        ]
        features = run_features(
            config.world,
            config.presets,
            jobs,
            config.observer.taps,
            observer(config),
            workers=config.workers,
            chunk_size=config.chunk_size,
        )
        self.width = features.shape[1]
        self.features = np.zeros((len(config.presets), realizations, cases, self.width))
        self.features[list(presets)] = features.reshape(len(presets), realizations, cases, self.width)
        self.targets = np.array([target for _, target in config.cases], dtype=np.int64)
        self.bits = np.array([bits for bits, _ in config.cases], dtype=np.int64)

    def design(self, assignment: tuple[int, ...], realizations: range) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Block features: each sample fills only the block of its routed preset.

        One ridge fit over block features equals one independent readout per
        preset, solved together.
        """
        presets, cases = len(self.config.presets), len(self.config.cases)
        x = np.zeros((len(realizations), cases, presets * self.width))
        for case, preset in enumerate(assignment):
            x[:, case, preset * self.width : (preset + 1) * self.width] = self.features[preset, realizations.start : realizations.stop, case]
        rows = len(realizations) * cases
        return (
            x.reshape(rows, -1),
            np.tile(self.targets, (len(realizations), 1)),
            np.tile(self.bits, (len(realizations), 1)),
        )


def observer(config: PlusConfig) -> MultiScaleObserver:
    return MultiScaleObserver(config.world.space_size, config.observer.grids, speed=config.observer.speed)


def train(config: PlusConfig, cache: FeatureCache | None = None) -> PlusReport:
    cache = cache or FeatureCache(config)
    train_range = range(0, config.train_realizations)
    test_range = range(config.train_realizations, config.train_realizations + config.test_realizations)

    if config.gate.mode == "linear":
        gate, validation, evaluations = search_gate(config, cache)
    else:
        gate = LinearGate.constant(config.gate.fixed_preset, len(config.presets), config.input_width)
        validation, evaluations = float("nan"), 0
    assignment = gate.assignment(config.cases)
    readout, train_metrics, test_metrics = fit_assignment(config, cache, assignment, train_range, test_range)

    controls: dict[str, dict[str, SplitMetrics | None]] = {}
    controls["baseline_linear_on_bits"] = _control_fit(config, cache, assignment, train_range, test_range, "bits")
    controls["gate_only"] = _control_fit(config, cache, assignment, train_range, test_range, "gate")
    if config.gate.mode == "linear":
        for index, preset in enumerate(config.presets):
            fixed = (index,) * len(config.cases)
            _, fixed_train, fixed_test = fit_assignment(config, cache, fixed, train_range, test_range)
            controls[f"fixed_{preset.name}"] = {"train": fixed_train, "test": fixed_test}

    return PlusReport(
        gate=gate,
        assignment=assignment,
        readout=readout,
        feature_count=cache.width,
        gate_validation_accuracy=validation,
        gate_evaluations=evaluations,
        train=train_metrics,
        test=test_metrics,
        controls=controls,
    )


def search_gate(config: PlusConfig, cache: FeatureCache) -> tuple[LinearGate, float, int]:
    """Random restarts + hill climbing over gate weights, scored on held-out
    training realizations. Only the induced case->preset assignment matters, so
    scores are memoized by assignment and the physics is never re-simulated.
    """
    spec = config.gate
    fit_range = range(0, config.train_realizations - spec.validation_realizations)
    validation_range = range(fit_range.stop, config.train_realizations)
    presets, inputs = len(config.presets), config.input_width
    rng = np.random.default_rng(spec.seed)
    scores: dict[tuple[int, ...], tuple[float, float, float, float]] = {}

    def score(gate: LinearGate) -> tuple[float, float, float, float]:
        # Ranked by whole-case accuracy, then per-output accuracy. Ties go to
        # the gate that leaks less of the answer on its own, then to lower
        # squared error. Without the leakage term, a gate that encodes the
        # answer always wins ties, because it makes the readout's job easier.
        assignment = gate.assignment(config.cases)
        if assignment not in scores:
            x, y, _ = cache.design(assignment, fit_range)
            readout = RidgeReadout(alpha=config.ridge_alpha).fit(x, y)
            vx, vy, _ = cache.design(assignment, validation_range)
            raw = readout.scores(vx)
            hits = (raw >= 0.5) == vy
            scores[assignment] = (
                float(np.all(hits, axis=1).mean()),
                float(hits.mean()),
                -gate_leakage(config, assignment),
                -float(np.mean((raw - vy) ** 2)),
            )
        return scores[assignment]

    # Every constant gate is a candidate, so routing can only win on validation
    # if it beats the best single preset.
    best = LinearGate.constant(0, presets, inputs)
    best_score = score(best)
    for preset in range(1, presets):
        candidate = LinearGate.constant(preset, presets, inputs)
        if score(candidate) > best_score:
            best, best_score = candidate, score(candidate)

    for _ in range(spec.restarts):
        current = LinearGate(rng.normal(size=(presets, inputs + 1)))
        current_score = score(current)
        for _ in range(spec.iterations):
            candidate = LinearGate(current.weights + rng.normal(0.0, 0.5, current.weights.shape))
            candidate_score = score(candidate)
            if candidate_score >= current_score:
                current, current_score = candidate, candidate_score
        if current_score > best_score:
            best, best_score = current, current_score
    return best, best_score[0], len(scores)


def gate_leakage(config: PlusConfig, assignment: tuple[int, ...]) -> float:
    """Per-output accuracy of predicting the targets from the chosen preset alone."""
    targets = np.array([target for _, target in config.cases], dtype=np.int64)
    one_hot = np.eye(len(config.presets))[list(assignment)]
    readout = RidgeReadout(alpha=config.ridge_alpha).fit(one_hot, targets)
    return float((readout.predict(one_hot) == targets).mean())


def fit_assignment(
    config: PlusConfig,
    cache: FeatureCache,
    assignment: tuple[int, ...],
    fit_range: range,
    eval_range: range,
) -> tuple[RidgeReadout, SplitMetrics, SplitMetrics | None]:
    x, y, bits = cache.design(assignment, fit_range)
    readout = RidgeReadout(alpha=config.ridge_alpha).fit(x, y)
    fit_metrics = split_metrics(config.task, config.cases, readout.predict(x), y, bits)
    if len(eval_range) == 0:
        return readout, fit_metrics, None
    ex, ey, ebits = cache.design(assignment, eval_range)
    return readout, fit_metrics, split_metrics(config.task, config.cases, readout.predict(ex), ey, ebits)


def _control_fit(
    config: PlusConfig,
    cache: FeatureCache,
    assignment: tuple[int, ...],
    train_range: range,
    test_range: range,
    kind: str,
) -> dict[str, SplitMetrics | None]:
    """Readouts that see no physics: raw bits, or only which preset was chosen."""
    one_hot = np.eye(len(config.presets))[list(assignment)]

    def inputs(realizations: range) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        reps = len(realizations)
        bits = np.tile(cache.bits, (reps, 1))
        features = bits.astype(np.float64) if kind == "bits" else np.tile(one_hot, (reps, 1))
        return features, np.tile(cache.targets, (reps, 1)), bits

    x, y, bits = inputs(train_range)
    readout = RidgeReadout(alpha=config.ridge_alpha).fit(x, y)
    result: dict[str, SplitMetrics | None] = {
        "train": split_metrics(config.task, config.cases, readout.predict(x), y, bits),
        "test": None,
    }
    if len(test_range):
        tx, ty, tbits = inputs(test_range)
        result["test"] = split_metrics(config.task, config.cases, readout.predict(tx), ty, tbits)
    return result


def split_metrics(
    task: str,
    cases: tuple[Case, ...],
    predictions: np.ndarray,
    targets: np.ndarray,
    bits: np.ndarray,
) -> SplitMetrics:
    hits = predictions == targets
    whole = np.all(hits, axis=1)
    names = output_names(task, targets.shape[1])
    case_accuracy = {}
    for case_bits, _ in cases:
        rows = np.all(bits == np.array(case_bits), axis=1)
        case_accuracy["".join(map(str, case_bits))] = float(whole[rows].mean())
    return SplitMetrics(
        samples=len(whole),
        accuracy=float(whole.mean()),
        output_accuracy={name: float(hits[:, index].mean()) for index, name in enumerate(names)},
        case_accuracy=case_accuracy,
    )
