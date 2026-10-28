"""Data collection, objective and bilevel training for Endophalon-2-Pro.

Reuses Endophalon-2's per-stream layout (``StreamData``) and its exact ridge
inner loop (``dataset`` / ``fit_score``), so the two versions are scored the
same way.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable

import numpy as np

from tele_neuron.endophalon2.evaluate import StreamData, dataset, fit_score
from tele_neuron.endophalon2.tasks import stream_bits
from tele_neuron.endophalon2_pro.config import E2PConfig
from tele_neuron.endophalon2_pro.engine import EventRow, run_events
from tele_neuron.endophalon2_pro.theta import SpringTheta


FAILED_J = 10.0  # score for a theta whose integration blew up


@dataclass(frozen=True, slots=True)
class Collected:
    data: StreamData  # ``wall`` holds mean relative spring stretch here
    finite: bool


def collect(config: E2PConfig, thetas: list[SpringTheta], streams: range, *, reset: bool = False) -> list[Collected]:
    """Run every theta on every stream. ``reset=True`` runs each event alone from a fresh start."""
    spec, pulses, seed = config.stream, config.stream.pulses, config.physics.seed
    bits = {s: stream_bits(seed, s, pulses) for s in streams}
    rows: list[EventRow] = []
    for index in range(len(thetas)):
        for s in streams:
            if reset:
                rows += [
                    EventRow(index, bits[s][k : k + 1], np.array([spec.warmup]), 1_000_000 + s * pulses + k, 1_000_000 + s * pulses + k)
                    for k in range(pulses)
                ]
            else:
                rows.append(EventRow(index, bits[s], spec.onsets(seed, s), s, s))
    run_spec = replace(spec, pulses=1) if reset else spec
    record = run_events(config.network, config.physics, thetas, rows, run_spec, workers=config.workers, chunk_size=config.chunk_size)
    per_theta = len(streams) * (pulses if reset else 1)
    out = []
    for index in range(len(thetas)):
        part = slice(index * per_theta, (index + 1) * per_theta)
        out.append(
            Collected(
                StreamData(
                    features=record.zones[part].reshape(len(streams), pulses, spec.interval, -1),
                    bits=np.stack([bits[s] for s in streams]),
                    energy=record.energy[part].reshape(len(streams), pulses),
                    wall=record.stretch[part].reshape(len(streams), pulses),
                    streams=tuple(streams),
                ),
                bool(record.finite[part].all()),
            )
        )
    return out


def drift(config: E2PConfig, data: StreamData, streams: range) -> float:
    """Per-stream drift: median over streams of |total change in log pre-event energy|.

    Each stream's trend is a least-squares slope over its scored events, so a
    stream that heats up and one that cools down no longer cancel (the flaw of
    Endophalon-2's pooled ratio).
    """
    rows = data.index(streams)
    k = np.arange(config.washout, config.stream.pulses, dtype=np.float64)
    log_energy = np.log(data.energy[rows, config.washout :] + 1e-9)
    centered = k - k.mean()
    slopes = (log_energy - log_energy.mean(axis=1, keepdims=True)) @ centered / (centered @ centered)
    return float(np.median(np.abs(slopes)) * len(k))


@dataclass(frozen=True, slots=True)
class Fitness:
    j: float
    l_val: float
    r_noise: float
    r_drift: float
    r_complexity: float
    val_accuracy: dict[str, float]
    finite: bool


def fitness(config: E2PConfig, item: Collected, train: range, val: range, tasks: tuple[str, ...]) -> Fitness:
    if not item.finite:
        return Fitness(FAILED_J, FAILED_J, 0.0, 0.0, 0.0, {task: 0.0 for task in tasks}, False)
    data = item.data
    losses, noises, weights, accuracies = [], [], [], {}
    for task in tasks:
        accuracy, mse, readout = fit_score(config, data, task, train, val, config.readout_delay)
        x, y = dataset(config, data, task, val, config.readout_delay)
        scores = readout.scores(x)[:, 0]
        ones, zeros = scores[y[:, 0] == 1], scores[y[:, 0] == 0]
        if len(ones) and len(zeros):
            noises.append(float(0.5 * (ones.std() + zeros.std()) / (abs(ones.mean() - zeros.mean()) + 1e-6)))
        assert readout.weights is not None
        weights.append(float(np.mean(readout.weights[:-1] ** 2)))
        losses.append(mse)
        accuracies[task] = accuracy
    spec = config.objective
    l_val, r_noise = float(np.mean(losses)), float(np.mean(noises)) if noises else 0.0
    r_drift, r_complexity = drift(config, data, range(train.start, val.stop)), float(np.mean(weights))
    j = (
        l_val
        + spec.get("lambda_noise", 0.0) * r_noise
        + spec.get("lambda_drift", 0.0) * r_drift
        + spec.get("lambda_complexity", 0.0) * r_complexity
    )
    return Fitness(j, l_val, r_noise, r_drift, r_complexity, accuracies, True)


@dataclass(frozen=True, slots=True)
class Report:
    theta: SpringTheta
    fitness: Fitness
    fitness_init: Fitness
    evaluations: int
    history: tuple[dict[str, Any], ...]
    test: dict[str, Any]
    random_search: dict[str, Any] | None = None


def train(
    config: E2PConfig,
    *,
    tasks: tuple[str, ...] | None = None,
    random_baseline: bool = False,
    log: Callable[[str], None] | None = None,
) -> Report:
    """Outer cross-entropy search over theta; inner exact ridge readout per task."""
    tasks = tasks or config.tasks
    train_range, val_range, _ = config.stream_ranges()
    outer = config.outer
    generations, population = int(outer.get("generations", 30)), int(outer.get("population", 16))
    elite, min_sigma = int(outer.get("elite", 4)), float(outer.get("min_sigma", 0.05))
    space = config.space
    rng = np.random.default_rng(int(outer.get("seed", 1)))
    mean = space.encode(config.theta_init)
    sigma = np.full(space.dim, float(outer.get("init_sigma", 0.8)))
    weights = np.log(elite + 0.5) - np.log(np.arange(1, elite + 1))
    weights /= weights.sum()
    best_u, best_fit, init_fit = mean.copy(), None, None
    history: list[dict[str, Any]] = []
    evaluations = 0

    for generation in range(generations + 1):
        candidates = mean[None, :] if generation == 0 else np.vstack(
            [mean, mean + rng.standard_normal((population - 1, space.dim)) * sigma]
        )
        items = collect(config, [space.decode(u) for u in candidates], range(0, val_range.stop))
        scored = [fitness(config, item, train_range, val_range, tasks) for item in items]
        evaluations += len(candidates)
        js = np.array([fit.j for fit in scored])
        if generation == 0:
            init_fit = scored[0]
        top = int(np.argmin(js))
        if best_fit is None or scored[top].j < best_fit.j:
            best_u, best_fit = candidates[top].copy(), scored[top]
        failed = int(sum(not fit.finite for fit in scored))
        history.append(
            {"generation": generation, "best_j": best_fit.j, "best_val_accuracy": best_fit.val_accuracy,
             "center_j": scored[0].j, "sigma": float(sigma.mean()), "failed": failed}
        )
        if log:
            accs = " ".join(f"{task}={value:.2f}" for task, value in best_fit.val_accuracy.items())
            log(f"  gen {generation:>3} best J={best_fit.j:.4f} [{accs}] center J={scored[0].j:.4f} sigma={sigma.mean():.3f} failed={failed}")
        if generation == 0:
            continue
        chosen = candidates[np.argsort(js)[:elite]]
        spread = np.sqrt(weights @ (chosen - mean) ** 2)
        mean = weights @ chosen
        sigma = np.maximum(0.7 * spread + 0.3 * sigma, min_sigma)

    assert best_fit is not None and init_fit is not None
    theta = space.decode(best_u)
    random = random_search(config, tasks, budget=evaluations, seed=int(outer.get("seed", 1)) + 1000) if random_baseline else None
    return Report(theta, best_fit, init_fit, evaluations, tuple(history), test_evaluation(config, theta, config.theta_init), random)


def test_evaluation(config: E2PConfig, theta: SpringTheta, theta_init: SpringTheta) -> dict[str, Any]:
    """Held-out test streams, continuous and reset, trained and initial theta, plus a bit-history baseline."""
    train_range, _, test_range = config.stream_ranges()
    streams = range(0, test_range.stop)
    half = config.washout + (config.stream.pulses - config.washout) // 2
    out: dict[str, Any] = {}
    for label, reset in (("continuous", False), ("reset", True)):
        trained, initial = collect(config, [theta, theta_init], streams, reset=reset)
        for name, item in (("trained", trained), ("initial", initial)):
            out[f"{name}/{label}"] = score_tasks(config, item.data, train_range, test_range, half)
            if not reset:
                out[f"{name}/drift"] = drift(config, item.data, test_range)
                out[f"{name}/finite"] = item.finite
    history_scores = {}
    for task in config.tasks:
        accuracy, _, _ = fit_score(config, trained.data, task, train_range, test_range, 1, history=True)
        _, y = dataset(config, trained.data, task, test_range, 1)
        history_scores[task] = {"test": accuracy, "majority": float(max(y.mean(), 1 - y.mean()))}
    out["history_baseline"] = history_scores
    return out


def score_tasks(config: E2PConfig, data: StreamData, fit: range, score: range, half: int) -> dict[str, Any]:
    scores = {}
    for task in config.tasks:
        accuracy, _, _ = fit_score(config, data, task, fit, score, config.readout_delay)
        early, _, _ = fit_score(config, data, task, fit, score, config.readout_delay, score_pulses=slice(config.washout, half))
        late, _, _ = fit_score(config, data, task, fit, score, config.readout_delay, score_pulses=slice(half, None))
        scores[task] = {"test": accuracy, "early": early, "late": late}
    return scores


def random_search(config: E2PConfig, tasks: tuple[str, ...], *, budget: int, seed: int) -> dict[str, Any]:
    """Control: the same number of theta evaluations drawn around theta_init, without updates."""
    train_range, val_range, test_range = config.stream_ranges()
    space = config.space
    rng = np.random.default_rng(seed)
    center = space.encode(config.theta_init)
    sigma = float(config.outer.get("init_sigma", 0.8))
    candidates = np.vstack([center, center + rng.standard_normal((max(1, budget - 1), space.dim)) * sigma])
    population = int(config.outer.get("population", 16))
    best_u, best_fit = None, None
    for start in range(0, len(candidates), population):
        block = candidates[start : start + population]
        for u, item in zip(block, collect(config, [space.decode(u) for u in block], range(0, val_range.stop))):
            fit = fitness(config, item, train_range, val_range, tasks)
            if best_fit is None or fit.j < best_fit.j:
                best_u, best_fit = u, fit
    theta = space.decode(best_u)
    [item] = collect(config, [theta], range(0, test_range.stop))
    test = {task: fit_score(config, item.data, task, train_range, test_range, config.readout_delay)[0] for task in config.tasks}
    return {"theta": theta.to_payload(), "j": best_fit.j, "val_accuracy": best_fit.val_accuracy, "test": test}
