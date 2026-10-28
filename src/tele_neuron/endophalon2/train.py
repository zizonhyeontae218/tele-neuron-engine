"""Stage 2: bilevel training for continuous computation.

Outer loop: cross-entropy search over theta (physics, rest dynamics, stimulus
points, readout zone centers and width). For each candidate theta, one
continuous run per stream. Inner loop: one exact ridge readout per task, read
``readout_delay`` steps after each pulse. Candidates are scored on validation
streams:

    J = mean_task L_val + l_noise * R_noise + l_drift * R_drift + l_complexity * R_complexity

Streams, not pulses, are split into train / val / test, so a readout is never
scored on the stream it was fit on. The outer loop never sees test streams.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from tele_neuron.endophalon2.config import E2Config
from tele_neuron.endophalon2.evaluate import StreamData, collect, dataset, fit_score
from tele_neuron.endophalon2.theta import Theta2


@dataclass(frozen=True, slots=True)
class Fitness2:
    j: float
    l_val: float  # mean over tasks of validation MSE
    r_noise: float  # within-class score spread / between-class score gap (scale-free)
    r_drift: float  # |log energy ratio late/early| + wall share late in the stream
    r_complexity: float  # mean squared readout weight
    val_accuracy: dict[str, float]


@dataclass(frozen=True, slots=True)
class E2Report:
    theta: Theta2
    fitness: Fitness2
    fitness_init: Fitness2
    evaluations: int
    history: tuple[dict[str, Any], ...]
    test: dict[str, Any]
    random_search: dict[str, Any] | None = None
    readouts: dict[str, Any] = field(default_factory=dict)


def drift(config: E2Config, data: StreamData, streams: range) -> float:
    rows = data.index(streams)
    start = config.washout
    quarter = max(1, (config.stream.pulses - start) // 4)
    early = data.energy[rows, start : start + quarter].mean()
    late = data.energy[rows, -quarter:].mean()
    return float(abs(np.log((late + 1e-9) / (early + 1e-9))) + data.wall[rows, -quarter:].mean())


def fitness(config: E2Config, data: StreamData, train: range, val: range, tasks: tuple[str, ...]) -> Fitness2:
    """Inner loop for every task, then the outer score J."""
    losses, noises, weights, accuracies = [], [], [], {}
    for task in tasks:
        accuracy, mse, readout = fit_score(config, data, task, train, val, config.readout_delay)
        x, y = dataset(config, data, task, val, config.readout_delay)
        scores = readout.scores(x)[:, 0]
        ones, zeros = scores[y[:, 0] == 1], scores[y[:, 0] == 0]
        if len(ones) and len(zeros):
            within = 0.5 * (ones.std() + zeros.std())
            noises.append(float(within / (abs(ones.mean() - zeros.mean()) + 1e-6)))
        assert readout.weights is not None
        weights.append(float(np.mean(readout.weights[:-1] ** 2)))
        losses.append(mse)
        accuracies[task] = accuracy
    spec = config.objective
    l_val = float(np.mean(losses))
    r_noise = float(np.mean(noises)) if noises else 0.0
    r_drift = drift(config, data, range(train.start, val.stop))
    r_complexity = float(np.mean(weights))
    j = (
        l_val
        + spec.get("lambda_noise", 0.0) * r_noise
        + spec.get("lambda_drift", 0.0) * r_drift
        + spec.get("lambda_complexity", 0.0) * r_complexity
    )
    return Fitness2(j, l_val, r_noise, r_drift, r_complexity, accuracies)


def train(
    config: E2Config,
    *,
    tasks: tuple[str, ...] | None = None,
    random_baseline: bool = False,
    log: Callable[[str], None] | None = None,
) -> E2Report:
    tasks = tasks or config.tasks
    train_range, val_range, test_range = config.stream_ranges()
    fit_streams = range(0, val_range.stop)
    outer = config.outer
    generations = int(outer.get("generations", 30))
    population = int(outer.get("population", 16))
    elite = int(outer.get("elite", 4))
    min_sigma = float(outer.get("min_sigma", 0.05))
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
        datas = collect(config, [space.decode(u) for u in candidates], fit_streams)
        scored = [fitness(config, data, train_range, val_range, tasks) for data in datas]
        evaluations += len(candidates)
        js = np.array([fit.j for fit in scored])
        if generation == 0:
            init_fit = scored[0]
        top = int(np.argmin(js))
        if best_fit is None or scored[top].j < best_fit.j:
            best_u, best_fit = candidates[top].copy(), scored[top]
        history.append(
            {"generation": generation, "best_j": best_fit.j, "best_val_accuracy": best_fit.val_accuracy,
             "center_j": scored[0].j, "sigma": float(sigma.mean())}
        )
        if log:
            accs = " ".join(f"{task}={value:.2f}" for task, value in best_fit.val_accuracy.items())
            log(f"  gen {generation:>3} best J={best_fit.j:.4f} [{accs}] center J={scored[0].j:.4f} sigma={sigma.mean():.3f}")
        if generation == 0:
            continue
        order = np.argsort(js)[:elite]
        chosen = candidates[order]
        spread = np.sqrt(weights @ (chosen - mean) ** 2)
        mean = weights @ chosen
        sigma = np.maximum(0.7 * spread + 0.3 * sigma, min_sigma)

    assert best_fit is not None and init_fit is not None
    theta = space.decode(best_u)
    report = E2Report(
        theta=theta,
        fitness=best_fit,
        fitness_init=init_fit,
        evaluations=evaluations,
        history=tuple(history),
        test=test_evaluation(config, theta, config.theta_init),
    )
    if random_baseline:
        random = random_search(config, tasks, budget=evaluations, seed=int(outer.get("seed", 1)) + 1000)
        return E2Report(**{**{f: getattr(report, f) for f in report.__dataclass_fields__}, "random_search": random})
    return report


def test_evaluation(config: E2Config, theta: Theta2, theta_init: Theta2) -> dict[str, Any]:
    """Held-out test streams: continuous and reset runs for trained and initial theta."""
    train_range, _, test_range = config.stream_ranges()
    streams = range(0, test_range.stop)
    half = config.washout + (config.stream.pulses - config.washout) // 2
    out: dict[str, Any] = {}
    for label, reset in (("continuous", False), ("reset", True)):
        trained, initial = collect(config, [theta, theta_init], streams, reset=reset)
        for name, data in (("trained", trained), ("initial", initial)):
            scores = {}
            for task in config.tasks:
                accuracy, _, _ = fit_score(config, data, task, train_range, test_range, config.readout_delay)
                early, _, _ = fit_score(config, data, task, train_range, test_range, config.readout_delay, score_pulses=slice(config.washout, half))
                late, _, _ = fit_score(config, data, task, train_range, test_range, config.readout_delay, score_pulses=slice(half, None))
                scores[task] = {"test": accuracy, "early": early, "late": late}
            out[f"{name}/{label}"] = scores
            if not reset:
                out[f"{name}/drift"] = drift(config, data, test_range)
    history_scores = {}
    for task in config.tasks:
        accuracy, _, _ = fit_score(config, trained, task, train_range, test_range, 1, history=True)
        _, y = dataset(config, trained, task, test_range, 1)
        history_scores[task] = {"test": accuracy, "majority": float(max(y.mean(), 1 - y.mean()))}
    out["history_baseline"] = history_scores
    return out


def random_search(config: E2Config, tasks: tuple[str, ...], *, budget: int, seed: int) -> dict[str, Any]:
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
        for u, data in zip(block, collect(config, [space.decode(u) for u in block], range(0, val_range.stop))):
            fit = fitness(config, data, train_range, val_range, tasks)
            if best_fit is None or fit.j < best_fit.j:
                best_u, best_fit = u, fit
    theta = space.decode(best_u)
    [data] = collect(config, [theta], range(0, test_range.stop))
    test = {task: fit_score(config, data, task, train_range, test_range, config.readout_delay)[0] for task in config.tasks}
    return {"theta": theta.to_payload(), "j": best_fit.j, "val_accuracy": best_fit.val_accuracy, "test": test}
