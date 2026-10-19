from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from tele_neuron.endophalon.readout import RidgeReadout
from tele_neuron.endophalon_pro.config import ProConfig
from tele_neuron.endophalon_pro.objective import CaseFeatures, Fitness, accuracy, fitness, observe
from tele_neuron.endophalon_pro.physics import ProJob, simulate
from tele_neuron.endophalon_pro.theta import Theta


@dataclass(frozen=True, slots=True)
class Generation:
    index: int
    best_j: float
    best: Fitness
    mean_candidate: Fitness  # fitness of the search distribution's center
    sigma: float  # mean search step size


@dataclass(frozen=True, slots=True)
class ProReport:
    theta: Theta
    theta_init: Theta
    fitness: Fitness
    fitness_init: Fitness
    readout: RidgeReadout
    evaluations: int
    history: tuple[Generation, ...]
    test: dict[str, float]  # test accuracy: trained / initial / bits-linear / random-search
    sweep: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    random_search: dict[str, Any] | None = None


def evaluate_thetas(config: ProConfig, thetas: list[Theta], realizations: range) -> list[CaseFeatures]:
    """Step 2: run the physics of every theta on every (realization, case), in one batch."""
    jobs = [
        ProJob(theta=index, bits=bits, realization=realization)
        for index in range(len(thetas))
        for realization in realizations
        for bits, _ in config.cases
    ]
    readings = simulate(
        config.world, thetas, jobs, config.readout.step, workers=config.workers, chunk_size=config.chunk_size
    )
    per_theta = len(realizations) * len(config.cases)
    targets = np.array([target for _, target in config.cases])
    out = []
    for index in range(len(thetas)):
        rows = slice(index * per_theta, (index + 1) * per_theta)
        part = _slice(readings, rows)
        out.append(
            CaseFeatures(
                observe(config, part, config.readout.grid, len(realizations)),
                part.wall_fraction.reshape(len(realizations), -1).mean(axis=1),
                targets,
            )
        )
    return out


def train(config: ProConfig, *, random_baseline: bool = False, log: Callable[[str], None] | None = None) -> ProReport:
    train_range, val_range, test_range = config.splits.ranges()
    fit_range = range(0, val_range.stop)  # outer loop never sees test realizations
    space = config.space
    spec = config.outer
    rng = np.random.default_rng(spec.seed)

    mean = space.encode(config.theta_init)
    sigma = np.full(space.dim, spec.init_sigma)
    weights = np.log(spec.elite + 0.5) - np.log(np.arange(1, spec.elite + 1))
    weights /= weights.sum()

    best_u, best_fit = mean.copy(), None
    init_fit = None
    history: list[Generation] = []
    evaluations = 0

    for generation in range(spec.generations + 1):
        # Step 1: propose thetas. The first candidate is always the current center.
        if generation == 0:
            candidates = mean[None, :]
        else:
            noise = rng.standard_normal((spec.population - 1, space.dim)) * sigma
            candidates = np.vstack([mean, mean + noise])
        thetas = [space.decode(u) for u in candidates]
        datas = evaluate_thetas(config, thetas, fit_range)
        # Steps 3-4: inner-optimal readout per theta, scored on validation.
        scored = [fitness(config, data, train_range, val_range)[0] for data in datas]
        evaluations += len(thetas)
        js = np.array([fit.j for fit in scored])
        if generation == 0:
            init_fit = scored[0]
        top = int(np.argmin(js))
        if best_fit is None or scored[top].j < best_fit.j:
            best_u, best_fit = candidates[top].copy(), scored[top]
        history.append(Generation(generation, best_fit.j, best_fit, scored[0], float(sigma.mean())))
        if log:
            log(
                f"  gen {generation:>3} best J={best_fit.j:.4f} val_acc={best_fit.val_accuracy:.3f} "
                f"center J={scored[0].j:.4f} sigma={sigma.mean():.3f}"
            )
        if generation == 0:
            continue
        # Step 5: cross-entropy update toward the best candidates of this generation.
        order = np.argsort(js)[: spec.elite]
        elite = candidates[order]
        new_mean = weights @ elite
        spread = np.sqrt(weights @ (elite - mean) ** 2)
        sigma = np.maximum(0.7 * spread + 0.3 * sigma, spec.min_sigma)
        mean = new_mean

    assert best_fit is not None and init_fit is not None
    theta = space.decode(best_u)
    report_parts = final_evaluation(config, theta, config.theta_init)

    random_result = None
    if random_baseline:
        random_result = random_search(config, rng_seed=spec.seed + 1000, budget=evaluations)
        report_parts["test"]["random_search"] = random_result["test_accuracy"]

    return ProReport(
        theta=theta,
        theta_init=config.theta_init,
        fitness=best_fit,
        fitness_init=init_fit,
        readout=report_parts["readout"],
        evaluations=evaluations,
        history=tuple(history),
        test=report_parts["test"],
        sweep=report_parts["sweep"],
        random_search=random_result,
    )


def final_evaluation(config: ProConfig, theta: Theta, theta_init: Theta) -> dict[str, Any]:
    """Score trained and initial physics on held-out test realizations, and sweep readout sizes."""
    train_range, val_range, test_range = config.splits.ranges()
    everything = range(0, test_range.stop)
    jobs = [
        ProJob(theta=index, bits=bits, realization=realization)
        for index in range(2)
        for realization in everything
        for bits, _ in config.cases
    ]
    readings = simulate(
        config.world, [theta, theta_init], jobs, config.readout.step, workers=config.workers, chunk_size=config.chunk_size
    )
    per_theta = len(everything) * len(config.cases)
    targets = np.array([target for _, target in config.cases])
    parts = [_slice(readings, slice(i * per_theta, (i + 1) * per_theta)) for i in range(2)]

    def data_for(part, grid):
        return CaseFeatures(
            observe(config, part, grid, len(everything)),
            part.wall_fraction.reshape(len(everything), -1).mean(axis=1),
            targets,
        )

    trained, initial = data_for(parts[0], config.readout.grid), data_for(parts[1], config.readout.grid)
    _, readout = fitness(config, trained, train_range, val_range)
    _, readout_init = fitness(config, initial, train_range, val_range)
    bits = np.array([b for b, _ in config.cases], dtype=np.float64)
    bits_data = CaseFeatures(np.tile(bits, (len(everything), 1, 1)), np.zeros(len(everything)), targets)
    _, readout_bits = fitness(config, bits_data, train_range, val_range)
    test = {
        "trained": accuracy(readout, trained, test_range) if len(test_range) else float("nan"),
        "initial": accuracy(readout_init, initial, test_range) if len(test_range) else float("nan"),
        "bits_linear": accuracy(readout_bits, bits_data, test_range) if len(test_range) else float("nan"),
    }

    sweep = []
    for grid in config.sweep_grids:
        row: dict[str, Any] = {"grid": list(grid), "features": int(np.prod(grid))}
        for label, part in (("trained", parts[0]), ("initial", parts[1])):
            data = data_for(part, grid)
            _, fitted = fitness(config, data, train_range, val_range)
            row[label] = accuracy(fitted, data, test_range) if len(test_range) else float("nan")
        sweep.append(row)
    return {"readout": readout, "test": test, "sweep": tuple(sweep)}


def random_search(config: ProConfig, *, rng_seed: int, budget: int) -> dict[str, Any]:
    """Control: the same number of theta evaluations, drawn at random around theta_init, no updates."""
    train_range, val_range, test_range = config.splits.ranges()
    space = config.space
    rng = np.random.default_rng(rng_seed)
    center = space.encode(config.theta_init)
    samples = center + rng.standard_normal((max(1, budget - 1), space.dim)) * config.outer.init_sigma
    candidates = np.vstack([center, samples])
    best_u, best_fit = None, None
    for start in range(0, len(candidates), config.outer.population):
        block = candidates[start : start + config.outer.population]
        datas = evaluate_thetas(config, [space.decode(u) for u in block], range(0, val_range.stop))
        for u, data in zip(block, datas):
            fit = fitness(config, data, train_range, val_range)[0]
            if best_fit is None or fit.j < best_fit.j:
                best_u, best_fit = u, fit
    theta = space.decode(best_u)
    test = final_evaluation(config, theta, config.theta_init)["test"]["trained"]
    return {"theta": theta.to_payload(), "j": best_fit.j, "val_accuracy": best_fit.val_accuracy, "test_accuracy": test}


def _slice(readings, rows: slice):
    from tele_neuron.endophalon_pro.physics import Readings

    return Readings(readings.positions[rows], readings.velocities[rows], readings.wall_fraction[rows])
