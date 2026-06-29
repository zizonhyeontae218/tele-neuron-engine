from __future__ import annotations

import numpy as np

from tele_neuron.learner import build_result_model, random_masses, result_better


ALGORITHM_NAME = "Novelty Search"


def train(context):
    rng = np.random.default_rng(context.settings.seed)
    archive = [np.array(context.base_model.masses, dtype=float)]
    best_masses = context.base_model.masses
    best_result = context.evaluate(best_masses)
    history = [_row(0, best_result, 0.0)]

    for iteration in range(1, context.settings.iterations + 1):
        best_novelty = -1.0
        iteration_best = best_masses
        for _ in range(context.settings.population):
            candidate = random_masses(
                rng,
                context.config.balls.count,
                min_mass=context.settings.min_mass,
                max_mass=context.settings.max_mass,
            )
            vector = np.array(candidate, dtype=float)
            novelty = min(float(np.linalg.norm(vector - item)) for item in archive)
            result = context.evaluate(candidate)
            if novelty > best_novelty or result_better(result, best_result):
                best_novelty = novelty
                iteration_best = candidate
            if result_better(result, best_result):
                best_masses = candidate
                best_result = result
        archive.append(np.array(iteration_best, dtype=float))
        history.append(_row(iteration, best_result, best_novelty))

    return build_result_model(
        context,
        masses=best_masses,
        algorithm="novelty_search",
        result=best_result,
        history=history,
    )


def _row(iteration, result, novelty):
    return {"iteration": iteration, "score": result.score, "accuracy": result.accuracy, "novelty": novelty}
