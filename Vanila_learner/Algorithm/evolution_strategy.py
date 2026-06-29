from __future__ import annotations

import numpy as np

from tele_neuron.learner import build_result_model, clip_masses, result_better


ALGORITHM_NAME = "Evolution Strategy"


def train(context):
    rng = np.random.default_rng(context.settings.seed)
    center = np.array(context.base_model.masses, dtype=float)
    best_masses = tuple(float(value) for value in center)
    best_result = context.evaluate(best_masses)
    history = [_row(0, best_result)]

    for iteration in range(1, context.settings.iterations + 1):
        candidates = []
        for _ in range(context.settings.population):
            noise = rng.normal(0.0, context.settings.mutation_scale, center.shape)
            masses = clip_masses(
                tuple(float(value) for value in center + noise),
                context.settings.min_mass,
                context.settings.max_mass,
            )
            result = context.evaluate(masses)
            candidates.append((result, masses))
            if result_better(result, best_result):
                best_result = result
                best_masses = masses

        candidates.sort(key=lambda item: item[0].score, reverse=True)
        elites = np.array([masses for _, masses in candidates[: max(1, len(candidates) // 4)]])
        center = elites.mean(axis=0)
        history.append(_row(iteration, best_result))

    return build_result_model(
        context,
        masses=best_masses,
        algorithm="evolution_strategy",
        result=best_result,
        history=history,
    )


def _row(iteration, result):
    return {"iteration": iteration, "score": result.score, "accuracy": result.accuracy}
