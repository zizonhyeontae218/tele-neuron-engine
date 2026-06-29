from __future__ import annotations

import numpy as np

from tele_neuron.learner import build_result_model, clip_masses, result_better


ALGORITHM_NAME = "Logistic Regression"


def train(context):
    rng = np.random.default_rng(context.settings.seed)
    masses = np.array(context.base_model.masses, dtype=float)
    best_masses = tuple(float(value) for value in masses)
    best_result = context.evaluate(best_masses)
    history = [_row(0, best_result)]

    for iteration in range(1, context.settings.iterations + 1):
        # Phase 1 approximation: logistic-shaped multiplicative mass update.
        direction = rng.normal(0.0, 1.0, masses.shape)
        gate = 1.0 / (1.0 + np.exp(-direction))
        candidate_values = masses * (1.0 + context.settings.mutation_scale * (gate - 0.5))
        candidate = clip_masses(tuple(float(value) for value in candidate_values), context.settings.min_mass, context.settings.max_mass)
        result = context.evaluate(candidate)
        if result_better(result, best_result):
            masses = np.array(candidate)
            best_masses = candidate
            best_result = result
        history.append(_row(iteration, best_result))

    return build_result_model(
        context,
        masses=best_masses,
        algorithm="logistic_regression",
        result=best_result,
        history=history,
        metadata={"approximation": "logistic-shaped mass search"},
    )


def _row(iteration, result):
    return {"iteration": iteration, "score": result.score, "accuracy": result.accuracy}
