from __future__ import annotations

import numpy as np

from tele_neuron.learner import build_result_model, clip_masses, result_better


ALGORITHM_NAME = "Linear Regression"


def train(context):
    rng = np.random.default_rng(context.settings.seed)
    base = np.array(context.base_model.masses, dtype=float)
    best_masses = tuple(float(value) for value in base)
    best_result = context.evaluate(best_masses)
    history = [_row(0, best_result)]

    for iteration in range(1, context.settings.iterations + 1):
        slope = rng.normal(0.0, context.settings.mutation_scale, base.shape)
        candidate_values = base + slope * (iteration / max(1, context.settings.iterations))
        candidate = clip_masses(tuple(float(value) for value in candidate_values), context.settings.min_mass, context.settings.max_mass)
        result = context.evaluate(candidate)
        if result_better(result, best_result):
            best_masses = candidate
            best_result = result
        history.append(_row(iteration, best_result))

    return build_result_model(
        context,
        masses=best_masses,
        algorithm="linear_regression",
        result=best_result,
        history=history,
        metadata={"approximation": "linear mass trend search"},
    )


def _row(iteration, result):
    return {"iteration": iteration, "score": result.score, "accuracy": result.accuracy}
