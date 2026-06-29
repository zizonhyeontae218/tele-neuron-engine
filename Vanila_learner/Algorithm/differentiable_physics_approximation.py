from __future__ import annotations

import numpy as np

from tele_neuron.learner import build_result_model, clip_masses, result_better


ALGORITHM_NAME = "Differentiable Physics Approximation"


def train(context):
    rng = np.random.default_rng(context.settings.seed)
    best_masses = context.base_model.masses
    best_result = context.evaluate(best_masses)
    history = [_row(0, best_result)]

    for iteration in range(1, context.settings.iterations + 1):
        values = np.array(best_masses, dtype=float)
        indexes = rng.choice(len(values), size=max(1, min(len(values), context.settings.population)), replace=False)
        improved = False
        for index in indexes:
            for sign in (-1.0, 1.0):
                trial = values.copy()
                trial[index] += sign * context.settings.mutation_scale
                candidate = clip_masses(tuple(float(value) for value in trial), context.settings.min_mass, context.settings.max_mass)
                result = context.evaluate(candidate)
                if result_better(result, best_result):
                    best_masses = candidate
                    best_result = result
                    values = np.array(candidate, dtype=float)
                    improved = True
        row = _row(iteration, best_result)
        row["finite_difference_improved"] = improved
        history.append(row)

    return build_result_model(
        context,
        masses=best_masses,
        algorithm="differentiable_physics_approximation",
        result=best_result,
        history=history,
        metadata={"approximation": "finite-difference mass perturbation"},
    )


def _row(iteration, result):
    return {"iteration": iteration, "score": result.score, "accuracy": result.accuracy}
