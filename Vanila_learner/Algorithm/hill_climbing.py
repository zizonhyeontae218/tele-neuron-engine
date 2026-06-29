from __future__ import annotations

import numpy as np

from tele_neuron.learner import build_result_model, perturb_masses, result_better


ALGORITHM_NAME = "Hill Climbing"


def train(context):
    rng = np.random.default_rng(context.settings.seed)
    best_masses = context.base_model.masses
    best_result = context.evaluate(best_masses)
    history = [_row(0, best_result)]

    for iteration in range(1, context.settings.iterations + 1):
        accepted = False
        for _ in range(context.settings.population):
            candidate = perturb_masses(
                rng,
                best_masses,
                scale=context.settings.mutation_scale,
                min_mass=context.settings.min_mass,
                max_mass=context.settings.max_mass,
            )
            result = context.evaluate(candidate)
            if result_better(result, best_result):
                best_masses = candidate
                best_result = result
                accepted = True
        row = _row(iteration, best_result)
        row["accepted"] = accepted
        history.append(row)

    return build_result_model(
        context,
        masses=best_masses,
        algorithm="hill_climbing",
        result=best_result,
        history=history,
    )


def _row(iteration, result):
    return {"iteration": iteration, "score": result.score, "accuracy": result.accuracy}
