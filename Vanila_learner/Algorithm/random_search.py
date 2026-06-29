from __future__ import annotations

import numpy as np

from tele_neuron.learner import build_result_model, random_masses, result_better


ALGORITHM_NAME = "Random Searching"


def train(context):
    rng = np.random.default_rng(context.settings.seed)
    best_masses = context.base_model.masses
    best_result = context.evaluate(best_masses)
    history = [_row(0, best_result)]

    for iteration in range(1, context.settings.iterations + 1):
        for _ in range(context.settings.population):
            masses = random_masses(
                rng,
                context.config.balls.count,
                min_mass=context.settings.min_mass,
                max_mass=context.settings.max_mass,
            )
            result = context.evaluate(masses)
            if result_better(result, best_result):
                best_masses = masses
                best_result = result
        history.append(_row(iteration, best_result))

    return build_result_model(
        context,
        masses=best_masses,
        algorithm="random_search",
        result=best_result,
        history=history,
    )


def _row(iteration, result):
    return {"iteration": iteration, "score": result.score, "accuracy": result.accuracy}
