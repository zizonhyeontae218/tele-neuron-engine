from __future__ import annotations

import numpy as np

from tele_neuron.learner import build_result_model, clip_masses, random_masses, result_better


ALGORITHM_NAME = "Genetic Algorithm"


def train(context):
    rng = np.random.default_rng(context.settings.seed)
    population = [context.base_model.masses]
    while len(population) < context.settings.population:
        population.append(
            random_masses(
                rng,
                context.config.balls.count,
                min_mass=context.settings.min_mass,
                max_mass=context.settings.max_mass,
            )
        )

    best_masses = population[0]
    best_result = context.evaluate(best_masses)
    history = [_row(0, best_result)]

    for iteration in range(1, context.settings.iterations + 1):
        scored = [(context.evaluate(masses), masses) for masses in population]
        scored.sort(key=lambda item: (item[0].score, sum(case.collisions for case in item[0].cases)), reverse=True)
        if result_better(scored[0][0], best_result):
            best_result, best_masses = scored[0]

        parents = [masses for _, masses in scored[: max(2, len(scored) // 3)]]
        next_population = parents[:]
        while len(next_population) < context.settings.population:
            left = np.array(parents[rng.integers(0, len(parents))])
            right = np.array(parents[rng.integers(0, len(parents))])
            mask = rng.random(left.shape) < 0.5
            child = np.where(mask, left, right)
            child += rng.normal(0.0, context.settings.mutation_scale, child.shape)
            next_population.append(
                clip_masses(tuple(float(value) for value in child), context.settings.min_mass, context.settings.max_mass)
            )
        population = next_population
        history.append(_row(iteration, best_result))

    return build_result_model(
        context,
        masses=best_masses,
        algorithm="genetic_algorithm",
        result=best_result,
        history=history,
    )


def _row(iteration, result):
    return {"iteration": iteration, "score": result.score, "accuracy": result.accuracy}
