from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Any

from tele_neuron.config import ExperimentConfig, ReadoutZoneConfig
from tele_neuron.physics import initialize_state, initialize_state_with_masses, step_state
from tele_neuron.readout import readout_counts


@dataclass(frozen=True, slots=True)
class CaseResult:
    bits: tuple[int, ...]
    target: tuple[int, ...]
    counts: tuple[int, ...]
    output: tuple[int, ...]
    correct: bool
    collisions: int


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    config_name: str
    task: str
    seed: int
    threshold: int
    score: int
    total: int
    cases: tuple[CaseResult, ...]

    @property
    def accuracy(self) -> float:
        return self.score / self.total


def run_cases(
    config: ExperimentConfig,
    threshold: int | None = None,
    masses: tuple[float, ...] | None = None,
) -> ExperimentResult:
    zone = config.readout[0]
    effective_threshold = zone.threshold if threshold is None else threshold
    cases: list[CaseResult] = []

    for case in config.cases:
        state = (
            initialize_state_with_masses(config, _mass_array(masses))
            if masses is not None
            else initialize_state(config)
        )
        for _ in range(config.simulation.steps):
            step_state(state, config, case.bits)

        counts = readout_counts(state.positions, config)
        output = _outputs_from_counts(config, counts, first_threshold=effective_threshold)
        cases.append(
            CaseResult(
                bits=case.bits,
                target=case.target,
                counts=counts,
                output=output,
                correct=output == case.target,
                collisions=state.collision_count,
            )
        )

    return ExperimentResult(
        config_name=config.name,
        task=config.task,
        seed=config.simulation.seed,
        threshold=effective_threshold,
        score=sum(1 for case in cases if case.correct),
        total=len(cases),
        cases=tuple(cases),
    )


def score_masses(config: ExperimentConfig, masses: tuple[float, ...]) -> ExperimentResult:
    return run_cases(config, masses=masses)


def search_best_seed(
    config: ExperimentConfig,
    *,
    seed_start: int,
    seed_end: int,
) -> ExperimentResult:
    best: ExperimentResult | None = None

    for seed in range(seed_start, seed_end + 1):
        seeded = replace(config, simulation=replace(config.simulation, seed=seed))
        counts_by_case = _case_counts(seeded)
        candidate_thresholds = _candidate_thresholds(counts_by_case)

        for threshold in candidate_thresholds:
            result = _result_from_counts(seeded, counts_by_case, threshold)
            if _is_better(result, best):
                best = result

    if best is None:
        raise ValueError("seed range produced no results")
    return best


def save_result(result: ExperimentResult, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(result)
    payload["accuracy"] = result.accuracy
    with output.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def print_result(result: ExperimentResult) -> None:
    print(
        f"{result.config_name} seed={result.seed} threshold={result.threshold} "
        f"score={result.score}/{result.total} accuracy={result.accuracy:.2%}"
    )
    for case in result.cases:
        bits = "".join(str(bit) for bit in case.bits)
        target = _format_bits(case.target)
        output = _format_bits(case.output)
        mark = "ok" if case.correct else "miss"
        print(
            f"  bits={bits} target={target} output={output} "
            f"counts={case.counts} collisions={case.collisions} {mark}"
        )


def apply_threshold(config: ExperimentConfig, threshold: int) -> ExperimentConfig:
    first = config.readout[0]
    updated_first = replace(first, threshold=threshold)
    readout: tuple[ReadoutZoneConfig, ...] = (updated_first, *config.readout[1:])
    return replace(config, readout=readout)


def _case_counts(config: ExperimentConfig) -> tuple[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], int], ...]:
    rows: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], int]] = []
    for case in config.cases:
        state = initialize_state(config)
        for _ in range(config.simulation.steps):
            step_state(state, config, case.bits)
        rows.append((readout_counts(state.positions, config), case.target, case.bits, state.collision_count))
    return tuple(rows)


def _candidate_thresholds(rows: tuple[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], int], ...]) -> tuple[int, ...]:
    counts = sorted({row[0][0] for row in rows})
    candidates = {0}
    for count in counts:
        candidates.add(count)
        candidates.add(count + 1)
    return tuple(sorted(candidates))


def _result_from_counts(
    config: ExperimentConfig,
    rows: tuple[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], int], ...],
    threshold: int,
) -> ExperimentResult:
    cases = tuple(
        CaseResult(
            bits=bits,
            target=target,
            counts=counts,
            output=_outputs_from_counts(config, counts, first_threshold=threshold),
            correct=_outputs_from_counts(config, counts, first_threshold=threshold) == target,
            collisions=collisions,
        )
        for counts, target, bits, collisions in rows
    )
    return ExperimentResult(
        config_name=config.name,
        task=config.task,
        seed=config.simulation.seed,
        threshold=threshold,
        score=sum(1 for case in cases if case.correct),
        total=len(cases),
        cases=cases,
    )


def _is_better(result: ExperimentResult, best: ExperimentResult | None) -> bool:
    if best is None:
        return True
    return (result.score, -result.threshold, -result.seed) > (
        best.score,
        -best.threshold,
        -best.seed,
    )


def _outputs_from_counts(
    config: ExperimentConfig,
    counts: tuple[int, ...],
    *,
    first_threshold: int | None = None,
) -> tuple[int, ...]:
    outputs: list[int] = []
    for index, (count, zone) in enumerate(zip(counts, config.readout, strict=True)):
        threshold = first_threshold if index == 0 and first_threshold is not None else zone.threshold
        outputs.append(1 if count >= threshold else 0)
    return tuple(outputs)


def _format_bits(bits: tuple[int, ...]) -> str:
    if len(bits) == 1:
        return str(bits[0])
    return "[" + ", ".join(str(bit) for bit in bits) + "]"


def _mass_array(masses: tuple[float, ...]) -> Any:
    import numpy as np

    return np.array(masses, dtype=np.float64)
