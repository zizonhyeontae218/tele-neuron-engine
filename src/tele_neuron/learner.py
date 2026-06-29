from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol

import numpy as np

from tele_neuron.config import ExperimentConfig, parse_config
from tele_neuron.model import TeleNeuronModel, model_from_payload, random_model, save_model
from tele_neuron.trainer import ExperimentResult, run_cases


@dataclass(frozen=True, slots=True)
class LearnerSettings:
    iterations: int
    population: int
    seed: int
    min_mass: float
    max_mass: float
    mutation_scale: float


@dataclass(slots=True)
class TrainingContext:
    config: ExperimentConfig
    base_model: TeleNeuronModel
    settings: LearnerSettings

    def evaluate(self, masses: tuple[float, ...]) -> ExperimentResult:
        clipped = clip_masses(masses, self.settings.min_mass, self.settings.max_mass)
        return run_cases(self.config, masses=clipped)


@dataclass(frozen=True, slots=True)
class TrainingResult:
    model: TeleNeuronModel
    result: ExperimentResult
    history: tuple[dict[str, Any], ...]


class AlgorithmModule(Protocol):
    ALGORITHM_NAME: str

    def train(self, context: TrainingContext) -> TrainingResult:
        ...


def list_algorithm_files(root: Path) -> list[dict[str, str]]:
    algorithm_dir = algorithm_root(root)
    return [
        {
            "name": path.stem,
            "label": path.stem.replace("_", " ").title(),
            "path": path.relative_to(root).as_posix(),
        }
        for path in sorted(algorithm_dir.glob("*.py"))
        if path.name != "__init__.py"
    ]


def train_model_payload(
    root: Path,
    config_payload: dict[str, Any],
    *,
    algorithm: str,
    settings_payload: dict[str, Any],
    model_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = parse_config(config_payload)
    settings = parse_settings(settings_payload)
    base_model = (
        model_from_payload(model_payload)
        if model_payload
        else random_model(config, seed=settings.seed, model_id=f"{config.name}_{algorithm}_model")
    )
    if len(base_model.masses) != config.balls.count:
        raise ValueError("model masses length must match config balls.count")

    module = load_algorithm(root, algorithm)
    context = TrainingContext(config=config, base_model=base_model, settings=settings)
    training = module.train(context)
    output_path = save_training_model(root, training.model)

    return {
        "algorithm": algorithm,
        "model": asdict(training.model),
        "result": _result_payload(training.result),
        "history": training.history,
        "path": output_path.relative_to(root).as_posix(),
    }


def parse_settings(payload: dict[str, Any]) -> LearnerSettings:
    return LearnerSettings(
        iterations=max(1, int(payload.get("iterations", 20))),
        population=max(1, int(payload.get("population", 12))),
        seed=int(payload.get("seed", 1)),
        min_mass=max(0.0001, float(payload.get("min_mass", 0.5))),
        max_mass=max(0.0001, float(payload.get("max_mass", 1.5))),
        mutation_scale=max(0.0, float(payload.get("mutation_scale", 0.15))),
    )


def algorithm_root(root: Path) -> Path:
    path = root / "Vanila_learner" / "Algorithm"
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_algorithm(root: Path, name: str) -> AlgorithmModule:
    safe = "".join(char for char in name if char.isalnum() or char in {"_", "-"}).replace("-", "_")
    path = algorithm_root(root) / f"{safe}.py"
    if not path.is_file():
        raise FileNotFoundError(f"algorithm not found: {safe}")
    spec = importlib.util.spec_from_file_location(f"tele_neuron_algorithm_{safe}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load algorithm: {safe}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "train"):
        raise AttributeError(f"algorithm {safe} must define train(context)")
    return module  # type: ignore[return-value]


def save_training_model(root: Path, model: TeleNeuronModel) -> Path:
    output_dir = root / "models" / "lab"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{_safe_name(model.model_id)}.json"
    save_model(model, output_path)
    return output_path


def clip_masses(masses: tuple[float, ...], min_mass: float, max_mass: float) -> tuple[float, ...]:
    clipped = np.clip(np.array(masses, dtype=np.float64), min_mass, max_mass)
    return tuple(float(value) for value in clipped)


def perturb_masses(
    rng: np.random.Generator,
    masses: tuple[float, ...],
    *,
    scale: float,
    min_mass: float,
    max_mass: float,
) -> tuple[float, ...]:
    values = np.array(masses, dtype=np.float64)
    values += rng.normal(0.0, scale, size=values.shape)
    return clip_masses(tuple(float(value) for value in values), min_mass, max_mass)


def random_masses(
    rng: np.random.Generator,
    count: int,
    *,
    min_mass: float,
    max_mass: float,
) -> tuple[float, ...]:
    return tuple(float(value) for value in rng.uniform(min_mass, max_mass, count))


def result_better(candidate: ExperimentResult, best: ExperimentResult | None) -> bool:
    if best is None:
        return True
    candidate_collisions = sum(case.collisions for case in candidate.cases)
    best_collisions = sum(case.collisions for case in best.cases)
    return (candidate.score, candidate_collisions) > (best.score, best_collisions)


def build_result_model(
    context: TrainingContext,
    *,
    masses: tuple[float, ...],
    algorithm: str,
    result: ExperimentResult,
    history: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> TrainingResult:
    model = TeleNeuronModel(
        model_id=f"{context.config.name}_{algorithm}_gen{context.base_model.generation + len(history)}",
        config_name=context.config.name,
        masses=masses,
        generation=context.base_model.generation + len(history),
        algorithm=algorithm,
        score=result.score,
        total=result.total,
        accuracy=result.accuracy,
        metadata={
            **context.base_model.metadata,
            "source_model_id": context.base_model.model_id,
            "settings": asdict(context.settings),
            **(metadata or {}),
        },
    )
    return TrainingResult(model=model, result=result, history=tuple(history))


def _result_payload(result: ExperimentResult) -> dict[str, Any]:
    payload = asdict(result)
    payload["accuracy"] = result.accuracy
    return payload


def _safe_name(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"_", ".", "-"} else "_" for char in value)
    return safe.strip("._") or "model"
