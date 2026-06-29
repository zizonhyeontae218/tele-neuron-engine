from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from tele_neuron.config import ExperimentConfig


@dataclass(frozen=True, slots=True)
class TeleNeuronModel:
    model_id: str
    config_name: str
    masses: tuple[float, ...]
    generation: int
    algorithm: str
    score: int
    total: int
    accuracy: float
    metadata: dict[str, Any]


def random_model(
    config: ExperimentConfig,
    *,
    seed: int,
    model_id: str,
    min_mass: float = 0.5,
    max_mass: float = 1.5,
) -> TeleNeuronModel:
    rng = np.random.default_rng(seed)
    masses = tuple(float(value) for value in rng.uniform(min_mass, max_mass, config.balls.count))
    return TeleNeuronModel(
        model_id=model_id,
        config_name=config.name,
        masses=masses,
        generation=0,
        algorithm="random_initialization",
        score=0,
        total=len(config.cases),
        accuracy=0.0,
        metadata={"seed": seed, "min_mass": min_mass, "max_mass": max_mass},
    )


def model_from_payload(payload: dict[str, Any]) -> TeleNeuronModel:
    masses = tuple(float(value) for value in payload["masses"])
    score = int(payload.get("score", 0))
    total = int(payload.get("total", 0))
    accuracy = float(payload.get("accuracy", score / total if total else 0.0))
    return TeleNeuronModel(
        model_id=str(payload["model_id"]),
        config_name=str(payload.get("config_name", "")),
        masses=masses,
        generation=int(payload.get("generation", 0)),
        algorithm=str(payload.get("algorithm", "unknown")),
        score=score,
        total=total,
        accuracy=accuracy,
        metadata=dict(payload.get("metadata", {})),
    )


def model_to_payload(model: TeleNeuronModel) -> dict[str, Any]:
    return {
        "model_id": model.model_id,
        "config_name": model.config_name,
        "masses": list(model.masses),
        "generation": model.generation,
        "algorithm": model.algorithm,
        "score": model.score,
        "total": model.total,
        "accuracy": model.accuracy,
        "metadata": model.metadata,
    }


def load_model(path: str | Path) -> TeleNeuronModel:
    with Path(path).open("r", encoding="utf-8") as file:
        return model_from_payload(json.load(file))


def save_model(model: TeleNeuronModel, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(model_to_payload(model), file, indent=2)
        file.write("\n")


def with_result(
    model: TeleNeuronModel,
    *,
    algorithm: str,
    generation: int,
    score: int,
    total: int,
    metadata: dict[str, Any] | None = None,
) -> TeleNeuronModel:
    return TeleNeuronModel(
        model_id=model.model_id,
        config_name=model.config_name,
        masses=model.masses,
        generation=generation,
        algorithm=algorithm,
        score=score,
        total=total,
        accuracy=score / total if total else 0.0,
        metadata={**model.metadata, **(metadata or {})},
    )


def replace_masses(model: TeleNeuronModel, masses: tuple[float, ...]) -> TeleNeuronModel:
    return TeleNeuronModel(
        model_id=model.model_id,
        config_name=model.config_name,
        masses=masses,
        generation=model.generation,
        algorithm=model.algorithm,
        score=model.score,
        total=model.total,
        accuracy=model.accuracy,
        metadata=model.metadata,
    )
