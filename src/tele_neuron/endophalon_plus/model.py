from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from tele_neuron.endophalon_plus.readout import WideRidgeReadout as RidgeReadout
from tele_neuron.endophalon_plus import FAMILY
from tele_neuron.endophalon_plus.config import PlusConfig
from tele_neuron.endophalon_plus.gate import LinearGate
from tele_neuron.endophalon_plus.train import PlusReport, observer
from tele_neuron.plus.runner import run_features
from tele_neuron.plus.world import Job, preset_to_payload


@dataclass(frozen=True, slots=True)
class PlusModel:
    model_id: str
    config_path: str
    task: str
    gate: LinearGate
    readout: RidgeReadout
    metrics: dict[str, Any]

    def predict(self, config: PlusConfig, bits: tuple[int, ...], realization: int = 0) -> tuple[int, ...]:
        preset = self.gate.choose(bits)
        features = run_features(
            config.world, config.presets, [Job(preset, bits, realization)], config.observer.taps, observer(config)
        )[0]
        block = np.zeros(len(config.presets) * features.size)
        block[preset * features.size : (preset + 1) * features.size] = features
        return tuple(int(value) for value in self.readout.predict(block[np.newaxis, :])[0])


def model_from_report(model_id: str, config_path: str, config: PlusConfig, report: PlusReport) -> PlusModel:
    return PlusModel(
        model_id=model_id,
        config_path=config_path,
        task=config.task,
        gate=report.gate,
        readout=report.readout,
        metrics=report_metrics(config, report),
    )


def report_metrics(config: PlusConfig, report: PlusReport) -> dict[str, Any]:
    return {
        "features_per_preset": report.feature_count,
        "noise": {
            "position_jitter": config.world.position_jitter,
            "velocity_jitter": config.world.velocity_jitter,
            "train_realizations": config.train_realizations,
            "test_realizations": config.test_realizations,
        },
        "assignment": {
            "".join(map(str, bits)): config.presets[preset].name
            for (bits, _), preset in zip(config.cases, report.assignment, strict=True)
        },
        "gate_validation_accuracy": report.gate_validation_accuracy,
        "gate_evaluations": report.gate_evaluations,
        "train": asdict(report.train),
        "test": asdict(report.test) if report.test else None,
        "controls": {
            name: {split: asdict(value) if value else None for split, value in splits.items()}
            for name, splits in report.controls.items()
        },
    }


def save_model(model: PlusModel, config: PlusConfig, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "family": FAMILY,
        "model_id": model.model_id,
        "config": model.config_path,
        "task": model.task,
        "presets": [preset_to_payload(preset) for preset in config.presets],
        "metrics": model.metrics,
        "gate": model.gate.to_payload(),
        "readout": model.readout.to_payload(),
        "notes": [
            "Half-learning: three fixed physics presets, a learned linear gate picks one per input.",
            "The presets are never trained; the readout is closed-form ridge regression.",
        ],
    }
    with output.open("w", encoding="utf-8") as file:
        json.dump(_finite(payload), file, indent=2)
        file.write("\n")


def load_model(path: str | Path) -> PlusModel:
    with Path(path).open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if payload.get("family") != FAMILY:
        raise ValueError(f"not a {FAMILY} model card: {path}")
    return PlusModel(
        model_id=str(payload["model_id"]),
        config_path=str(payload["config"]),
        task=str(payload["task"]),
        gate=LinearGate.from_payload(payload["gate"]),
        readout=RidgeReadout.from_payload(payload["readout"]),
        metrics=dict(payload["metrics"]),
    )


def _finite(value: Any) -> Any:
    """JSON has no NaN; store it as null."""
    if isinstance(value, float) and value != value:
        return None
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite(item) for item in value]
    return value
