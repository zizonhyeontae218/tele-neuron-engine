from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from tele_neuron.endophalon import FAMILY
from tele_neuron.endophalon.config import EndophalonConfig
from tele_neuron.endophalon.readout import RidgeReadout
from tele_neuron.endophalon.train import TrainingReport, build_pipeline


@dataclass(frozen=True, slots=True)
class EndophalonModel:
    model_id: str
    config_path: str
    task: str
    readout: RidgeReadout
    metrics: dict[str, Any]

    def predict(self, config: EndophalonConfig, bits: tuple[int, ...], realization: int = 0) -> tuple[int, ...]:
        reservoir, observer = build_pipeline(config)
        features = observer.features(reservoir.run(bits, realization))
        return tuple(int(value) for value in self.readout.predict(features[np.newaxis, :])[0])


def model_from_report(model_id: str, config_path: str, config: EndophalonConfig, report: TrainingReport) -> EndophalonModel:
    return EndophalonModel(
        model_id=model_id,
        config_path=config_path,
        task=config.task,
        readout=report.readout,
        metrics=report_metrics(report),
    )


def report_metrics(report: TrainingReport) -> dict[str, Any]:
    return {
        "feature_count": report.feature_count,
        "train": asdict(report.train),
        "test": asdict(report.test) if report.test else None,
        "baseline_linear_on_bits": {
            "train": asdict(report.baseline_train),
            "test": asdict(report.baseline_test) if report.baseline_test else None,
        },
    }


def save_model(model: EndophalonModel, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "family": FAMILY,
        "model_id": model.model_id,
        "config": model.config_path,
        "task": model.task,
        "metrics": model.metrics,
        "readout": model.readout.to_payload(),
        "notes": [
            "Separate model family from TeleNeuron-001: fixed physics reservoir, trained linear readout.",
            "No backpropagation: the readout is fitted with closed-form ridge regression.",
        ],
    }
    with output.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")


def load_model(path: str | Path) -> EndophalonModel:
    with Path(path).open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if payload.get("family") != FAMILY:
        raise ValueError(f"not a {FAMILY} model card: {path}")
    return EndophalonModel(
        model_id=str(payload["model_id"]),
        config_path=str(payload["config"]),
        task=str(payload["task"]),
        readout=RidgeReadout.from_payload(payload["readout"]),
        metrics=dict(payload["metrics"]),
    )
