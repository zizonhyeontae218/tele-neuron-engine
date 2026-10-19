from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from tele_neuron.endophalon.readout import RidgeReadout
from tele_neuron.endophalon_pro import FAMILY
from tele_neuron.endophalon_pro.config import ProConfig
from tele_neuron.endophalon_pro.physics import ProJob, simulate
from tele_neuron.endophalon_pro.theta import Theta
from tele_neuron.endophalon_pro.train import ProReport
from tele_neuron.plus.observe import MultiScaleObserver


@dataclass(frozen=True, slots=True)
class ProModel:
    model_id: str
    config_path: str
    task: str
    theta: Theta
    readout: RidgeReadout
    metrics: dict[str, Any]

    def predict(self, config: ProConfig, bits: tuple[int, ...], realization: int = 0) -> tuple[int, ...]:
        readings = simulate(config.world, [self.theta], [ProJob(0, bits, realization)], config.readout.step)
        observer = MultiScaleObserver(config.world.space_size, (config.readout.grid,), speed=False)
        features = observer.observe(readings.positions, readings.velocities)
        return tuple(int(v) for v in self.readout.predict(features)[0])


def model_from_report(model_id: str, config_path: str, config: ProConfig, report: ProReport) -> ProModel:
    return ProModel(
        model_id=model_id,
        config_path=config_path,
        task=config.task,
        theta=report.theta,
        readout=report.readout,
        metrics=report_metrics(config, report),
    )


def report_metrics(config: ProConfig, report: ProReport) -> dict[str, Any]:
    return {
        "readout_grid": list(config.readout.grid),
        "readout_features": int(np.prod(config.readout.grid)),
        "readout_step": config.readout.step,
        "noise": {
            "position_jitter": config.world.position_jitter,
            "velocity_jitter": config.world.velocity_jitter,
            "train": config.splits.train,
            "val": config.splits.val,
            "test": config.splits.test,
        },
        "objective": asdict(config.objective),
        "evaluations": report.evaluations,
        "fitness_trained": asdict(report.fitness),
        "fitness_initial": asdict(report.fitness_init),
        "test_accuracy": report.test,
        "readout_size_sweep": list(report.sweep),
        "random_search": report.random_search,
        "history": [
            {"generation": g.index, "best_j": g.best_j, "best_val_accuracy": g.best.val_accuracy,
             "center_j": g.mean_candidate.j, "sigma": g.sigma}
            for g in report.history
        ],
    }


def save_model(model: ProModel, config: ProConfig, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "family": FAMILY,
        "model_id": model.model_id,
        "config": model.config_path,
        "task": model.task,
        "theta": model.theta.to_payload(),
        "theta_init": config.theta_init.to_payload(),
        "metrics": model.metrics,
        "readout": model.readout.to_payload(),
        "notes": [
            "One trainable physics system: theta is learned by an outer cross-entropy search.",
            "Inner loop: the readout is the exact ridge optimum W*(theta) for each candidate theta.",
            "Test realizations are never seen by either loop.",
        ],
    }
    with output.open("w", encoding="utf-8") as file:
        json.dump(_finite(payload), file, indent=2)
        file.write("\n")


def load_model(path: str | Path) -> ProModel:
    with Path(path).open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if payload.get("family") != FAMILY:
        raise ValueError(f"not a {FAMILY} model card: {path}")
    return ProModel(
        model_id=str(payload["model_id"]),
        config_path=str(payload["config"]),
        task=str(payload["task"]),
        theta=Theta.from_payload(payload["theta"]),
        readout=RidgeReadout.from_payload(payload["readout"]),
        metrics=dict(payload["metrics"]),
    )


def _finite(value: Any) -> Any:
    if isinstance(value, float) and value != value:
        return None
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    return value
