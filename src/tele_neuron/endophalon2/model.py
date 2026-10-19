from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from tele_neuron.endophalon2 import FAMILY
from tele_neuron.endophalon2.config import E2Config
from tele_neuron.endophalon2.evaluate import collect, fit_score
from tele_neuron.endophalon2.train import E2Report


def report_payload(config: E2Config, report: E2Report, tasks: tuple[str, ...]) -> dict[str, Any]:
    return {
        "trained_tasks": list(tasks),
        "readout_delay": config.readout_delay,
        "readout_zones": config.zones,
        "stream": asdict(config.stream),
        "split": {
            "washout": config.washout,
            "train_streams": config.train_streams,
            "val_streams": config.val_streams,
            "test_streams": config.test_streams,
        },
        "objective": config.objective,
        "evaluations": report.evaluations,
        "fitness_initial": asdict(report.fitness_init),
        "fitness_trained": asdict(report.fitness),
        "test": report.test,
        "random_search": report.random_search,
        "history": list(report.history),
    }


def save_model(model_id: str, config_path: str, config: E2Config, report: E2Report, tasks: tuple[str, ...], path: str | Path) -> None:
    """Model card: the trained theta, one readout per task (fit on train streams), and all metrics."""
    train_range, _, _ = config.stream_ranges()
    [data] = collect(config, [report.theta], train_range)
    readouts = {
        task: fit_score(config, data, task, train_range, train_range, config.readout_delay)[2].to_payload()
        for task in config.tasks
    }
    payload = {
        "family": FAMILY,
        "model_id": model_id,
        "config": config_path,
        "theta": report.theta.to_payload(),
        "theta_init": config.theta_init.to_payload(),
        "readouts": readouts,
        "metrics": report_payload(config, report, tasks),
        "notes": [
            "Continuous run: the physics is never reset between pulses.",
            "Theta (physics, rest dynamics, stimulus points, readout zones) is learned by an outer cross-entropy search.",
            "Each task's readout is an exact ridge fit on train streams; test streams are never seen in training.",
        ],
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
