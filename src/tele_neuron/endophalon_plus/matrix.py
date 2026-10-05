"""Experiment matrix: Endophalon vs Endophalon-Plus across tasks and noise levels.

    python -m tele_neuron.endophalon_plus.matrix --output outputs/endophalon_plus_matrix.json
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time
from typing import Any

from tele_neuron.endophalon.config import parse_config as parse_endophalon
from tele_neuron.endophalon.train import train as train_endophalon
from tele_neuron.endophalon_plus.config import load_config as load_plus
from tele_neuron.endophalon_plus.train import train as train_plus


ROOT = Path(__file__).resolve().parents[3]
TASKS = ("parity3", "full_adder", "parity4", "adder2", "mult2")
NOISE_LEVELS = ((0.1, 0.02), (0.4, 0.05), (1.0, 0.1))
ENDOPHALON_POINTS = {
    3: [[0.75, 0.75, 1.5], [3.25, 0.75, 1.5], [2.0, 3.25, 1.5]],
    4: [[0.75, 0.75, 1.5], [3.25, 0.75, 1.5], [0.75, 3.25, 1.5], [3.25, 3.25, 1.5]],
}


def run_endophalon(task: str, position_jitter: float, velocity_jitter: float) -> dict[str, Any]:
    raw = json.loads((ROOT / "configs" / "endophalon" / "full_adder.json").read_text(encoding="utf-8"))
    raw = copy.deepcopy(raw)
    raw["name"] = f"endophalon_{task}"
    raw["task"] = task
    raw.pop("cases", None)
    inputs = 4 if task in {"parity4", "adder2", "mult2"} else 3
    raw["reservoir"]["input"]["points"] = ENDOPHALON_POINTS[inputs]
    raw["noise"]["position_jitter"] = position_jitter
    raw["noise"]["velocity_jitter"] = velocity_jitter
    config = parse_endophalon(raw)
    start = time.perf_counter()
    report = train_endophalon(config)
    return {
        "seconds": time.perf_counter() - start,
        "features": report.feature_count,
        "train": report.train.accuracy,
        "test": report.test.accuracy if report.test else None,
        "baseline_bits_test": report.baseline_test.accuracy if report.baseline_test else None,
    }


def run_plus(task: str, position_jitter: float, velocity_jitter: float, workers: int | None) -> dict[str, Any]:
    config = load_plus(ROOT / "configs" / "endophalon_plus" / f"{task}.json").with_noise(position_jitter, velocity_jitter)
    if workers:
        from dataclasses import replace

        config = replace(config, workers=workers)
    start = time.perf_counter()
    report = train_plus(config)
    controls = {name: splits["test"].accuracy if splits["test"] else None for name, splits in report.controls.items()}
    return {
        "seconds": time.perf_counter() - start,
        "features_per_preset": report.feature_count,
        "assignment": [config.presets[index].name for index in report.assignment],
        "presets_used": len(set(report.assignment)),
        "train": report.train.accuracy,
        "test": report.test.accuracy if report.test else None,
        "controls_test": controls,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Endophalon vs Endophalon-Plus experiment matrix.")
    parser.add_argument("--output", default="outputs/endophalon_plus_matrix.json")
    parser.add_argument("--tasks", nargs="*", default=list(TASKS))
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for task in args.tasks:
        for position_jitter, velocity_jitter in NOISE_LEVELS:
            row = {
                "task": task,
                "position_jitter": position_jitter,
                "velocity_jitter": velocity_jitter,
                "endophalon": run_endophalon(task, position_jitter, velocity_jitter),
                "plus": run_plus(task, position_jitter, velocity_jitter, args.workers),
            }
            rows.append(row)
            print(_line(row), flush=True)
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def _line(row: dict[str, Any]) -> str:
    endo, plus = row["endophalon"], row["plus"]
    controls = plus["controls_test"]
    fixed = max(value for name, value in controls.items() if name.startswith("fixed_"))
    return (
        f"{row['task']:<10} jitter={row['position_jitter']:<4} "
        f"endophalon={endo['test']:.3f} ({endo['seconds']:.0f}s) | "
        f"plus routed={plus['test']:.3f} best_fixed={fixed:.3f} "
        f"gate_only={controls['gate_only']:.3f} bits={controls['baseline_linear_on_bits']:.3f} "
        f"presets_used={plus['presets_used']} ({plus['seconds']:.0f}s)"
    )


if __name__ == "__main__":
    main()
