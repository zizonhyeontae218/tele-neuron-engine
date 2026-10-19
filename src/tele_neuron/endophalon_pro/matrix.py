"""Endophalon-Pro experiment matrix: does training the physics shrink the readout it needs?

    python -m tele_neuron.endophalon_pro.matrix --output outputs/endophalon_pro_matrix.json
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time
from typing import Any

from tele_neuron.endophalon_pro.config import ObjectiveSpec, load_config
from tele_neuron.endophalon_pro.train import train


ROOT = Path(__file__).resolve().parents[3]
TASKS = ("parity3", "full_adder", "parity4", "adder2")
READOUT_GRIDS = ((2, 2, 1), (3, 3, 1))


def run(task: str, grid: tuple[int, int, int], *, regularized: bool, workers: int | None) -> dict[str, Any]:
    config = load_config(ROOT / "configs" / "endophalon_pro" / f"{task}.json")
    config = replace(config, readout=replace(config.readout, grid=grid))
    if not regularized:
        config = replace(config, objective=ObjectiveSpec(0.0, 0.0, 0.0))
    if workers:
        config = replace(config, workers=workers)
    start = time.perf_counter()
    report = train(config, random_baseline=regularized)
    return {
        "task": task,
        "readout_grid": list(grid),
        "readout_features": grid[0] * grid[1] * grid[2],
        "regularized": regularized,
        "seconds": time.perf_counter() - start,
        "evaluations": report.evaluations,
        "theta": report.theta.to_payload(),
        "fitness_initial": asdict(report.fitness_init),
        "fitness_trained": asdict(report.fitness),
        "test": report.test,
        "sweep": list(report.sweep),
        "random_search": report.random_search,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Endophalon-Pro experiment matrix.")
    parser.add_argument("--output", default="outputs/endophalon_pro_matrix.json")
    parser.add_argument("--tasks", nargs="*", default=list(TASKS))
    parser.add_argument("--workers", type=int)
    parser.add_argument("--grids", nargs="*", help="Readout grids like 2x2x1 3x3x1 (default: both).")
    parser.add_argument("--ablation", action="store_true", help="Also run every cell with all lambdas set to 0.")
    args = parser.parse_args()

    grids = [tuple(int(v) for v in item.split("x")) for item in args.grids] if args.grids else list(READOUT_GRIDS)
    plan = [(task, grid, True) for task in args.tasks for grid in grids]
    if args.ablation:
        plan += [(task, grid, False) for task in args.tasks for grid in grids]
    rows: list[dict[str, Any]] = []
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for task, grid, regularized in plan:
        row = run(task, grid, regularized=regularized, workers=args.workers)
        rows.append(row)
        test = row["test"]
        print(
            f"{task:<10} k={row['readout_features']:<3} reg={regularized!s:<5} "
            f"initial={test['initial']:.3f} random={test.get('random_search', float('nan')):.3f} trained={test['trained']:.3f} "
            f"bits={test['bits_linear']:.3f} J {row['fitness_initial']['j']:.3f}->{row['fitness_trained']['j']:.3f} "
            f"({row['seconds']:.0f}s)",
            flush=True,
        )
        output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
