from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from typing import Any

from tele_neuron.endophalon2.evaluate import fit_score
from tele_neuron.endophalon2_pro import FAMILY
from tele_neuron.endophalon2_pro.config import E2PConfig, load_config
from tele_neuron.endophalon2_pro.core import Report, collect, train


def save_model(model_id: str, config_path: str, config: E2PConfig, report: Report, tasks: tuple[str, ...], path: str | Path) -> None:
    """Model card: trained theta, one readout per task (fit on train streams), and all metrics."""
    train_range, _, _ = config.stream_ranges()
    [item] = collect(config, [report.theta], train_range)
    readouts = {
        task: fit_score(config, item.data, task, train_range, train_range, config.readout_delay)[2].to_payload()
        for task in config.tasks
    }
    payload: dict[str, Any] = {
        "family": FAMILY,
        "model_id": model_id,
        "config": config_path,
        "theta": report.theta.to_payload(),
        "theta_init": config.theta_init.to_payload(),
        "readouts": readouts,
        "metrics": {
            "trained_tasks": list(tasks),
            "readout_delay": config.readout_delay,
            "events": asdict(config.stream),
            "network": {"nodes": config.network.nodes, "edges": config.network.edges, "lattice": list(config.lattice)},
            "evaluations": report.evaluations,
            "fitness_initial": asdict(report.fitness_init),
            "fitness_trained": asdict(report.fitness),
            "test": report.test,
            "random_search": report.random_search,
            "history": list(report.history),
        },
        "notes": [
            "Nonlinear spring network, never reset. Input events arrive at their own times.",
            "Theta (springs, damping, rest dynamics, stimulus points, readout zones) is learned by an outer cross-entropy search.",
            "Each task's readout is an exact ridge fit on train streams. Test streams are never seen in training.",
        ],
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Train {FAMILY}.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--tasks", nargs="*", help="Tasks the outer loop optimizes (default: all config tasks).")
    parser.add_argument("--generations", type=int)
    parser.add_argument("--seed", type=int, help="Override outer.seed.")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--random-baseline", action="store_true")
    parser.add_argument("--output", help="Model card JSON path.")
    parser.add_argument("--model-id")
    args = parser.parse_args()

    config = load_config(args.config)
    outer = dict(config.outer)
    if args.generations is not None:
        outer["generations"] = args.generations
    if args.seed is not None:
        outer["seed"] = args.seed
    config = replace(config, outer=outer, workers=args.workers or config.workers)
    tasks = tuple(args.tasks) if args.tasks else config.tasks
    spec = config.stream
    print(
        f"{FAMILY} {config.name}: train tasks={tasks}, nodes={config.network.nodes}, springs={config.network.edges}, "
        f"zones={config.zones}, delay n={config.readout_delay}, gaps {spec.gap_min}-{spec.gap_max}, "
        f"events/stream={spec.pulses}, theta dim={config.space.dim}, seed={outer.get('seed', 1)}"
    )
    report = train(config, tasks=tasks, random_baseline=args.random_baseline, log=print)
    print(f"  J {report.fitness_init.j:.4f} -> {report.fitness.j:.4f} ({report.evaluations} thetas)")
    for key in ("trained/continuous", "initial/continuous", "trained/reset", "initial/reset", "history_baseline"):
        cells = " ".join(f"{task}={value['test']:.2f}" for task, value in report.test[key].items())
        print(f"  {key:<20} {cells}")
    print(f"  drift test: trained {report.test['trained/drift']:.3f} initial {report.test['initial/drift']:.3f}")
    if report.random_search:
        print("  random search        " + " ".join(f"{t}={v:.2f}" for t, v in report.random_search["test"].items()))
    if args.output:
        save_model(args.model_id or config.name, Path(args.config).as_posix(), config, report, tasks, args.output)
        print(f"Saved model card to {args.output}")


if __name__ == "__main__":
    main()
