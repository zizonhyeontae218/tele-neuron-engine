from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from typing import Any

from tele_neuron.endophalon2_promax import variant_name
from tele_neuron.endophalon2_promax.config import ProMaxConfig, load_config
from tele_neuron.endophalon2_promax.core import Report, collect, single_readout, train
from tele_neuron.endophalon2_promax.tasks import OPS


def save_model(model_id: str, config_path: str, config: ProMaxConfig, report: Report, path: str | Path) -> None:
    """Model card: trained theta, the single readout (fit on train streams), and all metrics."""
    train_range, _, _ = config.stream_ranges()
    [data] = collect(config, [report.theta], train_range)
    readout = single_readout(config, data, train_range, train_range)["readout"]
    payload: dict[str, Any] = {
        "family": variant_name(config.zones),
        "model_id": model_id,
        "config": config_path,
        "operations": list(OPS),
        "input_channels": ["a", "b"] + [f"opcode:{name}" for name in OPS],
        "theta": report.theta.to_payload(),
        "theta_init": config.theta_init.to_payload(),
        "readout": readout.to_payload(),
        "metrics": {
            "readout_zones": config.zones,
            "readout_delay": config.readout_delay,
            "events": asdict(config.stream),
            "evaluations": report.evaluations,
            "fitness_initial": asdict(report.fitness_init),
            "fitness_trained": asdict(report.fitness),
            "test": report.test,
            "random_search": report.random_search,
            "history": list(report.history),
        },
        "notes": [
            "One spring network and ONE linear readout. The opcode input picks the operation for each event.",
            "Theta is learned by an outer cross-entropy search, warm-started from Endophalon-2-Pro.",
            "Test streams are never seen in training.",
        ],
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Teleneuron-Endophalon-2-ProMax8 / ProMax16.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--generations", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--random-baseline", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--model-id")
    args = parser.parse_args()

    config = load_config(args.config)
    outer = dict(config.outer)
    if args.generations is not None:
        outer["generations"] = args.generations
    if args.seed is not None:
        outer["seed"] = args.seed
    config = replace(config, outer=outer, workers=args.workers or config.workers)
    name = variant_name(config.zones)
    print(f"{name}: ops={OPS}, zones={config.zones}, delay n={config.readout_delay}, theta dim={config.space.dim}, seed={outer.get('seed', 1)}")
    report = train(config, random_baseline=args.random_baseline, log=print)
    print(f"  J {report.fitness_init.j:.4f} -> {report.fitness.j:.4f} ({report.evaluations} thetas)")
    for who in ("trained", "initial"):
        for key in ("single_readout", "single_readout_plus_opcode", "separate_readouts", "linear_nodes"):
            v = report.test[who][key]
            per_op = " ".join(f"{o}={a:.2f}" for o, a in v["per_op"].items())
            extra = f" (first half {v['early']:.2f} / second half {v['late']:.2f})" if "early" in v else ""
            print(f"  {who:<8} {key:<27} {v['accuracy']:.3f} [{per_op}]{extra}")
    print(f"  history baseline           {report.test['history_baseline']['accuracy']:.3f}")
    if report.random_search:
        print(f"  random search              {report.random_search['test']['accuracy']:.3f}")
    if args.output:
        save_model(args.model_id or name, Path(args.config).as_posix(), config, report, args.output)
        print(f"Saved model card to {args.output}")


if __name__ == "__main__":
    main()
