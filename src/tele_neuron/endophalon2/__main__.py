from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from tele_neuron.endophalon2 import FAMILY
from tele_neuron.endophalon2.config import load_config
from tele_neuron.endophalon2.model import save_model
from tele_neuron.endophalon2.train import train


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Train {FAMILY} for continuous computation.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--tasks", nargs="*", help="Tasks the outer loop optimizes (default: all config tasks).")
    parser.add_argument("--generations", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--random-baseline", action="store_true")
    parser.add_argument("--output", help="Model card JSON path.")
    parser.add_argument("--model-id")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.generations is not None:
        config = replace(config, outer={**config.outer, "generations": args.generations})
    if args.workers:
        config = replace(config, workers=args.workers)
    tasks = tuple(args.tasks) if args.tasks else config.tasks
    print(
        f"{FAMILY} {config.name}: train tasks={tasks}, zones={config.zones}, delay n={config.readout_delay}, "
        f"interval={config.stream.interval}, pulses/stream={config.stream.pulses}, theta dim={config.space.dim}"
    )
    report = train(config, tasks=tasks, random_baseline=args.random_baseline, log=print)
    print(f"  J {report.fitness_init.j:.4f} -> {report.fitness.j:.4f} ({report.evaluations} thetas)")
    for key in ("trained/continuous", "initial/continuous", "trained/reset", "initial/reset", "history_baseline"):
        cells = " ".join(f"{task}={value['test']:.2f}" for task, value in report.test[key].items())
        print(f"  {key:<20} {cells}")
    if report.random_search:
        print("  random search        " + " ".join(f"{t}={v:.2f}" for t, v in report.random_search["test"].items()))
    if args.output:
        save_model(args.model_id or config.name, Path(args.config).as_posix(), config, report, tasks, args.output)
        print(f"Saved model card to {args.output}")


if __name__ == "__main__":
    main()
