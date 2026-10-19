from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from tele_neuron.endophalon_pro import FAMILY
from tele_neuron.endophalon_pro.config import load_config
from tele_neuron.endophalon_pro.model import model_from_report, save_model
from tele_neuron.endophalon_pro.train import train


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Train the physics and readout of a {FAMILY} model.")
    parser.add_argument("--config", required=True, help="Path to an Endophalon-Pro JSON config.")
    parser.add_argument("--output", help="Optional model card JSON path.")
    parser.add_argument("--model-id", help="Model id stored in the card. Defaults to the config name.")
    parser.add_argument("--workers", type=int, help="Override runtime.workers (parallel processes).")
    parser.add_argument("--generations", type=int, help="Override outer.generations.")
    parser.add_argument("--jitter", type=float, nargs=2, metavar=("POSITION", "VELOCITY"), help="Override noise jitters.")
    parser.add_argument("--grid", help="Override readout.grid, for example 2x2x1.")
    parser.add_argument("--random-baseline", action="store_true", help="Also run random search with the same budget.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.workers:
        config = replace(config, workers=max(1, args.workers))
    if args.generations is not None:
        config = replace(config, outer=replace(config.outer, generations=max(0, args.generations)))
    if args.jitter:
        config = config.with_noise(*args.jitter)
    if args.grid:
        grid = tuple(int(v) for v in args.grid.split("x"))
        if len(grid) != 3 or min(grid) <= 0:
            parser.error("--grid must look like 2x2x1")
        config = replace(config, readout=replace(config.readout, grid=grid))
    print(
        f"{FAMILY} {config.name}: task={config.task}, balls={config.world.ball_count}, "
        f"readout grid={config.readout.grid} at step {config.readout.step}, theta dim={config.space.dim}"
    )
    report = train(config, random_baseline=args.random_baseline, log=print)
    print(f"  J: initial {report.fitness_init.j:.4f} -> trained {report.fitness.j:.4f} ({report.evaluations} thetas)")
    for label in ("l_val", "r_noise", "r_stability", "r_complexity", "val_accuracy"):
        print(f"    {label:<13} {getattr(report.fitness_init, label):.4f} -> {getattr(report.fitness, label):.4f}")
    print("  test accuracy: " + " ".join(f"{name}={value:.2%}" for name, value in report.test.items()))
    for row in report.sweep:
        print(f"  readout {row['features']:>4} features: initial {row['initial']:.2%}  trained {row['trained']:.2%}")
    if args.output:
        model = model_from_report(args.model_id or config.name, Path(args.config).as_posix(), config, report)
        save_model(model, config, args.output)
        print(f"Saved model card to {args.output}")


if __name__ == "__main__":
    main()
