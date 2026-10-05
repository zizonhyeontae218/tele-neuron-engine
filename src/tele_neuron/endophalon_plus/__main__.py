from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from tele_neuron.endophalon_plus import FAMILY
from tele_neuron.endophalon_plus.config import load_config
from tele_neuron.endophalon_plus.model import model_from_report, save_model
from tele_neuron.endophalon_plus.train import SplitMetrics, train


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Train and evaluate a {FAMILY} model.")
    parser.add_argument("--config", required=True, help="Path to an Endophalon-Plus JSON config.")
    parser.add_argument("--output", help="Optional model card JSON path.")
    parser.add_argument("--model-id", help="Model id stored in the card. Defaults to the config name.")
    parser.add_argument("--workers", type=int, help="Override runtime.workers (parallel processes).")
    parser.add_argument("--jitter", type=float, nargs=2, metavar=("POSITION", "VELOCITY"), help="Override noise jitters.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.workers:
        config = replace(config, workers=max(1, args.workers))
    if args.jitter:
        config = config.with_noise(*args.jitter)
    world = config.world
    print(
        f"{FAMILY} {config.name}: task={config.task}, balls={world.ball_count}, "
        f"presets={[preset.name for preset in config.presets]}, grids={config.observer.grids}, "
        f"taps={config.observer.taps}, jitter=({world.position_jitter}, {world.velocity_jitter})"
    )
    report = train(config)
    routes = " ".join(
        f"{''.join(map(str, bits))}->{config.presets[preset].name}"
        for (bits, _), preset in zip(config.cases, report.assignment, strict=True)
    )
    print(f"  features/preset={report.feature_count} gate evaluations={report.gate_evaluations}")
    print(f"  gate: {routes}")
    _print_split("routed train", report.train)
    _print_split("routed test ", report.test)
    for name, splits in report.controls.items():
        _print_split(f"{name} test", splits["test"])

    if args.output:
        model = model_from_report(args.model_id or config.name, Path(args.config).as_posix(), config, report)
        save_model(model, config, args.output)
        print(f"Saved model card to {args.output}")


def _print_split(label: str, metrics: SplitMetrics | None) -> None:
    if metrics is None:
        return
    outputs = " ".join(f"{name}={value:.2%}" for name, value in metrics.output_accuracy.items())
    print(f"  {label:<34} n={metrics.samples} acc={metrics.accuracy:.2%} [{outputs}]")


if __name__ == "__main__":
    main()
