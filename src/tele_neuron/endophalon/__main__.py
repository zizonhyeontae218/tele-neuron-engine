from __future__ import annotations

import argparse
from pathlib import Path

from tele_neuron.endophalon import FAMILY
from tele_neuron.endophalon.config import load_config
from tele_neuron.endophalon.model import model_from_report, save_model
from tele_neuron.endophalon.train import SplitMetrics, train


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Train and evaluate a {FAMILY} model.")
    parser.add_argument("--config", required=True, help="Path to an Endophalon JSON config.")
    parser.add_argument("--output", help="Optional model card JSON path.")
    parser.add_argument("--model-id", help="Model id stored in the card. Defaults to the config name.")
    args = parser.parse_args()

    config = load_config(args.config)
    print(
        f"{FAMILY} {config.name}: task={config.task}, balls={config.reservoir.ball_count}, "
        f"grid={config.observer.grid}, taps={config.observer.taps}"
    )
    report = train(config)
    print(f"  features={report.feature_count} ridge_alpha={config.readout.ridge_alpha}")
    _print_split("reservoir train", report.train)
    _print_split("reservoir test ", report.test)
    _print_split("baseline train ", report.baseline_train)
    _print_split("baseline test  ", report.baseline_test)

    if args.output:
        model = model_from_report(args.model_id or config.name, Path(args.config).as_posix(), config, report)
        save_model(model, args.output)
        print(f"Saved model card to {args.output}")


def _print_split(label: str, metrics: SplitMetrics | None) -> None:
    if metrics is None:
        return
    outputs = " ".join(f"{name}={value:.2%}" for name, value in metrics.output_accuracy.items())
    cases = " ".join(f"{bits}:{value:.0%}" for bits, value in metrics.case_accuracy.items())
    print(f"  {label} n={metrics.samples} acc={metrics.accuracy:.2%} [{outputs}] cases {cases}")


if __name__ == "__main__":
    main()
