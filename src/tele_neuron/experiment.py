from __future__ import annotations

import argparse

from tele_neuron.config import ExperimentConfig, load_config
from tele_neuron.input_encoder import parse_bits
from tele_neuron.trainer import (
    apply_threshold,
    print_result,
    run_cases,
    save_result,
    search_best_seed,
)
from tele_neuron.visualize import run_display, save_final_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Tele-Neuron Phase 1 experiment.")
    parser.add_argument("--config", required=True, help="Path to an experiment JSON config.")
    parser.add_argument("--display", action="store_true", help="Show a live x/y display with z as color.")
    parser.add_argument("--bits", help="Input bits to visualize, for example 01 or 11.")
    parser.add_argument("--frames", type=int, help="Animation frame count. Defaults to simulation.steps.")
    parser.add_argument("--save", help="Optional GIF path instead of opening a display window.")
    parser.add_argument("--snapshot", help="Optional PNG path for final K-step positions.")
    parser.add_argument("--search-seeds", help="Inclusive seed range, for example 0:200.")
    parser.add_argument("--output", help="Optional JSON result path.")
    args = parser.parse_args()

    config = load_config(args.config)
    print(
        f"Loaded {config.name}: task={config.task}, "
        f"balls={config.balls.count}, cases={len(config.cases)}"
    )
    if args.search_seeds:
        start, end = _parse_seed_range(args.search_seeds)
        result = search_best_seed(config, seed_start=start, seed_end=end)
        print_result(result)
        if args.output:
            save_result(result, args.output)
            print(f"Saved result to {args.output}")
        return

    result = run_cases(config)
    print_result(result)
    if args.output:
        save_result(result, args.output)
        print(f"Saved result to {args.output}")

    bits = parse_bits(args.bits) if args.bits else _default_visible_bits(config)

    if args.snapshot:
        save_final_snapshot(config, bits, args.snapshot)
        print(f"Saved final snapshot to {args.snapshot}")

    if args.display or args.save:
        config = apply_threshold(config, result.threshold)
        run_display(config, bits, frames=args.frames, save_path=args.save)
        if args.save:
            print(f"Saved display animation to {args.save}")
        return

    print(f"Ready. Use --display --bits {''.join(str(bit) for bit in bits)} to watch it move.")


def _parse_seed_range(value: str) -> tuple[int, int]:
    parts = value.split(":", maxsplit=1)
    if len(parts) != 2:
        raise ValueError("--search-seeds must look like START:END")
    start = int(parts[0])
    end = int(parts[1])
    if end < start:
        raise ValueError("--search-seeds END must be >= START")
    return start, end


def _default_visible_bits(config: ExperimentConfig) -> tuple[int, ...]:
    for case in config.cases:
        if any(case.bits):
            return case.bits
    return config.cases[0].bits


if __name__ == "__main__":
    main()
