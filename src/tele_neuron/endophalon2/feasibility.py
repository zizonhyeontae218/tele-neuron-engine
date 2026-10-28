"""Stage 1: can a running physics system compute at all, before any physics training?

For each physics variant and readout, every stream task is scored on held-out
test streams. The readout delay n is picked on validation streams. Each result
is compared with:

- the reset baseline (same physics, but every pulse starts fresh)
- a physics-free readout on the last three bit pairs
- the majority-class rate

    python -m tele_neuron.endophalon2.feasibility --config configs/endophalon2/base.json
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import numpy as np

from tele_neuron.endophalon2.config import E2Config, load_config
from tele_neuron.endophalon2.evaluate import StreamData, collect, dataset, fit_score


READOUTS: dict[str, tuple[tuple[int, int, int], ...]] = {
    "zones4": (),  # the theta's 4 soft readout zones
    "grid9": ((3, 3, 1),),
    "grid36": ((6, 6, 1),),
}


def variants(config: E2Config) -> dict[str, Any]:
    base = config.theta_init
    return {
        "no_rest": replace(base, anchor=1e-5, bath=1e-5),
        "anchor": replace(base, bath=1e-5),
        "anchor_bath": base,
    }


def score_all(config: E2Config, data: StreamData, *, history: bool = False) -> dict[str, Any]:
    train, val, test = config.stream_ranges()
    half = config.washout + (config.stream.pulses - config.washout) // 2
    out: dict[str, Any] = {}
    for task in config.tasks:
        delays = [1] if history else range(1, config.stream.interval + 1)
        best = max(delays, key=lambda n: fit_score(config, data, task, train, val, n, history=history)[0])
        accuracy, _, _ = fit_score(config, data, task, train, test, best, history=history)
        early, _, _ = fit_score(config, data, task, train, test, best, history=history, score_pulses=slice(config.washout, half))
        late, _, _ = fit_score(config, data, task, train, test, best, history=history, score_pulses=slice(half, None))
        _, y = dataset(config, data, task, test, best, history=history)
        out[task] = {
            "delay": best,
            "test": accuracy,
            "early": early,
            "late": late,
            "majority": float(max(y.mean(), 1 - y.mean())),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Endophalon-2 stage 1 feasibility check.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="outputs/endophalon2_feasibility.json")
    args = parser.parse_args()
    config = load_config(args.config)
    streams = range(0, sum(len(r) for r in config.stream_ranges()))

    results: dict[str, Any] = {"config": args.config, "variants": {}}
    history_data = collect(config, [config.theta_init], range(0, 1))[0]  # only bits are needed here
    history_data = replace(history_data, bits=np.stack([_bits(config, s) for s in streams]), streams=tuple(streams))
    results["history_baseline"] = score_all(config, history_data, history=True)
    _print("history (no physics)", results["history_baseline"])

    for name, theta in variants(config).items():
        entry: dict[str, Any] = {"theta": theta.to_payload()}
        for readout, grids in READOUTS.items():
            for mode in ("continuous", "reset"):
                data = collect(config, [theta], streams, reset=(mode == "reset"), grids=grids)[0]
                scored = score_all(config, data)
                scored["_energy_early_late"] = [float(data.energy[:, : config.washout + 4].mean()), float(data.energy[:, -4:].mean())]
                scored["_wall_early_late"] = [float(data.wall[:, : config.washout + 4].mean()), float(data.wall[:, -4:].mean())]
                entry[f"{readout}/{mode}"] = scored
                _print(f"{name:<11} {readout:<6} {mode:<10}", scored)
        results["variants"][name] = entry
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


def _bits(config: E2Config, stream: int) -> np.ndarray:
    from tele_neuron.endophalon2.tasks import stream_bits

    return stream_bits(config.world.seed, stream, config.stream.pulses)


def _print(label: str, scored: dict[str, Any]) -> None:
    cells = " ".join(
        f"{task}={value['test']:.2f}(n{value['delay']},{value['early']:.2f}/{value['late']:.2f})"
        for task, value in scored.items()
        if not task.startswith("_")
    )
    extra = ""
    if "_energy_early_late" in scored:
        e, w = scored["_energy_early_late"], scored["_wall_early_late"]
        extra = f" | E {e[0]:.4f}->{e[1]:.4f} wall {w[0]:.3f}->{w[1]:.3f}"
    print(f"{label} {cells}{extra}", flush=True)


if __name__ == "__main__":
    main()
