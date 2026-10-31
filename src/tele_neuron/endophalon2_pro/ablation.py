"""Where does the nonlinearity come from: the physics or the readout zones?

Soft readout zones are Gaussian, so the zones are a nonlinear function of node
positions. A trained model might compute XOR in the observation rather than in
the physics. This script rescores a trained theta four ways, on the same test
streams:

- soft zones (the model as trained)
- **linear observation**: a linear readout on raw node displacements (N x 3),
  which is linear in the physical state
- each of the above with the cubic spring term set to zero

If XOR stays high under linear observation, the physics did the nonlinear work.

    python -m tele_neuron.endophalon2_pro.ablation --config configs/endophalon2_pro/regular.json \\
        --model models/endophalon2_pro/teleneuron_endophalon2_pro_regular_multi.json
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import numpy as np

from tele_neuron.endophalon2.evaluate import StreamData, fit_score
from tele_neuron.endophalon2.tasks import stream_bits
from tele_neuron.endophalon2_pro.config import E2PConfig, load_config
from tele_neuron.endophalon2_pro.engine import EventRow, run_events
from tele_neuron.endophalon2_pro.theta import SpringTheta


def observe_both(config: E2PConfig, theta: SpringTheta, streams: range, *, reset: bool = False) -> tuple[StreamData, StreamData, bool]:
    """Soft-zone features and node-displacement features at the readout delay, for every event."""
    spec, pulses, seed = config.stream, config.stream.pulses, config.physics.seed
    bits = {s: stream_bits(seed, s, pulses) for s in streams}
    if reset:
        rows = [
            EventRow(0, bits[s][k : k + 1], np.array([spec.warmup]), 1_000_000 + s * pulses + k, 1_000_000 + s * pulses + k)
            for s in streams
            for k in range(pulses)
        ]
        run_spec = replace(spec, pulses=1)
    else:
        rows = [EventRow(0, bits[s], spec.onsets(seed, s), s, s) for s in streams]
        run_spec = spec
    record = run_events(
        config.network, config.physics, [theta], rows, run_spec,
        workers=config.workers, chunk_size=config.chunk_size, record_nodes=config.readout_delay,
    )
    assert record.nodes is not None
    shape = (len(streams), pulses)
    zones = record.zones[:, :, config.readout_delay - 1].reshape(*shape, 1, -1)
    nodes = record.nodes.reshape(*shape, 1, -1)
    common = dict(bits=np.stack([bits[s] for s in streams]), energy=record.energy.reshape(shape), wall=record.stretch.reshape(shape), streams=tuple(streams))
    return StreamData(features=zones, **common), StreamData(features=nodes, **common), bool(record.finite.all())


def ablate(config: E2PConfig, theta: SpringTheta) -> dict[str, Any]:
    train, _, test = config.stream_ranges()
    streams = range(0, test.stop)
    out: dict[str, Any] = {}
    for label, variant in (("trained", theta), ("trained_no_cubic", replace(theta, k_cubic=0.0))):
        for mode in ("continuous", "reset"):
            zones, nodes, finite = observe_both(config, variant, streams, reset=(mode == "reset"))
            for observation, data in (("soft_zones", zones), ("linear_nodes", nodes)):
                out[f"{label}/{mode}/{observation}"] = {
                    task: fit_score(config, data, task, train, test, 1)[0] for task in config.tasks
                } | {"_finite": finite}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Endophalon-2-Pro nonlinearity ablation.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True, help="Model card with a trained theta.")
    parser.add_argument("--output")
    args = parser.parse_args()
    config = load_config(args.config)
    theta = SpringTheta.from_payload(json.loads(Path(args.model).read_text(encoding="utf-8"))["theta"])
    result = ablate(config, theta)
    for key, scores in result.items():
        print(f"{key:<42} " + " ".join(f"{t}={v:.2f}" for t, v in scores.items() if not t.startswith("_")))
    if args.output:
        Path(args.output).write_text(json.dumps({"config": args.config, "model": args.model, "result": result}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
