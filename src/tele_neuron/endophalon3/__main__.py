"""python -m tele_neuron.endophalon3 --config configs/endophalon3/level_a.json [--probe | train options]"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from tele_neuron.endophalon3 import variant_name
from tele_neuron.endophalon3.config import load_config
from tele_neuron.endophalon3.core import capacity_probe, collect, report_payload, single_readout, train
from tele_neuron.endophalon3.isa import OPS


def main() -> None:
    parser = argparse.ArgumentParser(description="Teleneuron-Endophalon-3 (train level A or B separately).")
    parser.add_argument("--config", required=True)
    parser.add_argument("--probe", action="store_true", help="Stage 1 only: capacity probe of theta_init, no training.")
    parser.add_argument("--no-regions", action="store_true", help="Ablation: homogeneous physics (no slow region).")
    parser.add_argument("--generations", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--random-baseline", action="store_true")
    parser.add_argument("--output", help="Model card (training) or probe result JSON path. Never overwrite published files.")
    args = parser.parse_args()

    config = load_config(args.config)
    outer = dict(config.outer)
    if args.generations is not None:
        outer["generations"] = args.generations
    if args.seed is not None:
        outer["seed"] = args.seed
    config = replace(config, outer=outer, workers=args.workers or config.workers, regions=None if args.no_regions else config.regions)
    name = variant_name(config.level)
    print(f"{name}: encoding={config.encoding} channels={config.channels} ops={config.ops} zones={config.zones} "
          f"slow nodes={int(config.slow.sum())}/{config.network.nodes} theta dim={config.space.dim} seed={outer.get('seed', 1)}")

    if args.probe:
        result = capacity_probe(config, config.theta_init)
        print(json.dumps(result, indent=2))
        if args.output:
            Path(args.output).write_text(json.dumps({"config": args.config, "regions": config.regions, **result}, indent=2) + "\n")
        return

    report = train(config, random_baseline=args.random_baseline, log=print)
    print(f"  J {report.fitness_init.j:.4f} -> {report.fitness.j:.4f} ({report.evaluations} thetas)")
    for who in ("trained", "initial"):
        for key in ("single_readout", "single_readout_plus_instruction", "separate_readouts", "linear_nodes"):
            v = report.test[who][key]
            print(f"  {who:<8} {key:<32} {v['accuracy']:.3f} " + " ".join(f"{o}={a:.2f}" for o, a in v["per_op"].items()))
    if args.output:
        train_r, _, _ = config.stream_ranges()
        [data] = collect(config, [report.theta], train_r)
        readout = single_readout(config, data, train_r, train_r)["readout"]
        card = {"family": name, "model_id": Path(args.output).stem, "config": Path(args.config).as_posix(),
                "operations": list(config.ops), "all_ops": list(OPS), "input_channels": list(config.channels),
                "theta": report.theta.to_payload(), "theta_init": config.theta_init.to_payload(),
                "readout": readout.to_payload(), "metrics": report_payload(config, report)}
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(card, indent=2) + "\n", encoding="utf-8")
        print(f"Saved model card to {args.output}")


if __name__ == "__main__":
    main()
