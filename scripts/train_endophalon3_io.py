"""Train one current Endophalon 3 module with only stimulus and readout geometry open."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from tele_neuron.endophalon3_software.model import Module
from tele_neuron.endophalon3_software.theta import NAMES
from tele_neuron.endophalon3_software.training import (
    OPS, advance, fitness, flatten, metrics, new_state, observe, subset,
)

SOURCE = Path("models/endophalon3_software/current.json")
GENERATIONS = 120
POPULATION = 24
WEAK_OPS = {"A": ("recall2",), "B": ("recall2", "xor_mem", "parity3w")}
WEAK_WEIGHT = 0.5
TUNABLE = ("pulse_strength", "stimulus_width", "stimulus_points", "readout_centers")


def atomic_json(path, payload):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def search_space(model):
    theta = model.config.theta_init
    channels = len(theta.stimulus_points)
    active_channels = range(2 if model.level == "A" else 0, channels)
    centers_start = len(NAMES) + channels * 3
    indices = [
        NAMES.index("pulse_strength"),
        NAMES.index("stimulus_width"),
        *(len(NAMES) + 3 * channel + axis for channel in active_channels for axis in range(3)),
        *range(centers_start, centers_start + 3 * len(theta.readout_centers)),
    ]
    return model.config.space.encode(theta), np.array(indices), active_channels


def build_candidate(base, coordinates, encoded, indices, active_channels):
    vector = encoded.copy()
    vector[indices] = coordinates
    decoded = base.config.space.decode(vector).to_payload()
    raw = deepcopy(base.raw)
    theta = raw["theta_init"]
    for key in TUNABLE:
        if key == "stimulus_points":
            for channel in active_channels:
                theta[key][channel] = decoded[key][channel]
        else:
            theta[key] = decoded[key]
    return Module(raw)


def score_model(model, train, validation):
    data = observe(model, [*train, *validation])
    fitted, result = fitness(model, data, train, validation)
    x, y, ops = flatten(fitted, subset(data, validation))
    squared_error = (fitted.head.scores(x)[:, 0] - y[:, 0]) ** 2
    weak_mse = {
        name: float(squared_error[ops == OPS.index(name)].mean())
        for name in WEAK_OPS[model.level]
    }
    result["original_j"] = result["j"]
    result["weak_mse"] = weak_mse
    result["j"] += WEAK_WEIGHT * float(np.mean(list(weak_mse.values())))
    if not np.isfinite(result["j"]):
        raise FloatingPointError("Non-finite selection objective")
    return fitted, result


def evaluate_candidate(args):
    raw, coordinates, encoded, indices, active_channels, train, validation = args
    try:
        model = build_candidate(Module(raw), coordinates, encoded, indices, active_channels)
        return score_model(model, train, validation)[1]
    except (FloatingPointError, np.linalg.LinAlgError) as exc:
        return {"j": 1e100, "finite": False, "failure": str(exc)}


def run(level, output, workers, preflight):
    source = SOURCE.read_bytes()
    base = Module.from_payload(json.loads(source)["modules"][level])
    if base.config.encoding != "binary":
        raise ValueError("This experiment requires the current binary A/B modules")
    train = base.raw["training_stream_ids"]["train"]
    validation = base.raw["training_stream_ids"]["validation"]
    first_fresh = 62000 if level == "A" else 62100
    confirmation = list(range(first_fresh, first_fresh + 64))
    old_ids = {sid for ids in base.raw["training_stream_ids"].values() for sid in ids}
    if set(confirmation) & old_ids or set(train) & set(validation):
        raise ValueError("Training, validation and new confirmation must be disjoint")
    encoded, indices, active_channels = search_space(base)
    initial = encoded[indices]
    settings = {
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "bundle_sha256": hashlib.sha256(source).hexdigest(),
        "level": level,
        "generations": GENERATIONS,
        "population": POPULATION,
        "train": train,
        "validation": validation,
        "confirmation": confirmation,
        "weak_ops": list(WEAK_OPS[level]),
        "weak_weight": WEAK_WEIGHT,
        "coordinates": indices.tolist(),
        "numpy": np.__version__,
    }
    if preflight:
        probe = build_candidate(base, initial + 0.01, encoded, indices, active_channels)
        original_theta = base.raw["theta_init"]
        for name, value in probe.raw["theta_init"].items():
            if name not in TUNABLE:
                assert value == original_theta[name], name
        for channel in set(range(len(original_theta["stimulus_points"]))) - set(active_channels):
            assert probe.raw["theta_init"]["stimulus_points"][channel] == original_theta["stimulus_points"][channel]
        _, trial = score_model(probe, train, validation)
        print(f"{level} preflight: {len(indices)} coordinates, objective {trial['j']:.6f}")
        return

    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "checkpoint.json"
    settings_path = output / "settings.json"
    model_path = output / "model.json"
    if model_path.exists():
        raise FileExistsError("Completed run already exists")
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise ValueError("Checkpoint settings differ")
    atomic_json(settings_path, settings)
    if checkpoint.exists():
        state = json.loads(checkpoint.read_text())
    else:
        state = new_state(base.raw)
        state.update(mean=initial.tolist(), best_u=initial.tolist(), sigma=[0.5] * len(indices))

    with ProcessPoolExecutor(max_workers=workers) as pool:
        def score(rows):
            args = ((base.raw, row, encoded, indices, active_channels, train, validation) for row in rows)
            return list(pool.map(evaluate_candidate, args))

        while state["generation"] < GENERATIONS:
            state = advance(state, score, population=POPULATION)
            best = state["best"]
            if not best["finite"]:
                raise FloatingPointError("Initial candidate is invalid")
            atomic_json(checkpoint, state)
            print(f"{level} generation {state['generation']}/{GENERATIONS}, "
                  f"{state['evaluations']} evaluations, J={best['j']:.6f}, "
                  f"validation={best['validation']['accuracy']:.2%}", flush=True)

    assert state["evaluations"] == 1 + GENERATIONS * POPULATION
    selected = build_candidate(base, state["best_u"], encoded, indices, active_channels)
    fitted, result = score_model(selected, train, validation)
    np.testing.assert_allclose(result["j"], state["best"]["j"], atol=1e-10, rtol=1e-10)
    fitted.provenance = {"experiment": "io-only120", "settings": settings}
    candidate_data = observe(fitted, confirmation)
    baseline_data = observe(base, confirmation)
    measured = metrics(fitted, candidate_data)
    baseline = metrics(base, baseline_data)
    target = 0.90 if level == "A" else 0.80
    report = {
        "evaluations": state["evaluations"],
        "generations": state["generation"],
        "failed_candidates": sum(row["failed"] for row in state["history"]),
        "selected_validation": result,
        "baseline_confirmation": baseline,
        "confirmation": measured,
        "user_target": target,
        "meets_user_target": measured["accuracy"] >= target,
        "meets_strict_gate": measured["accuracy"] >= .9 and all(
            row["accuracy"] >= .85 for row in measured["per_op"].values()
        ),
    }
    np.savez_compressed(output / "confirmation_observations.npz", **candidate_data)
    np.savez_compressed(output / "baseline_observations.npz", **baseline_data)
    atomic_json(output / "evaluation.json", report)
    atomic_json(output / "model.pending.json", fitted.to_payload())
    reloaded = Module.load(output / "model.pending.json")
    assert metrics(reloaded, candidate_data) == measured
    (output / "model.pending.json").replace(model_path)
    print(f"{level} confirmation {baseline['accuracy']:.2%} -> "
          f"{measured['accuracy']:.2%}, target {target:.0%}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", choices=("A", "B"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    run(args.level, args.output_dir, args.workers, args.preflight)
