"""Execute each available model CLI using bounded, temporary smoke configs."""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MODELS = (
    ("endophalon", "xor"),
    ("endophalon_plus", "full_adder"),
    ("endophalon_pro", "full_adder"),
    ("endophalon2", "base"),
    ("endophalon2_pro", "regular"),
    ("endophalon2_promax", "promax8"),
    ("endophalon2_promax", "promax16"),
)


def reduced(raw: dict, family: str) -> dict:
    """Keep scientific defaults intact on disk; reduce only a temporary payload."""
    raw["runtime"] = {"workers": 1, "chunk_size": 64}
    if family == "endophalon":
        raw["reservoir"]["balls"]["count"] = 32
        raw["reservoir"]["simulation"]["steps"] = 12
        raw["observer"]["taps"] = [6, 12]
        raw["noise"].update(train_realizations=6, test_realizations=2)
    elif family == "endophalon_plus":
        raw["task"] = "xor"
        raw["world"]["input"]["points"] = [[1.0, 1.0, 1.5], [4.0, 1.0, 1.5]]
        raw["world"]["balls"]["count"] = 40
        raw["world"]["simulation"]["steps"] = 8
        raw["observer"] = {"grids": [[2, 2, 1]], "taps": [4, 8], "speed": True}
        raw["gate"].update(restarts=2, iterations=3, validation_realizations=2)
        raw["noise"].update(train_realizations=6, test_realizations=2)
    elif family == "endophalon_pro":
        raw["task"] = "xor"
        raw["theta_init"]["stimulus_points"] = [[1.0, 1.0, 1.5], [4.0, 1.0, 1.5]]
        raw["world"]["balls"]["count"] = 30
        raw["world"]["simulation"]["steps"] = 8
        raw["readout"] = {"step": 8, "grid": [2, 2, 1], "ridge_alpha": 1.0}
        raw["noise"].update(train_realizations=4, val_realizations=2, test_realizations=2)
        raw["sweep"] = {"grids": [[2, 2, 1]]}
    else:
        raw["split"].update(washout=1, train_streams=4, val_streams=2, test_streams=2)
        raw["readout"]["delay"] = 4
        if family == "endophalon2":
            raw["world"]["balls"]["count"] = 40
            raw["stream"].update(pulses=8, interval=6, pulse_steps=2, warmup=4)
        else:
            raw["network"]["lattice"] = [4, 3, 2]
            raw["events"].update(pulses=8, gap_min=6, gap_max=6, pulse_steps=2, warmup=4)
    if "outer" in raw:
        raw["outer"].update(generations=1, population=3, elite=2, seed=1)
    return raw


def main() -> None:
    # Support running without an editable install, just like pytest's pythonpath.
    sys.path.insert(0, str(ROOT / "src"))
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    count = 0
    with tempfile.TemporaryDirectory(prefix="endophalon-smoke-") as directory:
        for family, name in MODELS:
            source = ROOT / "configs" / family / f"{name}.json"
            # Generation-1 archives intentionally lack all generation-2 configs.
            if not source.exists():
                if (ROOT / "src" / "tele_neuron" / family).exists():
                    raise FileNotFoundError(source)
                if family.startswith("endophalon2"):
                    continue
                raise FileNotFoundError(source)
            importlib.import_module(f"tele_neuron.{family}")
            config = Path(directory) / f"{family}-{name}.json"
            card = config.with_suffix(".card.json")
            config.write_text(json.dumps(reduced(json.loads(source.read_text()), family)))
            result = subprocess.run(
                [sys.executable, "-m", f"tele_neuron.{family}", "--config", str(config), "--output", str(card)],
                cwd=ROOT, env=env, capture_output=True, text=True, timeout=60,
            )
            if result.returncode:
                raise RuntimeError(f"{family}/{name}:\n{result.stdout}\n{result.stderr}")
            payload = json.loads(card.read_text())
            if "metrics" not in payload:
                raise AssertionError(f"Missing card metrics: {family}/{name}")
            print(f"PASS {family}/{name}: import, CLI training/evaluation, model-card save")
            count += 1
    print(f"{count} model/config combinations passed")


if __name__ == "__main__":
    main()
