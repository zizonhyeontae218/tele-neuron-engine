"""Re-evaluate published ProMax readouts on their original held-out streams."""
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tele_neuron.endophalon.readout import RidgeReadout
from tele_neuron.endophalon2_pro.theta import SpringTheta
from tele_neuron.endophalon2_promax.config import load_config
from tele_neuron.endophalon2_promax.core import collect, dataset
from tele_neuron.endophalon2_promax.tasks import OPS


def main():
    for path in sorted((ROOT / "models/endophalon2_promax").glob("*.json")):
        card = json.loads(path.read_text())
        config = replace(load_config(ROOT / card["config"]), workers=1)
        _, _, test = config.stream_ranges()
        [data] = collect(config, [SpringTheta.from_payload(card["theta"])], test)
        assert data.finite, path.name
        x, y, ops = dataset(config, data, test)
        readout = RidgeReadout.from_payload(card["readout"])
        raw = readout.scores(x)
        hit = ((raw >= 0.5) == y)[:, 0]
        observed = {"accuracy": float(hit.mean()), "mse": float(np.mean((raw - y) ** 2)),
                    "per_op": {name: float(hit[ops == i].mean()) for i, name in enumerate(OPS)}}
        expected = card["metrics"]["test"]["trained"]["single_readout"]
        assert observed["accuracy"] == expected["accuracy"], (path.name, observed, expected)
        assert observed["per_op"] == expected["per_op"], (path.name, observed, expected)
        print(json.dumps({"card": path.name, "result": "PASS", **observed}), flush=True)


if __name__ == "__main__":
    main()
