from __future__ import annotations

import json
from pathlib import Path
import threading
from urllib.request import Request, urlopen

from tele_neuron.learner_lab import create_server, list_model_files


def _sample_config() -> dict:
    return {
        "name": "learner_http",
        "task": "LAB",
        "space": {"size": [4, 4, 3]},
        "simulation": {
            "steps": 1,
            "dt": 1,
            "damping": 1,
            "restitution": 1,
            "cell_size": 0.5,
            "seed": 1,
        },
        "balls": {"count": 4, "mass": 1, "radius": 0.05, "initial_speed": 0},
        "input": {"points": [[0.5, 0.5, 1]], "strength": 0, "epsilon": 0.001},
        "readout": [
            {"name": "out1", "center": [1, 1, 1], "half_extents": [0.5, 0.5, 1], "threshold": 1}
        ],
        "cases": [{"bits": [0], "target": 0}, {"bits": [1], "target": 1}],
    }


def test_list_model_files_includes_lab_models(tmp_path: Path) -> None:
    (tmp_path / "models" / "lab").mkdir(parents=True)
    (tmp_path / "models" / "root.json").write_text('{"masses": [1.0]}', encoding="utf-8")
    (tmp_path / "models" / "lab" / "child.json").write_text('{"masses": [1.0]}', encoding="utf-8")
    (tmp_path / "models" / "card.json").write_text("{}", encoding="utf-8")

    rows = list_model_files(tmp_path)

    assert {"name": "root", "path": "models/root.json"} in rows
    assert {"name": "child", "path": "models/lab/child.json"} in rows
    assert {"name": "card", "path": "models/card.json"} not in rows


def test_http_train_endpoint_returns_saved_model(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    (tmp_path / "models").mkdir()
    algorithm_dir = tmp_path / "Vanila_learner" / "Algorithm"
    algorithm_dir.mkdir(parents=True)
    source_algorithm = Path("Vanila_learner/Algorithm/random_search.py")
    (algorithm_dir / "random_search.py").write_text(source_algorithm.read_text(encoding="utf-8"), encoding="utf-8")
    server = create_server(tmp_path, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/api/train"
        body = json.dumps(
            {
                "config": _sample_config(),
                "algorithm": "random_search",
                "settings": {"iterations": 1, "population": 1, "seed": 1},
            }
        ).encode("utf-8")
        request = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        payload = json.loads(urlopen(request, timeout=10).read().decode("utf-8"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert payload["ok"] is True
    assert payload["training"]["path"].startswith("models/lab/")
    assert (tmp_path / payload["training"]["path"]).is_file()
