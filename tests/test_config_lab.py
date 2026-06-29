from __future__ import annotations

import json
from pathlib import Path
import threading
from urllib.request import Request, urlopen

from tele_neuron.config_lab import (
    create_server,
    preview_config_payload,
    save_config_payload,
    validate_config_payload,
)


def _sample_config(name: str = "sample") -> dict:
    return {
        "name": name,
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
        "balls": {"count": 6, "mass": 1, "radius": 0.05, "initial_speed": 0},
        "input": {"points": [[0.5, 0.5, 1]], "strength": 0, "epsilon": 0.001},
        "readout": [
            {"name": "out1", "center": [1, 1, 1], "half_extents": [0.5, 0.5, 1], "threshold": 1}
        ],
        "cases": [{"bits": [0], "target": 0}, {"bits": [1], "target": 1}],
    }


def test_validate_rejects_bits_target_length_mismatch() -> None:
    bad = _sample_config()
    bad["cases"][0]["bits"] = [0, 1]

    result = validate_config_payload(bad)

    assert result["valid"] is False
    assert "bits length" in result["errors"][0]


def test_save_sanitizes_name_into_lab_dir(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    payload = _sample_config("../escape")

    result = save_config_payload(tmp_path, payload)

    assert result["saved"] is True
    assert result["path"] == "configs/lab/escape.json"
    assert (tmp_path / "configs" / "lab" / "escape.json").is_file()
    assert not (tmp_path / "escape.json").exists()


def test_preview_returns_result_and_png(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()

    result = preview_config_payload(tmp_path, _sample_config(), bits="1")

    assert result["valid"] is True
    assert result["preview"]["result"]["total"] == 2
    assert (tmp_path / result["preview"]["image_path"]).is_file()


def test_http_configs_endpoint_lists_configs(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "sample.json").write_text(json.dumps(_sample_config()), encoding="utf-8")
    server = create_server(tmp_path, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/api/configs"
        payload = json.loads(urlopen(url, timeout=5).read().decode("utf-8"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert payload["configs"] == [{"name": "sample", "path": "configs/sample.json"}]


def test_http_preview_endpoint_returns_png_path(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    server = create_server(tmp_path, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/api/preview"
        body = json.dumps({"config": _sample_config(), "bits": "1"}).encode("utf-8")
        request = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        payload = json.loads(urlopen(request, timeout=10).read().decode("utf-8"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert payload["valid"] is True
    assert payload["preview"]["image_path"].endswith(".png")
