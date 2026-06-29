from __future__ import annotations

from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse
import webbrowser

from tele_neuron.config import parse_config
from tele_neuron.trainer import run_cases
from tele_neuron.visualize import save_final_snapshot


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


def main() -> None:
    root = _project_root()
    server = create_server(root=root, host=DEFAULT_HOST, port=DEFAULT_PORT)
    url = f"http://{DEFAULT_HOST}:{DEFAULT_PORT}"
    print(f"TeleNeuron Config Lab running at {url}")
    print("Press Ctrl+C to stop.")
    try:
        webbrowser.open(url)
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Config Lab.")
    finally:
        server.server_close()


def create_server(root: Path, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    class ConfigLabHandler(_ConfigLabHandler):
        project_root = root

    return ThreadingHTTPServer((host, port), ConfigLabHandler)


def list_config_files(root: Path) -> list[dict[str, str]]:
    configs_dir = root / "configs"
    paths = sorted(configs_dir.glob("*.json")) + sorted((configs_dir / "lab").glob("*.json"))
    return [
        {
            "name": path.stem,
            "path": path.relative_to(root).as_posix(),
        }
        for path in paths
        if path.is_file()
    ]


def load_config_payload(root: Path, path_value: str) -> dict[str, Any]:
    path = _resolve_config_path(root, path_value)
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def validate_config_payload(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        config = parse_config(payload)
    except Exception as exc:
        return {"valid": False, "errors": [str(exc)]}
    return {
        "valid": True,
        "errors": [],
        "summary": {
            "name": config.name,
            "task": config.task,
            "emitters": len(config.input.points),
            "readouts": len(config.readout),
            "cases": len(config.cases),
        },
    }


def save_config_payload(root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    validation = validate_config_payload(payload)
    if not validation["valid"]:
        return validation | {"saved": False}

    name = _safe_name(str(payload.get("name") or "config"))
    output_dir = root / "configs" / "lab"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (output_dir / f"{name}.json").resolve()
    if output_dir.resolve() not in output_path.parents:
        raise ValueError("refusing to save outside configs/lab")

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")

    return {
        "valid": True,
        "saved": True,
        "path": output_path.relative_to(root).as_posix(),
        "errors": [],
    }


def preview_config_payload(
    root: Path,
    payload: dict[str, Any],
    *,
    bits: str | list[int] | tuple[int, ...] | None = None,
) -> dict[str, Any]:
    validation = validate_config_payload(payload)
    if not validation["valid"]:
        return validation | {"preview": None}

    config = parse_config(payload)
    result = run_cases(config)
    preview_bits = _preview_bits(payload, bits)
    safe_bits = "".join(str(bit) for bit in preview_bits)
    safe_name = _safe_name(config.name)
    image_path = root / "outputs" / "lab" / f"{safe_name}_{safe_bits}.png"
    save_final_snapshot(config, preview_bits, image_path)

    return {
        "valid": True,
        "errors": [],
        "preview": {
            "result": _result_payload(result),
            "bits": preview_bits,
            "image_path": image_path.relative_to(root).as_posix(),
            "image_url": "/" + image_path.relative_to(root).as_posix().replace("\\", "/"),
        },
    }


class _ConfigLabHandler(BaseHTTPRequestHandler):
    project_root: Path

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_file(_static_dir() / "index.html", "text/html; charset=utf-8")
            return
        if parsed.path == "/api/configs":
            self._send_json({"configs": list_config_files(self.project_root)})
            return
        if parsed.path == "/api/config":
            query = parse_qs(parsed.query)
            path = query.get("path", [""])[0]
            self._send_json({"config": load_config_payload(self.project_root, path)})
            return
        if parsed.path.startswith("/outputs/"):
            self._send_project_file(unquote(parsed.path.lstrip("/")))
            return
        self._send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        payload = self._read_json()
        if parsed.path == "/api/validate":
            self._send_json(validate_config_payload(payload.get("config", payload)))
            return
        if parsed.path == "/api/save":
            self._send_json(save_config_payload(self.project_root, payload.get("config", payload)))
            return
        if parsed.path == "/api/preview":
            config_payload = payload.get("config", payload)
            self._send_json(
                preview_config_payload(
                    self.project_root,
                    config_payload,
                    bits=payload.get("bits"),
                )
            )
            return
        self._send_error(HTTPStatus.NOT_FOUND, "Not found")

    def log_message(self, format: str, *args: object) -> None:
        return

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw)

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, path: Path, content_type: str) -> None:
        if not path.is_file():
            self._send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_project_file(self, relative_path: str) -> None:
        try:
            path = (self.project_root / relative_path).resolve()
            outputs_dir = (self.project_root / "outputs").resolve()
            if outputs_dir not in path.parents or not path.is_file():
                raise ValueError("invalid output path")
        except ValueError:
            self._send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        content_type = "image/png" if path.suffix.lower() == ".png" else "application/octet-stream"
        self._send_file(path, content_type)

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        self._send_json({"error": message}, status=status)


def _project_root() -> Path:
    cwd = Path.cwd()
    if (cwd / "configs").is_dir() and (cwd / "src").is_dir():
        return cwd
    return Path(__file__).resolve().parents[2]


def _static_dir() -> Path:
    return Path(__file__).with_name("config_lab_static")


def _resolve_config_path(root: Path, path_value: str) -> Path:
    candidate = (root / path_value).resolve()
    configs_dir = (root / "configs").resolve()
    if configs_dir not in candidate.parents or candidate.suffix.lower() != ".json":
        raise ValueError("config path must point to configs/*.json")
    if not candidate.is_file():
        raise FileNotFoundError(path_value)
    return candidate


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    safe = safe.strip("._")
    return safe or "config"


def _preview_bits(payload: dict[str, Any], bits: str | list[int] | tuple[int, ...] | None) -> tuple[int, ...]:
    if bits is not None:
        if isinstance(bits, str):
            return tuple(int(char) for char in bits.strip())
        return tuple(int(bit) for bit in bits)

    for case in payload.get("cases", []):
        case_bits = tuple(int(bit) for bit in case.get("bits", []))
        if any(case_bits):
            return case_bits
    first = payload.get("cases", [{}])[0]
    return tuple(int(bit) for bit in first.get("bits", []))


def _result_payload(result: Any) -> dict[str, Any]:
    payload = asdict(result)
    payload["accuracy"] = result.accuracy
    return payload


if __name__ == "__main__":
    main()
