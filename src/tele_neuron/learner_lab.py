from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse
import webbrowser

from tele_neuron.config_lab import list_config_files, load_config_payload
from tele_neuron.learner import list_algorithm_files, train_model_payload


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8766


def main() -> None:
    root = _project_root()
    server = create_server(root=root, host=DEFAULT_HOST, port=DEFAULT_PORT)
    url = f"http://{DEFAULT_HOST}:{DEFAULT_PORT}"
    print(f"TeleNeuron Learner Lab running at {url}")
    print("Press Ctrl+C to stop.")
    try:
        webbrowser.open(url)
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Learner Lab.")
    finally:
        server.server_close()


def create_server(root: Path, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    class LearnerLabHandler(_LearnerLabHandler):
        project_root = root

    return ThreadingHTTPServer((host, port), LearnerLabHandler)


def list_model_files(root: Path) -> list[dict[str, str]]:
    model_dir = root / "models"
    paths = sorted(model_dir.glob("*.json")) + sorted((model_dir / "lab").glob("*.json"))
    return [
        {"name": path.stem, "path": path.relative_to(root).as_posix()}
        for path in paths
        if path.is_file() and _is_trainable_model(path)
    ]


def load_model_payload(root: Path, path_value: str) -> dict[str, Any]:
    path = _resolve_model_path(root, path_value)
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


class _LearnerLabHandler(BaseHTTPRequestHandler):
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
            self._send_json({"config": load_config_payload(self.project_root, query.get("path", [""])[0])})
            return
        if parsed.path == "/api/models":
            self._send_json({"models": list_model_files(self.project_root)})
            return
        if parsed.path == "/api/model":
            query = parse_qs(parsed.query)
            self._send_json({"model": load_model_payload(self.project_root, query.get("path", [""])[0])})
            return
        if parsed.path == "/api/algorithms":
            self._send_json({"algorithms": list_algorithm_files(self.project_root)})
            return
        if parsed.path.startswith("/models/") or parsed.path.startswith("/outputs/"):
            self._send_project_file(unquote(parsed.path.lstrip("/")))
            return
        self._send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        payload = self._read_json()
        if parsed.path == "/api/train":
            try:
                result = train_model_payload(
                    self.project_root,
                    payload["config"],
                    algorithm=payload["algorithm"],
                    settings_payload=payload.get("settings", {}),
                    model_payload=payload.get("model"),
                )
            except Exception as exc:
                self._send_json({"ok": False, "errors": [str(exc)]}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json({"ok": True, "training": result})
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
            allowed_roots = [(self.project_root / "models").resolve(), (self.project_root / "outputs").resolve()]
            if not any(root in path.parents for root in allowed_roots) or not path.is_file():
                raise ValueError("invalid project file path")
        except ValueError:
            self._send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        self._send_file(path, "application/json; charset=utf-8")

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        self._send_json({"error": message}, status=status)


def _project_root() -> Path:
    cwd = Path.cwd()
    if (cwd / "configs").is_dir() and (cwd / "src").is_dir():
        return cwd
    return Path(__file__).resolve().parents[2]


def _static_dir() -> Path:
    return Path(__file__).with_name("learner_lab_static")


def _resolve_model_path(root: Path, path_value: str) -> Path:
    candidate = (root / path_value).resolve()
    models_dir = (root / "models").resolve()
    if models_dir not in candidate.parents or candidate.suffix.lower() != ".json":
        raise ValueError("model path must point to models/*.json")
    if not candidate.is_file():
        raise FileNotFoundError(path_value)
    if not _is_trainable_model(candidate):
        raise ValueError("model file must contain per-ball masses")
    return candidate


def _is_trainable_model(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except (OSError, json.JSONDecodeError):
        return False
    masses = payload.get("masses")
    return isinstance(masses, list) and bool(masses)


if __name__ == "__main__":
    main()
