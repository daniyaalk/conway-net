"""
Standalone stdlib-only HTTP server for reduced_network_trials.html --
deliberately separate from server.py so this stress-test tool doesn't touch
any of the project's existing files. Serves the trials page and the routes
it calls, importing the real network.py / reduced_network.py computation
(not a second, hand-duplicated copy of either).

Runs on its own port so it doesn't conflict with server.py.
"""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from network import ConwayNet, run
from reduced_network import ReducedConwayNet, describe_reduced_network
from reduced_network import run as reduced_run

DIR = Path(__file__).parent
PORT = 8001


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/reduced_network_trials.html"):
            body = (DIR / "reduced_network_trials.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/reduced-network":
            self._send_json(200, describe_reduced_network())
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length))
            if self.path == "/api/step":
                self._handle_step(payload)
            elif self.path == "/api/reduced-step":
                self._handle_reduced_step(payload)
            else:
                self.send_error(404)
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            self._send_json(400, {"error": str(exc)})

    def _handle_step(self, payload):
        epochs = int(payload.get("epochs", 1))
        if epochs < 1:
            raise ValueError("epochs must be >= 1")
        history = run(payload["grid"], epochs)
        prior_grid = history[-2]  # the grid that fed the final epoch
        _, _, output = ConwayNet(prior_grid).forward()
        self._send_json(200, {"output": output, "priorGrid": prior_grid, "epochs": epochs})

    def _handle_reduced_step(self, payload):
        epochs = int(payload.get("epochs", 1))
        if epochs < 1:
            raise ValueError("epochs must be >= 1")
        history = reduced_run(payload["grid"], epochs)
        prior_grid = history[-2]
        _, output = ReducedConwayNet(prior_grid).forward()
        self._send_json(200, {"output": output, "priorGrid": prior_grid, "epochs": epochs})

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    server = HTTPServer(("localhost", PORT), Handler)
    print(f"Serving Reduced Network Trials at http://localhost:{PORT}")
    server.serve_forever()
