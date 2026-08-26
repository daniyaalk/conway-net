"""
Stdlib-only HTTP server. Serves index.html and runs the real Python
network for every step, so the browser calls it directly instead of a
second, hand-duplicated JS copy. Two interchangeable engines are
available -- network.py's per-cell ConwayNet and matrix_network.py's
flat-matrix FullGridNet -- selected per request via payload["engine"].
"""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from network import ConwayNet, describe_network, run
from matrix_network import FullGridNet

DIR = Path(__file__).parent
PORT = 8000


def _split(layer, n):
    """[[(a,b,c), ...], ...] (ConwayNet's per-cell tuples) -> n separate grids."""
    return [[[cell[k] for cell in row] for row in layer] for k in range(n)]


def _reshape_flat(flat, rows, cols, width):
    """A FullGridNet flat vector (width consecutive values per position) ->
    `width` separate rows x cols grids."""
    return [
        [[flat[(i * cols + j) * width + k] for j in range(cols)] for i in range(rows)]
        for k in range(width)
    ]


def _run_matrix_engine(grid, epochs):
    """FullGridNet, looped `epochs` times. Returns the last epoch's
    flat layer2/layer3, the final grid, and the grid that fed that last
    epoch -- same shape of information run()+ConwayNet gives the cell engine."""
    rows, cols = len(grid), len(grid[0])
    net = FullGridNet(rows, cols)
    current = grid
    for _ in range(epochs):
        prior = current
        flat_l2, flat_l3, current = net.forward(current)
    return flat_l2, flat_l3, current, prior


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = (DIR / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/network":
            self._send_json(200, describe_network())
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length))
            if self.path == "/api/step":
                self._handle_step(payload)
            else:
                self.send_error(404)
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            self._send_json(400, {"error": str(exc)})

    def _handle_step(self, payload):
        epochs = int(payload.get("epochs", 1))
        if epochs < 1:
            raise ValueError("epochs must be >= 1")
        engine = payload.get("engine", "cell")

        if engine == "cell":
            history = run(payload["grid"], epochs)
            prior_grid = history[-2]  # the grid that fed the final epoch
            layer2, layer3, output = ConwayNet(prior_grid).forward()
            l2_1, l2_2, l2_3, l2_4 = _split(layer2, 4)
            l3_1, l3_2, l3_3 = _split(layer3, 3)
        elif engine == "matrix":
            flat_l2, flat_l3, output, prior_grid = _run_matrix_engine(payload["grid"], epochs)
            rows, cols = len(prior_grid), len(prior_grid[0])
            l2_1, l2_2, l2_3, l2_4 = _reshape_flat(flat_l2, rows, cols, 4)
            l3_1, l3_2, l3_3 = _reshape_flat(flat_l3, rows, cols, 3)
        else:
            raise ValueError(f"unknown engine: {engine!r}")

        self._send_json(200, {
            "layer2": {"n1": l2_1, "n2": l2_2, "n3": l2_3, "n4": l2_4},
            "layer3": {"n1": l3_1, "n2": l3_2, "n3": l3_3},
            "output": output,
            "priorGrid": prior_grid,
            "epochs": epochs,
            "engine": engine,
        })

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    server = HTTPServer(("localhost", PORT), Handler)
    print(f"Serving Conway Net at http://localhost:{PORT}")
    server.serve_forever()
