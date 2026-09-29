"""Local player: one fold-10 ECG, a live CNN forward pass, three lead masks."""

import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch

from ecg_transformer.cnn import CNN, apply_condition, predict_logits
from ecg_transformer.dataset import PTBXLDataset
from ecg_transformer.ptbxl import SUPERCLASSES
from ecg_transformer.smoke import pick_device

ROOT = Path(__file__).resolve().parents[1]
PACKED = ROOT / "data" / "ptb-xl" / "packed"
CHECKPOINT = ROOT / "checkpoints" / "m5_cnn.pt"
PAGE = ROOT / "docs" / "player.html"
HOST = "127.0.0.1"
PORT = 8777


class Player:
    def __init__(self) -> None:
        self.device = pick_device()
        self.model = CNN().to(self.device)
        self.model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=True))
        self.model.eval()
        self.ds = PTBXLDataset(PACKED, "test")

    def payload(self, index: int, condition: str) -> dict:
        n = len(self.ds)
        index %= n
        j = int(self.ds._index[index])
        signal = torch.from_numpy(np.array(self.ds.signals[j], dtype=np.float32, copy=True)).unsqueeze(0)
        started = time.perf_counter()
        masked = apply_condition(signal, condition)
        logits = predict_logits(self.model, masked, 1, self.device)[0]
        probs = torch.sigmoid(torch.from_numpy(logits)).tolist()
        elapsed_ms = (time.perf_counter() - started) * 1000
        labels = np.array(self.ds.labels[j], dtype=np.float32)
        shown = np.round(masked[0].numpy(), 3)
        return {
            "index": index,
            "n": n,
            "ecg_id": int(self.ds.ecg_ids[j]),
            "condition": condition,
            "elapsed_ms": round(elapsed_ms, 1),
            "label": {name: float(labels[k]) for k, name in enumerate(SUPERCLASSES)},
            "probabilities": {name: float(probs[k]) for k, name in enumerate(SUPERCLASSES)},
            "signal": shown.tolist(),
        }


PLAYER = Player()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/record":
            qs = parse_qs(parsed.query)
            try:
                index = int(qs.get("i", ["0"])[0])
                condition = qs.get("condition", ["clean"])[0]
                body = json.dumps(PLAYER.payload(index, condition)).encode()
            except (ValueError, IndexError) as err:
                self._send(400, json.dumps({"error": str(err)}).encode(), "application/json")
                return
            self._send(200, body, "application/json")
            return
        if parsed.path in ("/", "/player.html"):
            self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            return
        self._send(404, b"not found", "text/plain")

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:
        print(fmt % args, flush=True)


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"device {PLAYER.device}", flush=True)
    print(f"open http://{HOST}:{PORT}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
