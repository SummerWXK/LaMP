"""Small JSON/HTTP server for explicitly authorized LaMP inference clients."""

from __future__ import annotations

import argparse
import base64
import binascii
import io
import json
import logging
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np
import torch
from PIL import Image

from starVLA import load_policy

LOGGER = logging.getLogger("lamp.server")
LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}
MAX_REQUEST_BYTES = 32 * 1024 * 1024


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--allow-remote", action="store_true", help="Acknowledge that this unauthenticated server is remote"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.host not in LOOPBACK_HOSTS and not args.allow_remote:
        raise ValueError("non-loopback binding requires the explicit --allow-remote flag")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    policy = load_policy(
        args.checkpoint,
        device=args.device,
        dtype=args.dtype,
    )
    handler = _make_handler(policy, args.device)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    LOGGER.info("listening on http://%s:%d", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("stopping")
    finally:
        server.server_close()
    return 0


def _make_handler(policy: Any, device: str) -> type[BaseHTTPRequestHandler]:
    inference_lock = threading.Lock()

    class PredictionHandler(BaseHTTPRequestHandler):
        server_version = "LaMP/0.1.0"

        def do_GET(self) -> None:
            if self.path != "/health":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            self._send_json(HTTPStatus.OK, {"status": "ok"})

        def do_POST(self) -> None:
            if self.path != "/predict_action":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            try:
                request = self._read_request()
                examples = [_decode_example(example) for example in request["examples"]]
                seed = request.get("seed")
                generator = None if seed is None else torch.Generator(device=device).manual_seed(int(seed))
                with inference_lock:
                    result = policy.predict_action(
                        examples,
                        unnorm_key=request.get("unnorm_key", "libero"),
                        generator=generator,
                        return_motion=bool(request.get("return_motion", False)),
                    )
                self._send_json(HTTPStatus.OK, {key: value.tolist() for key, value in result.items()})
            except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError, binascii.Error) as error:
                LOGGER.warning("invalid request: %s", error)
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_request"})
            except Exception:
                LOGGER.exception("prediction failed")
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "prediction_failed"})

        def _read_request(self) -> dict[str, Any]:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as error:
                raise ValueError("invalid Content-Length") from error
            if not 0 < length <= MAX_REQUEST_BYTES:
                raise ValueError("request body is empty or too large")
            request = json.loads(self.rfile.read(length))
            if not isinstance(request, dict) or not isinstance(request.get("examples"), list):
                raise TypeError("request must contain an examples list")
            return request

        def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self.send_response(status.value)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format_string: str, *args: Any) -> None:
            LOGGER.info("%s - %s", self.address_string(), format_string % args)

    return PredictionHandler


def _decode_example(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("each example must be an object")
    encoded_images = payload.get("image")
    if not isinstance(encoded_images, list) or len(encoded_images) != 2:
        raise ValueError("each example must contain two base64 PNG/JPEG images")
    images = []
    for encoded in encoded_images:
        raw = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(raw)) as image:
            images.append(image.convert("RGB").copy())
    state = np.asarray(payload.get("state"), dtype=np.float32)
    if state.shape != (8,):
        raise ValueError("state must have shape [8]")
    if not isinstance(payload.get("lang"), str):
        raise TypeError("lang must be a string")
    return {"image": images, "lang": payload["lang"], "state": state}


if __name__ == "__main__":
    raise SystemExit(main())
