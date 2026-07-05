#!/usr/bin/env python3
"""Stand-in for llama-server, used to test llama_priority_proxy.py's own
routing/auth/SSE-rewrite/capacity-queuing logic without the real binary (or a
GPU). Accepts the same CLI shape launch_server() constructs, ignores
everything except --model/--port, and serves /health, /slots, a minimal
OpenAI-shaped /v1/chat/completions (streaming-aware) and /v1/embeddings.

Usage — point the proxy at this instead of the real binary:
    LLAMA_SERVER_BIN=/path/to/fake_llama_server.py python3 llama_priority_proxy.py \\
        --models-dir ~/docker/appdata/llm-models --config-dir config --port 11444

To simulate a full --kv-unified pool (for testing capacity-queuing/heartbeats/
max-wait), set FAKE_NO_SLOT_UNTIL to a unix timestamp before starting the
proxy — every spawned instance inherits it, and GET /slots?fail_on_no_slot=1
returns 503 until that time passes:
    FAKE_NO_SLOT_UNTIL=$(python3 -c "import time; print(time.time()+25)") \\
        LLAMA_SERVER_BIN=... python3 llama_priority_proxy.py ...
"""
import argparse
import json
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

# Testing hook: if set, /slots?fail_on_no_slot=1 returns 503 until this unix
# timestamp passes, simulating a full --kv-unified pool for queuing tests.
NO_SLOT_UNTIL = float(os.environ.get("FAKE_NO_SLOT_UNTIL", "0"))

parser = argparse.ArgumentParser()
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--ctx-size", type=int, default=0)
parser.add_argument("--parallel", type=int, default=1)
parser.add_argument("--n-gpu-layers", type=int, default=0)
parser.add_argument("--flash-attn", default=None)
parser.add_argument("--kv-unified", action="store_true")
parser.add_argument("--cont-batching", action="store_true")
parser.add_argument("--no-context-shift", action="store_true")
parser.add_argument("--embedding", action="store_true")
args, _unknown = parser.parse_known_args()

MODEL_NAME = args.model.split("/")[-1].removesuffix(".gguf")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *a):
        pass

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        elif self.path.startswith("/slots"):
            if "fail_on_no_slot=1" in self.path and time.time() < NO_SLOT_UNTIL:
                self.send_response(503)
                self.end_headers()
                self.wfile.write(b'{"error":{"message":"no slot available"}}')
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'[{"id":0,"is_processing":false}]')
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        stream = bool(body.get("stream"))

        if "/embeddings" in self.path:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"model": MODEL_NAME, "data": [{"embedding": [0.1, 0.2]}]}).encode())
            return

        if not stream:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "model": MODEL_NAME,
                "choices": [{"message": {"role": "assistant", "content": f"hello from {MODEL_NAME}"}}],
            }).encode())
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for word in [f"hello", "from", MODEL_NAME]:
            chunk = {"model": MODEL_NAME, "choices": [{"delta": {"content": word + " "}}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
            time.sleep(0.05)
        self.wfile.write(b"data: [DONE]\n\n")


HTTPServer((args.host, args.port), Handler).serve_forever()
