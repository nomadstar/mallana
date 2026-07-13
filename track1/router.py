#!/usr/bin/env python3
"""
Mallana Local Router — 100% on-device inference, zero cloud, zero billable tokens.

This is a deliberately thin, dependency-free (stdlib only) OpenAI-compatible gateway
in front of mallana's `llama-server`. There is NO cloud path and no API keys: every
request is served by the local model, which means the token score is 0 by construction.

Why local-only? Because that is the entire thesis of this project (see MANIFESTO.md):

    "How much useful intelligence can we run on the hardware people already own?"

The goal is not to shuttle work to a datacenter — it is to reduce the *cost of intelligence
itself* so that even modest hardware can run a capable model. Mallana's TurboQuant KV
compression (V-cache 4.6–6.4x smaller) lets a larger, more accurate model fit in the same
VRAM, so more queries are answered well, on-device, at 0 tokens. On a "fewest tokens wins,
subject to an accuracy gate" leaderboard, running a stronger local model is exactly what wins.

Endpoints (OpenAI spec): /v1/chat/completions, /v1/completions, /v1/models, /health.
"""
import http.server
import json
import urllib.request
import urllib.error
import os
import glob
import subprocess
import time
import threading
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

# Configuration — all local. There is intentionally no cloud/provider configuration.
PORT = int(os.environ.get("PORT", "8080"))
LOCAL_PORT = int(os.environ.get("LOCAL_PORT", "8081"))
LOCAL_MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH", "/app/model.gguf")

# TurboQuant defaults — the capability-density knobs that let a bigger model fit on-device.
# Overridable via env, but the defaults embody the project's KV-compression thesis.
CACHE_TYPE_K = os.environ.get("CACHE_TYPE_K", "q8_0")
CACHE_TYPE_V = os.environ.get("CACHE_TYPE_V", "turbo3")
CTX_SIZE = os.environ.get("CTX_SIZE", "4096")
NGL = os.environ.get("LLAMA_NGL", "99")  # offload all layers if a GPU backend is present; ignored on CPU-only builds

# Global state
local_process = None
local_ready = False


def log(msg):
    logging.info(f"[ROUTER] {msg}")


# Resolve the llama-server binary for either the Docker image or a local checkout.
if os.path.exists("/app/llama-server"):
    LLAMA_SERVER_BIN = "/app/llama-server"
    LD_LIBRARY_PATH_ENV = "/app"
else:
    # Prefer the locally-built mallana binary. We deliberately do NOT silently fall back to a
    # `llama-server` on PATH: an upstream build there lacks TurboQuant cache types and would
    # reject `--cache-type-v turbo3` at startup. Better to fail loudly than serve from the wrong
    # binary. Override with LLAMA_SERVER_BIN if your build lives elsewhere.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    LLAMA_SERVER_BIN = os.environ.get("LLAMA_SERVER_BIN", "")
    if not LLAMA_SERVER_BIN:
        for b in [os.path.join(repo_root, "build/bin/llama-server"),
                  os.path.join(repo_root, "build/bin/Release/llama-server")]:
            if os.path.exists(b):
                LLAMA_SERVER_BIN = b
                break
    LD_LIBRARY_PATH_ENV = os.path.join(repo_root, "build/bin")

# Resolve the model path. LOCAL_MODEL_PATH wins; otherwise scan common locations for
# any *.gguf. We deliberately SKIP known-bad test models: qwen2.5-coder-1.5b-bf16 is a
# weak, fp-fragile model that collapses into repetition loops under any lossy KV and has
# repeatedly masqueraded as a "TurboQuant bug" — never auto-serve it. If nothing valid is
# found we keep LOCAL_MODEL_PATH so start_local_server() fails loudly rather than serving
# garbage from the wrong model.
_BAD_MODELS = {"qwen2.5-coder-1.5b-bf16.gguf"}
if os.path.exists(LOCAL_MODEL_PATH):
    resolved_model_path = LOCAL_MODEL_PATH
else:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    resolved_model_path = LOCAL_MODEL_PATH  # fail-loud default
    candidates = sorted(glob.glob(os.path.join(repo_root, "*.gguf")) +
                        glob.glob(os.path.join(repo_root, "models", "*.gguf")))
    for m in candidates:
        if os.path.basename(m) in _BAD_MODELS:
            log(f"skipping known-bad test model: {os.path.basename(m)}")
            continue
        resolved_model_path = m
        break


def check_local_health():
    """Poll the local llama-server health endpoint."""
    global local_ready
    url = f"http://127.0.0.1:{LOCAL_PORT}/health"
    while True:
        if local_process and local_process.poll() is not None:
            log(f"Local llama-server process terminated with code {local_process.returncode}")
            local_ready = False
            time.sleep(5)
            continue
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200 and not local_ready:
                    log("Local llama-server is healthy and ready!")
                    local_ready = True
        except Exception:
            local_ready = False
        time.sleep(2)


def start_local_server():
    global local_process
    if not LLAMA_SERVER_BIN or not os.path.exists(LLAMA_SERVER_BIN):
        log(f"FATAL: mallana llama-server binary not found (looked in build/bin/). "
            f"Build it first (cmake --build build --target llama-server) or set LLAMA_SERVER_BIN. "
            f"Refusing to fall back to a PATH binary, which would lack TurboQuant cache types.")
        return
    if not os.path.exists(resolved_model_path):
        log(f"FATAL: local model not found at {resolved_model_path}. "
            f"Set LOCAL_MODEL_PATH — this router is local-only and cannot serve without a model.")
        return

    log(f"Starting local llama-server on port {LOCAL_PORT} with model {resolved_model_path}")
    log(f"TurboQuant KV compression: -fa on, K={CACHE_TYPE_K}, V={CACHE_TYPE_V}, ctx={CTX_SIZE}, ngl={NGL}")

    cmd = [
        LLAMA_SERVER_BIN,
        "-m", resolved_model_path,
        "--port", str(LOCAL_PORT),
        "--host", "127.0.0.1",
        "-c", CTX_SIZE,
        "-ngl", NGL,
        "-fa", "on",
        "--cache-type-k", CACHE_TYPE_K,
        "--cache-type-v", CACHE_TYPE_V,
    ]
    extra_args = os.environ.get("LLAMA_ARGS", "")
    if extra_args:
        cmd.extend(extra_args.split())

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = LD_LIBRARY_PATH_ENV

    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        local_process = proc

        def log_streamer():
            if proc.stdout is not None:
                for line in proc.stdout:
                    logging.info(f"[LLAMA-SERVER] {line.strip()}")
        threading.Thread(target=log_streamer, daemon=True).start()
        threading.Thread(target=check_local_health, daemon=True).start()
    except Exception as e:
        log(f"Failed to start local llama-server: {e}")


class RouterHTTPHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        pass

    def _json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/health", "/healthcheck"):
            if local_ready:
                self._json(200, {"status": "ok", "engine": "local"})
            else:
                self._json(503, {"status": "starting", "engine": "local"})
            return

        if self.path == "/v1/models":
            self._json(200, {
                "object": "list",
                "data": [{
                    "id": "mallana-local",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "mallana",
                }],
            })
            return

        self._json(404, {"error": "Not Found"})

    def do_POST(self):
        if self.path not in ("/v1/chat/completions", "/v1/completions"):
            self._json(404, {"error": "Not Found"})
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        try:
            json.loads(body.decode("utf-8"))
        except Exception as e:
            self._json(400, {"error": f"Invalid JSON: {e}"})
            return

        # Local-only: every request is served on-device. No cloud, no tokens billed.
        if not local_ready:
            self._json(503, {"error": "Local model is still loading. Retry shortly."})
            return

        url = f"http://127.0.0.1:{LOCAL_PORT}{self.path}"
        self.proxy_request(url, body, {"Content-Type": "application/json"})

    def proxy_request(self, url, body_bytes, headers):
        req = urllib.request.Request(url, data=body_bytes, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                self.send_response(resp.status)
                is_stream = False
                for k, v in resp.getheaders():
                    if k.lower() in ("transfer-encoding", "content-length", "connection"):
                        continue
                    if k.lower() == "content-type" and "text/event-stream" in v.lower():
                        is_stream = True
                    self.send_header(k, v)
                if is_stream:
                    self.send_header("Transfer-Encoding", "chunked")
                    self.send_header("Connection", "keep-alive")
                else:
                    self.send_header("Connection", "close")
                self.end_headers()

                if is_stream:
                    while True:
                        line = resp.readline()
                        if not line:
                            break
                        self.wfile.write(f"{len(line):X}\r\n".encode("utf-8"))
                        self.wfile.write(line)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                else:
                    self.wfile.write(resp.read())
                    self.wfile.flush()
                return True
        except urllib.error.HTTPError as e:
            log(f"HTTPError from local backend {url}: {e.code} - {e.reason}")
            try:
                err_body = e.read()
                self.send_response(e.code)
                for k, v in e.headers.items():
                    if k.lower() not in ("content-length", "transfer-encoding", "connection"):
                        self.send_header(k, v)
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(err_body)
            except Exception:
                self.send_response(e.code)
                self.send_header("Connection", "close")
                self.end_headers()
            return True
        except Exception as e:
            log(f"Exception while proxying to local backend {url}: {e}")
            self._json(502, {"error": f"Local backend error: {e}"})
            return False


def run_router():
    server = http.server.HTTPServer(("0.0.0.0", PORT), RouterHTTPHandler)
    log(f"Mallana local router listening on 0.0.0.0:{PORT} — 100% on-device, 0 tokens.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if local_process:
            log("Stopping local llama-server...")
            local_process.terminate()
            local_process.wait()


if __name__ == "__main__":
    # Local-only by design: the on-device model is always the backend.
    start_local_server()
    run_router()
