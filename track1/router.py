#!/usr/bin/env python3
import http.server
import json
import urllib.request
import urllib.error
import urllib.parse
import os
import subprocess
import time
import threading
import logging

# Setup logging to stdout
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Configuration
PORT = int(os.environ.get("PORT", "8080"))
LOCAL_PORT = int(os.environ.get("LOCAL_PORT", "8081"))
LOCAL_MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH", "/app/model.gguf")
FIREWORKS_API_KEY = os.environ.get("FIREWORKS_API_KEY", "")
FIREWORKS_MODEL = os.environ.get("FIREWORKS_MODEL", "accounts/fireworks/models/llama-v3p1-8b-instruct")

# Global state
local_process = None
local_ready = False


def log(msg):
    logging.info(f"[ROUTER] {msg}")


# Resolve paths for local testing or inside Docker
if os.path.exists("/app/llama-server"):
    LLAMA_SERVER_BIN = "/app/llama-server"
    LD_LIBRARY_PATH_ENV = "/app"
else:
    # Local fallback for testing
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    possible_bins = [
        os.path.join(repo_root, "build/bin/llama-server"),
        os.path.join(repo_root, "build/bin/Release/llama-server"),
        "llama-server" # in PATH
    ]
    LLAMA_SERVER_BIN = "llama-server"
    for b in possible_bins:
        if os.path.exists(b):
            LLAMA_SERVER_BIN = b
            break
    LD_LIBRARY_PATH_ENV = os.path.join(repo_root, "build/bin")

# Resolve model path
if os.path.exists(LOCAL_MODEL_PATH):
    resolved_model_path = LOCAL_MODEL_PATH
else:
    # Look in repository root
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    possible_models = [
        os.path.join(repo_root, "qwen2.5-coder-1.5b-bf16.gguf"),
        os.path.join(repo_root, "models/your-model.gguf")
    ]
    resolved_model_path = LOCAL_MODEL_PATH
    for m in possible_models:
        if os.path.exists(m):
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
                if resp.status == 200:
                    if not local_ready:
                        log("Local llama-server is healthy and ready!")
                        local_ready = True
        except Exception:
            local_ready = False

        time.sleep(2)


def start_local_server():
    global local_process
    if not os.path.exists(resolved_model_path):
        log(f"Local model not found at {resolved_model_path}. Local fallback will not be available.")
        return

    log(f"Starting local llama-server on port {LOCAL_PORT} with model {resolved_model_path}...")

    # Base command
    cmd = [
        LLAMA_SERVER_BIN,
        "-m", resolved_model_path,
        "--port", str(LOCAL_PORT),
        "--host", "127.0.0.1",
        "-c", "2048",
        "-ngl", "0"
    ]

    # Add any extra args from environment
    extra_args = os.environ.get("LLAMA_ARGS", "")
    if extra_args:
        cmd.extend(extra_args.split())

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = LD_LIBRARY_PATH_ENV

    try:
        local_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env
        )

        # Stream logs in a separate thread to keep stdout clean
        def log_streamer():
            if local_process.stdout is not None:
                for line in local_process.stdout:
                    logging.info(f"[LLAMA-SERVER] {line.strip()}")
        threading.Thread(target=log_streamer, daemon=True).start()

        # Start health check thread
        threading.Thread(target=check_local_health, daemon=True).start()

    except Exception as e:
        log(f"Failed to start local llama-server: {e}")


class RouterHTTPHandler(http.server.BaseHTTPRequestHandler):

    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        # Prevent default logging to stderr/stdout to keep logs clean
        pass

    def do_GET(self):
        global local_ready

        # Health check
        if self.path in ("/health", "/healthcheck"):
            # If Fireworks is configured, we are immediately ready
            if FIREWORKS_API_KEY:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(b'{"status":"ok","engine":"fireworks"}')
                return

            # Otherwise we require local server to be ready
            if local_ready:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(b'{"status":"ok","engine":"local"}')
            else:
                self.send_response(503)
                self.send_header("Content-Type", "application/json")
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(b'{"status":"starting","engine":"local"}')
            return

        # Get list of models (OpenAI spec)
        if self.path == "/v1/models":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Connection", "close")
            self.end_headers()
            model_name = FIREWORKS_MODEL if FIREWORKS_API_KEY else "local-model"
            response_data = {
                "object": "list",
                "data": [
                    {
                        "id": model_name,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "router"
                    }
                ]
            }
            self.wfile.write(json.dumps(response_data).encode("utf-8"))
            return

        # Default fallback
        self.send_response(404)
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(b"Not Found")

    def do_POST(self):
        if self.path not in ("/v1/chat/completions", "/v1/completions"):
            self.send_response(404)
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(b"Not Found")
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            req_data = json.loads(body.decode("utf-8"))
        except Exception as e:
            self.send_response(400)
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(f"Invalid JSON: {e}".encode("utf-8"))
            return

        # Route request
        if FIREWORKS_API_KEY:
            log(f"Routing request to Fireworks AI (Model: {FIREWORKS_MODEL})...")
            req_data["model"] = FIREWORKS_MODEL
            url = "https://api.fireworks.ai/inference/v1/chat/completions" if self.path == "/v1/chat/completions" else "https://api.fireworks.ai/inference/v1/completions"

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {FIREWORKS_API_KEY}"
            }

            success = self.proxy_request(url, json.dumps(req_data).encode("utf-8"), headers)
            if success:
                return
            log("Fireworks request failed or timed out. Falling back to local model...")

        # Local fallback
        if not local_ready:
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(b'{"error":"No backend ready (local loading or Fireworks failed)"}')
            return

        log("Routing request to local llama-server...")
        url = f"http://127.0.0.1:{LOCAL_PORT}{self.path}"
        headers = {"Content-Type": "application/json"}
        self.proxy_request(url, body, headers)

    def proxy_request(self, url, body_bytes, headers):
        req = urllib.request.Request(url, data=body_bytes, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                self.send_response(resp.status)

                # Copy headers from backend response
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

                # Stream response
                if is_stream:
                    while True:
                        line = resp.readline()
                        if not line:
                            break
                        size_hex = f"{len(line):X}\r\n".encode("utf-8")
                        self.wfile.write(size_hex)
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
            log(f"HTTPError while proxying to {url}: {e.code} - {e.reason}")
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
            log(f"Exception while proxying to {url}: {e}")
            return False


def run_router():
    server = http.server.HTTPServer(("0.0.0.0", PORT), RouterHTTPHandler)
    log(f"Router listening on 0.0.0.0:{PORT}...")
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
    # Start local model server if Fireworks key is not present
    # Or if forced via environment variable
    if not FIREWORKS_API_KEY or os.environ.get("FORCE_LOCAL", "").lower() == "true":
        start_local_server()
    else:
        log("Fireworks API key detected. Running in cloud-only mode (local llama-server disabled).")

    run_router()
