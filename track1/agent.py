#!/usr/bin/env python3
"""
Mallana local-first batch agent — AMD Hackathon Track 1 (Model Router / Cost Optimizer).

Reads a fixed set of tasks and answers each one on-device with mallana's llama-server
(TurboQuant KV compression), consuming ZERO Fireworks tokens — which the scoring rules call
"the best possible outcome for ranking." Only if the local model genuinely cannot produce an
answer (and the optional Fireworks fallback is enabled) does a single task escalate to the
cheapest allowed cloud model, keeping token cost near zero while protecting the accuracy gate.

Contract (per the Track 1 spec, matched exactly):
  input : JSON array of {"task_id": str, "prompt": str}   at TASK_INPUT_PATH  (default /input/tasks.json)
  output: JSON array of {"task_id": str, "answer": str}    at TASK_OUTPUT_PATH (default /output/results.json)
  - task_ids are preserved exactly, one result per input task.
  - Output is written incrementally AND again before exit, so a partial run still scores
    (never a missing file → never a TIMEOUT-with-no-output).

Timeout safety (the scorer enforces a runtime limit):
  - Per-task wall-clock timeout (PER_TASK_TIMEOUT).
  - Global deadline (GLOBAL_DEADLINE): once exceeded, remaining tasks get a best-effort stub
    answer immediately so the output is always complete and written in time.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

# --- Contract paths (exact env var names from the Track 1 spec) ---
TASK_INPUT_PATH = os.environ.get("TASK_INPUT_PATH", "/input/tasks.json")
TASK_OUTPUT_PATH = os.environ.get("TASK_OUTPUT_PATH", "/output/results.json")

# --- Local model / server config ---
LOCAL_PORT = int(os.environ.get("LOCAL_PORT", "8081"))
MODELS_DIR = os.environ.get("MODELS_DIR", "/models")
LOCAL_MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH", "")
# Correctness-first defaults: Flash Attention is OFF and cache is f16. The fork's FA path
# currently corrupts generation on this build (repetitive-gibberish), and TurboQuant V-cache
# REQUIRES FA (its dequant happens inside the FA kernel) — so turbo is only usable once FA is
# fixed. Set FLASH_ATTN=on + CACHE_TYPE_V=turbo3 to opt back into the compressed path.
FLASH_ATTN = os.environ.get("FLASH_ATTN", "off")
CACHE_TYPE_K = os.environ.get("CACHE_TYPE_K", "f16")
CACHE_TYPE_V = os.environ.get("CACHE_TYPE_V", "f16")
CTX_SIZE = os.environ.get("CTX_SIZE", "4096")
NGL = os.environ.get("LLAMA_NGL", "99")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "512"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))
FREQUENCY_PENALTY = float(os.environ.get("FREQUENCY_PENALTY", "0.3"))
PRESENCE_PENALTY = float(os.environ.get("PRESENCE_PENALTY", "0.3"))

# --- Timeout safety ---
PER_TASK_TIMEOUT = float(os.environ.get("PER_TASK_TIMEOUT", "45"))       # seconds per task
GLOBAL_DEADLINE = float(os.environ.get("GLOBAL_DEADLINE", "1500"))       # seconds for the whole run
SERVER_START_TIMEOUT = float(os.environ.get("SERVER_START_TIMEOUT", "180"))

# --- Optional Fireworks fallback (OFF by default: pure-local = 0 tokens) ---
ENABLE_FIREWORKS_FALLBACK = os.environ.get("ENABLE_FIREWORKS_FALLBACK", "").lower() in ("1", "true", "yes")
FIREWORKS_API_KEY = os.environ.get("FIREWORKS_API_KEY", "")
FIREWORKS_BASE_URL = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
MODEL_CHEAP = os.environ.get("MODEL_CHEAP", "")

_start_time = time.monotonic()
_proc = None


def log(msg):
    print(f"[AGENT] {msg}", file=sys.stderr, flush=True)


def resolve_server_bin():
    if os.path.exists("/app/llama-server"):
        return "/app/llama-server", "/app"
    override = os.environ.get("LLAMA_SERVER_BIN", "")
    if override and os.path.exists(override):
        return override, os.path.dirname(override)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for b in [os.path.join(repo_root, "build/bin/llama-server"),
              os.path.join(repo_root, "build/bin/Release/llama-server")]:
        if os.path.exists(b):
            return b, os.path.dirname(b)
    return "", os.path.join(repo_root, "build/bin")


def resolve_model():
    if LOCAL_MODEL_PATH and os.path.exists(LOCAL_MODEL_PATH):
        return LOCAL_MODEL_PATH
    # Otherwise pick the first .gguf under MODELS_DIR (evaluator's mount), then the agent's own
    # dir (an optionally baked /app/model.gguf), then the repo root (local dev).
    here = os.path.dirname(os.path.abspath(__file__))
    search_dirs = [MODELS_DIR, here, os.path.dirname(here)]
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        ggufs = sorted(f for f in os.listdir(d) if f.endswith(".gguf"))
        if ggufs:
            return os.path.join(d, ggufs[0])
    return ""


def start_local_server():
    global _proc
    server_bin, ld_path = resolve_server_bin()
    model = resolve_model()
    if not server_bin:
        log("mallana llama-server binary not found (build it or set LLAMA_SERVER_BIN).")
        return False
    if not model:
        log(f"no .gguf model found (set LOCAL_MODEL_PATH or place one under {MODELS_DIR}).")
        return False

    log(f"starting llama-server: model={model} K={CACHE_TYPE_K} V={CACHE_TYPE_V} fa={FLASH_ATTN} ngl={NGL}")
    cmd = [server_bin, "-m", model, "--port", str(LOCAL_PORT), "--host", "127.0.0.1",
           "-c", CTX_SIZE, "-ngl", NGL, "-fa", FLASH_ATTN,
           "--cache-type-k", CACHE_TYPE_K, "--cache-type-v", CACHE_TYPE_V]
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ld_path + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    try:
        _proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
    except Exception as e:
        log(f"failed to launch llama-server: {e}")
        return False

    deadline = time.monotonic() + SERVER_START_TIMEOUT
    while time.monotonic() < deadline:
        if _proc.poll() is not None:
            log(f"llama-server exited early (code {_proc.returncode}).")
            return False
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{LOCAL_PORT}/health", timeout=2) as r:
                if r.status == 200:
                    log("llama-server healthy.")
                    return True
        except Exception:
            pass
        time.sleep(1)
    log("llama-server did not become healthy in time.")
    return False


def answer_local(prompt, timeout):
    # Greedy (temperature 0) collapses small quantized models into repetition loops on
    # open-ended prompts (repeated-token gibberish). A low-but-nonzero temperature plus
    # frequency/presence penalties keeps answers focused while avoiding the collapse.
    # NOTE: repeat_penalty is NOT an OpenAI-compatible parameter and causes HTTP 500 on
    # /v1/chat/completions. Use frequency_penalty + presence_penalty instead.
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": 0.9,
        "frequency_penalty": FREQUENCY_PENALTY,
        "presence_penalty": PRESENCE_PENALTY,
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{LOCAL_PORT}/v1/chat/completions",
        data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body_text = e.read().decode("utf-8", errors="replace")
        log(f"HTTP {e.code} from server: {body_text[:300]}")
        raise
    return data["choices"][0]["message"]["content"].strip()


def answer_fireworks(prompt, timeout):
    if not (ENABLE_FIREWORKS_FALLBACK and FIREWORKS_API_KEY and MODEL_CHEAP):
        return None
    body = json.dumps({
        "model": MODEL_CHEAP,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
    }).encode("utf-8")
    url = FIREWORKS_BASE_URL.rstrip("/") + "/chat/completions"
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {FIREWORKS_API_KEY}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        log(f"Fireworks fallback failed: {e}")
        return None


def load_tasks():
    with open(TASK_INPUT_PATH, "r") as f:
        tasks = json.load(f)
    if not isinstance(tasks, list):
        raise ValueError("tasks.json must be a JSON array")
    return tasks


def write_results(results):
    out_dir = os.path.dirname(TASK_OUTPUT_PATH)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    tmp = TASK_OUTPUT_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2)
    os.replace(tmp, TASK_OUTPUT_PATH)  # atomic — never a half-written results.json


def main():
    try:
        tasks = load_tasks()
    except Exception as e:
        log(f"FATAL: cannot read tasks from {TASK_INPUT_PATH}: {e}")
        write_results([])
        sys.exit(1)

    log(f"{len(tasks)} task(s) from {TASK_INPUT_PATH}")
    server_ok = start_local_server()

    # Warmup: the FIRST request to a freshly-loaded llama-server returns corrupted output (or a
    # 500) on this build — subsequent requests are correct. Absorb that first request with a
    # throwaway prompt so every real task is served warm. Retry once in case warmup itself 500s.
    if server_ok:
        for _ in range(2):
            try:
                answer_local("Hello.", min(30.0, PER_TASK_TIMEOUT))
                break
            except Exception as e:
                log(f"warmup request failed (absorbing first-request bug): {e}")
        log("warmup complete")

    results = []
    for i, task in enumerate(tasks):
        task_id = task.get("task_id", task.get("id", str(i)))
        prompt = task.get("prompt", task.get("question", task.get("input", "")))
        answer = ""

        # Global deadline guard: never risk blowing the runtime limit. Leave a margin to write.
        time_left = GLOBAL_DEADLINE - (time.monotonic() - _start_time)
        if time_left <= 5:
            log(f"global deadline reached at task {i}; filling remaining with stub answers")
            answer = ""
        elif server_ok:
            budget = min(PER_TASK_TIMEOUT, time_left - 3)
            try:
                answer = answer_local(prompt, budget)
            except Exception as e:
                log(f"local inference failed for {task_id}: {e}")
                fb = answer_fireworks(prompt, min(PER_TASK_TIMEOUT, time_left - 3))
                answer = fb if fb is not None else ""
        else:
            fb = answer_fireworks(prompt, PER_TASK_TIMEOUT)
            answer = fb if fb is not None else ""

        results.append({"task_id": task_id, "answer": answer})
        # Write after every task so a crash/kill still leaves a valid, scorable file.
        write_results(results)

    write_results(results)
    log(f"wrote {len(results)} result(s) to {TASK_OUTPUT_PATH}")


if __name__ == "__main__":
    try:
        main()
    finally:
        if _proc is not None:
            _proc.terminate()
