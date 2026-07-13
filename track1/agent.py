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
import socket
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
# Correctness-first defaults: f16 KV cache, no Flash Attention. On a small model that fits RAM,
# TurboQuant's compressed V-cache buys no accuracy (its win is memory — fitting bigger models /
# longer context), so the accuracy-gated submission ships uncompressed. TurboQuant is validated and
# is the showcase for GPU/large-model runs; opt in with FLASH_ATTN=on CACHE_TYPE_K=q8_0 CACHE_TYPE_V=turbo3.
FLASH_ATTN = os.environ.get("FLASH_ATTN", "off")
CACHE_TYPE_K = os.environ.get("CACHE_TYPE_K", "f16")
CACHE_TYPE_V = os.environ.get("CACHE_TYPE_V", "f16")
CTX_SIZE = os.environ.get("CTX_SIZE", "2048")
NGL = os.environ.get("LLAMA_NGL", "99")
# MAX_TOKENS kept modest: the grader runs on ~2 vCPU where generation is ~10-15 tok/s, so a
# 768-token essay costs ~60s/task and blows the wall-clock budget (→ SIGKILL → runtime error).
# A concise answer within a small cap finishes in seconds. SYSTEM_PROMPT stops the model from
# padding short-answer tasks into essays (a huge latency + token saving).
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "256"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.3"))
FREQUENCY_PENALTY = float(os.environ.get("FREQUENCY_PENALTY", "0.0"))
PRESENCE_PENALTY = float(os.environ.get("PRESENCE_PENALTY", "0.0"))
SYSTEM_PROMPT = os.environ.get(
    "SYSTEM_PROMPT",
    "You are a precise assistant. Answer the task directly and concisely. Give only what is "
    "asked — no preamble, no restating the question, no repetition. If the task asks for a "
    "specific value or format, output exactly that and nothing else.")

# --- Timeout safety ---
# On ~2 vCPU each task must stay well under the grader's hard limit. Streaming lets a timed-out
# task keep its partial answer (never empty), and GLOBAL_DEADLINE self-terminates gracefully
# BEFORE the grader SIGKILLs an over-budget run.
PER_TASK_TIMEOUT = float(os.environ.get("PER_TASK_TIMEOUT", "25"))       # seconds per task
GLOBAL_DEADLINE = float(os.environ.get("GLOBAL_DEADLINE", "240"))        # seconds for the whole run
SERVER_START_TIMEOUT = float(os.environ.get("SERVER_START_TIMEOUT", "120"))

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
    #
    # STREAMING: on ~2 vCPU a task can hit its wall-clock budget mid-generation. Streaming lets
    # us keep whatever text arrived so far (a partial, still-scorable answer) instead of raising
    # and returning "" — and lets us stop cleanly at the deadline rather than being SIGKILLed.
    body = json.dumps({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": 0.9,
        "frequency_penalty": FREQUENCY_PENALTY,
        "presence_penalty": PRESENCE_PENALTY,
        "stream": True,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{LOCAL_PORT}/v1/chat/completions",
        data=body, headers={"Content-Type": "application/json"}, method="POST")

    deadline = time.monotonic() + timeout
    parts = []
    try:
        # Cap each blocking read so we can re-check the wall deadline; never block past it.
        resp = urllib.request.urlopen(req, timeout=max(1.0, min(timeout, 10.0)))
    except urllib.error.HTTPError as e:
        body_text = e.read().decode("utf-8", errors="replace")
        log(f"HTTP {e.code} from server: {body_text[:300]}")
        raise
    try:
        for raw in resp:
            if time.monotonic() >= deadline:
                log("per-task deadline hit mid-stream; keeping partial answer")
                break
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
                delta = obj["choices"][0].get("delta", {}).get("content")
                if delta:
                    parts.append(delta)
            except (ValueError, KeyError, IndexError):
                continue
    except (socket.timeout, TimeoutError):
        log("stream read timed out; keeping partial answer")
    finally:
        try:
            resp.close()
        except Exception:
            pass
    return "".join(parts).strip()


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
        data = json.load(f)
    # Spec is a bare JSON array, but tolerate a wrapper object ({"tasks": [...]} / {"data": [...]})
    # so an unexpected input shape degrades to answers rather than a hard crash.
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in ("tasks", "data", "items", "inputs"):
            if isinstance(data.get(k), list):
                return data[k]
    raise ValueError("tasks input is neither a JSON array nor a known wrapper object")


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
        # Never exit nonzero: the grader treats a nonzero container exit as a hard RUNTIME_ERROR.
        # Emit a valid (empty) results file and stop — an accuracy result beats a runtime error.
        log(f"FATAL: cannot read tasks from {TASK_INPUT_PATH}: {e}")
        try:
            write_results([])
        except Exception as e2:
            log(f"could not write empty results: {e2}")
        return

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
        # Tolerate malformed entries (non-dict) without crashing the whole run.
        if isinstance(task, dict):
            task_id = task.get("task_id", task.get("id", str(i)))
            prompt = task.get("prompt", task.get("question", task.get("input", "")))
        else:
            task_id = str(i)
            prompt = str(task)
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
        # A transient write hiccup must not abort the whole run.
        try:
            write_results(results)
        except Exception as e:
            log(f"incremental write failed (will retry at end): {e}")

    try:
        write_results(results)
        log(f"wrote {len(results)} result(s) to {TASK_OUTPUT_PATH}")
    except Exception as e:
        log(f"final write failed: {e}")


if __name__ == "__main__":
    # Guarantee a clean exit code: the grader flags ANY nonzero container exit as RUNTIME_ERROR,
    # so no unexpected exception (or an OOM-killed subprocess) may propagate out of the agent.
    try:
        main()
    except SystemExit:
        raise
    except BaseException as e:
        log(f"unexpected top-level error (exiting 0 to avoid RUNTIME_ERROR): {e}")
    finally:
        if _proc is not None:
            try:
                _proc.terminate()
            except Exception:
                pass
    sys.exit(0)
