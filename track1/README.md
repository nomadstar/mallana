# Mallana Local Router — Track 1 (General-Purpose AI Agent)

> **100% on-device. Zero cloud. Zero billable tokens — by construction.**

This submission is an OpenAI-compatible gateway that serves **every** request from a model
running on the local machine. There is no cloud provider, no API key, and no escape hatch to a
remote endpoint. On a *"fewest tokens wins, subject to an accuracy gate"* leaderboard, our token
count is **0** — not because we route cleverly between local and cloud, but because we never
leave the device.

## Why local-only is the whole point

This is the thesis of the [MANIFESTO](../MANIFESTO.md):

> **How much useful intelligence can we run on the hardware people already own?**

The goal is to reduce the **cost of intelligence itself** — enough that a modest machine (a
laptop, a mini-PC, a *toaster*) can run a genuinely capable model. Shipping tokens to a
datacenter is the opposite of that. So we don't.

## Our edge: TurboQuant KV compression

A pure-local strategy is only as good as the model you can fit on-device. This is where mallana
wins: **TurboQuant** compresses the KV value cache **4.6–6.4×** (`turbo3` / `turbo2`), so a
larger, more accurate model — and a longer context — fits in the same RAM/VRAM. More capability
per byte means more questions answered *well*, locally, at 0 tokens. On an accuracy-gated,
token-minimizing leaderboard, running a stronger local model is exactly what breaks the tie.

Validated on real AMD hardware (Radeon gfx1100 / RDNA3): TurboQuant costs ~0.5% prompt and ~6%
generation throughput while shrinking the V-cache up to 6.4× (see [../docs/benchmarks.md](../docs/benchmarks.md)).

## Submission contract

The scorer runs the container as a **batch agent** (dependency-free, Python stdlib only):

- reads tasks from `TASK_INPUT_PATH` (default `/input/tasks.json`) — a JSON array of
  `{"task_id": str, "prompt": str}`;
- answers each task on-device with mallana's `llama-server`, **0 Fireworks tokens**. The shipped
  defaults enable **TurboQuant** (`-fa on --cache-type-k q8_0 --cache-type-v turbo3`) — the
  6.4×-compressed value cache, validated coherent on the CPU-only image. For a plain baseline set
  `FLASH_ATTN=off CACHE_TYPE_K=f16 CACHE_TYPE_V=f16`;
- writes `TASK_OUTPUT_PATH` (default `/output/results.json`) — a JSON array of
  `{"task_id": str, "answer": str}`, task IDs preserved exactly.

**Timeout-safe by design** (the scorer enforces a runtime limit): a per-task wall-clock timeout
(`PER_TASK_TIMEOUT`), a global deadline (`GLOBAL_DEADLINE`) that fills any remaining tasks
rather than overrun, and an **atomic write after every task** so a partial run still produces a
valid, scorable `results.json` — never a missing file, never a TIMEOUT-with-no-output.

An optional Fireworks fallback (`ENABLE_FIREWORKS_FALLBACK=1`, off by default) escalates only
tasks the local model fails, to `MODEL_CHEAP` — keeping tokens near zero while protecting the
accuracy gate.

## Run it

The published image is **precompiled and model-agnostic**: the `llama.cpp` build happens once at
push time (never during evaluation), and **no model is baked in** — so the image pulls fast and
you choose which GGUF to run. Mount a directory containing your `.gguf` at `/models`:

```bash
docker pull ghcr.io/nomadstar/mallana:track1-latest
docker run \
  -v "$PWD/models:/models" \
  -v "$PWD/input:/input" \
  -v "$PWD/output:/output" \
  ghcr.io/nomadstar/mallana:track1-latest
```

The agent picks the first `*.gguf` under `/models` (or set `LOCAL_MODEL_PATH` to an exact file).
A small instruct model such as
[`qwen2.5-1.5b-instruct-q4_k_m.gguf`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF)
runs comfortably on CPU; TurboQuant's compressed V-cache leaves headroom for larger models.

**Self-contained image (optional, for offline testing)** — bake a model at build time:

```bash
docker build -f track1/Dockerfile -t mallana-agent \
  --build-arg MODEL_URL=https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf .
docker run -v "$PWD/input:/input" -v "$PWD/output:/output" mallana-agent
```

**Directly against a local build:**

```bash
TASK_INPUT_PATH=tasks.json TASK_OUTPUT_PATH=results.json \
LOCAL_MODEL_PATH=/path/to/model.gguf python3 track1/agent.py
```

## Live-server alternative (`router.py`)

For interactive use, `router.py` exposes the same local model over an OpenAI-compatible API
(`/v1/chat/completions`, `/v1/models`, `/health`). It is **not** the submission entrypoint —
`agent.py` (the batch contract) is.

## Configuration (all optional, all local)

| Env var | Default | Purpose |
|---|---|---|
| `MODELS_DIR` | `/models` | Dir scanned for a `*.gguf` when `LOCAL_MODEL_PATH` is unset |
| `LOCAL_MODEL_PATH` | *(unset)* | Exact path to the GGUF served locally (overrides `MODELS_DIR`) |
| `CACHE_TYPE_K` | `q8_0` | K cache type (`f16` for uncompressed) |
| `CACHE_TYPE_V` | `turbo3` | V cache type — TurboQuant 6.4× compression (`f16` for uncompressed) |
| `FLASH_ATTN` | `on` | Flash Attention (required by TurboQuant's compressed V-cache) |
| `MAX_TOKENS` | `768` | Max generated tokens per task |
| `TEMPERATURE` | `0.3` | Sampling temperature |
| `CTX_SIZE` | `2048` | Context window |
| `LLAMA_NGL` | `99` | GPU layers to offload (ignored on CPU-only builds) |
| `PORT` | `8080` | Router listen port |

There is deliberately no variable to point at a cloud provider.
