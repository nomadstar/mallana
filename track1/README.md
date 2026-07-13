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

**Where TurboQuant applies — and where it doesn't.** Its win is *memory*: fitting a larger model or
a longer context in the same VRAM. That matters on GPU. The Track 1 grader, however, runs
**CPU-only** (2 vCPU / 4 GB), where a 3B already fits comfortably with lossless `f16` KV and the
only score that matters is the LLM-judge accuracy gate (our token count is 0 either way). So the
**submission image defaults to lossless `f16` KV with Flash Attention off** — the
accuracy-and-reliability-max, fastest-on-CPU path — and leaves nothing to a lossy cache near the
gate. TurboQuant remains a first-class GPU showcase: on AMD (Radeon gfx1100 / RDNA3, ROCm) the 3B
runs all sample tasks coherently with `-ngl 99` at ~2 s/task, and `turbo3` / `turbo2` V-cache
compression under Flash Attention is validated by `scripts/amd-validate.sh`. Enable it with
`FLASH_ATTN=on CACHE_TYPE_K=q8_0 CACHE_TYPE_V=turbo3`.

## Submission contract

The scorer runs the container as a **batch agent** (dependency-free, Python stdlib only):

- reads tasks from `TASK_INPUT_PATH` (default `/input/tasks.json`) — a JSON array of
  `{"task_id": str, "prompt": str}`;
- answers each task on-device with mallana's `llama-server`, **0 Fireworks tokens**. The shipped
  defaults use TurboQuant V-cache compression (`q8_0` K + `turbo3` V with Flash Attention) —
  validated on CUDA and ROCm (gfx1100). TurboQuant demonstrates the project's core
  capability-density thesis; disable with `FLASH_ATTN=off CACHE_TYPE_K=f16 CACHE_TYPE_V=f16`;
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

The published image is **precompiled and self-contained**: the `llama.cpp` build happens once at
push time (never during evaluation) and a general-purpose instruct model
([Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF)) is **baked in**,
so the scorer can just pull and run — no model to provide, no runtime download. The 3B is
materially more accurate than the 1.5B on the graded task mix (multi-step math, exact-format
summaries, sentiment nuance) yet still finishes the sample set within budget on 2 vCPU
(~14 s/task) thanks to the concise/reason-only-for-math prompt, low temperature, and streaming:

```bash
docker pull ghcr.io/nomadstar/mallana:track1-latest
docker run \
  -v "$PWD/input:/input" \
  -v "$PWD/output:/output" \
  ghcr.io/nomadstar/mallana:track1-latest
```

To run a **different** model, mount a directory containing your `.gguf` at `/models` (it takes
precedence over the baked model) or set `LOCAL_MODEL_PATH` to an exact file. Rebuild with a
different baked model via `--build-arg MODEL_URL=…`.

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
| `CACHE_TYPE_K` | `q8_0` | K cache type (`f16` for no compression) |
| `CACHE_TYPE_V` | `turbo3` | V cache type (`turbo3` for TurboQuant 4.6× compression, requires `FLASH_ATTN=on`) |
| `FLASH_ATTN` | `on` | Flash Attention (required for TurboQuant compressed V-cache) |
| `MAX_TOKENS` | `448` | Max generated tokens per task (room for brief step-by-step math) |
| `SYSTEM_PROMPT` | *(reason-only-for-math default)* | Concise answers; shows steps only for arithmetic/logic; obeys exact formats |
| `PER_TASK_TIMEOUT` | `45` | Per-task wall-clock budget; a timeout keeps the streamed partial answer |
| `GLOBAL_DEADLINE` | `280` | Whole-run budget; self-terminates cleanly before the grader can SIGKILL |
| `TEMPERATURE` | `0.1` | Low: graded tasks are deterministic (higher temp flips borderline math) |
| `CTX_SIZE` | `2048` | Context window |
| `LLAMA_NGL` | `99` | GPU layers to offload (ignored on CPU-only builds) |
| `PORT` | `8080` | Router listen port |

There is deliberately no variable to point at a cloud provider.
