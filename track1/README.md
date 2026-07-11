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

## What it is (and isn't)

- **Is:** a dependency-free (Python stdlib only) HTTP proxy in front of mallana's `llama-server`,
  launched with TurboQuant defaults (`-fa on --cache-type-k q8_0 --cache-type-v turbo3`).
- **Isn't:** a cloud router, a cascade to a remote 70B, or anything that consumes billable tokens.

## Run it

**Docker (self-contained, CPU-only, bakes in the model):**

```bash
docker build -f track1/Dockerfile -t mallana-router .
docker run -p 8080:8080 mallana-router
```

**Directly against a local build:**

```bash
LOCAL_MODEL_PATH=/path/to/model.gguf python3 track1/router.py
```

## API

OpenAI-compatible: `POST /v1/chat/completions`, `POST /v1/completions`, `GET /v1/models`,
`GET /health`.

```bash
curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello!"}]}'
```

## Configuration (all optional, all local)

| Env var | Default | Purpose |
|---|---|---|
| `LOCAL_MODEL_PATH` | `/app/model.gguf` | Path to the GGUF served locally |
| `CACHE_TYPE_K` | `q8_0` | K cache type |
| `CACHE_TYPE_V` | `turbo3` | V cache type (TurboQuant compression) |
| `CTX_SIZE` | `4096` | Context window |
| `LLAMA_NGL` | `99` | GPU layers to offload (ignored on CPU-only builds) |
| `PORT` | `8080` | Router listen port |

There is deliberately no variable to point at a cloud provider.
