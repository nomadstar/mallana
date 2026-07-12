# syntax=docker/dockerfile:1
#
# Track 1 submission image — PULL-AND-RUN, model-agnostic.
#
# The evaluator PULLS this prebuilt image (ghcr.io/nomadstar/mallana:track1-latest) and mounts
# their own model, so:
#   - The llama.cpp compile happens HERE at push time, never during evaluation.
#   - NO model is baked in (that was ~1 GB of dead weight slowing the pull). The evaluator
#     mounts a model dir at /models (or sets LOCAL_MODEL_PATH); agent.py picks it up.
# Net effect: a small image that pulls fast and a batch agent that exits promptly — no build,
# no runtime download, nothing left hanging.

# ---------------------------------------------------------------------------
# Stage 1: build llama-server (CPU-only). Runs at push time on the maintainer's
# machine — its cost never counts against the evaluation runtime limit.
# ---------------------------------------------------------------------------
FROM ubuntu:22.04 AS builder

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Static-ish CPU build of llama-server (no GPU deps, portable across hosts).
RUN cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_NATIVE=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DGGML_BACKEND_DL=OFF \
    -DGGML_CPU_ALL_VARIANTS=OFF \
    -DBUILD_SHARED_LIBS=OFF \
    && cmake --build build --config Release -j$(nproc) --target llama-server

# ---------------------------------------------------------------------------
# Stage 2: slim runtime. Precompiled binary + batch agent only — no model.
# ---------------------------------------------------------------------------
FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Precompiled llama-server (statically linked) — instant start, no build at eval time.
COPY --from=builder /app/build/bin/llama-server /app/llama-server

# OPTIONAL self-contained model bake, OFF by default. The evaluation image ships WITHOUT a model
# (the scorer mounts one at /models). For a self-contained local-test image, pass a URL:
#   docker build --build-arg MODEL_URL=https://.../qwen2.5-1.5b-instruct-q4_k_m.gguf ...
ARG MODEL_URL=""
RUN if [ -n "$MODEL_URL" ]; then \
        echo "Baking model from $MODEL_URL" && \
        curl -L -o /app/model.gguf "$MODEL_URL"; \
    else \
        echo "No MODEL_URL: shipping model-agnostic (evaluator mounts /models)."; \
    fi

# Batch agent (the Track 1 contract entrypoint) + optional live-server alternative.
COPY track1/agent.py /app/agent.py
COPY track1/router.py /app/router.py
RUN chmod +x /app/agent.py /app/router.py

# --- Runtime config (all local, zero Fireworks tokens by default) ---
# Model resolution order in agent.py: LOCAL_MODEL_PATH -> first *.gguf in MODELS_DIR (/models)
# -> first *.gguf in the baked /app dir (only present if MODEL_URL was passed at build).
ENV MODELS_DIR=/models
ENV LOCAL_PORT=8081
ENV CACHE_TYPE_K=f16
ENV CACHE_TYPE_V=f16
ENV CTX_SIZE=2048
ENV FLASH_ATTN=off
ENV MAX_TOKENS=256
ENV TEMPERATURE=0.7
ENV FREQUENCY_PENALTY=0.3
ENV PRESENCE_PENALTY=0.3
ENV PER_TASK_TIMEOUT=30
ENV GLOBAL_DEADLINE=900
ENV TASK_INPUT_PATH=/input/tasks.json
ENV TASK_OUTPUT_PATH=/output/results.json

# The scorer mounts /input (+ a model), runs the container, reads /output, then it exits.
ENTRYPOINT ["python3", "/app/agent.py"]
