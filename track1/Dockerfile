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
    ccache \
    cmake \
    git \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Static-ish CPU build of llama-server (no GPU deps, portable across hosts).
# ccache lives in a BuildKit cache mount, so it persists across builds: the first build is a full
# compile, every rebuild after only recompiles the translation units that actually changed
# (near-instant even though `COPY . .` invalidates the layer). Needs BuildKit (default on 6.x).
RUN --mount=type=cache,target=/root/.cache/ccache \
    cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DGGML_NATIVE=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DGGML_BACKEND_DL=OFF \
    -DGGML_CPU_ALL_VARIANTS=OFF \
    -DBUILD_SHARED_LIBS=OFF \
    && cmake --build build --config Release -j$(nproc) --target llama-server \
    && ccache --show-stats

# ---------------------------------------------------------------------------
# Stage 2: runtime. Precompiled binary + batch agent + a baked instruct model
# (self-contained: the scorer does not mount a model).
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

# Bake a general-purpose instruct model into the image — the scorer does NOT mount a model, so the
# submission MUST be self-contained (a model-less image produces empty answers → 0% accuracy gate).
# Qwen2.5-1.5B-Instruct (Apache-2.0): capable on general tasks yet small/fast enough to stay well
# within the CPU per-task time budget (a 3B times out on CPU). If the scorer DOES mount /models, that
# takes precedence (agent.py resolves /models first, then this baked /app/model.gguf as the fallback).
ARG MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"
RUN echo "Baking model from $MODEL_URL" && curl -L --fail -o /app/model.gguf "$MODEL_URL"

# Batch agent (the Track 1 contract entrypoint) + optional live-server alternative.
COPY track1/agent.py /app/agent.py
COPY track1/router.py /app/router.py
RUN chmod +x /app/agent.py /app/router.py

# --- Runtime config (all local, zero Fireworks tokens by default) ---
# Model resolution order in agent.py: LOCAL_MODEL_PATH -> first *.gguf in MODELS_DIR (/models)
# -> first *.gguf in the baked /app dir (only present if MODEL_URL was passed at build).
ENV MODELS_DIR=/models
ENV LOCAL_PORT=8081
# Correctness-first for the accuracy gate: f16 KV cache, no FA. On a 1.5B model that fits RAM easily,
# TurboQuant's compressed V-cache (turbo3) buys no accuracy — its value is fitting BIGGER models / longer
# context in less memory, which this small-model-on-CPU submission isn't bound by. TurboQuant is validated
# and is the showcase for GPU/large-model runs; enable it here with FLASH_ATTN=on CACHE_TYPE_V=turbo3.
ENV FLASH_ATTN=off
ENV CACHE_TYPE_K=f16
ENV CACHE_TYPE_V=f16
ENV CTX_SIZE=2048
# 768 leaves room for multi-step reasoning to finish (256 truncated math/explanation answers).
ENV MAX_TOKENS=768
# Low-but-nonzero temp for deterministic-ish tasks; penalties off (they distort exact-format asks).
ENV TEMPERATURE=0.3
ENV FREQUENCY_PENALTY=0.0
ENV PRESENCE_PENALTY=0.0
ENV PER_TASK_TIMEOUT=30
ENV GLOBAL_DEADLINE=900
ENV TASK_INPUT_PATH=/input/tasks.json
ENV TASK_OUTPUT_PATH=/output/results.json

# The scorer mounts /input (+ a model), runs the container, reads /output, then it exits.
ENTRYPOINT ["python3", "/app/agent.py"]
