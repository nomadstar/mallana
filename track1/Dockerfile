# Stage 1: Build llama.cpp (CPU only)
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

# Build CPU-only version of llama-server (fast build, no GPU dependencies)
RUN cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_NATIVE=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DGGML_BACKEND_DL=OFF \
    -DGGML_CPU_ALL_VARIANTS=OFF \
    -DBUILD_SHARED_LIBS=OFF \
    && cmake --build build --config Release -j$(nproc) --target llama-server

# Stage 2: Final lightweight runtime image
FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the statically compiled llama-server
COPY --from=builder /app/build/bin/llama-server /app/llama-server

# Bake in a general-purpose instruct model during build (instant start, no runtime downloads).
# Qwen2.5-1.5B-Instruct (Apache-2.0) — a real capability jump over the old 0.5B coder model for
# a general-purpose agent, still small enough to stay responsive on CPU. Overridable at build
# time (e.g. --build-arg MODEL_URL=.../qwen2.5-3b-instruct-q4_k_m.gguf) since TurboQuant's
# compressed V-cache leaves headroom for a larger model on capable hardware.
ARG MODEL_URL=https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf
RUN curl -L -o /app/model.gguf "${MODEL_URL}"

# Copy the batch agent (submission entrypoint) and the local server it drives.
# agent.py is the Track 1 contract: reads TASK_INPUT_PATH, writes TASK_OUTPUT_PATH.
# router.py is kept as an optional live-server alternative for interactive use.
COPY track1/agent.py /app/agent.py
COPY track1/router.py /app/router.py
RUN chmod +x /app/agent.py /app/router.py

# Env vars — all local. TurboQuant KV compression on by default. Answering on-device costs
# ZERO Fireworks tokens (the optimal token score); the agent only escalates to Fireworks if
# ENABLE_FIREWORKS_FALLBACK is set and the local model fails a task.
ENV LOCAL_PORT=8081
ENV LOCAL_MODEL_PATH=/app/model.gguf
ENV CACHE_TYPE_K=q8_0
ENV CACHE_TYPE_V=turbo3
ENV CTX_SIZE=4096
ENV TASK_INPUT_PATH=/input/tasks.json
ENV TASK_OUTPUT_PATH=/output/results.json

# The scorer runs the container, mounts /input, and reads /output before the runtime limit.
ENTRYPOINT ["python3", "/app/agent.py"]

