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

# Bake in the small 0.5B model during build to ensure instant start and no runtime downloads
RUN curl -L -o /app/model.gguf "https://huggingface.co/bartowski/Qwen2.5-Coder-0.5B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-0.5B-Instruct-Q4_K_M.gguf"

# Copy the lightweight python router
COPY track1/router.py /app/router.py
RUN chmod +x /app/router.py

# Env vars
ENV PORT=8080
ENV LOCAL_PORT=8081
ENV LOCAL_MODEL_PATH=/app/model.gguf

EXPOSE 8080

ENTRYPOINT ["python3", "/app/router.py"]

