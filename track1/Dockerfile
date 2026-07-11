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
    -DGGML_BACKEND_DL=ON \
    -DGGML_CPU_ALL_VARIANTS=ON \
    && cmake --build build --config Release -j$(nproc) --target llama-server

RUN mkdir -p /app/lib && \
    find build -name "*.so*" -exec cp -P {} /app/lib \;

# Stage 2: Final lightweight runtime image
FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the compiled llama-server and its shared libraries
COPY --from=builder /app/build/bin/llama-server /app/llama-server
COPY --from=builder /app/lib/ /app/

# Bake in the small 0.5B model during build to ensure instant start and no runtime downloads
RUN curl -L -o /app/model.gguf "https://huggingface.co/bartowski/Qwen2.5-Coder-0.5B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-0.5B-Instruct-Q4_K_M.gguf"

# Copy the lightweight python router
COPY track1/router.py /app/router.py
RUN chmod +x /app/router.py

# Env vars
ENV LD_LIBRARY_PATH=/app
ENV PORT=8080
ENV LOCAL_PORT=8081
ENV LOCAL_MODEL_PATH=/app/model.gguf

EXPOSE 8080

ENTRYPOINT ["python3", "/app/router.py"]
