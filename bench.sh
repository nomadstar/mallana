#!/bin/bash
set -x

# ── Compilar ambas ramas ────────────────────────────────────────
git checkout feature/triattention-paged
cmake -B build-tri -DGGML_CUDA=ON
cmake --build build-tri --config Release --parallel 6

git checkout feature/supermerge
cmake -B build-super -DGGML_CUDA=ON
cmake --build build-super --config Release --parallel 6

# ── Correctitud (una vez por rama) ─────────────────────────────
./build-tri/bin/test-quantize-fns  > triattention_results.txt     2>&1 || true
./build-tri/bin/test-backend-ops   > triattention_backend_ops.txt  2>&1 || true
./build-super/bin/test-quantize-fns > supermerge_results.txt       2>&1 || true
./build-super/bin/test-backend-ops  > supermerge_backend_ops.txt   2>&1 || true

# ── Benchmarks por modelo ───────────────────────────────────────
OLLAMA_MANIFESTS_DIR="$HOME/.ollama/models/manifests/registry.ollama.ai/library"

if [ ! -d "$OLLAMA_MANIFESTS_DIR" ]; then
    echo "No se encontró el directorio de manifiestos de Ollama."
    exit 1
fi

find "$OLLAMA_MANIFESTS_DIR" -type f | while read -r manifest; do
    model_path="${manifest#$OLLAMA_MANIFESTS_DIR/}"
    model_name=$(echo "$model_path" | tr '/' '_')

    # Buscar el layer de tipo "model" (no license/template/params)
    digest=$(python3 -c "
import json, sys
try:
    data = json.load(open('$manifest'))
    for layer in data.get('layers', []):
        if layer.get('mediaType','').endswith('.model'):
            print(layer['digest']); break
except: pass
" 2>/dev/null)

    if [ -z "$digest" ]; then
        echo "Sin blob de modelo en $model_name, omitiendo."
        continue
    fi

    blob_file="$HOME/.ollama/models/blobs/${digest/:/-}"
    if [ ! -f "$blob_file" ]; then
        echo "Blob no encontrado: $blob_file"
        continue
    fi

    echo "=== $model_name ==="

    ./build-tri/bin/llama-bench \
      -m "$blob_file" --cache-type-v turbo3 --cache-type-k q8_0 -r 3 \
      > "triattention_bench_${model_name}.txt" 2>&1

    ./build-super/bin/llama-bench \
      -m "$blob_file" --cache-type-v turbo3 --cache-type-k q8_0 -r 3 \
      > "supermerge_bench_${model_name}.txt" 2>&1

    TRIATTENTION_STATS="${model_name}.triattention"
    if [ -f "$TRIATTENTION_STATS" ]; then
        ./build-super/bin/llama-bench \
          -m "$blob_file" --cache-type-v turbo3 --cache-type-k q8_0 \
          --triattention-stats "$TRIATTENTION_STATS" \
          --triattention-budget 2048 -r 3 \
          > "supermerge_bench_triattention_${model_name}.txt" 2>&1
    fi
done

echo "Listo."
ls -lh *bench*.txt *results*.txt *backend*.txt 2>/dev/null