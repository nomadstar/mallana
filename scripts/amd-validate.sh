#!/usr/bin/env bash
# ============================================================================
# mallana — validación AMD/ROCm (Track 1 sample tasks) con TurboQuant en GPU
# Pega esto en una terminal de la instancia Radeon (JupyterLab).
# Ajusta REPO y GFX si hace falta. Corre y pégame el bloque "RESUMEN" del final.
# ============================================================================
set -uo pipefail

REPO="${REPO:-$HOME/llama.cpp}"          # ruta del repo mallana en la máquina AMD
GFX="${GFX:-gfx1100}"                     # arquitectura RDNA3 de tu Radeon
MODEL_DIR="${MODEL_DIR:-$HOME/models}"
MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"
MODEL="$MODEL_DIR/qwen2.5-3b-instruct-q4_k_m.gguf"
PORT=8244
PROMPTS_JSON="$HOME/mallana_tasks.json"
OUT="$HOME/mallana_results.json"

echo "############ 0. Entorno ROCm ############"
command -v rocminfo >/dev/null && rocminfo 2>/dev/null | grep -m1 'Name:.*gfx' || echo "(rocminfo no encontrado)"
echo "REPO=$REPO  GFX=$GFX"
cd "$REPO" || { echo "!! No existe $REPO — ajusta REPO="; exit 1; }

echo "############ 1. Binario llama-server (HIP) ############"
BIN=""
for b in build/bin/llama-server build-hip/bin/llama-server; do
  [ -x "$b" ] && BIN="$PWD/$b" && break
done
if [ -z "$BIN" ]; then
  echo ">> No hay binario; compilando con HIP para $GFX (ROCWMMA FATTN OFF = baseline validado)..."
  HIPCXX="$(hipconfig -l 2>/dev/null)/clang" cmake -S . -B build \
    -DGGML_HIP=ON -DAMDGPU_TARGETS="$GFX" -DGGML_HIP_ROCWMMA_FATTN=OFF \
    -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=OFF || { echo "!! cmake falló"; exit 1; }
  cmake --build build --target llama-server -j"$(nproc)" || { echo "!! build falló"; exit 1; }
  BIN="$PWD/build/bin/llama-server"
fi
echo "BIN=$BIN"

echo "############ 2. Modelo 3B ############"
mkdir -p "$MODEL_DIR"
[ -f "$MODEL" ] || { echo ">> bajando 3B..."; curl -L --fail -o "$MODEL" "$MODEL_URL" || exit 1; }
ls -lh "$MODEL"

echo "############ 3. Tasks (10 sample) ############"
cat > "$PROMPTS_JSON" <<'JSON'
[
 {"task_id":"T01","prompt":"Name the three primary colors in the RGB color model and briefly explain why displays use RGB instead of RYB."},
 {"task_id":"T01b","prompt":"What is the difference between machine learning and deep learning? Briefly explain how each works."},
 {"task_id":"T01c","prompt":"Explain the difference between RAM and ROM in a computer. What is each type used for?"},
 {"task_id":"T02","prompt":"A warehouse starts with 2,400 units. In Q1 it sells 37% of stock. In Q2 it restocks 800 units. In Q3 it sells 640 units. How many units remain at the end of Q3?"},
 {"task_id":"T02b","prompt":"A recipe requires 3/4 cup of sugar for 12 cookies. How much sugar is needed for 30 cookies? If sugar costs $2.40 per cup, what is the total cost of sugar for 30 cookies?"},
 {"task_id":"T03","prompt":"Classify the sentiment of this customer review as Positive, Negative, or Neutral and give a one-sentence reason: 'The product arrived two days late and the packaging was damaged, but the item worked perfectly and customer support resolved my complaint within an hour.'"},
 {"task_id":"T03b","prompt":"Classify the sentiment of this tweet as Positive, Negative, or Neutral and give a one-sentence reason: 'Just got my order. Box was dented and the manual was missing, but honestly the device itself is flawless and set up in under 5 minutes.'"},
 {"task_id":"T04","prompt":"Summarize the following passage in exactly two sentences:\n\n'Machine learning is increasingly deployed in healthcare for diagnosis, treatment planning, and patient monitoring. These systems analyse medical images, predict patient deterioration, and spot patterns in electronic health records that might be missed by human clinicians. However, concerns remain about model interpretability, data privacy, liability when errors occur, and the potential for algorithmic bias to worsen existing healthcare disparities. Regulatory frameworks are still catching up with the pace of deployment, creating uncertainty for healthcare providers and technology developers alike.'"},
 {"task_id":"T04b","prompt":"Summarize the following passage in exactly three bullet points, each no longer than 15 words:\n\n'Remote work has transformed how companies operate globally. Employees gain flexibility and reduced commute times, leading to reported improvements in work-life balance. However, challenges persist around collaboration, company culture, and the blurring of personal and professional boundaries. Organisations are responding by investing in digital collaboration tools and rethinking office space as a hub for social and creative work rather than daily attendance.'"},
 {"task_id":"T05","prompt":"Extract all named entities from the following text and label each as PERSON, ORGANIZATION, LOCATION, or DATE:\n\n'On March 15 2023, Sundar Pichai announced that Google would open a new AI research lab in Zurich, partnering with ETH Zurich to focus on large language model safety.'"}
]
JSON

run_sweep() { # $1=label  $2=ctk  $3=ctv  $4=fa
  local label="$1" ctk="$2" ctv="$3" fa="$4"
  echo "############ CONFIG: $label  (fa=$fa K=$ctk V=$ctv, -ngl 99 GPU) ############"
  "$BIN" -m "$MODEL" --port $PORT --host 127.0.0.1 -c 2048 -ngl 99 \
      -fa "$fa" --cache-type-k "$ctk" --cache-type-v "$ctv" >/tmp/srv_$PORT.log 2>&1 &
  local pid=$!
  for i in $(seq 1 90); do
    curl -s -m2 http://127.0.0.1:$PORT/health 2>/dev/null | grep -q ok && break
    kill -0 $pid 2>/dev/null || { echo "!! server murió:"; tail -15 /tmp/srv_$PORT.log; return; }
    sleep 1
  done
  # warmup (absorbe la primera request)
  curl -s -m30 http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"Hi"}],"max_tokens":8}' >/dev/null 2>&1
  local t0 nsec
  t0=$(date +%s.%N)
  python3 - "$PORT" "$PROMPTS_JSON" "$OUT" <<'PY'
import sys,json,time,urllib.request
port,inp,out=sys.argv[1],sys.argv[2],sys.argv[3]
tasks=json.load(open(inp)); res=[]
for t in tasks:
    body=json.dumps({"messages":[{"role":"user","content":t["prompt"]}],
        "max_tokens":768,"temperature":0.3,"top_p":0.9,"stream":False}).encode()
    r=urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,headers={"Content-Type":"application/json"},method="POST")
    s=time.time()
    try:
        with urllib.request.urlopen(r,timeout=120) as resp:
            a=json.loads(resp.read())["choices"][0]["message"]["content"].strip()
    except Exception as e: a=f"[ERR {e}]"
    dt=time.time()-s
    print(f"  {t['task_id']:5s} {dt:5.1f}s  {a[:90]!r}")
    res.append({"task_id":t["task_id"],"answer":a})
json.dump(res,open(out,"w"),indent=2)
PY
  nsec=$(python3 -c "print(f'{ $(date +%s.%N) - $t0 :.1f}')")
  echo "  >> tiempo total 10 tasks: ${nsec}s   (resultados: $OUT)"
  kill $pid 2>/dev/null; wait $pid 2>/dev/null
  echo
}

run_sweep "A f16 baseline"  f16  f16    on
run_sweep "B TurboQuant"    q8_0 turbo3 on

echo "############ RESUMEN — pégame esto ############"
echo "GPU: $(rocminfo 2>/dev/null | grep -m1 'Marketing Name' | sed 's/.*: *//' || echo '?')"
echo "Modelo: $(basename "$MODEL")"
echo "Resultados turbo3 en: $OUT"
echo "(arriba: por-tarea con segundos y preview; y el tiempo total de cada config)"
