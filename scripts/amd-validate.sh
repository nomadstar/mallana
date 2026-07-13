#!/usr/bin/env bash
# ============================================================================
# mallana — validación AMD/ROCm (Track 1 sample tasks) con TurboQuant en GPU
#
# Pégalo en una terminal de la instancia Radeon (JupyterLab) y córrelo:
#     REPO=~/mallana bash scripts/amd-validate.sh
#
# Auto-detecta la arquitectura GPU, compila llama-server con HIP en un dir
# dedicado (build-hip/), VERIFICA que el binario está enlazado a ROCm y que
# las capas se descargan a GPU, y corre dos configs (f16 baseline y TurboQuant
# turbo3). Al final pega el bloque "RESUMEN".
#
# Variables (todas opcionales):
#   REPO   ruta del repo mallana         (default: intenta ~/mallana, ~/llama.cpp, y el cwd)
#   GFX    arquitectura GPU, p.ej gfx1100 (default: auto-detecta)
#   ROCWMMA=1  activa rocWMMA FATTN (mejor FA en RDNA3+, requiere rocwmma-dev)
#   HSA_OVERRIDE_GFX_VERSION=11.0.0  si tu GPU no está soportada oficialmente
# ============================================================================
set -uo pipefail

export PATH="/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH"
GFX_OVERRIDE="${GFX:-}"
MODEL_DIR="${MODEL_DIR:-$HOME/models}"
MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"
MODEL="$MODEL_DIR/qwen2.5-3b-instruct-q4_k_m.gguf"
PORT=8244
PROMPTS_JSON="$HOME/mallana_tasks.json"
OUT="$HOME/mallana_results.json"
ROCWMMA="${ROCWMMA:-0}"          # 0 = baseline robusto; 1 = FA acelerado (RDNA3+, necesita headers)

die() { echo "!! $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
echo "############ 0. Entorno ROCm ############"
command -v hipconfig >/dev/null 2>&1 || die "No encuentro hipconfig — ¿está ROCm instalado y en PATH? \
Instala ROCm: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html"
echo "ROCm path : $(hipconfig -R 2>/dev/null)"
echo "hip clang : $(hipconfig -l 2>/dev/null)/clang"
if command -v rocminfo >/dev/null 2>&1; then
  # El PRIMER "Marketing Name" suele ser el CPU; reporta el nombre del agente GPU (bloque con Name: gfx)
  rocminfo 2>/dev/null | awk '/^ *Name: *gfx/{f=1} f&&/Marketing Name/{sub(/^ *Marketing Name: */,"GPU: ");print;exit}' || true
  rocminfo 2>/dev/null | grep -qE '^ *Name: *gfx' \
    || echo "(AVISO: rocminfo NO lista un agente GPU 'gfx' — ¿la Radeon está asignada a este contenedor? Sin GPU, -ngl 99 corre en CPU.)"
fi
[ -e /dev/kfd ] || echo "(aviso: /dev/kfd no existe — el contenedor/host quizá no expone la GPU)"

# --- localizar el repo mallana ---
if [ -n "${REPO:-}" ]; then :; else
  for cand in "$HOME/mallana" "$HOME/llama.cpp" "$PWD"; do
    [ -f "$cand/CMakeLists.txt" ] && REPO="$cand" && break
  done
fi
[ -n "${REPO:-}" ] && [ -f "$REPO/CMakeLists.txt" ] || die "No encuentro el repo mallana. Pásalo con REPO=/ruta/al/repo"
cd "$REPO" || die "No puedo entrar a $REPO"
echo "REPO=$REPO"

# --- auto-detectar la arquitectura GPU (gfxNNNN) ---
detect_gfx() {
  local g=""
  if command -v rocminfo >/dev/null 2>&1; then
    g=$(rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-f]+' || true)
  fi
  if [ -z "$g" ] && command -v amdgpu-arch >/dev/null 2>&1; then
    g=$(amdgpu-arch 2>/dev/null | grep -m1 -oE 'gfx[0-9a-f]+' || true)
  fi
  if [ -z "$g" ] && command -v offload-arch >/dev/null 2>&1; then
    g=$(offload-arch 2>/dev/null | grep -m1 -oE 'gfx[0-9a-f]+' || true)
  fi
  echo "$g"
}
GFX="${GFX_OVERRIDE:-$(detect_gfx)}"
[ -n "$GFX" ] || GFX="gfx1100"   # último recurso: RDNA3 (RX 7900)
echo "GFX=$GFX  (override con GFX=... si es incorrecto)"

# ---------------------------------------------------------------------------
echo "############ 1. Binario llama-server (HIP) ############"
# IMPORTANTE: usamos un dir DEDICADO build-hip/ para no reutilizar por error un
# build CPU en build/ (eso correría en CPU y parecería que ROCm "funciona").
BIN="$PWD/build-hip/bin/llama-server"
NEED_BUILD=1
if [ -x "$BIN" ]; then
  # sólo reutilizar si ese binario está realmente enlazado a HIP/ROCm
  if ldd "$BIN" 2>/dev/null | grep -qiE 'amdhip|hsa-runtime|rocm'; then
    NEED_BUILD=0
    echo ">> Reuso binario HIP existente: $BIN"
  else
    echo ">> build-hip/ existe pero el binario NO está enlazado a ROCm — recompilo."
  fi
fi

if [ "$NEED_BUILD" = 1 ]; then
  ROCWMMA_FLAG="OFF"; [ "$ROCWMMA" = 1 ] && ROCWMMA_FLAG="ON"
  echo ">> Compilando HIP para $GFX (ROCWMMA FATTN=$ROCWMMA_FLAG)..."
  HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
    cmake -S . -B build-hip \
      -DGGML_HIP=ON \
      -DGPU_TARGETS="$GFX" \
      -DGGML_HIP_ROCWMMA_FATTN="$ROCWMMA_FLAG" \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLAMA_BUILD_TESTS=OFF \
    || die "cmake (configuración HIP) falló — revisa que ROCm esté completo (rocm-dev/hip-dev)."
  cmake --build build-hip --target llama-server -j"$(nproc)" \
    || die "build HIP falló — mira los errores arriba (típico: falta rocm-device-libs o GPU_TARGETS incorrecto)."
fi
[ -x "$BIN" ] || die "No se generó el binario $BIN"

echo ">> Verificando enlace ROCm del binario:"
if ldd "$BIN" 2>/dev/null | grep -iE 'amdhip|hsa-runtime|rocm'; then
  echo "   OK: llama-server está enlazado a ROCm."
else
  die "El binario NO está enlazado a ROCm (correría en CPU). Aborto para no dar un falso positivo."
fi
echo "BIN=$BIN"

# ---------------------------------------------------------------------------
echo "############ 2. Modelo 3B ############"
mkdir -p "$MODEL_DIR"
HF_REPO="Qwen/Qwen2.5-3B-Instruct-GGUF"
HF_FILE="qwen2.5-3b-instruct-q4_k_m.gguf"

fetch_model() {
  # 1) ya está, o el usuario dejó CUALQUIER *.gguf en MODEL_DIR (descarga manual / scp)
  [ -s "$MODEL" ] && { echo ">> modelo ya presente"; return 0; }
  local existing
  existing=$(ls "$MODEL_DIR"/*.gguf 2>/dev/null | head -1 || true)
  [ -n "$existing" ] && { MODEL="$existing"; echo ">> uso gguf existente: $MODEL"; return 0; }
  # 2) CLI nativo de HF (respeta HF_TOKEN/HF_ENDPOINT y reintenta/resume — suele pasar donde curl no)
  for cli in "hf download" "huggingface-cli download"; do
    if command -v ${cli%% *} >/dev/null 2>&1; then
      echo ">> intentando '$cli'..."
      if $cli "$HF_REPO" "$HF_FILE" --local-dir "$MODEL_DIR" 2>&1 | tail -3; then
        [ -s "$MODEL_DIR/$HF_FILE" ] && { MODEL="$MODEL_DIR/$HF_FILE"; return 0; }
      fi
    fi
  done
  # 3) curl con reintentos, luego mirror hf-mirror.com
  echo ">> intentando curl (reintentos)..."
  curl -L --fail --retry 5 --retry-delay 3 --connect-timeout 20 -o "$MODEL" "$MODEL_URL" && return 0
  echo ">> intentando mirror hf-mirror.com..."
  curl -L --fail --retry 3 --retry-delay 3 --connect-timeout 20 -o "$MODEL" "${MODEL_URL/huggingface.co/hf-mirror.com}" && return 0
  return 1
}

if fetch_model; then
  ls -lh "$MODEL"
else
  die "No pude obtener el modelo (la instancia parece sin salida a huggingface.co).
   Opciones: (a) exporta HF_TOKEN y reintenta; (b) descarga $HF_FILE por otra vía y déjalo en
   $MODEL_DIR (el script toma cualquier *.gguf de ahí); (c) usa MODEL_DIR=/ruta/con/gguf.
   URL directa: $MODEL_URL"
fi

# ---------------------------------------------------------------------------
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
  local up=0
  for i in $(seq 1 180); do   # HIP compila kernels al cargar: la 1a vez puede tardar
    if curl -s -m2 http://127.0.0.1:$PORT/health 2>/dev/null | grep -q '"ok"\|ok'; then up=1; break; fi
    kill -0 $pid 2>/dev/null || { echo "!! el server murió al arrancar:"; tail -20 /tmp/srv_$PORT.log; return; }
    sleep 1
  done
  [ "$up" = 1 ] || { echo "!! el server no quedó healthy en 180s:"; tail -20 /tmp/srv_$PORT.log; kill $pid 2>/dev/null; return; }

  # Confirmar que las capas REALMENTE se descargaron a GPU (no CPU fallback silencioso)
  if grep -qiE 'offloaded .*layers to GPU|ROCm[0-9]|using ROCm' /tmp/srv_$PORT.log; then
    grep -iE 'offloaded .*layers to GPU|ROCm[0-9].*:' /tmp/srv_$PORT.log | head -3 | sed 's/^/   [GPU] /'
  else
    echo "   (aviso: no vi confirmación de offload a GPU en el log — revisa /tmp/srv_$PORT.log)"
  fi

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
echo "GPU     : $(rocminfo 2>/dev/null | awk '/^ *Name: *gfx/{f=1} f&&/Marketing Name/{sub(/^ *Marketing Name: */,"");print;exit}' || echo '?')"
echo "GFX     : $GFX"
echo "Binario : $BIN  ($(ldd "$BIN" 2>/dev/null | grep -oiE 'libamdhip[^ ]*' | head -1))"
echo "Modelo  : $(basename "$MODEL")"
echo "Salida  : $OUT (última config = turbo3)"
echo "(arriba: por-tarea con segundos y preview; y el tiempo total de cada config)"
