#!/bin/bash
# scripts/diagnose-model.sh — classify generation failures as model-vs-runtime suspects.
#
# This is intentionally a wrapper around gen-smoke.sh: it keeps the load-bearing
# smoke test unchanged, then compares the failing model against an optional known-good
# reference model using the same server binary and settings.

set -u

GREEN="\033[38;5;82m"; RED="\033[38;5;196m"; YELLOW="\033[38;5;214m"; BLUE="\033[38;5;39m"; RESET="\033[0m"

SERVER_BIN="${SERVER_BIN:-build/bin/llama-server}"
MODEL="${1:-${MODEL:-}}"
REFERENCE_MODEL="${REFERENCE_MODEL:-${2:-}}"
PORT="${GEN_SMOKE_PORT:-8199}"

if [ -z "$MODEL" ]; then
    echo -e "${YELLOW}diagnose-model: set MODEL=/path/to/model.gguf or pass it as arg 1.${RESET}"
    exit 2
fi

if [ ! -f "$MODEL" ]; then
    echo -e "${RED}diagnose-model: model not found: $MODEL${RESET}"
    exit 2
fi

if [ ! -x "$SERVER_BIN" ]; then
    echo -e "${YELLOW}diagnose-model: '$SERVER_BIN' not found — build llama-server first.${RESET}"
    exit 2
fi

run_smoke() {
    local model="$1"
    local out="$2"

    GEN_SMOKE_PORT="$PORT" SERVER_BIN="$SERVER_BIN" MODEL="$model" \
        bash scripts/gen-smoke.sh > "$out" 2>&1
}

print_metadata_hints() {
    local log="/tmp/gen-smoke-server.log"

    [ -f "$log" ] || return 0

    echo "model metadata hints from $log:"
    grep -E "general.name|general.finetune|tokenizer.ggml.eos_token_id|BOS token|EOS token|EOT token|control-looking token" "$log" || true
}

echo -e "${BLUE}diagnose-model: testing candidate model${RESET}"
echo "  model=$MODEL"
echo "  server=$SERVER_BIN"

CANDIDATE_OUT="/tmp/diagnose-model-candidate.out"
if run_smoke "$MODEL" "$CANDIDATE_OUT"; then
    cat "$CANDIDATE_OUT"
    echo -e "${GREEN}diagnose-model: PASS — candidate model passed generation smoke.${RESET}"
    exit 0
fi

cat "$CANDIDATE_OUT"
echo -e "${YELLOW}diagnose-model: candidate failed; checking whether this is model-specific.${RESET}"
print_metadata_hints

if [ -z "$REFERENCE_MODEL" ]; then
    echo -e "${YELLOW}diagnose-model: inconclusive — provide REFERENCE_MODEL=/path/to/known-good.gguf to classify.${RESET}"
    exit 1
fi

if [ ! -f "$REFERENCE_MODEL" ]; then
    echo -e "${RED}diagnose-model: reference model not found: $REFERENCE_MODEL${RESET}"
    exit 2
fi

echo -e "${BLUE}diagnose-model: testing reference model${RESET}"
echo "  reference=$REFERENCE_MODEL"

REFERENCE_OUT="/tmp/diagnose-model-reference.out"
if run_smoke "$REFERENCE_MODEL" "$REFERENCE_OUT"; then
    cat "$REFERENCE_OUT"
    echo -e "${YELLOW}diagnose-model: MODEL SUSPECT — candidate failed, reference passed with the same runtime.${RESET}"
    exit 3
fi

cat "$REFERENCE_OUT"
echo -e "${RED}diagnose-model: RUNTIME/SCRIPT SUSPECT — candidate and reference both failed.${RESET}"
exit 4
