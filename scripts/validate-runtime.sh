#!/bin/bash
# scripts/validate-runtime.sh — prove the built runtime generates coherently with a known-good GGUF.

set -u

GREEN="\033[38;5;82m"; RED="\033[38;5;196m"; YELLOW="\033[38;5;214m"; BLUE="\033[38;5;39m"; RESET="\033[0m"

SERVER_BIN="${SERVER_BIN:-build/bin/llama-server}"
KNOWN_GOOD_GEN_MODEL="${KNOWN_GOOD_GEN_MODEL:-${REFERENCE_MODEL:-}}"
MODEL="${MODEL:-}"

if [ -z "$KNOWN_GOOD_GEN_MODEL" ]; then
    echo -e "${YELLOW}validate-runtime: set KNOWN_GOOD_GEN_MODEL=/path/to/known-good.gguf.${RESET}"
    echo "Example (use a ROBUST instruct model as the reference — NOT a coder-1.5b, which is"
    echo " fp-fragile and collapses into loops under any lossy KV, faking a runtime bug):"
    echo "  KNOWN_GOOD_GEN_MODEL=~/models/qwen2.5-3b-instruct-q4_k_m.gguf scripts/validate-runtime.sh"
    exit 2
fi

if [ ! -f "$KNOWN_GOOD_GEN_MODEL" ]; then
    echo -e "${RED}validate-runtime: known-good model not found: $KNOWN_GOOD_GEN_MODEL${RESET}"
    exit 2
fi

if [ ! -x "$SERVER_BIN" ]; then
    echo -e "${YELLOW}validate-runtime: '$SERVER_BIN' not found — build llama-server first.${RESET}"
    exit 2
fi

echo -e "${BLUE}validate-runtime: proving runtime health with known-good model${RESET}"
echo "  server=$SERVER_BIN"
echo "  known_good=$KNOWN_GOOD_GEN_MODEL"

if ! SERVER_BIN="$SERVER_BIN" MODEL="$KNOWN_GOOD_GEN_MODEL" scripts/diagnose-model.sh; then
    echo -e "${RED}validate-runtime: FAIL — known-good model failed, runtime/build is suspect.${RESET}"
    exit 1
fi

echo -e "${GREEN}validate-runtime: PASS — runtime generates coherently with known-good model.${RESET}"

if [ -n "$MODEL" ] && [ "$MODEL" != "$KNOWN_GOOD_GEN_MODEL" ]; then
    echo -e "${BLUE}validate-runtime: classifying candidate model${RESET}"
    echo "  candidate=$MODEL"

    SERVER_BIN="$SERVER_BIN" MODEL="$MODEL" REFERENCE_MODEL="$KNOWN_GOOD_GEN_MODEL" \
        scripts/diagnose-model.sh
    rc=$?

    if [ "$rc" -eq 3 ]; then
        echo -e "${YELLOW}validate-runtime: candidate is model-suspect; runtime remains healthy.${RESET}"
        exit 0
    fi

    exit "$rc"
fi

exit 0
