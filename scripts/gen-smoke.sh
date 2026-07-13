#!/bin/bash
# scripts/gen-smoke.sh — end-to-end generation smoke test.
#
# WHY THIS EXISTS: the unit tests (test-turbo-quant, test-llama-archs) only check codec math and
# single-forward-pass NMSE; llama-bench only measures speed. NONE of them run real autoregressive
# generation through llama-server and check that the text is coherent. That gap let a severe
# "first request after model load returns repeated-garbage (e.g. 'etheusetheus...')" bug ship
# undetected. This test closes the gap: it starts llama-server, sends a COLD first request, and
# fails if the answer is wrong/garbage — exactly the failure mode the other tests miss.
#
# Usage: scripts/gen-smoke.sh [model.gguf]   (model also via $MODEL; else first *.gguf in cwd)
set -u

GREEN="\033[38;5;82m"; RED="\033[38;5;196m"; YELLOW="\033[38;5;214m"; RESET="\033[0m"

SERVER_BIN="${SERVER_BIN:-build/bin/llama-server}"
MODEL="${1:-${MODEL:-}}"
if [ -z "$MODEL" ]; then
    MODEL=$(ls -1 ./*.gguf 2>/dev/null | head -1)
fi
PORT="${GEN_SMOKE_PORT:-8199}"
NGL="${LLAMA_NGL:-0}"

if [ ! -x "$SERVER_BIN" ]; then
    echo -e "${YELLOW}gen-smoke: '$SERVER_BIN' not found — build llama-server first. SKIPPING.${RESET}"
    exit 0
fi
if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
    echo -e "${YELLOW}gen-smoke: no .gguf model found (set MODEL=/path). SKIPPING.${RESET}"
    exit 0
fi

echo "gen-smoke: model=$MODEL ngl=$NGL port=$PORT"
LD_LIBRARY_PATH="$(dirname "$SERVER_BIN"):${LD_LIBRARY_PATH:-}" \
    "$SERVER_BIN" -m "$MODEL" --port "$PORT" --host 127.0.0.1 -c 2048 -ngl "$NGL" \
    > /tmp/gen-smoke-server.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT

for i in $(seq 1 120); do
    if ! kill -0 $SRV 2>/dev/null; then
        echo -e "${RED}gen-smoke: server exited during load. See /tmp/gen-smoke-server.log${RESET}"; exit 1
    fi
    curl -s "localhost:$PORT/health" 2>/dev/null | grep -q '"ok"' && break
    sleep 1
done

ask() {  # $1 = prompt ; echoes the assistant answer (lowercased)
    curl -s "localhost:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
        -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$1\"}],\"temperature\":0.3,\"max_tokens\":40}" \
        2>/dev/null | python3 -c "import sys,json
try: print(json.load(sys.stdin)['choices'][0]['message']['content'].lower())
except Exception: print('__error__')"
}

looks_garbage() {  # $1 = answer text ; exit 0 (true) if it looks like a repetition loop / garbage.
    # Ported from scripts/amd-validate.sh so both validators share ONE objective criterion. A pure
    # substring check ("does the answer contain 'toast'?") is fooled by a loop that repeats the
    # keyword ("toast toast toast…") — the exact failure mode this smoke test claims to catch.
    python3 - "$1" <<'PY'
import sys, re
from collections import Counter
s = sys.argv[1] if len(sys.argv) > 1 else ""
def looks_garbage(s):
    if not s: return False
    if len(s) >= 20:
        letters = sum(c.isalpha() or c.isspace() for c in s)
        if letters/len(s) < 0.60: return True            # non-text junk
    toks = re.findall(r"\w+", s.lower())
    run = 1                                              # 3+ identical tokens in a row = loop
    for i in range(1, len(toks)):
        run = run+1 if toks[i] == toks[i-1] else 1
        if run >= 3: return True
    if len(toks) >= 4:
        top, n = Counter(toks).most_common(1)[0]
        if n >= 4 and n > 0.35*len(toks): return True    # one token dominates
        if len(toks) >= 8 and len(set(toks))/len(toks) < 0.25: return True
    return False
sys.exit(0 if looks_garbage(s) else 1)
PY
}

FAIL=0

# COLD first request. An open-ended instruction like this reliably regressed to
# repeated-token garbage ("etheus etheus...") on the cold request while short high-probability
# completions sometimes survived — so this is the sensitive, load-bearing check.
A1=$(ask "Name three uses for a toaster in one line.")
if looks_garbage "$A1"; then
    echo -e "  ${RED}FAIL cold request: repetition-loop/garbage, got: ${A1:0:120}${RESET}"; FAIL=1
elif echo "$A1" | grep -qiE "toast|bread|bagel|heat|warm"; then
    echo -e "  ${GREEN}PASS cold request: coherent toaster answer${RESET}"
else
    echo -e "  ${RED}FAIL cold request: incoherent, got: ${A1:0:120}${RESET}"; FAIL=1
fi

# WARM second request — sanity that ongoing generation stays coherent.
A2=$(ask "What is 2+2? Reply with only the number.")
if looks_garbage "$A2"; then
    echo -e "  ${RED}FAIL warm request: repetition-loop/garbage, got: ${A2:0:120}${RESET}"; FAIL=1
elif echo "$A2" | grep -q "4"; then
    echo -e "  ${GREEN}PASS warm request: contains '4'${RESET}"
else
    echo -e "  ${RED}FAIL warm request: expected '4', got: ${A2:0:120}${RESET}"; FAIL=1
fi

if [ "$FAIL" -eq 0 ]; then
    echo -e "${GREEN}gen-smoke: PASS${RESET}"; exit 0
else
    echo -e "${RED}gen-smoke: FAIL — generation is broken (see above)${RESET}"; exit 1
fi
