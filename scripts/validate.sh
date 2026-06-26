#!/bin/bash
# scripts/validate.sh — Quality & correctness validation script
# Runs build verification, unit tests, and quality gates.

set -e

# ANSI Color codes
BOLD="\033[1m"
GREEN="\033[38;5;82m"
BLUE="\033[38;5;39m"
YELLOW="\033[38;5;214m"
RED="\033[38;5;196m"
RESET="\033[0m"

echo -e "${BOLD}${BLUE}================================================================${RESET}"
echo -e "${BOLD}${GREEN}               RESEARCH OS VALIDATION ENGINE                    ${RESET}"
echo -e "${BOLD}${BLUE}================================================================${RESET}"
echo ""

FAIL=0

# Step 1: Compile Check
echo -e "${BOLD}[1/3] Verifying code builds correctly...${RESET}"
if [ -d "build" ]; then
    echo "Build directory exists. Running incremental compilation..."
    if cmake --build build -j$(nproc 2>/dev/null || echo 4); then
        echo -e "  ${GREEN}PASS: Compilation successful.${RESET}"
    else
        echo -e "  ${RED}FAIL: Compilation failed.${RESET}"
        FAIL=1
    fi
else
    echo -e "${YELLOW}Warning: 'build' directory not found. Skipping build step.${RESET}"
    echo -e "Please configure CMake first (e.g., 'cmake -B build -DGGML_CUDA=ON')."
fi
echo ""

# Step 2: Run Round-Trip Unit Tests
echo -e "${BOLD}[2/3] Running TurboQuant numerical unit tests...${RESET}"
TEST_BIN="build/bin/test-turbo-quant"
if [ ! -f "$TEST_BIN" ]; then
    # Search for it
    TEST_BIN=$(find . -name "test-turbo-quant" -type f -print -quit)
fi

if [ -f "$TEST_BIN" ] && [ -x "$TEST_BIN" ]; then
    echo "Executing $TEST_BIN..."
    if "$TEST_BIN"; then
        echo -e "  ${GREEN}PASS: Numerical tests match expectation.${RESET}"
    else
        echo -e "  ${RED}FAIL: Numerical unit tests failed.${RESET}"
        FAIL=1
    fi
else
    echo -e "${YELLOW}Warning: Unit test binary 'test-turbo-quant' not found.${RESET}"
    echo -e "Compile the tests target to enable this check."
fi
echo ""

# Step 3: Run Quality Gate Script
echo -e "${BOLD}[3/3] Running quality and perplexity gates...${RESET}"
if [ -f "scripts/turbo-quality-gate.sh" ]; then
    # We call it. If it fails, fail this script too.
    # Check if dependencies (model, wikitext-2) are available in default locations.
    # If not, warn but don't fail unless explicitly requested.
    WIKI_FILE="${WIKI:-~/local_llms/llama.cpp/wikitext-2-raw/wiki.test.raw}"
    MODEL_FILE="${MODEL:-~/local_llms/models/Qwen3.5-35B-A3B-Q8_0.gguf}"
    
    if [ ! -f "$MODEL_FILE" ] && [ -z "$FORCE_GATE" ]; then
        echo -e "${YELLOW}Warning: Quality Gate model '$MODEL_FILE' not found.${RESET}"
        echo -e "To run perplexity gates, download the model or set MODEL=/path/to/model.gguf."
        echo -e "Skipping perplexity check."
    else
        if bash scripts/turbo-quality-gate.sh; then
            echo -e "  ${GREEN}PASS: Quality Gate holds.${RESET}"
        else
            echo -e "  ${RED}FAIL: Quality Gate regression detected.${RESET}"
            FAIL=1
        fi
    fi
else
    echo -e "${RED}Error: scripts/turbo-quality-gate.sh not found.${RESET}"
    FAIL=1
fi
echo ""

echo -e "${BOLD}${BLUE}================================================================${RESET}"
if [ "$FAIL" -eq 0 ]; then
    echo -e "  ${BOLD}${GREEN}ALL VALIDATION TESTS PASSED${RESET}"
    echo -e "${BOLD}${BLUE}================================================================${RESET}"
    exit 0
else
    echo -e "  ${BOLD}${RED}VALIDATION FAILED — FIX REGRESSIONS BEFORE PROCEEDING${RESET}"
    echo -e "${BOLD}${BLUE}================================================================${RESET}"
    exit 1
fi
