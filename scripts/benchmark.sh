#!/bin/bash
# scripts/benchmark.sh — Research OS benchmark wrapper
# Runs performance benchmarks comparing TurboQuant configs to standard types.

set -e

# Default binary locations
BUILD_DIR="build"
LLAMA_BENCH="./${BUILD_DIR}/bin/llama-bench"

# Helper for text styling
BOLD="\033[1m"
GREEN="\033[38;5;82m"
BLUE="\033[38;5;39m"
YELLOW="\033[38;5;214m"
RESET="\033[0m"

# Verify build directory and llama-bench binary
if [ ! -f "$LLAMA_BENCH" ]; then
    # Try alternate location build-turbo/bin
    LLAMA_BENCH="./build-turbo/bin/llama-bench"
    if [ ! -f "$LLAMA_BENCH" ]; then
        echo -e "${YELLOW}Warning: llama-bench binary not found in standard paths.${RESET}"
        echo -e "Attempting to locate any llama-bench in workspace..."
        FOUND=$(find . -name "llama-bench" -type f -print -quit)
        if [ -n "$FOUND" ]; then
            LLAMA_BENCH="$FOUND"
            echo "Found: $LLAMA_BENCH"
        else
            echo -e "${YELLOW}Could not locate llama-bench. Please compile first using:${RESET}"
            echo -e "  cmake -B build -DGGML_CUDA=ON && cmake --build build -j"
            exit 1
        fi
    fi
fi

# Print Header
echo -e "${BOLD}${BLUE}================================================================${RESET}"
echo -e "${BOLD}${GREEN}               TURBOQUANT BENCHMARK RUNNER                      ${RESET}"
echo -e "${BOLD}${BLUE}================================================================${RESET}"
echo ""

MODEL_ARG=""
if [ -n "$1" ]; then
    if [ -f "$1" ]; then
        MODEL_ARG="-m $1"
        echo -e "Using user-specified model: ${BOLD}$1${RESET}"
    else
        echo -e "${YELLOW}Specified file '$1' does not exist. Running standard synthetic bench...${RESET}"
    fi
else
    echo -e "No model specified. Running standard synthetic benchmark tests..."
fi

echo -e "${BOLD}Running comparison suite...${RESET}"
echo "--------------------------------------------------------"
echo "1. Baseline FP16 Cache"
echo "2. Baseline Q8_0 Cache"
echo "3. Turbo3 Cache (Asymmetric: q8_0 K + turbo3 V)"
echo "4. Turbo4 Cache (Asymmetric: q8_0 K + turbo4 V)"
echo "5. Turbo2 Cache (Asymmetric: q8_0 K + turbo2 V)"
echo "--------------------------------------------------------"
echo ""

# Run llama-bench with specific cache options
# -ctk: cache type K, -ctv: cache type V
# Using 512, 2048, and 4096 context sizes to measure context scaling behavior
$LLAMA_BENCH $MODEL_ARG \
    -ctk f16 -ctv f16 \
    -ctk q8_0 -ctv q8_0 \
    -ctk q8_0 -ctv turbo3 \
    -ctk q8_0 -ctv turbo4 \
    -ctk q8_0 -ctv turbo2 \
    -p 512,2048 -n 128 -ngl 99

echo ""
echo -e "${BOLD}${GREEN}Benchmark complete.${RESET}"
echo -e "${BOLD}${BLUE}================================================================${RESET}"
