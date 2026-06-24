# Roadmap

> Detailed milestones and priorities for the TurboQuant project.

---

## Priority Levels

| Priority | Description |
|---|---|
| **P1 — Correctness** | Bugs that produce wrong answers, crash, or violate numerical guarantees |
| **P2 — Performance** | Throughput, latency, and memory efficiency |
| **P3 — Portability** | Backend support (ROCm, Vulkan, SYCL, WebGPU) |
| **P4 — Research** | Exploratory features, future extensions |

---

## Phase 1: TurboQuant (Complete)

### Type Definitions and Core Kernels

| Task | Priority | Status |
|---|---|---|
| Define `GGML_TYPE_TURBO3_0`, `TURBO4_0`, `TURBO2_0` enums | P1 | ✅ |
| Define block structures with static assertions | P1 | ✅ |
| Implement CPU quantize/dequantize for turbo2 | P1 | ✅ |
| Implement CPU quantize/dequantize for turbo3 | P1 | ✅ |
| Implement CPU quantize/dequantize for turbo4 (4-bit) | P1 | ✅ |
| Implement CPU quantize/dequantize for turbo4 (3-bit+QJL legacy) | P1 | ✅ |
| Implement CUDA device functions for WHT (turbo_fwht_128/64) | P1 | ✅ |
| Implement CUDA device functions for centroid lookup (2/3/4-bit) | P1 | ✅ |
| Implement CUDA WHT kernel (k_turbo_wht_f32) | P1 | ✅ |
| Implement CUDA SET_ROWS for turbo3 (warp-cooperative) | P1 | ✅ |
| Implement CUDA SET_ROWS for turbo2 (warp-cooperative) | P1 | ✅ |
| Implement CUDA SET_ROWS for turbo4 (warp-cooperative) | P1 | ✅ |
| Metal WHT rotation (turbo-wht.h) | P1 | ✅ |

### WHT Consistency Fix

| Task | Priority | Status |
|---|---|---|
| Replace dense rotation with FWHT in CPU turbo4 quantization | P1 | ✅ |
| Remove inverse dense rotation from CPU turbo4 dequantization | P1 | ✅ |
| Validate CPU-GPU bit-exactness for all turbo types | P1 | ✅ |

### Flash Attention Integration

| Task | Priority | Status |
|---|---|---|
| Implement on-the-fly K dequant for turbo types in FA (VEC kernel) | P1 | ✅ |
| Implement on-the-fly V dequant for turbo types in FA (VEC kernel) | P1 | ✅ |
| 16 VEC FA template instantiations for turbo type combos | P1 | ✅ |
| Turbo-specific LUT scoring optimization | P2 | ✅ |
| Sparse V dequantization (Metal) | P2 | ✅ |

### KV Cache Infrastructure

| Task | Priority | Status |
|---|---|---|
| Head dimension padding to 128 | P1 | ✅ |
| Pre-rotate Q (forward WHT) before FA | P1 | ✅ |
| Post-rotate V output (inverse WHT) after FA | P1 | ✅ |
| Layer-adaptive quantization modes (0-7) | P2 | ✅ |
| Boundary V protection for turbo2-V | P2 | ✅ |
| InnerQ per-channel equalization | P2 | ✅ |

---

## Phase 2: Validation (Complete)

| Task | Priority | Status |
|---|---|---|
| Round-trip quantization test (tests/test-turbo-quant.c) | P1 | ✅ |
| Automated quality gate (scripts/turbo-quality-gate.sh) | P1 | ✅ |
| Perplexity validation on Llama-3.2-3B | P1 | ✅ |
| Quality gate pass on Qwen3.5-35B-A3B | P1 | ✅ |
| HIP/ROCm port for turbo3/turbo2 | P3 | ✅ |
| Add missing FA template instances for HIP/ROCm build | P3 | ✅ |

---

## Phase 3: Paged Attention (In Progress)

| Task | Priority | Status |
|---|---|---|
| Define `GGML_OP_GATHER_PAGED_V` op | P1 | ✅ |
| Reshape V tensor to block-pool layout | P1 | ✅ |
| Implement page table data structures | P1 | ✅ |
| Implement dynamic page allocation | P1 | ✅ |
| Implement `set_input_v_idxs` through page table | P1 | ✅ |
| Implement CUDA gather kernel | P1 | ✅ |
| Wire gather into compute graph | P1 | ✅ |
| CPU fallback for gather | P1 | ✅ |
| `LLAMA_NO_PAGING` env var gate | P1 | ✅ |
| Fix device sync race in paged KV write | P1 | ✅ |
| Fix K-cache isolation (flat pool indices) | P1 | ✅ |
| Phase 2: Native paged FA (page table in kernel) | P2 | 🔄 In Progress |
| Phase 3: TurboQuant-aware block alignment | P2 | ⬜ Pending |
| Sliding window support | P2 | ⬜ Pending |
| Continuous batching | P2 | ⬜ Pending |
| Prefix sharing between sequences | P2 | ⬜ Pending |
| K cache paging | P2 | ⬜ Pending |

---

## Phase 4: Backend Portability (Pending)

| Task | Priority | Status |
|---|---|---|
| Vulkan TurboQuant quantize/dequantize shaders | P3 | ⬜ Pending |
| Full HIP/ROCm turbo4 support | P3 | ⬜ Pending |
| HIP/ROCm ROCWMMA FA for turbo types | P3 | ⬜ Pending |
| SYCL TurboQuant kernels | P3 | ⬜ Pending |
| WebGPU TurboQuant kernels | P3 | ⬜ Pending |
| CUDA block_size=128 validation for all GPUs | P2 | ⬜ Pending |
| Multi-GPU paged attention | P3 | ⬜ Pending |

---

## Phase 5: Future Research (Pending)

| Task | Priority | Status |
|---|---|---|
| Variable bit-width per layer | P4 | ⬜ Pending |
| Learned boundary detection for layer-adaptive | P4 | ⬜ Pending |
| Zero-overhead FA dequant (eliminate gather) | P2 | ⬜ Pending |
| QJL training on calibration data | P4 | ⬜ Pending |
| Adaptive bit-width selection per layer | P4 | ⬜ Pending |
| Weight + KV joint compression (TQ3_1S, TQ4_1S formats) | P4 | ⬜ Pending |
| Cross-layer KV cache coordination | P4 | ⬜ Pending |
| Benchmark automation and regression dashboard | P2 | ⬜ Pending |

---

## Immediate Next Steps

1. **Complete Paged Attention Phase 2**: Integrate page table lookup directly into the
   Flash Attention VEC kernel, eliminating the gather overhead.
2. **Vulkan TurboQuant**: Implement compute shaders for turbo quantize/dequantize to
   enable access on integrated GPUs.
3. **Benchmark automation**: Create a systematic benchmark harness to track performance
   across commits and configurations.
4. **Expand validation**: Add more model families to the validation suite (Gemma, Mistral,
   DeepSeek).
