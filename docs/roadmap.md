# Roadmap

> Detailed milestones and priorities for the TurboQuant project.

---

## Milestones

### Completed

| Milestone | Status |
|---|---|
| TurboQuant CPU correctness | ✅ |
| TurboQuant CUDA correctness | ✅ |
| CPU/CUDA mathematical equivalence audit | ✅ |
| Llama validation (PPL within 0.4 for turbo4) | ✅ |
| Explicit turbo4 NaN validation (Hito 004) | ✅ |
| ROCm backend completion (HIP warp/ballot fixes) | ✅ |
| Initial ROCm portability audit | ✅ |
| TriAttention KV eviction implementation (M006) | ✅ |
| TriAttention calibration infrastructure (M007) | ✅ |
| Phase 2 FA n_seq OOB fix | ✅ |
| CI correctness fixes (state serialization guards, EditorConfig) | ✅ |
| Repo detached from `atomicmilkshake/llama-cpp-turboquant` fork network, renamed to `nomadstar/mallana` | ✅ |
| Phase 2 FA tensor-graph wiring hardened (recursive DFS + explicit abort; H1 ruled out empirically) | ✅ |
| multiswarm.py: live build/compile progress from `/proc`, pause-until-build-done, agy `--print-timeout` fix | ✅ |

### Upcoming

| Milestone | Priority |
|---|---|
| **Paged Attention Phase 2 — PPL=35125+ bug, root cause still open** (see below) | P1 |
| TriAttention H6.1 validation (generation-mode eval) | P4 |
| Large-scale benchmarks (multi-GPU, multi-model) | P2 |
| Upstream synchronization | P3 |

The project has transitioned from debugging the implementation to building new capabilities
on a validated foundation.

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
| Phase 2: Native paged FA — PPL=35125-38849 with `-fa on`+paging (baseline ~9.7). Tensor-graph wiring, page-table address math, and write/read row agreement all verified correct with empirical evidence (raw pool bytes at the resolved physical row are real, non-garbage data). Bug is downstream in the kernel compute path — see `research/milestone-008/phase2-fa-debug-handoff.md` (2026-07-01 update) for ruled-out hypotheses and next leads (`flash_attn_mask_to_KV_max`, gqa_ratio/mask-stride in the paged branch). | P1 | 🔴 Blocked |
| Phase 3: TriAttention KV eviction (H6.1 pending gen-mode eval) | P4 | 🔄 Implemented |
| Phase 4: TurboQuant-aware block alignment | P2 | ⬜ Pending |
| Sliding window support | P2 | ⬜ Pending |
| Continuous batching | P2 | ⬜ Pending |
| Prefix sharing between sequences | P2 | ⬜ Pending |
| K cache paging | P2 | ⬜ Pending |

---

## Phase 4: Backend Portability (In Progress)

| Task | Priority | Status |
|---|---|---|
| Vulkan TurboQuant quantize/dequantize shaders | P3 | ⬜ Pending |
| HIP/ROCm turbo4 support | P3 | 🔄 Prepared — <br/>low remaining effort |
| HIP/ROCm ROCWMMA FA for turbo types | P3 | ⬜ Pending |
| SYCL TurboQuant kernels | P3 | ⬜ Pending |
| WebGPU TurboQuant kernels | P3 | ⬜ Pending |
| CUDA block_size=128 validation for all GPUs | P2 | ⬜ Pending |
| Multi-GPU paged attention | P3 | ⬜ Pending |

### ROCm Portability Audit

An initial portability audit has been completed. The architecture is already largely backend
portable. Current blockers are minimal:

- **API renames**: CUDA-specific API calls need HIP equivalents (e.g., `cudaMemcpy` → `hipMemcpy`).
- **HIP compatibility wrappers**: The existing `ggml-cuda/vendors/hip.h` header handles most
  of these, but some CUDA-isms remain in the turbo-specific code paths.
- **Debug-only CUDA pointer inspection**: A small number of diagnostic code paths use
  `cudaPointerGetAttributes` for debug logging, which has no direct HIP equivalent.

The remaining effort to reach a functional HIP backend is relatively low, but completion
has not yet been scheduled.

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
| Layer-wise inference / weight streaming (AirLLM-style) | P4 | ⬜ Pending |
| Prefetch-compute overlap for layer streaming | P4 | ⬜ Pending |

### Research OS Tooling

| Task | Priority | Status |
|---|---|---|
| Integrate codebase-memory-mcp for agent code navigation | P3 | ⬜ Pending |
| Persistent knowledge graph across multiswarm agent runs | P3 | ⬜ Pending |

---

## Immediate Next Steps

1. **[P1] Continue Phase 2 paged FA debugging.** Root cause is not the tensor-graph
   wiring, not the page-table address math, and not a write/read row mismatch — all
   three were empirically disproven (see `research/milestone-008/phase2-fa-debug-handoff.md`,
   2026-07-01 update). Next: instrument `flash_attn_mask_to_KV_max`
   (`ggml/src/ggml-cuda/fattn-common.cuh`) and the gqa_ratio/mask-stride handling in the
   paged branch of `build_attn` (`src/llama-graph.cpp` ~2130-2183), comparing against the
   working `-fa off` path with the same prompt. Use empirical probes (printed/dumped real
   values), not static reading alone — that already produced two plausible-but-wrong
   hypotheses this round.
2. **H6.1 generation-mode evaluation**: `scripts/triattention_generation_eval.py` is written;
   run it once Phase 2 paged FA is fixed (it depends on `-fa on` + paging being correct).
3. **Large-scale benchmarks**: Systematic benchmark harness across multiple GPUs, model
   sizes, and context lengths. Blocked on Phase 2 FA fix for the `-fa on` numbers.
4. **Upstream synchronization**: Rebase against latest `ggml-org/llama.cpp` master to
   incorporate upstream fixes and features.
5. **Expand validation**: Add more model families to the validation suite (Gemma, Mistral,
   DeepSeek).
