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
| Test-suite migration & CI integration — migrated 4 upstream tests (`test-save-load-state`, `test-recurrent-state-rollback`, `test-quant-type-selection`, `test-col2im-1d`), fixed the paged-FA graph-wiring abort on non-FA attention paths (T5 cross-attention), and wired `ctest -L main` into `scripts/validate.sh` | ✅ |
| **Paged Attention Phase 2 — fixed.** Two independent bugs, both required for correct PPL: (1) tensor-core FA kernels silently ignored the page table (dispatch/addressing bug — fixed by routing to VEC/TILE when paging is active), and (2) a missing unconditional scheduler synchronization in `llama_context::process_ubatch` let a still-in-flight kernel read a torn page table. See below and `research/milestone-008/phase2-fa-debug-handoff.md` (2026-07-02 update). | ✅ |
| CI restored & green-path fixes (2026-07-03) — workflows re-registered after the repo rename (push touching `.github/workflows` required), new lean `fork-tests.yml` (CPU build + `ctest -L main` on every push/PR), and fixes for every compile failure in the hosted matrix: `GGML_OP_COUNT` static_assert → 99 in `ggml-rpc.h` (`RPC_PROTO_PATCH_VERSION` → 3), MSVC `M_PI` fallback in `ggml-turbo-quant.c`, unused-variable removals for `LLAMA_FATAL_WARNINGS=ON` (`llama-kv-cache.cpp`, `set-rows.cu`), C++20 enum-arithmetic casts (`test-backend-ops.cpp`, `clip-graph.h`), MUSA `cudaMemcpyTo/FromSymbol` mappings, `ggml-org/vocabs` test repo pinned to `a40cfbe` | ✅ |
| **PagedAttention flipped to opt-in (`LLAMA_PAGING=1`, 2026-07-03).** Default-on paging corrupted all CPU inference (only the CUDA FA kernel understands the page table; CPU `FLASH_ATTN_EXT` read the paged V pool as linear memory — root cause of the deterministic-gibberish `test_load_split_model` failures in the Server workflows) and produced per-arch CUDA divergences (the former `test-llama-archs` P1: qwen, glm4, olmo, gemma3n NMSE 0.47, etc.). Paging now requires `LLAMA_PAGING=1` **and** a KV cache fully resident on CUDA devices (warns and disables otherwise). With paging off by default, `test-llama-archs` passes on CUDA (11 failures → 0) and the full ctest suite is 56/56 | ✅ |
| multiswarm.py: `--ci-status` / `--ci-fix` modes — query GitHub Actions runs via `gh`, download failed-step logs, and auto-compose a swarm task to fix them (no auto-push; owner reviews) | ✅ |
| Sched-reset audit High finding fixed: added missing scheduler synchronization after `mctx->apply()` in `llama_context::memory_update()` so the async K-shift graph completes before `graph_reserve()` resets the scheduler | ✅ |
| **Paged per-arch divergences fixed (2026-07-09).** All `LLAMA_PAGING=1` `test-llama-archs` failures (plamo2/3, gemma3n NMSE 0.85, olmo2, nemotron_h(+moe), granitehybrid, gpt-oss) had one root cause: the ISWA (`llm_graph_input_attn_kv_iswa`) and hybrid (`llm_graph_input_mem_hybrid*`) input paths never wired the V page table — the ISWA `build_attn` read the paged V pool as linear memory via `get_v`, and the hybrid `set_input`s never uploaded the page table. Fixed in `src/llama-graph.{h,cpp}` by mirroring the standard-KV paged wiring (build/set page tables for base+SWA caches, paged V view / gather fallback, `ggml_flash_attn_ext_set_page_table`). `LLAMA_PAGING=1 test-llama-archs` now 0 failures (was 8); paging-off suite unchanged (44/44). Default flip is a separate owner decision (P1 below). | ✅ |
| **ROCm/HIP validated on real AMD hardware (2026-07-11).** Built with `-DGGML_HIP=ON` on a gfx1100 (RDNA3, ROCm 7.2.4) host: fixed the last HIP link failure — the FA VEC dispatch referenced the `turbo3_0↔f16` template instances but `ggml/src/ggml-hip/CMakeLists.txt` omitted those two `.cu` files (present in the CUDA list), leaving `ggml_cuda_flash_attn_ext_vec_case<D,TURBO3_0,F16>` and its transpose undefined. After the fix, `test-turbo-quant` passes (round-trip + NaN/Inf/outlier robustness) and `test-llama-archs` runs entirely on `AMD Radeon Graphics` with NMSE 1e-8–1e-12 across all architectures. `.devops/rocm.Dockerfile` base pinned to ROCm 7.2.4 (the `7.2` tag → 7.2.0 hits an nvfp4 HIP fp8 header error). | ✅ |

### Upcoming

| Milestone | Priority |
|---|---|
| Re-enable PagedAttention by default: the per-arch CUDA divergences under `LLAMA_PAGING=1` are now fixed (ISWA/hybrid page-table wiring, 2026-07-09; `test-llama-archs` passes with paging on) — remaining step is the owner decision to flip the default | P1 |
| ROCm/HIP remaining work: PagedAttention (`LLAMA_PAGING=1`) on AMD, ROCWMMA Flash Attention for turbo types (`-DGGML_HIP_ROCWMMA_FATTN=ON`), and a self-hosted AMD runner for CI | P2 |
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
| Paging env var gate — now **opt-in** `LLAMA_PAGING=1` (2026-07-03; replaced the opt-out `LLAMA_NO_PAGING`) | P1 | ✅ |
| Fix device sync race in paged KV write | P1 | ✅ |
| Fix K-cache isolation (flat pool indices) | P1 | ✅ |
| Phase 2: Native paged FA — **fixed 2026-07-02, two independent bugs, both required for correct PPL:** (1) **kernel dispatch/addressing** — tensor-core FA kernels (MMA_F16/WMMA_F16) received the page table but silently ignored it (`GGML_UNUSED(v_ptable)`), reading unpaged addresses against a pool with no valid per-sequence stride when paged; fixed by routing to the VEC/TILE kernels (the only ones implementing `v_paged_ptr`) whenever a page table is attached (`fattn.cu`). (2) **missing scheduler synchronization** — `llama_context::process_ubatch` (`src/llama-context.cpp`) only called `ggml_backend_sched_synchronize()` before graph reuse when `cparams.pipeline_parallel` was set, on the incorrect assumption that async in-flight compute across ubatches only happens with pipeline parallelism; `ggml_backend_sched_graph_compute_async` is always asynchronous (single-GPU included), so a still-running previous-ubatch FA kernel could read a torn/stale page table — uniquely exposed by paging because, unlike K/V cache cells (append-only, non-overlapping), the page table is fully overwritten at the same address every ubatch. Fixed by making the sync unconditional. Also added a defensive `k_abs < k_max` bounds guard in `v_paged_ptr` (`fattn-common.cuh`) for the tail-tile case, uncovered once (1) and (2) were fixed. **Evidence:** PPL now correct (~9.7-12.3, matching baseline) under normal (non-blocking) execution for both VEC and TILE kernels, with and without CUDA graphs; `compute-sanitizer --tool memcheck` reports 0 errors; PPL reproduced 3× for VEC. **Known gap:** the byte-level V comparison between paged `-fa on` and gather `-fa off` at `sequence≥1` (requested explicitly in the 2026-07-02 debugging round) was not performed — the PPL-match evidence above was used instead; if a byte-level comparison harness gets built later, running it here would close that gap. See `research/milestone-008/phase2-fa-debug-handoff.md` (2026-07-02 update) for the full diagnostic trail. | P1 | ✅ |
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

1. **[P1] ~~Audit other `ggml_backend_sched_reset`/graph-rebuild call sites~~ — audit
   completed 2026-07-03** (`multiswarm.py --audit`, report in `.multiswarm_audit.md`).
   Result: the `process_ubatch` fix is correct; paged state save/restore safely rejects
   `pg_enabled`. **High finding fixed**: the K-shift context-shift path now calls
   `synchronize()` after `mctx->apply()` in `llama_context::memory_update()`, so the async
   K-shift graph completes before `graph_reserve()` resets the scheduler
   (`ggml_backend_sched_reset()`, llama-context.cpp).
2. **H6.1 generation-mode evaluation**: `scripts/triattention_generation_eval.py` is written
   and unblocked now that Phase 2 paged FA is fixed — run it.
3. **Large-scale benchmarks**: Systematic benchmark harness across multiple GPUs, model
   sizes, and context lengths. Unblocked now that Phase 2 FA is fixed.
4. **Upstream synchronization**: Rebase against latest `ggml-org/llama.cpp` master to
   incorporate upstream fixes and features.
5. **Expand validation**: Add more model families to the validation suite (Gemma, Mistral,
   DeepSeek).

---

## Known Failures & Pending Technical Debt

### CI/CD Failures
- **Docker Publish Workflow (`Publish Docker image`)**: The scheduled daily build fails on the ROCm/CUDA caching step (`failed to configure registry cache importer`). This was triggered by the repository rename from `atomicmilkshake/llama-cpp-turboquant` to `nomadstar/mallana`, causing permissions/not found errors for the cached GHCR registry layers under the old org.
  - *Mitigation*: The workflow utilizes dynamic repository owner variables, so future runs on the new repository owner (`nomadstar`) should auto-correct once clean cache layers are written, but manual cache eviction/invalidation may be needed if GitHub Actions continues referencing stale repository cache scopes.

### Codebase TODOs & Key Technical Debt
A scan of the implementation paths reveals the following high-priority pending items and assertions:
1. **Flash Attention (CUDA)**:
   - **Vector FA Kernel Optimization**: Replace heavy preprocessor macros with C++ templates, and switch to FP32 accumulate for BF16 validation in [fattn-common.cuh](file:///home/ignatus/GitHub/mallana/ggml/src/ggml-cuda/fattn-common.cuh#L135).
   - **Architectural Tuning**: Optimize TILE and MMA kernel parameters specifically for RDNA and legacy NVIDIA GPUs (e.g., Pascal P100) in [fattn-tile.cuh](file:///home/ignatus/GitHub/mallana/ggml/src/ggml-cuda/fattn-tile.cuh#L8-L9) and [fattn-mma-f16.cuh](file:///home/ignatus/GitHub/mallana/ggml/src/ggml-cuda/fattn-mma-f16.cuh#L112).
2. **KV Cache & Graph Wiring**:
   - **Multiple Streams**: Hard assertions block multi-stream execution and non-sequential batching: `GGML_ASSERT(n_stream == 1 && "TODO: support multiple streams")` in [llama-kv-cache.cpp](file:///home/ignatus/GitHub/mallana/src/llama-kv-cache.cpp#L2027) and `GGML_ASSERT(!ubatch->equal_seqs())` in [llama-graph.cpp](file:///home/ignatus/GitHub/mallana/src/llama-graph.cpp#L134).
   - **Unified Cache Assertions**: Several validations for input attention indices (`self_v_idxs->ne[0] == params.ubatch.n_tokens`) are currently commented out in [llama-graph.cpp](file:///home/ignatus/GitHub/mallana/src/llama-graph.cpp#L446) and need to be moved to the unified cache.
3. **CUDA Numerical Divergences in `test-llama-archs` (P1) — root-caused 2026-07-03**:
   - The divergences (qwen 2.7e-02, glm4 1.4e-01, olmo 6.4e-01, phi2, gemma3n NMSE 0.47, etc.)
     were caused by default-on PagedAttention, not by TurboQuant or the graph wiring. With
     paging now opt-in (`LLAMA_PAGING=1`), the test passes on CUDA and the `validate.sh`
     exclusion has been removed.
   - The divergences still reproduce **with `LLAMA_PAGING=1`** — fixing them is the
     prerequisite for re-enabling paging by default (tracked as P1 in Upcoming).
4. **Paging & Verification Gaps**:
   - **Byte-level V-pool comparison**: The validation harness to compare the exact layout of the paged V pool (`-fa on`) vs the gather V pool (`-fa off`) byte-by-byte for `sequence >= 1` was bypassed in favor of end-to-end perplexity (PPL) verification. This remains a validation gap.
   - **Unconditional Synchronization Performance Debt**: The `ggml_backend_sched_synchronize()` call added in `llama_context::process_ubatch()` prevents the page table race condition but is unconditional, which halts async overlap across micro-batches (especially on single-GPU setups). It should eventually be replaced by a conditional synchronization (only when mutable page tables/host buffers are shared) or by double-buffering the page tables.
