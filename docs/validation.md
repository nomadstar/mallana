# Validation

> Methodology, infrastructure, and results for TurboQuant validation.

---

## Methodology

### Perplexity Testing

Perplexity (PPL) is the primary quality metric. We measure PPL on a test corpus
(wikitext-2) using `llama-perplexity` with fixed parameters:

- Sequence length: 512 tokens
- Chunks: 8
- Backend: GPU (Metal or CUDA)
- Comparison: same model, same corpus, same chunking

The quality gate (`scripts/turbo-quality-gate.sh`) enforces:

```
PPL_turbo < PPL_baseline × 1.05
```

This 5% threshold is calibrated to reject regressions while accommodating the inherent
variance from quantization noise.

### Round-Trip Fidelity

The `tests/test-turbo-quant.c` program tests the quantize-dequantize round trip:

1. **Basis vector**: `[1, 0, 0, ...]` — Tests that the dominant component is preserved.
2. **Sinusoidal**: `sin(i × 0.1 + 0.5) × 10` — Tests realistic activation patterns with
   moderate amplitude variation.
3. **Cosine**: `cos(i × 0.2) × 5` — Tests phase diversity.

Metrics reported: MSE, cosine similarity, input norm, output norm.

### CPU/CUDA Mathematical Consistency Audit

A complete audit of all CPU and CUDA TurboQuant paths was conducted to verify mathematical
equivalence. Every contract between the two backends was checked independently:

| Checked Item | Finding |
|---|---|
| WHT sign arrays | Identical values in CPU (`turbo_cpu_s1/s2`) and CUDA (`turbo_gpu_s1/s2`) |
| WHT butterfly order | Identical stage ordering (1→2→4→...→64→128) |
| WHT normalization | Identical `1/sqrt(d)` scaling after butterfly |
| Turbo2 centroids | `{-0.133462, -0.039994, 0.039994, 0.133462}` — CPU and CUDA match |
| Turbo3 centroids | 8 centroids from Lloyd-Max — CPU and CUDA match |
| Turbo4 centroids | 16 centroids from Lloyd-Max — CPU and CUDA match |
| Turbo2 packing | 2-bit indices packed 4 per byte — identical |
| Turbo3 packing | 2-bit lo in `qs[]`, 1-bit hi in `signs[]` — identical |
| Turbo4 packing | 4-bit nibble packed (default) — identical |
| Norm correction | `corrected_norm = block_norm / recon_norm` — identical |
| `vec_dot` contract | Dequant+dot accumulator semantics — correct |
| InnerQ scaling contract | Per-channel RMS ratio application — correct |

**Engineering verdict:**

> No correctness-critical CPU/CUDA mathematical mismatches remain.

This audit supersedes all prior concerns about backend divergence. The turbo4 dense-rotation
bug (documented below) was the only systematic inconsistency, and its fix (commit `6457eac19`)
fully restored mathematical equivalence.

### Reference Implementation

The CPU implementation in `ggml/src/ggml-turbo-quant.c` serves as the reference. CUDA and
Metal implementations are validated against CPU outputs.

---

## Test Infrastructure

### Automated Quality Gate

**File**: `scripts/turbo-quality-gate.sh`

Two tests:

1. **Perplexity**: `llama-perplexity` with `-ctk turbo3 -ctv turbo3 -fa on` on 8 chunks of
   wikitext-2. Reference model: Qwen3.5-35B-A3B-Q8_0. Threshold: PPL < 6.111 × 1.05 = 6.417.

2. **Context Scaling**: 4K prefill speed comparison of turbo3 vs q8_0.
   Threshold: turbo3_tps / q8_0_tps > 0.95.

The gate exits with code 1 (FAIL) if either test fails.

```bash
bash scripts/turbo-quality-gate.sh
```

### Round-Trip Test

**File**: `tests/test-turbo-quant.c`

Compile and run:

```bash
gcc -o test-turbo-quant tests/test-turbo-quant.c -lm -I. -lggml
./test-turbo-quant
```

Reports MSE, cosine similarity, and norm for turbo3 and turbo4.

---

## Model Selection

### Test Models

| Model | Purpose | Notes |
|---|---|---|
| Qwen3.5-35B-A3B-Q8_0 | Quality gate reference | Large MoE, tests real-world behavior |
| Llama-3.2-3B | Baseline validation | Small dense model, rapid iteration |
| GLM-4.7 | Edge-case: non-128 head dim | head_dim = 576, verifies padding path |

### Reference Results

#### Llama-3.2-3B F16

| Configuration | PPL | vs Baseline |
|---|---|---|
| Baseline (F16) | 8.68 | — |
| turbo2 K | 13.42 | +4.74 |
| turbo3 K | 9.45 | +0.77 |
| turbo4 K | 8.99 | +0.31 |
| turbo4 V | 8.76 | +0.08 |
| turbo4 K + turbo4 V | 8.99 | +0.31 |

CPU TurboQuant is validated for Llama-family models:

- `turbo4` types degrade PPL by less than 0.4.
- `turbo3` degrades PPL by less than 0.8.
- The asymmetric policy (`q8_0` K + `turbo4` V) is within 0.1 PPL of baseline.
- `turbo2` on K shows significant degradation (+4.74 PPL) and is not recommended for K cache.

#### Qwen3.5-35B-A3B (Quality Gate)

| Configuration | PPL |
|---|---|
| Baseline (q8_0) | 6.111 |
| turbo3 + turbo3 | ≤6.417 (pass) |

The quality gate confirms turbo3 stays within 5% of the q8_0 baseline on this large MoE model.

---

## Known Limitations

### Qwen Family Compatibility

Testing on Qwen models reveals anomalous degradation across all low-bit quantization methods,
not just TurboQuant:

| Configuration | PPL |
|---|---|
| Baseline (F16) | 11.79 |
| q8_0 K + q8_0 V | 9.20 |
| q4_0 K + q4_0 V | 531 |
| turbo3 K + turbo3 V | 4098 |
| turbo4 K + turbo4 V | 1658 |

**Current evidence**: This appears to be a model/quantizer compatibility issue rather than a
TurboQuant implementation defect. The key observation is that plain `q4_0` K quantization
already fails dramatically on the same model, indicating the root cause lies in the data
distribution rather than the codec. The likely mechanism is large K activation outliers that
cannot be represented by any low-bit quantization scheme.

> **Note**: This conclusion is based on current evidence. The investigation remains open
> if new data emerges.

**Workaround**: Use `f16` or `q8_0` for K cache, apply turbo types only to V cache.

### Non-128 Head Dimensions

Models with head dimension not divisible by 128 (e.g., head_dim = 64, 80, 96, 576) trigger a
zero-padding path that wastes computation and memory. The padded dimensions are trimmed post-
attention, but the overhead is measurable.

### Gemma Family (Known Bug)

Combining `sinks=1`, turbo3 KV cache, and `logit_softcap` (Gemma family) produces wrong
results on CUDA. Workaround: avoid using `sinks` with turbo3 KV on CUDA with Gemma models.

### Flash Attention Constraints on ROCm (AMD GPUs)

Validation on **gfx1100 (RDNA3)** under ROCm 7.2.4 showed that Flash Attention (`-fa 1`) has strict format constraints in the backend kernels:
* **Compatible**: `ctk f16` + `ctv f16` or `ctv turbo3`.
* **Incompatible**: Any configuration utilizing `turbo2` or `q8_0` for K cache under Flash Attention triggers a fatal abort at `ggml-cuda/fattn.cu:334`.

**Workaround**: When running with `-fa 1` on AMD GPUs, use `ctk f16` and restrict V cache to `turbo3`.

---

## Regression Testing

Before every push, run the quality gate:

```bash
bash scripts/turbo-quality-gate.sh
```

Expected behavior:

- **PASS**: Both perplexity and context scaling checks are within thresholds.
- **FAIL**: Either test exceeds its threshold. Do not push until the failure is understood
  and resolved.

The gate runs against Qwen3.5-35B-A3B-Q8_0 by default. To use a different model:

```bash
MODEL=/path/to/model.gguf bash scripts/turbo-quality-gate.sh
```

---

## Validation Milestones

### Complete

| Milestone | Date |
|---|---|
| Round-trip quantization test | ✅ |
| Automated quality gate | ✅ |
| Llama-3.2-3B perplexity validation | ✅ |
| CPU/CUDA mathematical equivalence audit | ✅ |
| Qwen3.5-35B-A3B quality gate pass | ✅ |

### Planned

| Milestone | Target |
|---|---|
| Explicit turbo4 NaN validation | Priority P1 |
| Multi-model validation suite (Gemma, Mistral, DeepSeek) | Priority P2 |
| Cross-backend output bit-exactness test | Priority P2 |
| Regression benchmark automation | Priority P2 |
