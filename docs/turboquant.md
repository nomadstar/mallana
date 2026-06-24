# TurboQuant: Algorithm and Implementation

> In-depth description of the TurboQuant KV cache compression algorithm and its
> implementation across CPU, CUDA, and Metal backends.

---

## Motivation

KV cache memory is the primary bottleneck for long-context LLM inference. For a model with
32 layers, 32 attention heads, head_dim = 128, and 16-bit KV cache, each token requires
32 × 2 × 32 × 128 × 2 = 512 KB of cache memory. At 128K context, this reaches 64 GB —
exceeding the VRAM of most GPUs.

Standard quantization methods (q4_0, q8_0) underperform on KV cache data because they
minimize MSE on the original distribution, which is not Gaussian and has heavy tails. The
KV cache's distribution after RoPE exhibits structured patterns that are poorly captured by
uniform or power-of-two quantization grids.

TurboQuant addresses this by applying a Walsh-Hadamard Transform (WHT) before quantization,
which Gaussianizes the data and enables optimal scalar quantization via polar codebooks.

---

## Algorithm

### 1. WHT Rotation

The core insight of TurboQuant is that applying an orthogonal rotation before quantization
Gaussianizes the data distribution. The rotated distribution approaches N(0, σ²/d) where
σ is the original vector norm and d is the dimension.

The rotation uses the **Fast Walsh-Hadamard Transform (FWHT)**:

```
fwht(x):
    for len = 1 to d/2:
        for i = 0 to d-1 step 2*len:
            for j = 0 to len-1:
                u = x[i + j]
                v = x[i + j + len]
                x[i + j]       = u + v
                x[i + j + len] = u - v
    return x
```

The full rotation is: `x_rotated = signs2 ⊙ fwht(signs1 ⊙ x_normalized)`

where `signs1` and `signs2` are pre-computed ±1 sign arrays (512 bytes total) derived from
a seeded random process and QR-orthogonalized. This replaces the original dense 128×128
rotation matrix (64 KB) with an O(d log d) transform (896 operations for d=128).

**Why WHT instead of dense rotation:**

| Aspect | Dense Rotation | WHT Rotation |
|---|---|---|
| Operations | O(d²) = 16384 | O(d log d) = 896 |
| Storage | 64 KB (128×128 fp32) | 512 bytes (128×2 signs) |
| Gaussianization | ✓ | ✓ (same orthogonal group) |
| CPU/GPU consistency | ✗ (dense vs FWHT) | ✓ (identical transform) |

### 2. Normalization

Before rotation, each 128-element block is L2-normalized:

```
x_normalized = x / ‖x‖
```

The norm `‖x‖` is stored as fp16 in the quantized block header. During dequantization,
the reconstructed values are multiplied back by this norm.

### 3. Polar Codebook Quantization

After rotation, the data follows N(0, 1/128). Lloyd-Max optimal centroids are pre-computed:

**2-bit (4 centroids):**
```
{-0.133462, -0.039994, 0.039994, 0.133462}
```

**3-bit (8 centroids):**
```
{-0.190685, -0.117832, -0.065717, -0.021460,
  0.021460,  0.065717,  0.117832,  0.190685}
```

**4-bit (16 centroids):**
```
[Lloyd-Max optimal for N(0, 1/128), pre-computed]
```

Each rotated element is quantized to the nearest centroid. The centroid index is stored in
the packed block format.

### 4. Corrected Norm

After quantization, the reconstruction norm is computed:

```
recon_norm = ‖dequantize(quantize(x_normalized_rotated))‖
corrected_norm = block_norm / recon_norm
```

This ensures that the dequantized vector has the correct L2 norm despite quantization error
in the direction.

---

## Block Structures

### turbo2 (2-bit, 6.4× compression)

| Field | Size | Description |
|---|---|---|
| `norm` | 2 bytes (fp16) | Per-block L2 norm |
| `qs[8]` | 8 bytes | 2-bit indices, 4 per byte |
| **Total** | **10 bytes** | For 128 elements |

4 centroids, no QJL residual.

### turbo3 (3-bit, 4.6× compression)

| Field | Size | Description |
|---|---|---|
| `norm` | 2 bytes (fp16) | Per-block L2 norm |
| `qs[8]` | 8 bytes | Lower 2 bits of 3-bit index (4 per byte) |
| `signs[4]` | 4 bytes | Upper 1 bit of 3-bit index (8 per byte) |
| **Total** | **14 bytes** | For 128 elements |

8 centroids. The 3-bit index is split: lower 2 bits in `qs`, upper 1 bit in `signs`.

### turbo4 (4-bit, 3.8× compression) — Default Mode

| Field | Size | Description |
|---|---|---|
| `norm` | 2 bytes (fp16) | Per-block L2 norm |
| `rnorm` | 2 bytes (fp16) | Reserved (unused in 4-bit mode) |
| `qs[64]` | 64 bytes | 4-bit indices, nibble packed |
| **Total** | **68 bytes** | For 128 elements |

16 centroids, controlled by `TURBO4_USE_4BIT = 1`.

### turbo4 (4-bit, 3.8× compression) — Legacy Mode (`TURBO4_USE_4BIT = 0`)

| Field | Size | Description |
|---|---|---|
| `norm` | 2 bytes (fp16) | Per-block L2 norm |
| `rnorm` | 2 bytes (fp16) | Residual norm for QJL scale |
| `qs[48]` | 48 bytes | 3-bit PolarQuant indices |
| `signs[16]` | 16 bytes | 1-bit QJL signs |
| **Total** | **68 bytes** | For 128 elements |

3-bit PolarQuant + 1-bit QJL residual (original paper design).

---

## CPU Implementation

**File**: `ggml/src/ggml-turbo-quant.c`

### Quantization Path

For each 128-element block:

1. Compute L2 norm.
2. Normalize block by norm.
3. Apply forward WHT (signs → FWHT butterfly → normalize → signs).
4. For each element, find nearest centroid index (2/3/4-bit).
5. Pack indices into block format.
6. Compute corrected norm = grp_norm / recon_norm.
7. Store block (norm + packed indices).

### Dequantization Path

For each 128-element block:

1. Unpack centroid indices from block format.
2. Lookup centroids to reconstruct normalized values.
3. Multiply by stored norm.

### InnerQ (Per-Channel Equalization)

The InnerQ module computes per-channel RMS ratios during calibration and applies them as
scales before quantization. This equalizes channels with different activation statistics,
reducing quantization error on outlier channels. Active when `TURBO_INNERQ=N` env var is set.

---

## CUDA Implementation

### Core Device Functions (`ggml/src/ggml-cuda/turbo-quant.cuh`)

- `turbo_fwht_128()` / `turbo_fwht_64()` — Device-side FWHT butterfly with
  `__syncthreads()` between stages. Uses shared memory for intermediate values.
- `turbo_rotate_forward()` / `turbo_rotate_forward_64()` — signs1 → FWHT → signs2.
- `quantize_f32_turboN_0_block()` — Per-block quantize with centroid lookup via midpoint
  comparisons.
- `turboN_dequant_element()` — Per-element dequant with centroid lookup.

### WHT Kernel (`ggml/src/ggml-cuda/turbo-wht.cu`)

```
k_turbo_wht_f32<direction, group_size>:
  1. Load data from global to shared memory
  2. Apply InnerQ scale (if active)
  3. Apply signs1
  4. FWHT butterfly (macro-based, unrolled stages)
  5. Normalize + apply signs2
  6. Apply InnerQ inverse scale
  7. Write to global memory
```

Dispatch: `ggml_cuda_turbo_wht()` selects the template based on direction and group_size.
Tail elements (when head_dim has remainder after group_size division) are passed through
unmodified.

### SET_ROWS Kernels (`ggml/src/ggml-cuda/set-rows.cu`)

These warp-cooperative kernels handle KV cache row insertion with on-the-fly quantization:

1. Load row data from GPU memory.
2. InnerQ calibration / scale.
3. Parallel L2 norm via warp shuffle reduction.
4. Normalize and forward WHT.
5. Nearest centroid search.
6. Pack indices (warp shuffle for nibble/2-bit, `__ballot_sync` for 1-bit signs).
7. Reconstruction norm computation.
8. Write corrected norm + packed block.

### InnerQ State

Cross-translation-unit state in `turbo-innerq.cuh`:

- `d_innerq_scale[128]` — Device-side per-channel scales.
- `g_innerq_finalized` — Flag indicating calibration is complete.
- `turbo_innerq_publish()` — Uploads finalized scales to device.

---

## Flash Attention Integration

When the KV cache uses turbo types, Flash Attention is automatically enabled
(`src/llama-context.cpp`). The FA kernel dequantizes K and V on the fly:

### KQ Dot Product

```cuda
// For each K element in the tile:
float k_val = dequantize_K(block, element_offset);
float q_val = q_base[element_offset];
dot += k_val * q_val;
```

### V Aggregation

```cuda
// For each V element in the tile:
float v_val = dequantize_V(block, element_offset);
attn_out[i] += weight * v_val;
```

16 template instantiations cover all turbo type combinations for K and V
(turbo2/3/4 × turbo2/3/4/q8_0) plus cross-type combos.

---

## Current Limitations

1. **Head dimension must be a multiple of 128** for full WHT efficiency. Non-128 head
   dimensions (e.g., 64, 96) are zero-padded to the next multiple of 128.
2. **Qwen-family outliers** — Models with large K activation outliers (Qwen, some MoE)
   degrade under low-bit K quantization. The asymmetric policy (q8_0 K + turbo V) is
   recommended for these models.
3. **No group-size flexibility** — WHT group size is fixed at 128 or 64. Dynamic group
   size selection could improve quality on variable-length sequences.
4. **Boundary V is heuristic** — The layer-adaptive policy uses fixed layer indices rather
   than learned importance scores.

---

## Future Optimizations

1. **Variable-bit per layer** — Select bit-width based on layer-specific sensitivity.
2. **Learned boundary detection** — Replace fixed boundary layers with adaptive scoring.
3. **Zero-overhead FA dequant** — Eliminate gather overhead by integrating page table
   lookup directly into the FA kernel.
4. **QJL training on calibration data** — Improve the 1-bit QJL residual for turbo4 legacy
   mode using calibration-set statistics.
5. **Weight + KV joint compression** — Extend TurboQuant to weight tensor quantization
   (TQ3_1S, TQ4_1S formats).
