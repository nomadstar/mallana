# Benchmarks

> Quality, performance, and memory benchmarks for TurboQuant KV cache compression.

---

## Quality Benchmarks

Perplexity degradation relative to F16 baseline across TurboQuant configurations.

### Llama-3.2-3B

| Configuration | PPL | ΔPPL | Compression vs F16 |
|---|---|---|---|
| F16 (baseline) | 8.68 | — | 1.0× |
| turbo4 V | 8.76 | +0.08 | 3.8× (V only) |
| turbo4 K + turbo4 V | 8.99 | +0.31 | 3.8× |
| turbo4 K | 8.99 | +0.31 | 3.8× (K only) |
| turbo3 K | 9.45 | +0.77 | 4.6× (K only) |
| turbo2 K | 13.42 | +4.74 | 6.4× (K only) |

### Qwen3.5-35B-A3B

| Configuration | PPL | vs q8_0 Baseline |
|---|---|---|
| q8_0 (baseline) | 6.111 | — |
| turbo3 + turbo3 | ≤6.417 | ≤+5% (pass) |

See [validation.md](validation.md) for detailed methodology.

### Qwen Compatibility

| Configuration | PPL | Notes |
|---|---|---|
| F16 (baseline) | 11.79 | — |
| q8_0 + q8_0 | 9.20 | Baseline for comparison |
| q4_0 + q4_0 | 531 | Fails — outlier issue |
| turbo3 + turbo3 | 4098 | Fails — same root cause |
| turbo4 + turbo4 | 1658 | Fails — same root cause |

Qwen-family models exhibit K activation outliers that break all low-bit quantization.

---

## Performance Benchmarks

### Metal (Apple Silicon M5 Max, Qwen3.5-35B-A3B-Q8_0)

**Baseline (no SMEM pre-dequant):**

| Type | Context | Test | t/s |
|---|---|---|---|
| q8_0 | 8K | pp8192 | 2106.47 |
| q8_0 | 8K | tg128 | 76.72 |
| turbo3 | 8K | pp8192 | 2144.16 |
| turbo3 | 8K | tg128 | 78.90 |
| turbo4 | 8K | pp8192 | 2048.90 |
| turbo4 | 8K | tg128 | 79.84 |

**SMEM pre-dequant enabled:**

| Type | Context | Test | t/s |
|---|---|---|---|
| turbo3 | 8K | pp8192 | 1442.03 |
| turbo3 | 8K | tg128 | 40.38 |
| turbo4 | 8K | pp8192 | 1098.56 |
| turbo4 | 8K | tg128 | 50.13 |

> **TODO**: Systematic benchmarks across multiple GPUs, model sizes, and context lengths
> will be maintained in `benches/` as they become available.

---

## Memory Benchmarks

> **TODO**: Memory benchmark data will be collected once the testing infrastructure is
> complete. Placeholder for expected measurements:

| Model | Layers | Heads | Head Dim | Context | F16 Cache | turbo3 Cache | turbo2 Cache |
|---|---|---|---|---|---|---|---|
| Llama-3.2-3B | 28 | 24 | 128 | 128K | ~42 GB | ~9 GB | ~6.5 GB |
| Qwen3.5-35B-A3B | 64 | 16 | 128 | 128K | ~32 GB | ~7 GB | ~5 GB |

Estimates based on formula:
```
cache_size = 2 × layers × (head_dim × heads × context × bytes_per_element)
```

---

## GPU Benchmarks

### CUDA

> **TODO**: CUDA benchmarks to be added. Expected profile:
> - TurboQuant adds negligible decode latency overhead (<5%) for turbo3 and turbo4.
> - Prefill throughput is comparable to q8_0.
> - Memory savings scale linearly with compression ratio.

### Metal

See performance data above for Apple Silicon M5 Max.

### HIP/ROCm

> **TODO**: ROCm benchmarks (RDNA3/CDNA3) to be added for turbo3/turbo2.

---

## CPU Benchmarks

> **TODO**: CPU benchmarks to be added. The CPU path serves as a reference implementation
> and is not optimized for throughput. Expected to be 2-10× slower than GPU for turbo
> quantize/dequantize operations.

---

## Benchmark Methodology

### Measurement

All benchmarks use `llama-perplexity` or `llama-bench` with:

- Fixed model weights (q8_0 or F16)
- Flash Attention enabled (`-fa on`)
- 1 thread for GPU, as many threads as cores for CPU
- 5 runs minimum, report mean ± stddev

### Configurations Tested

For each supported backend and model:

1. **Baseline**: `q8_0` for both K and V.
2. **Turbo variants**: `turbo2`, `turbo3`, `turbo4` for V cache with `q8_0` K.
3. **Symmetric turbo**: Both K and V at the same turbo type.
4. **Asymmetric**: Mixed configurations from the recommended ladder.
