#include "paged-gather.cuh"
#include "common.cuh"

// One CUDA block per output row.  Threads within the block copy 4 bytes each
// for aligned portions, then byte-copy any tail (block_turbo3_0 is 50 bytes,
// not a multiple of 4; the 4-byte-only loop read 2 bytes past end per row).
static __global__ void paged_gather_v_kernel(
        const char    * __restrict__ pool,
        const int32_t * __restrict__ ptable,   // [ns][n_lpage]
        char          * __restrict__ out,
        int32_t n_kv,
        int32_t ns,
        int32_t n_lpage,
        int32_t block_size,
        int64_t row_bytes) {
    // blockIdx.x  = output row index in [0, n_kv * ns)
    const int32_t s = blockIdx.x / n_kv;
    const int32_t r = blockIdx.x % n_kv;

    const int32_t lpage  = r / block_size;
    const int32_t within = r % block_size;
    const int32_t pblock = ptable[s * n_lpage + lpage];

    const int64_t src_row = (int64_t)pblock * block_size + within;

    const char * src = pool + src_row  * row_bytes;
    char       * dst = out  + (int64_t)blockIdx.x * row_bytes;

#if defined(TURBO_DIAG_V_READS)
    // Round 4: dump raw V bytes actually read by the gather (-fa off) path for
    // sequence rows s>=1, r=0 (k_abs=0), so they can be diffed byte-for-byte
    // against the paged FA (-fa on) kernel's [V_READ_FA] dump at the same
    // logical coordinates. Prior rounds only verified this for s=0/page 0.
    if (r == 0 && s >= 1 && s <= 3 && threadIdx.x == 0) {
        uint8_t raw_bytes[8];
        for (int b = 0; b < 8 && b < row_bytes; ++b) raw_bytes[b] = (uint8_t) src[b];
        printf("[V_READ_GATHER] seq=%d k_abs=0 lpage=%d pblock=%d src_row=%lld raw_bytes=%02x%02x%02x%02x%02x%02x%02x%02x\n",
               s, lpage, pblock, (long long) src_row,
               raw_bytes[0], raw_bytes[1], raw_bytes[2], raw_bytes[3],
               raw_bytes[4], raw_bytes[5], raw_bytes[6], raw_bytes[7]);
    }
#endif

    // Fast 4-byte aligned copies for the bulk of the row.
    const int64_t n_full = row_bytes & ~(int64_t)3;
    for (int64_t i = (int64_t)threadIdx.x * 4; i < n_full; i += (int64_t)blockDim.x * 4) {
        *reinterpret_cast<int32_t *>(dst + i) = *reinterpret_cast<const int32_t *>(src + i);
    }
    // Byte-copy the 0–3 remaining bytes (e.g. turbo3: 50 % 4 == 2).
    const int64_t tail = row_bytes - n_full;
    if (tail > 0 && (int64_t)threadIdx.x < tail) {
        dst[n_full + threadIdx.x] = src[n_full + threadIdx.x];
    }
}

void ggml_cuda_gather_paged_v(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * v_pool     = dst->src[0];
    const ggml_tensor * page_table = dst->src[1];

    int32_t block_size, n_kv;
    memcpy(&block_size, dst->op_params + 0, sizeof(int32_t));
    memcpy(&n_kv,       dst->op_params + 4, sizeof(int32_t));

    const int32_t ns      = (int32_t) page_table->ne[1];
    const int32_t n_lpage = (int32_t) page_table->ne[0];
    const int32_t ptable_elems = ns * n_lpage;

    const int64_t row_bytes = (int64_t) ggml_row_size(v_pool->type, v_pool->ne[0]);

    const char * d_pool = (const char *) v_pool->data;
    char       * d_out  = (char *)       dst->data;

    const void * ptable_src = page_table->data;

    ggml_cuda_pool_alloc<int32_t> ptable_buf(ctx.pool(), ptable_elems);
    CUDA_CHECK(cudaMemcpyAsync(ptable_buf.get(), ptable_src,
                               ptable_elems * sizeof(int32_t),
                               cudaMemcpyDefault, ctx.stream()));

    const int32_t n_rows    = n_kv * ns;
    const int32_t n_threads = (int32_t) std::min((int64_t)128, (row_bytes + 3) / 4);

    paged_gather_v_kernel<<<n_rows, n_threads, 0, ctx.stream()>>>(
        d_pool, ptable_buf.get(), d_out,
        n_kv, ns, n_lpage, block_size, row_bytes);
}
