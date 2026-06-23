#include "paged-gather.cuh"
#include "common.cuh"

// One CUDA block per output row.  Threads within the block copy 4 bytes each.
// All quantized row sizes are multiples of 4 bytes.
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

    // 4-byte aligned loop (row_bytes is always a multiple of 4)
    for (int64_t i = (int64_t)threadIdx.x * 4; i < row_bytes; i += (int64_t)blockDim.x * 4) {
        *reinterpret_cast<int32_t *>(dst + i) = *reinterpret_cast<const int32_t *>(src + i);
    }
}

void ggml_cuda_gather_paged_v(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * v_pool     = dst->src[0];
    const ggml_tensor * page_table = dst->src[1];

    int32_t block_size, n_kv;
    memcpy(&block_size, dst->op_params + 0, sizeof(int32_t));
    memcpy(&n_kv,       dst->op_params + 4, sizeof(int32_t));

    const int32_t ns     = (int32_t) page_table->ne[1];
    const int32_t n_lpage = (int32_t) page_table->ne[0];

    const int64_t row_bytes = (int64_t) ggml_row_size(v_pool->type, v_pool->ne[0]);

    const char    * d_pool   = (const char *)    v_pool->data;
    const int32_t * d_ptable = (const int32_t *) page_table->data;
    char          * d_out    = (char *)           dst->data;

    const int32_t n_rows  = n_kv * ns;
    const int32_t n_threads = (int32_t) std::min((int64_t)128, (row_bytes + 3) / 4);

    paged_gather_v_kernel<<<n_rows, n_threads, 0, ctx.stream()>>>(
        d_pool, d_ptable, d_out,
        n_kv, ns, n_lpage, block_size, row_bytes);
}
