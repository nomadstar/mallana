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

    const int32_t ns      = (int32_t) page_table->ne[1];
    const int32_t n_lpage = (int32_t) page_table->ne[0];
    const int32_t ptable_elems = ns * n_lpage;

    const int64_t row_bytes = (int64_t) ggml_row_size(v_pool->type, v_pool->ne[0]);

    const char * d_pool = (const char *) v_pool->data;
    char       * d_out  = (char *)       dst->data;

    // page_table may be on host (CUDA_Host) or device, depending on scheduler placement.
    // Use cudaMemcpyDefault so the copy works in both cases without a CPU read.
    const void * ptable_src = page_table->data;

    // [DIAG] Print gather params for first layer of every decode call.
    {
        static int  s_decode_call = 0;
        static bool s_first_layer = true;
        if (s_first_layer) {
            cudaPointerAttributes ptbl_attr{};
            cudaPointerGetAttributes(&ptbl_attr, ptable_src);
            const bool ptable_on_host = (ptbl_attr.type == cudaMemoryTypeHost ||
                                         ptbl_attr.type == cudaMemoryTypeUnregistered);
            fprintf(stderr,
                "[PGATHER#%d] n_kv=%d bs=%d ns=%d n_lpage=%d ptable_mem=%d ptable=[",
                s_decode_call, n_kv, block_size, ns, n_lpage, (int)ptbl_attr.type);
            if (ptable_on_host) {
                const int32_t * h = (const int32_t *) ptable_src;
                for (int i = 0; i < ptable_elems; ++i) {
                    fprintf(stderr, "%d%s", h[i], (i+1<ptable_elems)?",":"");
                }
            } else {
                fprintf(stderr, "<on device>");
            }
            cudaPointerAttributes out_attr{};
            cudaPointerGetAttributes(&out_attr, d_out);
            fprintf(stderr, "] dst_mem=%d pool=%p out=%p\n",
                    (int)out_attr.type, (void*)d_pool, (void*)d_out);
            ++s_decode_call;
        }
        s_first_layer = !s_first_layer;
    }

    ggml_cuda_pool_alloc<int32_t> ptable_buf(ctx.pool(), ptable_elems);
    CUDA_CHECK(cudaMemcpyAsync(ptable_buf.get(), ptable_src,
                               ptable_elems * sizeof(int32_t),
                               cudaMemcpyDefault, ctx.stream()));

    // [DIAG2] Synchronous readback of ptable to verify actual kernel values.
    {
        static int  s_decode_call2 = 0;
        static bool s_first_layer2 = true;
        if (s_first_layer2) {
            CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
            std::vector<int32_t> h_verify(ptable_elems);
            CUDA_CHECK(cudaMemcpy(h_verify.data(), ptable_buf.get(),
                                  ptable_elems * sizeof(int32_t),
                                  cudaMemcpyDeviceToHost));
            fprintf(stderr, "[PGATHER2#%d] kernel_ptable=[", s_decode_call2);
            for (int i = 0; i < ptable_elems; ++i) {
                fprintf(stderr, "%d%s", h_verify[i], (i+1<ptable_elems)?",":"");
            }
            fprintf(stderr, "]\n");
            ++s_decode_call2;
        }
        s_first_layer2 = !s_first_layer2;
    }

    const int32_t n_rows    = n_kv * ns;
    const int32_t n_threads = (int32_t) std::min((int64_t)128, (row_bytes + 3) / 4);

    paged_gather_v_kernel<<<n_rows, n_threads, 0, ctx.stream()>>>(
        d_pool, ptable_buf.get(), d_out,
        n_kv, ns, n_lpage, block_size, row_bytes);
}
