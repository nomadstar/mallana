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

    // [DIAG2] Synchronous readback of ptable + pool rows to verify actual kernel values.
    {
        static int  s_decode_call2 = 0;
        static bool s_first_layer2 = true;
        if (s_first_layer2 && ns > 1) {
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

            // Read first 4 F16 values from pool rows 0 and 64 to verify cpy_v wrote different data
            auto read_pool_row = [&](int64_t row_idx, const char * label) {
                std::vector<uint16_t> h_row(4);
                CUDA_CHECK(cudaMemcpy(h_row.data(), d_pool + row_idx * row_bytes,
                                      4 * sizeof(uint16_t), cudaMemcpyDeviceToHost));
                fprintf(stderr, "  pool_row[%s]=[", label);
                for (int j = 0; j < 4; ++j) {
                    uint16_t v = h_row[j];
                    uint32_t sign=(v>>15)&1, exp=(v>>10)&0x1f, mant=v&0x3ff;
                    uint32_t f32 = (exp==0) ? ((sign<<31)|(mant<<13)) :
                                   (exp==31) ? ((sign<<31)|(0xff<<23)|(mant<<13)) :
                                               ((sign<<31)|((exp+112)<<23)|(mant<<13));
                    float fv; memcpy(&fv,&f32,4);
                    fprintf(stderr,"%.4f,",fv);
                }
                fprintf(stderr,"]\n");
            };
            // Stream 0: pblock 0 → pool row 0. Stream 1: pblock 2 → pool row 64.
            read_pool_row(0, "pblk0=row0");
            read_pool_row(1, "pblk0=row1");   // pos 1 of stream0 (different from pos 0)
            if (h_verify[8] >= 0) {  // strm1 lpage0 pblock
                const int32_t pblk_s1 = h_verify[8];
                read_pool_row((int64_t)pblk_s1 * block_size,     "strm1_lp0_first");
                read_pool_row((int64_t)pblk_s1 * block_size + 1, "strm1_lp0_second");
            }
            ++s_decode_call2;
        }
        s_first_layer2 = !s_first_layer2;
    }

    const int32_t n_rows    = n_kv * ns;
    const int32_t n_threads = (int32_t) std::min((int64_t)128, (row_bytes + 3) / 4);

    paged_gather_v_kernel<<<n_rows, n_threads, 0, ctx.stream()>>>(
        d_pool, ptable_buf.get(), d_out,
        n_kv, ns, n_lpage, block_size, row_bytes);

    // [DIAG3] After gather: readback first 4 F16 values for each stream's first KV pos
    {
        static int  s_dc3 = 0;
        static bool s_fl3 = true;
        if (s_fl3 && ns > 1) {
            CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
            // Also read pool rows to see what was in there before gather
            // Read 8 values from out (4 F16 values per stream for first KV position)
            const int n_f16_per_row = (int)(row_bytes / 2);
            const int n_sample = std::min(4, n_f16_per_row);
            std::vector<uint16_t> h_out(ns * n_sample);
            for (int s = 0; s < ns; ++s) {
                CUDA_CHECK(cudaMemcpy(h_out.data() + s*n_sample,
                                      d_out + (int64_t)s * n_kv * row_bytes,
                                      n_sample * sizeof(uint16_t),
                                      cudaMemcpyDeviceToHost));
            }
            // Also read pool rows corresponding to each stream's first pblock
            fprintf(stderr, "[PGATHER3#%d] ns=%d, gathered out stream0[kv=0][0..%d]=",
                    s_dc3, ns, n_sample-1);
            for (int j = 0; j < n_sample; ++j) {
                uint16_t v = h_out[j];
                float fv;
                // F16 → F32 manual: sign, exp, mantissa
                uint32_t sign = (v >> 15) & 1;
                uint32_t exp  = (v >> 10) & 0x1f;
                uint32_t mant = v & 0x3ff;
                uint32_t f32;
                if (exp == 0) { f32 = (sign << 31) | ((mant) << 13); }
                else if (exp == 31) { f32 = (sign << 31) | (0xff << 23) | (mant << 13); }
                else { f32 = (sign << 31) | ((exp + 112) << 23) | (mant << 13); }
                memcpy(&fv, &f32, 4);
                fprintf(stderr, "%.3f,", fv);
            }
            for (int s = 1; s < ns; ++s) {
                fprintf(stderr, " strm%d[kv=0][0..%d]=", s, n_sample-1);
                for (int j = 0; j < n_sample; ++j) {
                    uint16_t v = h_out[s*n_sample + j];
                    float fv;
                    uint32_t sign = (v >> 15) & 1;
                    uint32_t exp  = (v >> 10) & 0x1f;
                    uint32_t mant = v & 0x3ff;
                    uint32_t f32;
                    if (exp == 0) { f32 = (sign << 31) | ((mant) << 13); }
                    else if (exp == 31) { f32 = (sign << 31) | (0xff << 23) | (mant << 13); }
                    else { f32 = (sign << 31) | ((exp + 112) << 23) | (mant << 13); }
                    memcpy(&fv, &f32, 4);
                    fprintf(stderr, "%.3f,", fv);
                }
            }
            fprintf(stderr, "\n");
            ++s_dc3;
        }
        s_fl3 = !s_fl3;
    }
}
