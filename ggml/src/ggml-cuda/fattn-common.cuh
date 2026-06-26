#pragma once

#include "common.cuh"
#include "convert.cuh"
#include "vecdotq.cuh"
#include "turbo-quant.cuh"

#include <cstdint>
#include <cuda_fp16.h>

#define FATTN_KQ_STRIDE       256
#define HALF_MAX_HALF         __float2half(65504.0f/2) // Use neg. of this instead of -INFINITY to initialize KQ max vals to avoid NaN upon subtraction.
#define SOFTMAX_FTZ_THRESHOLD -20.0f                   // Softmax exp. of values smaller than this are flushed to zero to avoid NaNs.

// log(2) = 0.6931, by adding this to the KQ maximum used for the softmax the numerical range representable
//     by the VKQ accumulators is effectively being shifted up by a factor of 2.
// This reduces issues with numerical overflow but also causes larger values to be flushed to zero.
// However, as the output from FlashAttention will usually be used as an input for a matrix multiplication this should be negligible.
// Still, the value range should be shifted as much as necessary but as little as possible.
// The macro on the following line shifts it by a factor of 2**3=8, as was needed to fix https://github.com/ggml-org/llama.cpp/issues/18606 .
#define FATTN_KQ_MAX_OFFSET (3.0f*0.6931f)

typedef void (* fattn_kernel_t)(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        const char * __restrict__ sinks,
        const int  * __restrict__ KV_max,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33,
        const int32_t * __restrict__ v_ptable,
        const int32_t               v_ptable_ne0,
        const int32_t               v_block_size);

typedef float (*vec_dot_KQ_t)(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8 , const void * __restrict__ Q_ds);

// Resolve physical V row from logical token index using page table.
// Shared by fattn-vec.cuh and fattn-tile.cuh.
static __device__ __forceinline__ const char * v_paged_ptr(
        const char * __restrict__ V_base,
        const int64_t             nb21,
        const int32_t * __restrict__ v_ptable,
        const int32_t             seq,
        const int32_t             n0,
        const int32_t             bs,
        const int32_t             k_abs) {
    if (v_ptable) {
        const int32_t lpage  = k_abs / bs;
        const int32_t within = k_abs % bs;
        const int32_t pblock = v_ptable[seq * n0 + lpage];
        return V_base + ((int64_t)pblock * bs + within) * nb21;
    }
    return V_base + (int64_t)k_abs * nb21;
}

template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_f16(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8 , const void * __restrict__ Q_ds_v) {

    const half2 * K_h2 = (const half2 *) K_c;
    GGML_UNUSED(Q_q8);
    GGML_UNUSED(Q_ds_v);

    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads*cpy_ne) {
        __align__(16) half2 tmp[cpy_ne];
        ggml_cuda_memcpy_1<sizeof(tmp)>(tmp, K_h2 + k_KQ_0 + (threadIdx.x % nthreads)*cpy_ne);
#pragma unroll
        for (int k_KQ_1 = 0; k_KQ_1 < cpy_ne; ++k_KQ_1) {
#ifdef V_DOT2_F32_F16_AVAILABLE
            ggml_cuda_mad(sum,                tmp[k_KQ_1] , ((const half2  *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#else
            ggml_cuda_mad(sum, __half22float2(tmp[k_KQ_1]), ((const float2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#endif // V_DOT2_F32_F16_AVAILABLE
        }
    }

    return sum;
}

template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_bf16(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8 , const void * __restrict__ Q_ds_v) {

    const nv_bfloat162 * K_bf16 = (const nv_bfloat162 *) K_c;
    GGML_UNUSED(Q_q8);
    GGML_UNUSED(Q_ds_v);

    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads*cpy_ne) {
        __align__(16) nv_bfloat162 tmp[cpy_ne];
        ggml_cuda_memcpy_1<sizeof(tmp)>(tmp, K_bf16 + k_KQ_0 + (threadIdx.x % nthreads)*cpy_ne);
#pragma unroll
        for (int k_KQ_1 = 0; k_KQ_1 < cpy_ne; ++k_KQ_1) {
#ifdef V_DOT2_F32_F16_AVAILABLE
            // FIXME replace macros in vector FA kernel with templating and use FP32 for BF16
            ggml_cuda_mad(sum, ggml_cuda_cast<float2>(tmp[k_KQ_1]), __half22float2(((const half2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]));
#else
            ggml_cuda_mad(sum, ggml_cuda_cast<float2>(tmp[k_KQ_1]), ((const float2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#endif // V_DOT2_F32_F16_AVAILABLE
        }
    }

    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q4_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q4_0 * K_q4_0 = (const block_q4_0 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI4_0;
        const int shift = k_KQ & (QI8_1/2);

        int v;
        ggml_cuda_memcpy_1<sizeof(int), 2>(&v, K_q4_0[ib].qs + sizeof(int)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;
        const int u = Q_q8[k_KQ_0/nthreads];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];
        sum += __half2float(K_q4_0[ib].d) * (sumi*Q_ds.x - (8/QI8_1)*Q_ds.y);
    }

    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q4_1(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q4_1 * K_q4_1 = (const block_q4_1 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI4_1;
        const int shift = k_KQ & (QI8_1/2);

        int v;
        ggml_cuda_memcpy_1<sizeof(int)>(&v, K_q4_1[ib].qs + sizeof(int)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;
        const int u = Q_q8[k_KQ_0/nthreads];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

        const float2 K_dm = __half22float2(K_q4_1[ib].dm);
        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];

        sum += K_dm.x*Q_ds.x*sumi + K_dm.y*Q_ds.y/QI8_1;
    }

    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q5_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q5_0 * K_q5_0 = (const block_q5_0 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI5_0;
        const int iqs8  = k_KQ %  QI8_1;
        const int shift = k_KQ & (QI8_1/2);

        int v;
        ggml_cuda_memcpy_1<sizeof(int), 2>(&v, K_q5_0[ib].qs + sizeof(int)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;

        {
            int vh;
            ggml_cuda_memcpy_1<sizeof(int), 2>(&vh, K_q5_0[ib].qh);
            vh >>= iqs8 * QI5_0;

            v |= (vh <<  4) & 0x00000010; // 0 ->  4
            v |= (vh << 11) & 0x00001000; // 1 -> 12
            v |= (vh << 18) & 0x00100000; // 2 -> 20
            v |= (vh << 25) & 0x10000000; // 3 -> 28
        }

        const int u = Q_q8[k_KQ_0/nthreads];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];

        sum += __half2float(K_q5_0[ib].d) * (sumi*Q_ds.x - (16/QI8_1)*Q_ds.y);
    }

    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q5_1(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q5_1 * K_q5_1 = (const block_q5_1 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI5_1;
        const int iqs8  = k_KQ %  QI8_1;
        const int shift = k_KQ & (QI8_1/2);

        int v;
        ggml_cuda_memcpy_1<sizeof(int)>(&v, K_q5_1[ib].qs + sizeof(int)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;

        {
            int vh;
            ggml_cuda_memcpy_1<sizeof(int)>(&vh, K_q5_1[ib].qh);
            vh >>= iqs8 * QI5_0;

            v |= (vh <<  4) & 0x00000010; // 0 ->  4
            v |= (vh << 11) & 0x00001000; // 1 -> 12
            v |= (vh << 18) & 0x00100000; // 2 -> 20
            v |= (vh << 25) & 0x10000000; // 3 -> 28
        }

        const int u = Q_q8[k_KQ_0/nthreads];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

        const float2 K_dm = __half22float2(K_q5_1[ib].dm);
        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];

        sum += K_dm.x*Q_ds.x*sumi + K_dm.y*Q_ds.y/QI8_1;
    }

    return sum;
}

template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q8_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q8_0 * K_q8_0 = (const block_q8_0 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib  = k_KQ / QI8_0;
        const int iqs = k_KQ % QI8_0;

        int v;
        ggml_cuda_memcpy_1<sizeof(v), 2>(&v, K_q8_0[ib].qs + 4*iqs);

        const float2 * Q_ds = (const float2 *) Q_ds_v;
        const float Q_d = Q_ds[k_KQ_0/nthreads].x;

        sum += vec_dot_q8_0_q8_1_impl<float, 1>(&v, &Q_q8[k_KQ_0/nthreads], K_q8_0[ib].d, Q_d);
    }

    return sum;
}

// Turbo3 KQ dot product: dequantize K from turbo3 blocks, dot with Q (float2/half2)
// Uses float Q path (like f16), not q8_1 integer path.
// Q_v is half2[] or float2[] with D/2 pairs, partitioned nthreads-strided.
//
// Matches the f16 pattern: outer loop steps by nthreads*cpy_ne, inner loop
// processes cpy_ne pairs per thread per iteration so Q_v and K indices stay aligned.
// elem0 = 2*k_KQ is always even, so elem0 and elem0+1 always share the same
// turbo3 block (ib), qs byte, and signs byte — loaded once per pair.
template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_turbo3_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_turbo3_0 * K_turbo = (const block_turbo3_0 *) K_c;
    GGML_UNUSED(Q_q8);
    GGML_UNUSED(Q_ds_v);

    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    float sum = 0.0f;

#if defined(TURBO_DIAG_KQ)
    if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0) {
        printf("TURBO3_KQ_DIAG_START norm=%g qs[0]=0x%02x qs[1]=0x%02x qs[2]=0x%02x signs[0]=0x%02x\n",
               __half2float(K_turbo[0].norm),
               (unsigned)K_turbo[0].qs[0], (unsigned)K_turbo[0].qs[1], (unsigned)K_turbo[0].qs[2],
               (unsigned)K_turbo[0].signs[0]);
    }
#endif

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads*cpy_ne) {
#pragma unroll
        for (int k_KQ_1 = 0; k_KQ_1 < cpy_ne; ++k_KQ_1) {
            const int k_KQ = k_KQ_0 + (threadIdx.x % nthreads)*cpy_ne + k_KQ_1;

            // elem0 is always even; elem0 and elem1 are always in the same block,
            // the same qs byte (j0%4 ∈ {0,2}), and the same signs byte (j0%8 ∈ {0,2,4,6}).
            const int elem0 = k_KQ * 2;                  // always even
            const int ib    = elem0 / QK_TURBO3;          // shared block index
            const int j0    = elem0 % QK_TURBO3;          // always even, 0..30

            // Single loads for the shared block fields
            const float     norm     = __half2float(K_turbo[ib].norm);
            const uint8_t   qs_byte  = K_turbo[ib].qs[j0 / 4];      // covers both j0 and j0+1
            const uint8_t   sgn_byte = K_turbo[ib].signs[j0 / 8];   // covers both j0 and j0+1

            // Extract 3-bit indices for elem0 and elem1 from shared bytes
            const int     shift  = (j0 % 4) * 2;                     // 0 or 4
            const uint8_t idx0   = ((qs_byte >> shift)     & 0x3) | (((sgn_byte >> (j0 % 8))     & 0x1) << 2);
            const uint8_t idx1   = ((qs_byte >> (shift+2)) & 0x3) | (((sgn_byte >> (j0 % 8 + 1)) & 0x1) << 2);

            float2 kv;
            kv.x = TURBO_CENTROIDS_3BIT[idx0] * norm;
            kv.y = TURBO_CENTROIDS_3BIT[idx1] * norm;

#ifdef V_DOT2_F32_F16_AVAILABLE
            const half2 qv = ((const half2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1];
            ggml_cuda_mad(sum, make_float2(kv.x, kv.y), __half22float2(qv));
#else
            const float2 qv = ((const float2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1];
            sum += kv.x * qv.x + kv.y * qv.y;
#endif // V_DOT2_F32_F16_AVAILABLE

#if defined(TURBO_DIAG_KQ)
            if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0 && k_KQ_0 == 0 && k_KQ_1 == 0) {
                printf("TURBO3_KQ_DIAG_LOOP ib=%d j0=%d norm=%g qs_byte=0x%02x sgn_byte=0x%02x idx0=%d idx1=%d kv=(%g,%g) partial_sum=%g\n",
                       ib, j0, norm, (unsigned)qs_byte, (unsigned)sgn_byte,
                       (int)idx0, (int)idx1, kv.x, kv.y, sum);
            }
#endif
        }
    }

#if defined(TURBO_DIAG_KQ)
    if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0) {
        printf("TURBO3_KQ_DIAG_END final_sum=%g\n", sum);
    }
#endif

    return sum;
}

// Turbo2 KQ dot product: dequantize K from turbo2 blocks, dot with Q (float2/half2)
// Same structure as turbo3 but reads 2-bit indices from qs only (no signs).
template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_turbo2_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_turbo2_0 * K_turbo = (const block_turbo2_0 *) K_c;
    GGML_UNUSED(Q_q8);
    GGML_UNUSED(Q_ds_v);

    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads*cpy_ne) {
#pragma unroll
        for (int k_KQ_1 = 0; k_KQ_1 < cpy_ne; ++k_KQ_1) {
            const int k_KQ = k_KQ_0 + (threadIdx.x % nthreads)*cpy_ne + k_KQ_1;

            const int elem0 = k_KQ * 2;
            const int ib    = elem0 / QK_TURBO2;
            const int j0    = elem0 % QK_TURBO2;

            const float     norm     = __half2float(K_turbo[ib].norm);
            const uint8_t   qs_byte  = K_turbo[ib].qs[j0 / 4];

            const int     shift  = (j0 % 4) * 2;
            const uint8_t idx0   = (qs_byte >> shift)     & 0x3;
            const uint8_t idx1   = (qs_byte >> (shift+2)) & 0x3;

            float2 kv;
            kv.x = TURBO_CENTROIDS_2BIT[idx0] * norm;
            kv.y = TURBO_CENTROIDS_2BIT[idx1] * norm;

#ifdef V_DOT2_F32_F16_AVAILABLE
            const half2 qv = ((const half2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1];
            ggml_cuda_mad(sum, make_float2(kv.x, kv.y), __half22float2(qv));
#else
            const float2 qv = ((const float2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1];
            sum += kv.x * qv.x + kv.y * qv.y;
#endif // V_DOT2_F32_F16_AVAILABLE
        }
    }

    return sum;
}

// Turbo4 KQ dot product: dequantize K from turbo4 blocks, dot with Q (float2/half2)
// 4-bit nibble packed: qs[j/2] >> ((j%2)*4) & 0xF
template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_turbo4_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_turbo4_0 * K_turbo = (const block_turbo4_0 *) K_c;
    GGML_UNUSED(Q_q8);
    GGML_UNUSED(Q_ds_v);

    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads*cpy_ne) {
#pragma unroll
        for (int k_KQ_1 = 0; k_KQ_1 < cpy_ne; ++k_KQ_1) {
            const int k_KQ = k_KQ_0 + (threadIdx.x % nthreads)*cpy_ne + k_KQ_1;

            const int elem0 = k_KQ * 2;                   // always even
            const int ib    = elem0 / QK_TURBO4;           // block index
            const int j0    = elem0 % QK_TURBO4;           // always even

            const float   norm    = __half2float(K_turbo[ib].norm);
            // Both j0 and j0+1 are adjacent nibbles: j0/2 == (j0+1)/2 when j0 is even
            const uint8_t qs_byte = K_turbo[ib].qs[j0 / 2];

            const uint8_t idx0 = (qs_byte >> 0) & 0xF;    // low nibble = j0
            const uint8_t idx1 = (qs_byte >> 4) & 0xF;    // high nibble = j0+1

            float2 kv;
            kv.x = TURBO_CENTROIDS_4BIT[idx0] * norm;
            kv.y = TURBO_CENTROIDS_4BIT[idx1] * norm;

#ifdef V_DOT2_F32_F16_AVAILABLE
            const half2 qv = ((const half2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1];
            ggml_cuda_mad(sum, make_float2(kv.x, kv.y), __half22float2(qv));
#else
            const float2 qv = ((const float2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1];
            sum += kv.x * qv.x + kv.y * qv.y;
#endif // V_DOT2_F32_F16_AVAILABLE
        }
    }

    return sum;
}

template <typename Tds, int ni>
static __device__ __forceinline__ void quantize_q8_1_to_shared(
    const float * __restrict__ x, const float scale, int * __restrict__ yq32, void * __restrict__ yds) {

    float vals[sizeof(int)] = {0.0f};
#pragma unroll
    for (int l = 0; l < int(sizeof(int)); ++l) {
        vals[l] = (ni == WARP_SIZE || threadIdx.x < ni) ? scale * x[4*threadIdx.x + l] : 0.0f;
    }

    float amax = fabsf(vals[0]);
    float sum  = vals[0];
#pragma unroll
    for (int l = 1; l < int(sizeof(int)); ++l) {
        amax = fmaxf(amax, fabsf(vals[l]));
        sum += vals[l];
    }
#pragma unroll
    for (int mask = QI8_1/2; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, mask, 32));
        sum +=             __shfl_xor_sync(0xFFFFFFFF, sum,  mask, 32);
    }

    const float d = amax / 127;
    int q32 = 0;
    int8_t * q8 = (int8_t *) &q32;

    if (d != 0.0f) {
#pragma unroll
        for (int l = 0; l < int(sizeof(int)); ++l) {
            q8[l] = roundf(vals[l] / d);
        }
    }

    yq32[threadIdx.x] = q32;
    if (threadIdx.x % QI8_1 == 0 && (ni == WARP_SIZE || threadIdx.x < ni)) {
        if (std::is_same<Tds, half2>::value) {
            ((half2  *) yds)[threadIdx.x/QI8_1] =  make_half2(d, sum);
        } else {
            ((float2 *) yds)[threadIdx.x/QI8_1] = make_float2(d, sum);
        }
    }
}

typedef void (*dequantize_V_t)(const void *, void *, const int64_t);

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_f16(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    if constexpr (std::is_same_v<T, half>) {
        ggml_cuda_memcpy_1<ne*sizeof(half)>(dst, (const half *) vx + i0);
    } else if constexpr (std::is_same_v<T, float>) {
        static_assert(ne % 2 == 0, "bad ne");
        __align__(16) half2 tmp[ne/2];
        ggml_cuda_memcpy_1<ne*sizeof(half)>(tmp, (const half *) vx + i0);
        float2 * dst_f2 = (float2 *) dst;
#pragma unroll
        for (int l = 0; l < ne/2; ++l) {
            dst_f2[l] = __half22float2(tmp[l]);
        }
    } else {
        static_assert(std::is_same_v<T, void>, "unsupported type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_bf16(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    static_assert(std::is_same_v<T, float>, "BF16 V dequantization only supports float output");
    static_assert(ne % 2 == 0, "bad ne");
    __align__(16) nv_bfloat162 tmp[ne/2];
    ggml_cuda_memcpy_1<ne*sizeof(nv_bfloat16)>(tmp, (const nv_bfloat16 *) vx + i0);
    float2 * dst_f2 = (float2 *) dst;
#pragma unroll
    for (int l = 0; l < ne/2; ++l) {
        dst_f2[l] = ggml_cuda_cast<float2>(tmp[l]);
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q4_0(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const int64_t ib    =  i0          /  QK4_0;
    const int     iqs   =  i0          % (QK4_0/2);
    const int     shift = (i0 % QK4_0) / (QK4_0/2);

    int q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_cuda_memcpy_1<ne, 2>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;
    q = __vsubss4(q, 0x08080808);

    const int8_t * q8 = (const int8_t *) &q;

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        const half2 d = __half2half2(x[ib].d);

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]);
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
        const float d = x[ib].d;

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = d * q8[l];
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q4_1(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const int64_t ib    =  i0          /  QK4_1;
    const int     iqs   =  i0          % (QK4_1/2);
    const int     shift = (i0 % QK4_1) / (QK4_1/2);

    int q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_cuda_memcpy_1<ne>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    const int8_t * q8 = (const int8_t *) &q;

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        const half2 dm = x[ib].dm;
        const half2 d  = __half2half2( __low2half(dm));
        const half2 m  = __half2half2(__high2half(dm));

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]) + m;
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
        const float2 dm = __half22float2(x[ib].dm);

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = dm.x * q8[l] + dm.y;
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q5_0(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q5_0 * x = (const block_q5_0 *) vx;

    const int64_t ib    =  i0          /  QK5_0;
    const int     idq   =  i0          %  QK5_0;
    const int     iqs   =  i0          % (QK5_0/2);
    const int     shift = (i0 % QK5_0) / (QK5_0/2);

    int q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_cuda_memcpy_1<ne, 2>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    {
        int qh;
        ggml_cuda_memcpy_1<ne, 2>(&qh, x[ib].qh);
#pragma unroll
        for (int l = 0; l < ne; ++l) {
            q |= ((qh >> (idq + l)) & 0x00000001) << (8*l + 4);
        }
    }

    q = __vsubss4(q, 0x10101010);

    const int8_t * q8 = (const int8_t *) &q;

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        const half2 d = __half2half2(x[ib].d);

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]);
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
        const float d = x[ib].d;

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = d * q8[l];
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q5_1(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const int64_t ib    =  i0          /  QK5_1;
    const int     idq   =  i0          %  QK5_1;
    const int     iqs   =  i0          % (QK5_1/2);
    const int     shift = (i0 % QK5_1) / (QK5_1/2);

    int q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_cuda_memcpy_1<ne>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    {
        int qh;
        ggml_cuda_memcpy_1<ne>(&qh, x[ib].qh);
#pragma unroll
        for (int l = 0; l < ne; ++l) {
            q |= ((qh >> (idq + l)) & 0x00000001) << (8*l + 4);
        }
    }

    const int8_t * q8 = (const int8_t *) &q;

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        const half2 dm = x[ib].dm;
        const half2 d  = __half2half2( __low2half(dm));
        const half2 m  = __half2half2(__high2half(dm));

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]) + m;
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
        const float2 dm = __half22float2(x[ib].dm);

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = dm.x * q8[l] + dm.y;
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q8_0(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const int64_t ib  = i0 / QK8_0;
    const int     iqs = i0 % QK8_0;

    static_assert(ne % 2 == 0, "bad ne");
    int8_t qs[ne];
    ggml_cuda_memcpy_1<ne, 2>(qs, x[ib].qs + iqs);

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same<T, half>::value) {
        const half2 d = __half2half2(x[ib].d);

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(qs[l0 + 0], qs[l0 + 1]);
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same<T, float>::value) {
        const float d = x[ib].d;

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = d * qs[l];
        }
    } else {
        static_assert(std::is_same_v<T, void>, "unsupported type");
    }
}

// Turbo3 V dequantize: extract `ne` float/half values at position i0.
//
// Optimised for the ne==4 path (used by the VEC kernel with turbo3 V):
// i0 is always a multiple of 4 from the VEC kernel access pattern, so all 4
// elements share one qs byte and one signs byte — we load each once.
template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_turbo3_0(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_turbo3_0 * x = (const block_turbo3_0 *) vx;

    const int64_t ib   = i0 / QK_TURBO3;
    const int     j0   = i0 % QK_TURBO3;
    const float   norm = __half2float(x[ib].norm);

    static_assert(ne == 2 || ne == 4, "bad ne");

    if constexpr (ne == 4) {
        // When j0 % 4 == 0 (always true from VEC kernel), all 4 elements share one
        // qs byte (4 elements per byte) and one signs byte (8 elements per byte).
        const uint8_t qs_byte  = x[ib].qs[j0 / 4];
        const uint8_t sgn_byte = x[ib].signs[j0 / 8];
        const int     shift_s  = j0 % 8;   // 0 or 4

        const uint8_t idx0 = ((qs_byte >> 0) & 0x3) | (((sgn_byte >> (shift_s+0)) & 0x1) << 2);
        const uint8_t idx1 = ((qs_byte >> 2) & 0x3) | (((sgn_byte >> (shift_s+1)) & 0x1) << 2);
        const uint8_t idx2 = ((qs_byte >> 4) & 0x3) | (((sgn_byte >> (shift_s+2)) & 0x1) << 2);
        const uint8_t idx3 = ((qs_byte >> 6) & 0x3) | (((sgn_byte >> (shift_s+3)) & 0x1) << 2);

#ifdef FP16_AVAILABLE
        if constexpr (std::is_same_v<T, half>) {
            ((half2 *) dst)[0] = make_half2(
                __float2half(TURBO_CENTROIDS_3BIT[idx0] * norm),
                __float2half(TURBO_CENTROIDS_3BIT[idx1] * norm));
            ((half2 *) dst)[1] = make_half2(
                __float2half(TURBO_CENTROIDS_3BIT[idx2] * norm),
                __float2half(TURBO_CENTROIDS_3BIT[idx3] * norm));
        } else
#endif // FP16_AVAILABLE
        if constexpr (std::is_same_v<T, float>) {
            ((float2 *) dst)[0] = make_float2(
                TURBO_CENTROIDS_3BIT[idx0] * norm,
                TURBO_CENTROIDS_3BIT[idx1] * norm);
            ((float2 *) dst)[1] = make_float2(
                TURBO_CENTROIDS_3BIT[idx2] * norm,
                TURBO_CENTROIDS_3BIT[idx3] * norm);
        } else {
            static_assert(std::is_same_v<T, void>, "unsupported type");
        }
    } else { // ne == 2
#ifdef FP16_AVAILABLE
        if constexpr (std::is_same_v<T, half>) {
            float v0 = turbo3_dequant_element(&x[ib], j0,   norm);
            float v1 = turbo3_dequant_element(&x[ib], j0+1, norm);
            ((half2 *) dst)[0] = make_half2(__float2half(v0), __float2half(v1));
        } else
#endif // FP16_AVAILABLE
        if constexpr (std::is_same_v<T, float>) {
            ((float *) dst)[0] = turbo3_dequant_element(&x[ib], j0,   norm);
            ((float *) dst)[1] = turbo3_dequant_element(&x[ib], j0+1, norm);
        } else {
            static_assert(std::is_same_v<T, void>, "unsupported type");
        }
    }
}

// Turbo2 V dequantize: extract `ne` float/half values at position i0.
template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_turbo2_0(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_turbo2_0 * x = (const block_turbo2_0 *) vx;

    const int64_t ib   = i0 / QK_TURBO2;
    const int     j0   = i0 % QK_TURBO2;
    const float   norm = __half2float(x[ib].norm);

    static_assert(ne == 2 || ne == 4, "bad ne");

    if constexpr (ne == 4) {
        const uint8_t qs_byte = x[ib].qs[j0 / 4];

        const uint8_t idx0 = (qs_byte >> 0) & 0x3;
        const uint8_t idx1 = (qs_byte >> 2) & 0x3;
        const uint8_t idx2 = (qs_byte >> 4) & 0x3;
        const uint8_t idx3 = (qs_byte >> 6) & 0x3;

#ifdef FP16_AVAILABLE
        if constexpr (std::is_same_v<T, half>) {
            ((half2 *) dst)[0] = make_half2(
                __float2half(TURBO_CENTROIDS_2BIT[idx0] * norm),
                __float2half(TURBO_CENTROIDS_2BIT[idx1] * norm));
            ((half2 *) dst)[1] = make_half2(
                __float2half(TURBO_CENTROIDS_2BIT[idx2] * norm),
                __float2half(TURBO_CENTROIDS_2BIT[idx3] * norm));
        } else
#endif // FP16_AVAILABLE
        if constexpr (std::is_same_v<T, float>) {
            ((float2 *) dst)[0] = make_float2(
                TURBO_CENTROIDS_2BIT[idx0] * norm,
                TURBO_CENTROIDS_2BIT[idx1] * norm);
            ((float2 *) dst)[1] = make_float2(
                TURBO_CENTROIDS_2BIT[idx2] * norm,
                TURBO_CENTROIDS_2BIT[idx3] * norm);
        } else {
            static_assert(std::is_same_v<T, void>, "unsupported type");
        }
    } else { // ne == 2
#ifdef FP16_AVAILABLE
        if constexpr (std::is_same_v<T, half>) {
            float v0 = turbo2_dequant_element(&x[ib], j0,   norm);
            float v1 = turbo2_dequant_element(&x[ib], j0+1, norm);
            ((half2 *) dst)[0] = make_half2(__float2half(v0), __float2half(v1));
        } else
#endif // FP16_AVAILABLE
        if constexpr (std::is_same_v<T, float>) {
            ((float *) dst)[0] = turbo2_dequant_element(&x[ib], j0,   norm);
            ((float *) dst)[1] = turbo2_dequant_element(&x[ib], j0+1, norm);
        } else {
            static_assert(std::is_same_v<T, void>, "unsupported type");
        }
    }
}

// Turbo4 V dequantize: extract `ne` float/half values at position i0.
// 4-bit nibble packed, block size 128.
template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_turbo4_0(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_turbo4_0 * x = (const block_turbo4_0 *) vx;

    const int64_t ib   = i0 / QK_TURBO4;
    const int     j0   = i0 % QK_TURBO4;
    const float   norm = __half2float(x[ib].norm);

    static_assert(ne == 2 || ne == 4, "bad ne");

    if constexpr (ne == 4) {
        // j0 is always a multiple of 4 from the VEC kernel access pattern.
        // 4 consecutive elements span 2 qs bytes: j0/2 and j0/2+1.
        const uint8_t qs_byte0 = x[ib].qs[j0 / 2];      // elements j0, j0+1
        const uint8_t qs_byte1 = x[ib].qs[j0 / 2 + 1];  // elements j0+2, j0+3

        const uint8_t idx0 = (qs_byte0 >> 0) & 0xF;
        const uint8_t idx1 = (qs_byte0 >> 4) & 0xF;
        const uint8_t idx2 = (qs_byte1 >> 0) & 0xF;
        const uint8_t idx3 = (qs_byte1 >> 4) & 0xF;

#ifdef FP16_AVAILABLE
        if constexpr (std::is_same_v<T, half>) {
            ((half2 *) dst)[0] = make_half2(
                __float2half(TURBO_CENTROIDS_4BIT[idx0] * norm),
                __float2half(TURBO_CENTROIDS_4BIT[idx1] * norm));
            ((half2 *) dst)[1] = make_half2(
                __float2half(TURBO_CENTROIDS_4BIT[idx2] * norm),
                __float2half(TURBO_CENTROIDS_4BIT[idx3] * norm));
        } else
#endif // FP16_AVAILABLE
        if constexpr (std::is_same_v<T, float>) {
            ((float2 *) dst)[0] = make_float2(
                TURBO_CENTROIDS_4BIT[idx0] * norm,
                TURBO_CENTROIDS_4BIT[idx1] * norm);
            ((float2 *) dst)[1] = make_float2(
                TURBO_CENTROIDS_4BIT[idx2] * norm,
                TURBO_CENTROIDS_4BIT[idx3] * norm);
        } else {
            static_assert(std::is_same_v<T, void>, "unsupported type");
        }
    } else { // ne == 2
#ifdef FP16_AVAILABLE
        if constexpr (std::is_same_v<T, half>) {
            float v0 = turbo4_dequant_element(&x[ib], j0,   norm);
            float v1 = turbo4_dequant_element(&x[ib], j0+1, norm);
            ((half2 *) dst)[0] = make_half2(__float2half(v0), __float2half(v1));
        } else
#endif // FP16_AVAILABLE
        if constexpr (std::is_same_v<T, float>) {
            ((float *) dst)[0] = turbo4_dequant_element(&x[ib], j0,   norm);
            ((float *) dst)[1] = turbo4_dequant_element(&x[ib], j0+1, norm);
        } else {
            static_assert(std::is_same_v<T, void>, "unsupported type");
        }
    }
}

template <ggml_type type_K, int D, int nthreads>
constexpr __device__ vec_dot_KQ_t get_vec_dot_KQ() {
    if constexpr (type_K == GGML_TYPE_F16) {
        return vec_dot_fattn_vec_KQ_f16<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q4_0) {
        return vec_dot_fattn_vec_KQ_q4_0<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q4_1) {
        return vec_dot_fattn_vec_KQ_q4_1<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q5_0) {
        return vec_dot_fattn_vec_KQ_q5_0<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q5_1) {
        return vec_dot_fattn_vec_KQ_q5_1<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q8_0) {
        return vec_dot_fattn_vec_KQ_q8_0<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_BF16) {
        return vec_dot_fattn_vec_KQ_bf16<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_TURBO3_0) {
        return vec_dot_fattn_vec_KQ_turbo3_0<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_TURBO2_0) {
        return vec_dot_fattn_vec_KQ_turbo2_0<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_TURBO4_0) {
        return vec_dot_fattn_vec_KQ_turbo4_0<D, nthreads>;
    } else {
        static_assert(type_K == -1, "bad type");
        return nullptr;
    }
}

template <ggml_type type_V, typename T, int ne>
constexpr __device__ dequantize_V_t get_dequantize_V() {
    if constexpr (type_V == GGML_TYPE_F16) {
        return dequantize_V_f16<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q4_0) {
        return dequantize_V_q4_0<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q4_1) {
        return dequantize_V_q4_1<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q5_0) {
        return dequantize_V_q5_0<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q5_1) {
        return dequantize_V_q5_1<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q8_0) {
        return dequantize_V_q8_0<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_BF16) {
        return dequantize_V_bf16<float, ne>;
    } else if constexpr (type_V == GGML_TYPE_TURBO3_0) {
        return dequantize_V_turbo3_0<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_TURBO2_0) {
        return dequantize_V_turbo2_0<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_TURBO4_0) {
        return dequantize_V_turbo4_0<T, ne>;
    } else {
        static_assert(type_V == -1, "bad type");
        return nullptr;
    }
}

template <int ncols1>
__launch_bounds__(FATTN_KQ_STRIDE/2, 1)
static __global__ void flash_attn_mask_to_KV_max(
        const half2 * __restrict__ mask, int * __restrict__ KV_max, const int ne30, const int s31, const int s33) {
    const int ne31     = gridDim.x;
    const int tid      = threadIdx.x;
    const int sequence = blockIdx.y;
    const int jt       = blockIdx.x;

    mask += sequence*s33 + jt*ncols1*s31;

    __shared__ int buf_iw[WARP_SIZE];
    if (tid < WARP_SIZE) {
        buf_iw[tid] = 1;
    }
    __syncthreads();

    int KV_max_sj = (ne30 - 1) * FATTN_KQ_STRIDE;
    for (; KV_max_sj >= 0; KV_max_sj -= FATTN_KQ_STRIDE) {
        int all_inf = 1;

#pragma unroll
        for (int j = 0; j < ncols1; ++j) {
            const float2 tmp = __half22float2(mask[j*s31 + KV_max_sj/2 + tid]);
            all_inf = all_inf && int(isinf(tmp.x)) && int(isinf(tmp.y));
        }

        all_inf = warp_reduce_all(all_inf);
        if (tid % WARP_SIZE == 0) {
            buf_iw[tid / WARP_SIZE] = all_inf;
        }
        __syncthreads();
        all_inf = buf_iw[tid % WARP_SIZE];
        __syncthreads();
        all_inf = warp_reduce_all(all_inf);

        if (!all_inf) {
            break;
        }
    }

    // If the break in the loop was not triggered, KV_max_sj is now -FATTN_KQ_STRIDE.
    // If the break was triggered it's the lower edge of the tile with the first non-masked values.
    // In either case, walk back the decrementation by FATTN_KQ_STRIDE.
    KV_max_sj += FATTN_KQ_STRIDE;

    if (threadIdx.x != 0) {
        return;
    }

    KV_max[sequence*ne31 + jt] = KV_max_sj;
}

template<int D, int ncols1, int ncols2> // D == head size
__launch_bounds__(D, 1)
static __global__ void flash_attn_stream_k_fixup(
        float * __restrict__ dst, const float2 * __restrict__ dst_fixup, const int ne01, const int ne02, const int ne03,
        const int ne11, const int ne12, const int nbatch_fa) {
    constexpr int ncols = ncols1*ncols2;

    const int bidx0 = blockIdx.x;
    const int j     = blockIdx.y;
    const int c     = blockIdx.z;
    const int jc    = j*ncols2 + c;
    const int tid   = threadIdx.x;

    const float * dst_fixup_data = ((const float *) dst_fixup) + gridDim.x*(2*2*ncols);

    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.

    const int iter_k     = (ne11      + (nbatch_fa - 1)) / nbatch_fa;
    const int iter_j     = (ne01      + (ncols1    - 1)) / ncols1;
    const int iter_z_gqa = (gqa_ratio + (ncols2    - 1)) / ncols2;

    const int kbc0      = int64_t(bidx0 + 0)*(iter_k*iter_j*iter_z_gqa*ne12*ne03) / gridDim.x;
    const int kbc0_stop = int64_t(bidx0 + 1)*(iter_k*iter_j*iter_z_gqa*ne12*ne03) / gridDim.x;

    const bool did_not_have_any_data   = kbc0 == kbc0_stop;
    const bool wrote_beginning_of_tile = kbc0 % iter_k == 0;
    const bool did_not_write_last      = kbc0/iter_k == kbc0_stop/iter_k && kbc0_stop % iter_k != 0;
    if (did_not_have_any_data || wrote_beginning_of_tile || did_not_write_last) {
        return;
    }

    // z_KV == K/V head index, zt_gqa = Q head start index per K/V head, jt = token position start index
    const int sequence =  kbc0 /(iter_k*iter_j*iter_z_gqa*ne12);
    const int z_KV     = (kbc0 - iter_k*iter_j*iter_z_gqa*ne12 * sequence)/(iter_k*iter_j*iter_z_gqa);
    const int zt_gqa   = (kbc0 - iter_k*iter_j*iter_z_gqa*ne12 * sequence - iter_k*iter_j*iter_z_gqa * z_KV)/(iter_k*iter_j);
    const int jt       = (kbc0 - iter_k*iter_j*iter_z_gqa*ne12 * sequence - iter_k*iter_j*iter_z_gqa * z_KV - iter_k*iter_j * zt_gqa) / iter_k;

    const int zt_Q = z_KV*gqa_ratio + zt_gqa*ncols2; // Global Q head start index.

    if (jt*ncols1 + j >= ne01 || zt_gqa*ncols2 + c >= gqa_ratio) {
        return;
    }

    dst += sequence*ne02*ne01*D + jt*ne02*(ncols1*D) + zt_Q*D + (j*ne02 + c)*D + tid;

    // Load the partial result that needs a fixup:
    float dst_val = 0.0f;
    float max_val = 0.0f;
    float rowsum  = 0.0f;
    {
        dst_val = *dst;

        const float2 tmp = dst_fixup[bidx0*ncols + jc];
        max_val = tmp.x;
        rowsum  = tmp.y;
    }

    // Iterate over previous blocks and compute the combined results.
    // All CUDA blocks that get here must have a previous block that needs a fixup.
    int bidx = bidx0 - 1;
    int kbc_stop = kbc0;
    while(true) {
        const int kbc = int64_t(bidx)*(iter_k*iter_j*iter_z_gqa*ne12*ne03) / gridDim.x;
        if (kbc == kbc_stop) { // Did not have any data.
            bidx--;
            kbc_stop = kbc;
            continue;
        }

        const float dst_add = dst_fixup_data[bidx*ncols*D + jc*D + tid];

        const float2 tmp = dst_fixup[(gridDim.x + bidx)*ncols + jc];

        // Scale the current and new value accumulators depending on the max. values.
        const float max_val_new = fmaxf(max_val, tmp.x);

        const float diff_val = max_val - max_val_new;
        const float diff_add = tmp.x   - max_val_new;

        const float scale_val = diff_val >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_val) : 0.0f;
        const float scale_add = diff_add >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_add) : 0.0f;

        dst_val = scale_val*dst_val + scale_add*dst_add;
        rowsum  = scale_val*rowsum  + scale_add*tmp.y;

        max_val = max_val_new;

        // If this block started in a previous tile we are done and don't need to combine additional partial results.
        if (kbc % iter_k == 0 || kbc/iter_k < kbc0/iter_k) {
            break;
        }
        bidx--;
        kbc_stop = kbc;
    }

    // Write back final result:
    *dst = dst_val / rowsum;
}

template<int D> // D == head size
__launch_bounds__(D, 1)
static __global__ void flash_attn_combine_results(
        const float  * __restrict__ VKQ_parts,
        const float2 * __restrict__ VKQ_meta,
        float * __restrict__ dst,
        const int parallel_blocks) {
    // Dimension 0: threadIdx.x
    // Dimension 1: blockIdx.x
    // Dimension 2: blockIdx.y
    // Dimension 3: blockIdx.z
    // Memory layout is permuted with [0, 2, 1, 3]

    const int ne01 = gridDim.x;
    const int ne02 = gridDim.y;

    const int col      = blockIdx.x;
    const int head     = blockIdx.y;
    const int sequence = blockIdx.z;

    const int j_dst_unrolled = (sequence*ne01 + col)*ne02 + head;

    VKQ_parts += j_dst_unrolled * parallel_blocks*D;
    VKQ_meta  += j_dst_unrolled * parallel_blocks;
    dst       += j_dst_unrolled *                 D;

    const int tid = threadIdx.x;
    __builtin_assume(tid < D);

    extern __shared__ float2 meta[];
    for (int i = tid; i < 2*parallel_blocks; i += D) {
        ((float *) meta)[i] = ((const float *)VKQ_meta) [i];
    }

    __syncthreads();

    float kqmax = meta[0].x;
    for (int l = 1; l < parallel_blocks; ++l) {
        kqmax = max(kqmax, meta[l].x);
    }

    float VKQ_numerator   = 0.0f;
    float VKQ_denominator = 0.0f;
    for (int l = 0; l < parallel_blocks; ++l) {
        const float KQ_max_scale = expf(meta[l].x - kqmax);

        VKQ_numerator   += KQ_max_scale * VKQ_parts[l*D + tid];
        VKQ_denominator += KQ_max_scale * meta[l].y;
    }

    dst[tid] = VKQ_numerator / VKQ_denominator;
}

template <int DV, int ncols1, int ncols2>
void launch_fattn(
    ggml_backend_cuda_context & ctx, ggml_tensor * dst, fattn_kernel_t fattn_kernel, const int nwarps, const size_t nbytes_shared,
    const int nbatch_fa, const bool need_f16_K, const bool need_f16_V, const bool stream_k, const int warp_size = WARP_SIZE
) {
    constexpr int ncols = ncols1 * ncols2;

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    const bool V_is_K_view = V->view_src && (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));

    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    const ggml_tensor * v_ptable_tensor = dst->src[5];
    const int32_t * v_ptable_data  = v_ptable_tensor ? (const int32_t *)v_ptable_tensor->data : nullptr;
    int32_t         v_ptable_ne0   = v_ptable_tensor ? (int32_t)v_ptable_tensor->ne[0] : 0;
    int32_t         v_block_size   = 0;
    if (v_ptable_tensor) {
        memcpy(&v_block_size, v_ptable_tensor->op_params, sizeof(int32_t));
    }

    ggml_tensor * KQV = dst;

    GGML_ASSERT(Q->type == GGML_TYPE_F32);
    GGML_ASSERT(KQV->type == GGML_TYPE_F32);

    GGML_ASSERT(Q->nb[0] == ggml_element_size(Q));
    GGML_ASSERT(K->nb[0] == ggml_element_size(K));
    GGML_ASSERT(V->nb[0] == ggml_element_size(V));

    GGML_ASSERT(!mask || mask->type == GGML_TYPE_F16);

    ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t main_stream = ctx.stream();
    const int id  = ggml_cuda_get_device();
    const int cc  = ggml_cuda_info().devices[id].cc;
    const int nsm = ggml_cuda_info().devices[id].nsm;

    ggml_cuda_pool_alloc<half>   K_f16(pool);
    ggml_cuda_pool_alloc<half>   V_f16(pool);
    ggml_cuda_pool_alloc<int>    KV_max(pool);
    ggml_cuda_pool_alloc<float>  dst_tmp(pool);
    ggml_cuda_pool_alloc<float2> dst_tmp_meta(pool);

    const char * K_data = (const char *) K->data;
#if defined(TURBO_DIAG_K_TILE)
    const char * K_data_orig = K_data;
#endif
    size_t nb11 = K->nb[1];
    size_t nb12 = K->nb[2];
    size_t nb13 = K->nb[3];

    const char * V_data = (const char *) V->data;
    size_t nb21 = V->nb[1];
    size_t nb22 = V->nb[2];
    size_t nb23 = V->nb[3];

    if (need_f16_K && K->type != GGML_TYPE_F16) {
#if defined(TURBO_DIAG_K_TILE)
        fprintf(stderr, "[TURBO_ENTER_DEQUANT_K] type=%s ne=[%lld,%lld,%lld,%lld] need_f16=%d\n",
                ggml_type_name(K->type), (long long)K->ne[0], (long long)K->ne[1], (long long)K->ne[2], (long long)K->ne[3], (int)need_f16_K);
#endif
        const size_t bs = ggml_blck_size(K->type);
        const size_t ts = ggml_type_size(K->type);

        K_f16.alloc(ggml_nelements(K));
        if (ggml_is_contiguously_allocated(K)) {
            to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(K->type);
            to_fp16(K_data, K_f16.ptr, ggml_nelements(K), main_stream);

            nb11 = nb11*bs*sizeof(half)/ts;
            nb12 = nb12*bs*sizeof(half)/ts;
            nb13 = nb13*bs*sizeof(half)/ts;
        } else {
            GGML_ASSERT(K->nb[0] == ts);
            to_fp16_nc_cuda_t to_fp16 = ggml_get_to_fp16_nc_cuda(K->type);
            const int64_t s01 = nb11 / ts;
            const int64_t s02 = nb12 / ts;
            const int64_t s03 = nb13 / ts;
            to_fp16(K_data, K_f16.ptr, K->ne[0], K->ne[1], K->ne[2], K->ne[3], s01, s02, s03, main_stream);

            nb11 = K->ne[0] * sizeof(half);
            nb12 = K->ne[1] * nb11;
            nb13 = K->ne[2] * nb12;
        }
        K_data = (char *) K_f16.ptr;

#if defined(TURBO_DIAG_KQ)
        {
            static int s_count = 0;
            if (s_count++ < 3) {
                CUDA_CHECK(cudaStreamSynchronize(main_stream));
                // First 128 elements of K (first KV position, first head)
                half K_tmp[128];
                cudaMemcpy(K_tmp, K_f16.ptr, sizeof(K_tmp), cudaMemcpyDeviceToHost);
                // First 128 floats of Q (first Q token, first head) — Q is float32
                float Q_tmp[128];
                cudaMemcpy(Q_tmp, Q->data, sizeof(Q_tmp), cudaMemcpyDeviceToHost);
                // Manual dot product Q · K for first KV position
                double dot_manual = 0.0;
                for (int _i = 0; _i < 128; _i++) {
                    dot_manual += (double)Q_tmp[_i] * (double)__half2float(K_tmp[_i]);
                }
                // Q L2 norm and K L2 norm
                double q_sumsq = 0.0, k_sumsq = 0.0;
                for (int _i = 0; _i < 128; _i++) {
                    q_sumsq += (double)Q_tmp[_i] * Q_tmp[_i];
                    k_sumsq += (double)__half2float(K_tmp[_i]) * __half2float(K_tmp[_i]);
                }
                int orig_contig = ggml_is_contiguously_allocated(K);
                fprintf(stderr, "[DIAG_KDEQ] s=%d K: ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] "
                        "nb11(adj)=%zu nb12(adj)=%zu nb13(adj)=%zu "
                        "contig=%d type=%s nelem=%lld "
                        "K_first8: %g %g %g %g %g %g %g %g "
                        "K_l2=%.4g Q_l2=%.4g dot=%.6g "
                        "Q_first8: %g %g %g %g %g %g %g %g\n",
                        s_count-1,
                        (long long)K->ne[0], (long long)K->ne[1], (long long)K->ne[2], (long long)K->ne[3],
                        K->nb[0], K->nb[1], K->nb[2], K->nb[3],
                        nb11, nb12, nb13,
                        orig_contig, ggml_type_name(K->type),
                        (long long)ggml_nelements(K),
                        __half2float(K_tmp[0]), __half2float(K_tmp[1]), __half2float(K_tmp[2]), __half2float(K_tmp[3]),
                        __half2float(K_tmp[4]), __half2float(K_tmp[5]), __half2float(K_tmp[6]), __half2float(K_tmp[7]),
                        sqrt(k_sumsq), sqrt(q_sumsq), dot_manual,
                        Q_tmp[0], Q_tmp[1], Q_tmp[2], Q_tmp[3],
                        Q_tmp[4], Q_tmp[5], Q_tmp[6], Q_tmp[7]);
            }
        }
#endif
    }

#if defined(TURBO_DIAG_KQ)
    // For F16 K baseline: read first 8 K elements directly (no dequant needed)
    if (!need_f16_K && K->type == GGML_TYPE_F16) {
        static int s_f16_count = 0;
        if (s_f16_count++ < 3) {
            CUDA_CHECK(cudaStreamSynchronize(main_stream));
            half K_f16_raw[8];
            CUDA_CHECK(cudaMemcpy(K_f16_raw, K->data, sizeof(K_f16_raw), cudaMemcpyDeviceToHost));
            double k_l2_partial = 0.0;
            half K_128[128];
            CUDA_CHECK(cudaMemcpy(K_128, K->data, sizeof(K_128), cudaMemcpyDeviceToHost));
            for (int _i = 0; _i < 128; _i++) k_l2_partial += (double)__half2float(K_128[_i]) * __half2float(K_128[_i]);
            fprintf(stderr, "[F16_K_DIAG] s=%d K_l2=%.4g K_first8: %g %g %g %g %g %g %g %g\n",
                    s_f16_count-1, sqrt(k_l2_partial),
                    __half2float(K_f16_raw[0]), __half2float(K_f16_raw[1]),
                    __half2float(K_f16_raw[2]), __half2float(K_f16_raw[3]),
                    __half2float(K_f16_raw[4]), __half2float(K_f16_raw[5]),
                    __half2float(K_f16_raw[6]), __half2float(K_f16_raw[7]));
        }
    }
#endif
    if (need_f16_V && V->type != GGML_TYPE_F16) {
        if (V_is_K_view) {
            V_data = K_data;
            nb21   = nb11;
            nb22   = nb12;
            nb23   = nb13;
        } else {
            const size_t bs = ggml_blck_size(V->type);
            const size_t ts = ggml_type_size(V->type);

            V_f16.alloc(ggml_nelements(V));
            if (ggml_is_contiguously_allocated(V)) {
                to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(V->type);
                to_fp16(V_data, V_f16.ptr, ggml_nelements(V), main_stream);
                V_data = (char *) V_f16.ptr;

                nb21 = nb21*bs*sizeof(half)/ts;
                nb22 = nb22*bs*sizeof(half)/ts;
                nb23 = nb23*bs*sizeof(half)/ts;
            } else {
                GGML_ASSERT(V->nb[0] == ts);
                to_fp16_nc_cuda_t to_fp16 = ggml_get_to_fp16_nc_cuda(V->type);
                const int64_t s01 = nb21 / ts;
                const int64_t s02 = nb22 / ts;
                const int64_t s03 = nb23 / ts;
                to_fp16(V_data, V_f16.ptr, V->ne[0], V->ne[1], V->ne[2], V->ne[3], s01, s02, s03, main_stream);

                nb21 = V->ne[0] * sizeof(half);
                nb22 = V->ne[1] * nb21;
                nb23 = V->ne[2] * nb22;
            }
            V_data = (char *) V_f16.ptr;

#if defined(TURBO_DIAG_KQ)
            {
                static int s_count = 0;
                if (s_count++ < 5) {
                    CUDA_CHECK(cudaStreamSynchronize(main_stream));
                    half tmp[16];
                    cudaMemcpy(tmp, V_f16.ptr, sizeof(tmp), cudaMemcpyDeviceToHost);
                    int orig_contig = ggml_is_contiguously_allocated(V);
                    fprintf(stderr, "[DIAG_VDEQ] V dequant: ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] "
                            "contig=%d type=%s nelem=%lld nbytes=%zu "
                            "first16: %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g\n",
                            (long long)V->ne[0], (long long)V->ne[1], (long long)V->ne[2], (long long)V->ne[3],
                            V->nb[0], V->nb[1], V->nb[2], V->nb[3],
                            orig_contig, ggml_type_name(V->type),
                            (long long)ggml_nelements(V),
                            (long long)ggml_nbytes(V),
                            __half2float(tmp[0]), __half2float(tmp[1]), __half2float(tmp[2]), __half2float(tmp[3]),
                            __half2float(tmp[4]), __half2float(tmp[5]), __half2float(tmp[6]), __half2float(tmp[7]),
                            __half2float(tmp[8]), __half2float(tmp[9]), __half2float(tmp[10]), __half2float(tmp[11]),
                            __half2float(tmp[12]), __half2float(tmp[13]), __half2float(tmp[14]), __half2float(tmp[15]));
                }
            }
#endif
        }
    }

    const int ntiles_x     = ((Q->ne[1] + ncols1 - 1) / ncols1);
    const int gqa_ratio    = Q->ne[2] / K->ne[2];
    const int ntiles_z_gqa = ((gqa_ratio + ncols2 - 1) / ncols2);
    const int ntiles_dst   = ntiles_x * ntiles_z_gqa * K->ne[2] * Q->ne[3];

    // Optional optimization where the mask is scanned to determine whether part of the calculation can be skipped.
    // Only worth the overhead if there is at lease one FATTN_KQ_STRIDE x FATTN_KQ_STRIDE square to be skipped or
    //     multiple sequences of possibly different lengths.
    if (mask && K->ne[1] % FATTN_KQ_STRIDE == 0 && (Q->ne[1] >= 1024 || Q->ne[3] > 1)) {
        const int s31 = mask->nb[1] / sizeof(half2);
        const int s33 = mask->nb[3] / sizeof(half2);

        const dim3 blocks_num_KV_max(ntiles_x, Q->ne[3], 1);
        const dim3 block_dim_KV_max(FATTN_KQ_STRIDE/2, 1, 1);

        const int ne_KV_max = blocks_num_KV_max.x*blocks_num_KV_max.y;
        const int iter_k = K->ne[1] / FATTN_KQ_STRIDE;

        KV_max.alloc(ne_KV_max);
        flash_attn_mask_to_KV_max<ncols1><<<blocks_num_KV_max, block_dim_KV_max, 0, main_stream>>>
            ((const half2 *) mask->data, KV_max.ptr, iter_k, s31, s33);
        CUDA_CHECK(cudaGetLastError());
    }

    const dim3 block_dim(warp_size, nwarps, 1);
    int max_blocks_per_sm = 1; // Max. number of active blocks limited by occupancy.
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, fattn_kernel, block_dim.x * block_dim.y * block_dim.z, nbytes_shared));
    GGML_ASSERT(max_blocks_per_sm > 0);
    int parallel_blocks = max_blocks_per_sm;

    const int ntiles_KV = (K->ne[1] + nbatch_fa - 1) / nbatch_fa; // Max. number of parallel blocks limited by KV cache length.

    dim3 blocks_num;
    if (stream_k) {
        // For short contexts it can be faster to have the SMs work on whole tiles because this lets us skip the fixup.
        const int max_blocks = max_blocks_per_sm*nsm;
        const int tiles_nwaves = (ntiles_dst + max_blocks - 1) / max_blocks;
        const int tiles_efficiency_percent = 100 * ntiles_dst / (max_blocks*tiles_nwaves);

        const int nblocks_stream_k = std::min(max_blocks, ntiles_KV*ntiles_dst);

        const bool use_stream_k = cc >= GGML_CUDA_CC_ADA_LOVELACE || amd_wmma_available(cc) || tiles_efficiency_percent < 75;

        blocks_num.x = use_stream_k ? nblocks_stream_k : ntiles_dst;
        blocks_num.y = 1;
        blocks_num.z = 1;

        if (ntiles_dst % blocks_num.x != 0) { // Fixup is only needed if the SMs work on fractional tiles.
            dst_tmp_meta.alloc((size_t(blocks_num.x) * ncols * (2 + DV/2)));
        }
    } else {
        // parallel_blocks must not be larger than what the tensor size allows:
        parallel_blocks = std::min(parallel_blocks, ntiles_KV);

        // If ntiles_total % blocks_per_wave != 0 then some efficiency is lost due to tail effects.
        // Test whether parallel_blocks can be set to a higher value for better efficiency.
        const int blocks_per_wave = nsm * max_blocks_per_sm;
        int nwaves_best = 0;
        int efficiency_percent_best = 0;
        for (int parallel_blocks_test = parallel_blocks; parallel_blocks_test <= ntiles_KV; ++parallel_blocks_test) {
            const int nblocks_total = ntiles_dst * parallel_blocks_test;
            const int nwaves = (nblocks_total + blocks_per_wave - 1) / blocks_per_wave;
            const int efficiency_percent = 100 * nblocks_total / (nwaves*blocks_per_wave);

            // Stop trying configurations with more waves if we already have good efficiency to avoid excessive overhead.
            if (efficiency_percent_best >= 95 && nwaves > nwaves_best) {
                break;
            }

            if (efficiency_percent > efficiency_percent_best) {
                nwaves_best = nwaves;
                efficiency_percent_best = efficiency_percent;
                parallel_blocks = parallel_blocks_test;
            }
        }

        blocks_num.x = ntiles_x;
        blocks_num.y = parallel_blocks;
        blocks_num.z = ntiles_z_gqa*K->ne[2]*Q->ne[3];

        if (parallel_blocks > 1) {
            dst_tmp.alloc(parallel_blocks*ggml_nelements(KQV));
            dst_tmp_meta.alloc(parallel_blocks*ggml_nrows(KQV));
        }
    }

    float scale         = 1.0f;
    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;

    memcpy(&scale,         (const float *) KQV->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (const float *) KQV->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    const uint32_t n_head      = Q->ne[2];
    const uint32_t n_head_log2 = 1u << uint32_t(floorf(log2f(float(n_head))));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    // TODO other tensor dimensions after removal of WMMA kernel:
    const uint3 ne01 = init_fastdiv_values(Q->ne[1]);

#if defined(TURBO_DIAG_K_TILE)
    if (K->type == GGML_TYPE_TURBO3_0 && need_f16_K) {
        cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
        CUDA_CHECK(cudaStreamIsCapturing(main_stream, &capture_status));
        if (capture_status == cudaStreamCaptureStatusNone) {

        static constexpr float turbo3_centroids_host[8] = {
            -0.190685f, -0.117832f, -0.065717f, -0.021460f,
             0.021460f,  0.065717f,  0.117832f,  0.190685f
        };

        const int gqa_ratio_diag    = Q->ne[2] / K->ne[2];
        const int iter_k_diag       = (K->ne[1] + (nbatch_fa - 1)) / nbatch_fa;
        const int iter_j_diag       = (Q->ne[1] + (ncols1    - 1)) / ncols1;
        const int iter_z_gqa_diag   = (gqa_ratio_diag + (ncols2 - 1)) / ncols2;
        const int64_t kbc_diag      = int64_t(0)*(iter_k_diag*iter_j_diag*iter_z_gqa_diag*K->ne[2]*Q->ne[3]) / blocks_num.x;
        const int kb0_start_diag    = kbc_diag % iter_k_diag;
        const int sequence_diag     = kbc_diag /(iter_k_diag*iter_j_diag*iter_z_gqa_diag*K->ne[2]);
        const int head_diag         = (kbc_diag - int64_t(iter_k_diag)*iter_j_diag*iter_z_gqa_diag*K->ne[2]*sequence_diag)/(iter_k_diag*iter_j_diag*iter_z_gqa_diag);
        const int zt_gqa_diag       = (kbc_diag - int64_t(iter_k_diag)*iter_j_diag*iter_z_gqa_diag*K->ne[2]*sequence_diag - int64_t(iter_k_diag)*iter_j_diag*iter_z_gqa_diag*head_diag)/(iter_k_diag*iter_j_diag);
        const int jt_diag           = (kbc_diag - int64_t(iter_k_diag)*iter_j_diag*iter_z_gqa_diag*K->ne[2]*sequence_diag - int64_t(iter_k_diag)*iter_j_diag*iter_z_gqa_diag*head_diag - int64_t(iter_k_diag)*iter_j_diag*zt_gqa_diag) / iter_k_diag;
        const int kv_index_diag     = kb0_start_diag * nbatch_fa;
        const int block_index_diag  = 0;

        int layer_diag = -1;
        sscanf(K->name, "cache_k_l%d", &layer_diag);

        const char * blk_dev  = K_data_orig + K->nb[3]*sequence_diag + K->nb[2]*head_diag + K->nb[1]*kv_index_diag + sizeof(block_turbo3_0)*block_index_diag;
        const char * tile_dev = K_data      + nb13*sequence_diag    + nb12*head_diag    + nb11*kv_index_diag    + sizeof(half)*QK_TURBO3*block_index_diag;

        block_turbo3_0 blk_host;
        half k_gpu[QK_TURBO3];
        CUDA_CHECK(cudaMemcpyAsync(&blk_host, blk_dev, sizeof(blk_host), cudaMemcpyDeviceToHost, main_stream));
        CUDA_CHECK(cudaMemcpyAsync(k_gpu, tile_dev, sizeof(k_gpu), cudaMemcpyDeviceToHost, main_stream));
        CUDA_CHECK(cudaStreamSynchronize(main_stream));

        const float norm = __half2float(blk_host.norm);
        float k_cpu[QK_TURBO3];
        for (int j = 0; j < QK_TURBO3; ++j) {
            const uint8_t low2 = (blk_host.qs[j / 4] >> ((j % 4) * 2)) & 0x3;
            const uint8_t hi1  = (blk_host.signs[j / 8] >> (j % 8)) & 0x1;
            const uint8_t idx  = low2 | (hi1 << 2);
            k_cpu[j] = __half2float(__float2half(turbo3_centroids_host[idx] * norm));
        }

        double max_abs = 0.0;
        double rms = 0.0;
        int first_mismatch = -1;
        for (int j = 0; j < QK_TURBO3; ++j) {
            const double err = fabs((double)__half2float(k_gpu[j]) - (double)k_cpu[j]);
            rms += err * err;
            if (err > max_abs) {
                max_abs = err;
            }
            if (first_mismatch < 0 && err >= 1e-4) {
                first_mismatch = j;
            }
        }
        rms = sqrt(rms / QK_TURBO3);

        fprintf(stderr,
                "[TURBO_DIAG_K_TILE] layer=%d head=%d kv_index=%d block_index=%d original_block=%p fp16_tile=%p jt=%d norm=%g max_abs=%.9g rms=%.9g first_mismatch=%d\n",
                layer_diag, head_diag, kv_index_diag, block_index_diag, (const void *) blk_dev, (const void *) tile_dev,
                jt_diag, norm, max_abs, rms, first_mismatch);
        if (first_mismatch >= 0) {
            const int j = first_mismatch;
            const uint8_t low2 = (blk_host.qs[j / 4] >> ((j % 4) * 2)) & 0x3;
            const uint8_t hi1  = (blk_host.signs[j / 8] >> (j % 8)) & 0x1;
            const uint8_t idx  = low2 | (hi1 << 2);
            fprintf(stderr,
                    "[TURBO_DIAG_K_TILE] mismatch j=%d cpu=%g gpu=%g idx=%u\n",
                    j, k_cpu[j], __half2float(k_gpu[j]), (unsigned) idx);
        }
        GGML_ASSERT(max_abs < 1e-4f);
        }
    }
#endif

    GGML_ASSERT(block_dim.x % warp_size == 0);
    fattn_kernel<<<blocks_num, block_dim, nbytes_shared, main_stream>>>(
        (const char *) Q->data,
        K_data,
        V_data,
        mask ? ((const char *) mask->data) : nullptr,
        sinks ? ((const char *) sinks->data) : nullptr,
        KV_max.ptr,
        !stream_k && parallel_blocks > 1 ? dst_tmp.ptr : (float *) KQV->data, dst_tmp_meta.ptr,
        scale, max_bias, m0, m1, n_head_log2, logit_softcap,
        Q->ne[0], ne01,     Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3],
        K->ne[0], K->ne[1], K->ne[2], K->ne[3], nb11, nb12, nb13,
        nb21, nb22, nb23,
        mask ? mask->ne[1] : 0, mask ? mask->ne[2] : 0, mask ? mask->ne[3] : 0,
        mask ? mask->nb[1] : 0, mask ? mask->nb[2] : 0, mask ? mask->nb[3] : 0,
        v_ptable_data, v_ptable_ne0, v_block_size
    );
    CUDA_CHECK(cudaGetLastError());

#if defined(TURBO_DIAG_KQ)
    // CPU-side diagnostic: for K=turbo3 VEC path, verify KQ dot product after first kernel call.
    if (!need_f16_K && K->type == GGML_TYPE_TURBO3_0) {
        static int s_vec_count = 0;
        if (s_vec_count++ < 1) {
            CUDA_CHECK(cudaStreamSynchronize(main_stream));
            // Read K block 0 (first KV position, head 0) from device
            block_turbo3_0 blk0;
            CUDA_CHECK(cudaMemcpy(&blk0, K->data, sizeof(blk0), cudaMemcpyDeviceToHost));
            // Read Q head 0 (first 128 floats = 64 float2) from device
            float Q_h[128];
            CUDA_CHECK(cudaMemcpy(Q_h, Q->data, sizeof(Q_h), cudaMemcpyDeviceToHost));
            // Manually dequantize K block 0
            float K_dequant[128];
            float kblk_norm = __half2float(blk0.norm);
            for (int j = 0; j < 128; j++) {
                uint8_t low2 = (blk0.qs[j / 4] >> ((j % 4) * 2)) & 0x3;
                uint8_t hi1  = (blk0.signs[j / 8] >> (j % 8)) & 0x1;
                uint8_t idx  = low2 | (hi1 << 2);
                static const float C[8] = {-0.190685f, -0.117832f, -0.065717f, -0.021460f,
                                            0.021460f,  0.065717f,  0.117832f,  0.190685f};
                K_dequant[j] = C[idx] * kblk_norm;
            }
            // Compute dot product Q_h · K_dequant
            double dot_cpu = 0.0;
            double q_l2 = 0.0, k_l2 = 0.0;
            for (int j = 0; j < 128; j++) {
                dot_cpu += (double)Q_h[j] * K_dequant[j];
                q_l2 += (double)Q_h[j] * Q_h[j];
                k_l2 += (double)K_dequant[j] * K_dequant[j];
            }
            // Read attention output for head 0 (first 128 floats of KQV)
            float out_h[128];
            CUDA_CHECK(cudaMemcpy(out_h, KQV->data, sizeof(out_h), cudaMemcpyDeviceToHost));
            fprintf(stderr, "[VEC_KQ_CPU_DIAG] K_ne=[%lld,%lld,%lld] Q_ne=[%lld,%lld,%lld] "
                    "kblk_norm=%g K_l2=%.4g Q_l2=%.4g dot_cpu=%.6g "
                    "K_dq8=(%g %g %g %g %g %g %g %g) Q8=(%g %g %g %g %g %g %g %g) "
                    "out8=(%g %g %g %g %g %g %g %g)\n",
                    (long long)K->ne[0], (long long)K->ne[1], (long long)K->ne[2],
                    (long long)Q->ne[0], (long long)Q->ne[1], (long long)Q->ne[2],
                    kblk_norm, sqrt(k_l2), sqrt(q_l2), dot_cpu,
                    K_dequant[0], K_dequant[1], K_dequant[2], K_dequant[3],
                    K_dequant[4], K_dequant[5], K_dequant[6], K_dequant[7],
                    Q_h[0], Q_h[1], Q_h[2], Q_h[3], Q_h[4], Q_h[5], Q_h[6], Q_h[7],
                    out_h[0], out_h[1], out_h[2], out_h[3],
                    out_h[4], out_h[5], out_h[6], out_h[7]);
        }
    }
#endif

    if (stream_k) {
        if (ntiles_dst % blocks_num.x != 0) { // Fixup is only needed if the SMs work on fractional tiles.
            const dim3 block_dim_combine(DV, 1, 1);
            const dim3 blocks_num_combine = {blocks_num.x, ncols1, ncols2};

            flash_attn_stream_k_fixup<DV, ncols1, ncols2>
                <<<blocks_num_combine, block_dim_combine, 0, main_stream>>>
                ((float *) KQV->data, dst_tmp_meta.ptr, Q->ne[1], Q->ne[2], Q->ne[3], K->ne[1], K->ne[2], nbatch_fa);
        }
    } else if (parallel_blocks > 1) {
        const dim3 block_dim_combine(DV, 1, 1);
        const dim3 blocks_num_combine(Q->ne[1], Q->ne[2], Q->ne[3]);
        const size_t nbytes_shared_combine = parallel_blocks*sizeof(float2);

        flash_attn_combine_results<DV>
            <<<blocks_num_combine, block_dim_combine, nbytes_shared_combine, main_stream>>>
            (dst_tmp.ptr, dst_tmp_meta.ptr, (float *) KQV->data, parallel_blocks);
    }
    CUDA_CHECK(cudaGetLastError());
}
