// turbo-cuda-stubs.cpp
// Weak fallback implementations for CUDA-specific TurboQuant symbols.
//
// When libggml-cuda.so is loaded (CUDA GPU available), its strong symbol
// definitions override these weak stubs, enabling full GPU acceleration.
// Without CUDA, these stubs provide safe no-op behavior so libllama.so
// never has unresolved symbol references.
//
// GCC/Clang __attribute__((weak)) guarantees:
//   - If a strong definition exists in any linked library → use that one
//   - Otherwise → use this fallback

#ifndef INNERQ_MAX_CHANNELS
#define INNERQ_MAX_CHANNELS 128
#endif

#include <cstdint>
#include <cstring>
#include "ggml.h"
#include "ggml-backend.h"

// Dummy strong symbol to force the linker to extract this object file
// from a static library archive when building llama-server.
extern "C" {
GGML_API int ggml_turbo_cuda_stubs_dummy = 1;
}

// ── InnerQ host-side globals ────────────────────────────────────────────────
// Defined as weak so libggml-cuda.so can provide the real ones.

extern "C" {
__attribute__((weak)) GGML_API bool  g_innerq_finalized = false;
__attribute__((weak)) GGML_API float g_innerq_scale_inv_host[INNERQ_MAX_CHANNELS] = {
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
};

__attribute__((weak)) GGML_API bool turbo_innerq_needs_tensor_update(void) {
    return false;
}
__attribute__((weak)) GGML_API void turbo_innerq_mark_tensor_updated(void) {
    // no-op on CPU-only builds
}
}

// ── TriAttention GPU stubs ───────────────────────────────────────────────────
// These mirror the signatures from ggml/include/ggml-cuda.h.
// They are never called in practice on CPU-only builds because
// llama-triattention.cpp checks gpu_enabled before invoking them.

struct triattention_gpu_state;
struct triattention_gpu_head_calib;
struct triattention_gpu_config;

extern "C" {

__attribute__((weak)) GGML_BACKEND_API
triattention_gpu_state * triattention_gpu_init(
    const triattention_gpu_config *,
    const triattention_gpu_head_calib *,
    const float *, const float *, const float *, void *)
{
    return nullptr;
}

__attribute__((weak)) GGML_BACKEND_API
void triattention_gpu_score_head(
    triattention_gpu_state *, const void *, uint64_t, size_t,
    uint32_t, uint32_t,
    const uint32_t *, const int32_t *, uint32_t, int64_t, int,
    float *, void *)
{
    // no-op
}

__attribute__((weak)) GGML_BACKEND_API
void triattention_gpu_scores_to_host(
    float *, const float *, uint32_t, void *)
{
    // no-op
}

__attribute__((weak)) GGML_BACKEND_API
void triattention_gpu_upload_cells(
    uint32_t **, int32_t **,
    const uint32_t *, const int32_t *,
    uint32_t, void *)
{
    // no-op
}

__attribute__((weak)) GGML_BACKEND_API
float * triattention_gpu_alloc_scores(uint32_t, void *)
{
    return nullptr;
}

__attribute__((weak)) GGML_BACKEND_API
void triattention_gpu_free_dev(void *)
{
    // no-op
}

__attribute__((weak)) GGML_BACKEND_API
void triattention_gpu_free(triattention_gpu_state *)
{
    // no-op
}

} // extern "C"
