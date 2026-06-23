#pragma once

#include "common.cuh"

void ggml_cuda_gather_paged_v(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
