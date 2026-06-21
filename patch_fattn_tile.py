import re

def patch_tile():
    file_path = "ggml/src/ggml-cuda/fattn-tile.cuh"
    with open(file_path, "r") as f:
        content = f.read()

    # Update overload 1 signature
    old_sig_1 = """static __device__ __forceinline__ void flash_attn_tile_load_tile(
        const half2 * const __restrict__ KV, half2 * const __restrict__ tile_KV, const int stride_KV, const int i_sup) {"""
    new_sig_1 = """static __device__ __forceinline__ void flash_attn_tile_load_tile(
        const half2 * const __restrict__ KV, half2 * const __restrict__ tile_KV, const int stride_KV, const int i_sup,
        const char * block_table, const int k_VKQ_0, const int block_size, const int sequence, const int ne11) {"""
    content = content.replace(old_sig_1, new_sig_1)

    # Update overload 1 memory access
    old_acc_1 = """                    const __align__(16) half2 zero[cpy_ne] = {{0.0f, 0.0f}};
                    ggml_cuda_memcpy_1<cpy_nb>(
                        tile_KV + i*(J/2 + J_padding) + j,
                        !oob_check || i < i_sup ? KV + i*stride_KV + j : zero);"""
    new_acc_1 = """                    const __align__(16) half2 zero[cpy_ne] = {{0.0f, 0.0f}};
                    int64_t physical_i = get_physical_token_idx(block_table, k_VKQ_0 + i, block_size, sequence, ne11);
                    ggml_cuda_memcpy_1<cpy_nb>(
                        tile_KV + i*(J/2 + J_padding) + j,
                        !oob_check || i < i_sup ? KV + physical_i*stride_KV + j : zero);"""
    content = content.replace(old_acc_1, new_acc_1)

    # Update overload 2 memory access
    # Overload 2 has a very complex physical_i manual calculation which can be replaced by get_physical_token_idx
    old_acc_2 = """                    int64_t physical_i = i + k_VKQ_0;
                    if (block_table && (!oob_check || i < i_sup)) {
                        const int32_t * bt = (const int32_t *) block_table;
                        const int max_blocks = (ne11 + block_size - 1) / block_size;
                        const int logical_idx = k_VKQ_0 + i;
                        const int physical_block = bt[sequence * max_blocks + logical_idx / block_size];
                        physical_i = (physical_block >= 0) ? physical_block * block_size + (logical_idx % block_size) : logical_idx;
                    } else if (!block_table) {
                        physical_i = k_VKQ_0 + i;
                    }"""
    new_acc_2 = """                    int64_t physical_i = get_physical_token_idx(block_table, k_VKQ_0 + i, block_size, sequence, ne11);"""
    content = content.replace(old_acc_2, new_acc_2)

    with open(file_path, "w") as f:
        f.write(content)

if __name__ == "__main__":
    patch_tile()
    print("Done")
