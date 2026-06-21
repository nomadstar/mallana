import re

def patch_mma():
    file_path = "ggml/src/ggml-cuda/fattn-mma-f16.cuh"
    with open(file_path, "r") as f:
        content = f.read()

    # The missed V_h2 load
    content = content.replace(
        """(V_h2 + int64_t(k_VKQ_0)*stride_V + i0_start/2, tile_V, i0_diff/2, stride_V, k_VKQ_sup);""",
        """(V_h2 + i0_start/2, tile_V, i0_diff/2, stride_V, k_VKQ_sup, block_table, k_VKQ_0, block_size, sequence, ne11);"""
    )

    with open(file_path, "w") as f:
        f.write(content)

if __name__ == "__main__":
    patch_mma()
    print("Done")
