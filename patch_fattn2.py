import os
import re

def patch_mma():
    file_path = "ggml/src/ggml-cuda/fattn-mma-f16.cuh"
    with open(file_path, "r") as f:
        content = f.read()

    # flash_attn_ext_f16_iter calls with k_VKQ_sup parameter as last arg
    # there are 4 such calls in process_tile
    content = re.sub(
        r"KQ_max, KQ_rowsum, jt, kb0, k_VKQ_sup\);",
        r"KQ_max, KQ_rowsum, jt, kb0, k_VKQ_sup, block_table, block_size, sequence, ne11);",
        content
    )

    with open(file_path, "w") as f:
        f.write(content)

if __name__ == "__main__":
    patch_mma()
    print("Done")
