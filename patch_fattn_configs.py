import re

def patch_configs():
    file_path = "ggml/src/ggml-cuda/fattn-mma-f16.cuh"
    with open(file_path, "r") as f:
        content = f.read()

    # Volta
    content = content.replace(
        "    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 64, 256, 1,  32, 160, 128,  64, 1, false);",
        "    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 64, 256, 1,  32, 160, 128,  64, 1, false);\n\n" +
        "    GGML_CUDA_FATTN_MMA_CONFIG_CASE(640, 512,  8,  64, 4,  32, 288, 256,  64, 1, false);\n" +
        "    GGML_CUDA_FATTN_MMA_CONFIG_CASE(640, 512, 16,  64, 4,  32, 288, 256,  64, 1, false);\n" +
        "    GGML_CUDA_FATTN_MMA_CONFIG_CASE(640, 512, 32, 128, 2,  32, 160, 128,  64, 1, false);\n" +
        "    GGML_CUDA_FATTN_MMA_CONFIG_CASE(640, 512, 64, 256, 1,  32, 160, 128,  64, 1, false);"
    )

    # RDNA
    content = content.replace(
        "    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 64, 128, 2,  32, 160, 128, 128, 1, true);",
        "    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 64, 128, 2,  32, 160, 128, 128, 1, true);\n\n" +
        "    GGML_CUDA_FATTN_MMA_CONFIG_CASE(640, 512,  8, 128, 3,  64,  96,  64, 128, 1, true);\n" +
        "    GGML_CUDA_FATTN_MMA_CONFIG_CASE(640, 512, 16, 128, 3,  64,  96,  64, 128, 1, true);\n" +
        "    GGML_CUDA_FATTN_MMA_CONFIG_CASE(640, 512, 32, 128, 2,  32, 160, 128, 128, 1, true);\n" +
        "    GGML_CUDA_FATTN_MMA_CONFIG_CASE(640, 512, 64, 128, 2,  32, 160, 128, 128, 1, true);"
    )

    # CDNA
    content = content.replace(
        "    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 64, 256, 1,  64, 160, 128, 128, 1, true);",
        "    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 64, 256, 1,  64, 160, 128, 128, 1, true);\n\n" +
        "    GGML_CUDA_FATTN_MMA_CONFIG_CASE(640, 512,  8, 256, 1,  64, 128, 128, 128, 1, true);\n" +
        "    GGML_CUDA_FATTN_MMA_CONFIG_CASE(640, 512, 16, 256, 1,  64, 128, 128, 128, 1, true);\n" +
        "    GGML_CUDA_FATTN_MMA_CONFIG_CASE(640, 512, 32, 256, 1,  64, 160, 128, 128, 1, true);\n" +
        "    GGML_CUDA_FATTN_MMA_CONFIG_CASE(640, 512, 64, 256, 1,  64, 160, 128, 128, 1, true);"
    )

    with open(file_path, "w") as f:
        f.write(content)

if __name__ == "__main__":
    patch_configs()
    print("Done")
