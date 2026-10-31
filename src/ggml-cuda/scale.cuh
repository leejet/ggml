#include "common.cuh"

#define CUDA_SCALE_BLOCK_SIZE 256

void ggml_cuda_op_scale(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
void ggml_cuda_op_scale_permute_f16(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst, float scale);
#endif
