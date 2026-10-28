#include "common.cuh"

#define CUDA_ROPE_BLOCK_SIZE 256

void ggml_cuda_op_rope_apply(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
constexpr int GGML_CUDA_ROPE_APPLY_HEAD_SIZE       = 128;
constexpr int GGML_CUDA_ROPE_APPLY_WARPS_PER_BLOCK = 4;

void ggml_cuda_op_rms_norm_mul_rope_apply(
    ggml_backend_cuda_context & ctx, ggml_tensor * norm, ggml_tensor * mul, ggml_tensor * rope, ggml_tensor * dst);
#endif

void ggml_cuda_op_rope(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rope_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rope_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * set_rows);

void ggml_cuda_op_rms_norm_mul_rope_fused(ggml_backend_cuda_context & ctx, ggml_tensor * rms_norm, ggml_tensor * mul, ggml_tensor * rope, ggml_tensor * set_rows);
