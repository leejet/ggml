#include "common.cuh"

#define CUDA_CONCAT_BLOCK_SIZE 256

void ggml_cuda_op_concat(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_concat_cast_f16(ggml_backend_cuda_context & ctx, const ggml_tensor * concat,
                                  bool permuted, void * output);

void ggml_cuda_op_concat_prefix_cast_f16(ggml_backend_cuda_context & ctx, const ggml_tensor * concat,
                                         bool permuted, void * output);
