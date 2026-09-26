#pragma once

#include "common.cuh"
#include "ggml-cuda.h"

// packed-w4 convrot kernels (see ggml_mul_mat_w4_convrot); w rows are nibble
// pairs, k = 2 * packed_k. W4A4 accumulates int32 (epilogue by the caller),
// W4A8 writes y in-kernel (act scale + bias applied there).
void ggml_cuda_mul_mat_w4a4_convrot_dp4a(ggml_backend_cuda_context & ctx, const int8_t * w, const int8_t * x,
        int32_t * y, int64_t n, int64_t rows, int64_t packed_k, cudaStream_t stream);
void ggml_cuda_mul_mat_w4a8_convrot_dp4a(ggml_backend_cuda_context & ctx, const int8_t * w, const int8_t * x,
        const float * codebook, const float * s_channel, const uint8_t * s_rel, const float * act_scales,
        const float * bias, float * y, int64_t n, int64_t rows, int64_t k, cudaStream_t stream);
