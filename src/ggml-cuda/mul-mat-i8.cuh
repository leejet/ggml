#pragma once

#include "common.cuh"

bool ggml_cuda_mul_mat_i8_cutlass(
    ggml_backend_cuda_context& ctx,
    const int8_t* x,
    const int8_t* w,
    const float* x_scales,
    const float* w_scales,
    float* dst,
    int64_t rows,
    int64_t n,
    int64_t k);

bool ggml_cuda_mul_mat_i8_cutlass_residual(
    ggml_backend_cuda_context& ctx,
    const int8_t* x,
    const int8_t* w,
    const float* x_scales,
    const float* w_scales,
    float* dst,
    int64_t rows,
    int64_t n,
    int64_t k,
    const float* gate,
    const float* residual,
    float output_scale);

// dst retains rows * n storage; the fused result occupies its first rows * (n / 2) elements.
bool ggml_cuda_mul_mat_i8_swiglu_cutlass(
    ggml_backend_cuda_context& ctx,
    const int8_t* x,
    const int8_t* w,
    const float* x_scales,
    const float* w_scales,
    float* dst,
    int64_t rows,
    int64_t n,
    int64_t k);

// output retains rows * n floats of storage, including when its result is FP16.
bool ggml_cuda_mul_mat_i8_rms_rope_cutlass(
    ggml_backend_cuda_context& ctx, const int8_t* x, const int8_t* w,
    const float* x_scales, const float* w_scales, float* output, int64_t rows, int64_t n, int64_t k,
    const float* norm_weight, const float* pe, int64_t tokens, float eps, bool f16, float scale);

// workspace supplies rows * n FP32 address space for the visitor; only output is written.
bool ggml_cuda_mul_mat_i8_kv_cutlass(
    ggml_backend_cuda_context& ctx, const int8_t* x, const int8_t* w,
    const float* x_scales, const float* w_scales, float* workspace, half* output, int64_t rows, int64_t n, int64_t k,
    const float* norm_weight, const float* pe, int64_t tokens, int64_t prefix_tokens, float eps);
