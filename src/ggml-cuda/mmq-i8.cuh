#pragma once

#include "common.cuh"
#include "ggml-cuda.h"

void ggml_cuda_mul_mat_i8_mmq(ggml_backend_cuda_context & ctx, const int8_t * w, const int8_t * x,
        int32_t * y, int64_t n, int64_t rows, int64_t k, cudaStream_t stream);
