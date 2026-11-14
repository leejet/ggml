#pragma once

#include "common.cuh"

struct ggml_fp8_e5m2_cuda {
    uint8_t value;

    __device__ operator float() const {
        const uint32_t sign     = value >> 7;
        const uint32_t exponent = (value >> 2) & 0x1f;
        const uint32_t mantissa = value & 0x03;

        if (exponent == 0x1f) {
            const float result = mantissa == 0 ? INFINITY : nanf("");
            return sign ? -result : result;
        }

        float result;
        if (exponent == 0) {
            result = ldexpf((float) mantissa, -16);
        } else {
            result = ldexpf(1.0f + (float) mantissa / 4.0f, (int) exponent - 15);
        }
        return sign ? -result : result;
    }
};

static_assert(sizeof(ggml_fp8_e5m2_cuda) == 1, "unexpected E5M2 storage size");

bool ggml_cuda_mul_mat_fp8(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const ggml_tensor * weight_scale = nullptr);
