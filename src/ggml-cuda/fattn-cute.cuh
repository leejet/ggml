#pragma once

#include "common.cuh"

bool ggml_cuda_fattn_cute_supported(int device, const ggml_tensor* dst);
bool ggml_cuda_fattn_cute_preferred(int device, const ggml_tensor* dst);
// Optional buffers use contiguous FP16 storage; query already includes the attention scale.
bool ggml_cuda_try_fattn_cute(ggml_backend_cuda_context& ctx, ggml_tensor* dst, half* output = nullptr, const half* query = nullptr);
