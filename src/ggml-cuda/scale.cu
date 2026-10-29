#include "scale.cuh"
#include "convert.cuh"

#define MAX_GRIDDIM_X 0x7FFFFFFF

static __global__ void scale_permute_f16(const float * src, half * dst,
        int64_t dim, int64_t tokens, int64_t heads, int64_t count,
        int64_t nb1, int64_t nb2, int64_t nb3, float scale) {
    const int64_t index = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
    if (index >= count) {
        return;
    }
    const int64_t column = index % dim;
    const int64_t row = index / dim;
    const int64_t token = row % tokens;
    const int64_t head = row / tokens % heads;
    const int64_t batch = row / (tokens * heads);
    const float4 value = *(const float4 *) (src + column + head * nb1 + token * nb2 + batch * nb3);
    half2 * output = (half2 *) (dst + index);
    output[0] = __floats2half2_rn(__fmul_rn(value.x, scale), __fmul_rn(value.y, scale));
    output[1] = __floats2half2_rn(__fmul_rn(value.z, scale), __fmul_rn(value.w, scale));
}

void ggml_cuda_op_scale_permute_f16(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst, float scale) {
    const int64_t count = ggml_nelements(dst);
    scale_permute_f16<<<(count + 1023) / 1024, 256, 0, ctx.stream()>>>(
        (const float *) src->data, (half *) dst->data, src->ne[0], src->ne[2], src->ne[1], count,
        src->nb[1] / sizeof(float), src->nb[2] / sizeof(float), src->nb[3] / sizeof(float), scale);
    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
static __global__ void scale_f32(const T * x, T * dst, const float scale, const float bias, const int64_t nelements) {
    ggml_cuda_pdl_lc();
    int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * (int64_t)gridDim.x;

    ggml_cuda_pdl_sync();
    for (int64_t i = tid; i < nelements; i += stride) {
        dst[i] = ggml_cuda_cast<T>(scale * ggml_cuda_cast<float>(x[i]) + bias);
    }
}

template <typename T>
static void scale_f32_cuda(const T * x, T * dst, const float scale, const float bias, const int64_t nelements, cudaStream_t stream) {
    const int64_t num_blocks = (nelements + CUDA_SCALE_BLOCK_SIZE - 1) / CUDA_SCALE_BLOCK_SIZE;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(MIN(MAX_GRIDDIM_X, num_blocks), CUDA_SCALE_BLOCK_SIZE, 0, stream);
    ggml_cuda_kernel_launch(scale_f32<T>, launch_params, x, dst, scale, bias, nelements);
}

void ggml_cuda_op_scale(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16);
    GGML_ASSERT(dst->type == src0->type);

    float scale;
    float bias;
    memcpy(&scale, (float *) dst->op_params + 0, sizeof(float));
    memcpy(&bias,  (float *) dst->op_params + 1, sizeof(float));

    if (src0->type == GGML_TYPE_BF16) {
        scale_f32_cuda((const nv_bfloat16 *) src0->data, (nv_bfloat16 *) dst->data, scale, bias, ggml_nelements(src0), stream);
    } else {
        scale_f32_cuda((const float *) src0->data, (float *) dst->data, scale, bias, ggml_nelements(src0), stream);
    }
}
