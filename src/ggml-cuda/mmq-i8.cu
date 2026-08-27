#include "ggml-cuda.h"
#include "mmq-i8.cuh"

#include "common.cuh"

// DP4A MMQ for the INT8 convrot tensorwise mulmat on devices without INT8 BLAS
// (RDNA2). Computes y[r*n + o] = sum_k W[o*k + t] * X[r*k + t].
// W: [n, k] row-major I8. X: [rows_padded, k] row-major I8, padded rows zeroed.
// Only r < rows is written; the dequantize epilogue never reads padded rows.
// k and n are multiples of 4 (guaranteed by the backend supports_op gate).

#define MUL_MAT_I8_BLOCK 16
#define MUL_MAT_I8_STAGE 8 // int32 packs staged per K step (32 bytes)

static __global__ void mul_mat_i8_tensorwise_dp4a(
        const char * __restrict__ w, const char * __restrict__ x, int * __restrict__ y,
        const uint32_t n, const uint32_t rows, const uint32_t k) {

    const uint32_t o  = blockIdx.x * MUL_MAT_I8_BLOCK + threadIdx.x;
    const uint32_t r  = blockIdx.y * MUL_MAT_I8_BLOCK + threadIdx.y;
    const uint32_t lx = threadIdx.x;
    const uint32_t ly = threadIdx.y;

    __shared__ int w_tile[MUL_MAT_I8_BLOCK * MUL_MAT_I8_STAGE];
    __shared__ int x_tile[MUL_MAT_I8_BLOCK * MUL_MAT_I8_STAGE];

    const uint32_t k_pack = k / 4;
    const int * wi = (const int *) w;
    const int * xi = (const int *) x;

    int acc = 0;

    for (uint32_t kb = 0; kb < k_pack; kb += MUL_MAT_I8_STAGE) {
        // out-of-range slots are zero-filled so the dot product stays exact
        if (ly < MUL_MAT_I8_STAGE) {
            w_tile[lx * MUL_MAT_I8_STAGE + ly] =
                (o < n && kb + ly < k_pack) ? wi[(size_t) o * k_pack + kb + ly] : 0;
        }
        if (lx < MUL_MAT_I8_STAGE) {
            x_tile[ly * MUL_MAT_I8_STAGE + lx] =
                (r < rows && kb + lx < k_pack) ? xi[(size_t) r * k_pack + kb + lx] : 0;
        }
        __syncthreads();

#pragma unroll
        for (uint32_t i = 0; i < MUL_MAT_I8_STAGE; ++i) {
            acc = ggml_cuda_dp4a(w_tile[lx * MUL_MAT_I8_STAGE + i], x_tile[ly * MUL_MAT_I8_STAGE + i], acc);
        }
        __syncthreads();
    }

    if (o < n && r < rows) {
        y[(size_t) r * n + o] = acc;
    }
}

void ggml_cuda_mul_mat_i8_mmq(ggml_backend_cuda_context & ctx, const int8_t * w, const int8_t * x,
        int32_t * y, int64_t n, int64_t rows, int64_t k, cudaStream_t stream) {

    GGML_UNUSED(ctx);

    const dim3 block_nums((n + MUL_MAT_I8_BLOCK - 1) / MUL_MAT_I8_BLOCK,
                          (rows + MUL_MAT_I8_BLOCK - 1) / MUL_MAT_I8_BLOCK);
    const dim3 block_dims(MUL_MAT_I8_BLOCK, MUL_MAT_I8_BLOCK, 1);

    mul_mat_i8_tensorwise_dp4a<<<block_nums, block_dims, 0, stream>>>(
        (const char *) w, (const char *) x, y, (uint32_t) n, (uint32_t) rows, (uint32_t) k);
}
