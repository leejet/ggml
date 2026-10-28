/*
 * Adapted from FlashAttention v2.8.3:
 * https://github.com/Dao-AILab/flash-attention/tree/v2.8.3/csrc/flash_attn/src
 * Sources: kernel_traits.h, utils.h, softmax.h, flash_fwd_kernel.h.
 * Copyright (c) 2023-2024, Tri Dao.
 *
 * BSD 3-Clause License
 *
 * Copyright (c) 2022, the respective contributors, as shown by the AUTHORS file.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include "fattn-cute.cuh"

#ifdef GGML_CUDA_USE_CUTLASS
#include <climits>
#include <mutex>
#include "convert.cuh"
#include <cutlass/numeric_conversion.h>
#include <cute/tensor.hpp>
namespace ggml_fattn_cute {
    using namespace cute;
    template <bool A_in_regs = false, bool B_in_regs = false, typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3, typename Tensor4, typename TiledMma, typename TiledCopyA, typename TiledCopyB, typename ThrCopyA, typename ThrCopyB>
    __forceinline__ __device__ void gemm(Tensor0& acc, Tensor1& tCrA, Tensor2& tCrB, Tensor3 const& tCsA, Tensor4 const& tCsB, TiledMma tiled_mma, TiledCopyA smem_tiled_copy_A, TiledCopyB smem_tiled_copy_B, ThrCopyA smem_thr_copy_A, ThrCopyB smem_thr_copy_B) {
        CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));
        CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));
        CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));
        Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
        CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));
        Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
        CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));
        if (!A_in_regs) {
            cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{}));
        }
        if (!B_in_regs) {
            cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
        }
#pragma unroll
        for (int i = 0; i < size<2>(tCrA); ++i) {
            if (i < size<2>(tCrA) - 1) {
                if (!A_in_regs) {
                    cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1));
                }
                if (!B_in_regs) {
                    cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
                }
            }
            cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
        }
    }

    template <typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3, typename TiledMma, typename TiledCopy, typename ThrCopy>
    __forceinline__ __device__ void gemm_rs(Tensor0& acc, Tensor1& tCrA, Tensor2& tCrB, Tensor3 const& tCsB, TiledMma tiled_mma, TiledCopy smem_tiled_copy_B, ThrCopy smem_thr_copy_B) {
        CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));
        CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));
        CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));
        Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
        CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));
        cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
#pragma unroll
        for (int i = 0; i < size<2>(tCrA); ++i) {
            if (i < size<2>(tCrA) - 1) {
                cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
            }
            cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
        }
    }

    template <typename Layout>
    __forceinline__ __device__ auto convert_layout_acc_rowcol(Layout acc_layout) {
        static_assert(decltype(size<0>(acc_layout))::value == 4);
        static_assert(decltype(rank(acc_layout))::value == 3);
        auto l = logical_divide(acc_layout, Shape<_2>{});
        return make_layout(make_layout(get<0, 1>(l), get<1>(l)), make_layout(get<0, 0>(l), get<2>(l)));
    };

    template <typename MMA_traits, typename Layout>
    __forceinline__ __device__ auto convert_layout_acc_Aregs(Layout acc_layout) {
        using X = Underscore;
        static_assert(decltype(size<0>(acc_layout))::value == 4);
        static_assert(decltype(rank(acc_layout))::value == 3);
        constexpr int mma_shape_K = get<2>(typename MMA_traits::Shape_MNK{});
        static_assert(mma_shape_K == 8 || mma_shape_K == 16);
        if constexpr (mma_shape_K == 8) {
            return acc_layout;
        } else {
            auto l = logical_divide(acc_layout, Shape<X, X, _2>{});
            return make_layout(make_layout(get<0>(l), get<2, 0>(l)), get<1>(l), get<2, 1>(l));
        }
    };

    template <typename TensorT>
    __device__ __forceinline__ auto to_half(const TensorT& src) {
        constexpr int count = decltype(size(src))::value;
        auto dst            = make_tensor<cutlass::half_t>(src.layout());
        cutlass::NumericArrayConverter<cutlass::half_t, float, count> convert;
        *reinterpret_cast<cutlass::Array<cutlass::half_t, count>*>(dst.data()) =
            convert(*reinterpret_cast<const cutlass::Array<float, count>*>(src.data()));
        return dst;
    }

    template <bool Full, bool Clear, bool FullCols = true, typename Copy, typename Src, typename Dst, typename Id>
    __device__ __forceinline__ void copy_rows(Copy copier, const Src& src, Dst& dst, const Id& id, int rows, int cols = 0) {
#pragma unroll
        for (int m = 0; m < size<1>(src); ++m) {
#pragma unroll
            for (int k = 0; k < size<2>(src); ++k) {
                if ((Full || get<0>(id(0, m, k)) < rows) && (FullCols || get<1>(id(0, m, k)) < cols)) {
                    cute::copy(copier, src(_, m, k), dst(_, m, k));
                } else if constexpr (Clear) {
                    cute::clear(dst(_, m, k));
                }
            }
        }
    }

    template <int Rows>
    struct Softmax {
        decltype(make_tensor<float>(Shape<Int<Rows>>{})) maximum, sum;
        template <bool First, typename Scores, typename Output>
        __device__ __forceinline__ void update(Scores& acc_s, Output& acc_o) {
            auto scores = make_tensor(acc_s.data(), convert_layout_acc_rowcol(acc_s.layout()));
            auto output = make_tensor(acc_o.data(), convert_layout_acc_rowcol(acc_o.layout()));
#pragma unroll
            for (int m = 0; m < Rows; ++m) {
                const float old_max = First ? -INFINITY : maximum(m);
                float next_max      = First ? scores(m, 0) : fmaxf(old_max, scores(m, 0));
#pragma unroll
                for (int n = 1; n < size<1>(scores); ++n)
                    next_max = fmaxf(next_max, scores(m, n));
                next_max      = fmaxf(next_max, __shfl_xor_sync(0xffffffff, next_max, 2));
                next_max      = fmaxf(next_max, __shfl_xor_sync(0xffffffff, next_max, 1));
                maximum(m)    = next_max;
                float row_sum = 0.f;
                if constexpr (!First) {
                    row_sum = sum(m);
                    if (old_max != next_max) {
                        const float factor = exp2f((old_max - next_max) * 1.4426950408889634f);
                        row_sum *= factor;
#pragma unroll
                        for (int n = 0; n < size<1>(output); ++n)
                            output(m, n) *= factor;
                    }
                }
#pragma unroll
                for (int n = 0; n < size<1>(scores); ++n) {
                    const float value = exp2f(scores(m, n) * 1.4426950408889634f - next_max * 1.4426950408889634f);
                    scores(m, n)      = value;
                    row_sum += value;
                }
                // Defer the four-lane sum reduction until the final output normalization.
                sum(m) = row_sum;
            }
        }
        template <typename Output>
        __device__ __forceinline__ void normalize(Output& acc_o) {
            auto output = make_tensor(acc_o.data(), convert_layout_acc_rowcol(acc_o.layout()));
#pragma unroll
            for (int m = 0; m < Rows; ++m) {
                float total = sum(m);
                total += __shfl_xor_sync(0xffffffff, total, 2);
                total += __shfl_xor_sync(0xffffffff, total, 1);
                const float inverse = 1.f / total;
#pragma unroll
                for (int n = 0; n < size<1>(output); ++n)
                    output(m, n) *= inverse;
            }
        }
    };

    template <int BM, int BN, int D = 128, int Warps = 4>
    struct Traits {
        using Element                    = cutlass::half_t;
        static constexpr int Threads     = Warps * 32;
        using Mma                        = TiledMMA<MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
                             Layout<Shape<Int<Warps>, _1, _1>>,
                             Tile<Int<16 * Warps>, _16, _16>>;
        static constexpr int SmemWidth   = D % 64 == 0 ? 64 : D % 32 == 0 ? 32 : 16;
        static constexpr int SwizzleBits = SmemWidth == 64 ? 3 : SmemWidth == 32 ? 2 : 1;
        using SmemAtom                   = decltype(composition(Swizzle<SwizzleBits, 3, 3>{},
            Layout<Shape<_8, Int<SmemWidth>>, Stride<Int<SmemWidth>, _1>>{}));
        using QLayout                    = decltype(tile_to_shape(SmemAtom{}, Shape<Int<BM>, Int<D>>{}));
        using KVLayout                   = decltype(tile_to_shape(SmemAtom{}, Shape<Int<BN>, Int<D>>{}));
        using VtLayout                   = decltype(composition(KVLayout{}, make_layout(Shape<Int<D>, Int<BN>>{}, GenRowMajor{})));
        using CopyAtom                   = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
        using CopyAtomT                  = Copy_Atom<SM75_U16x8_LDSM_T, Element>;
        using GlobalLayout               = Layout<Shape<Int<Threads / (SmemWidth / 8)>, Int<SmemWidth / 8>>, Stride<Int<SmemWidth / 8>, _1>>;
        using GlobalCopy                 = decltype(make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, Element>{},
                                                                    GlobalLayout{},
                                                                    Layout<Shape<_1, _8>>{}));
        using OutputCopy                 = decltype(make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
                                                                    GlobalLayout{},
                                                                    Layout<Shape<_1, _8>>{}));
        static constexpr int SharedBytes = (BM + 2 * BN) * D * sizeof(Element);
    };

    struct Params {
        const cutlass::half_t *q, *k, *v;
        cutlass::half_t* o;
        int nq, nk, heads, kv_heads, dim;
    };

    template <int BM, int BN, int D = 128, bool FullDim = true, int Warps = 4>
    __global__ __launch_bounds__(Warps * 32) void kernel(Params p) {
        using T       = Traits<BM, BN, D, Warps>;
        using Element = typename T::Element;
        extern __shared__ char storage[];
        const int tid = threadIdx.x, m_block = blockIdx.x, batch = blockIdx.y, head = blockIdx.z;
        const int64_t q_offset  = (int64_t(batch) * p.heads + head) * p.nq * p.dim;
        const int64_t kv_offset = (int64_t(batch) * p.kv_heads + head / (p.heads / p.kv_heads)) * p.nk * p.dim;
        auto mQ                 = make_tensor(make_gmem_ptr(p.q + q_offset), make_shape(p.nq, p.dim), make_stride(p.dim, _1{}));
        auto mK                 = make_tensor(make_gmem_ptr(p.k + kv_offset), make_shape(p.nk, p.dim), make_stride(p.dim, _1{}));
        auto mV                 = make_tensor(make_gmem_ptr(p.v + kv_offset), make_shape(p.nk, p.dim), make_stride(p.dim, _1{}));
        auto gQ                 = local_tile(mQ, Shape<Int<BM>, Int<D>>{}, make_coord(m_block, 0));
        auto gK                 = local_tile(mK, Shape<Int<BN>, Int<D>>{}, make_coord(_, 0));
        auto gV                 = local_tile(mV, Shape<Int<BN>, Int<D>>{}, make_coord(_, 0));
        auto sQ                 = make_tensor(make_smem_ptr(reinterpret_cast<Element*>(storage)), typename T::QLayout{});
        auto sK                 = make_tensor(sQ.data() + size(sQ), typename T::KVLayout{});
        auto sV                 = make_tensor(sK.data() + size(sK), typename T::KVLayout{});
        auto sVt                = make_tensor(sV.data(), typename T::VtLayout{});
        auto sVplain            = make_tensor(sV.data().get(), get_nonswizzle_portion(typename T::VtLayout{}));
        typename T::GlobalCopy global_copy;
        auto thread_copy = global_copy.get_thread_slice(tid);
        auto tQgQ        = thread_copy.partition_S(gQ);
        auto tQsQ        = thread_copy.partition_D(sQ);
        auto tKgK        = thread_copy.partition_S(gK);
        auto tKsK        = thread_copy.partition_D(sK);
        auto tVgV        = thread_copy.partition_S(gV);
        auto tVsV        = thread_copy.partition_D(sV);
        auto tQcQ        = thread_copy.partition_S(make_identity_tensor(Shape<Int<BM>, Int<D>>{}));
        auto tKVcKV      = thread_copy.partition_S(make_identity_tensor(Shape<Int<BN>, Int<D>>{}));
        typename T::Mma mma;
        auto thread_mma = mma.get_thread_slice(tid);
        auto rQ         = thread_mma.partition_fragment_A(sQ);
        auto rK         = thread_mma.partition_fragment_B(sK);
        auto rV         = thread_mma.partition_fragment_B(sVplain);
        auto acc_o      = partition_fragment_C(mma, Shape<Int<BM>, Int<D>>{});
        auto copyQ      = make_tiled_copy_A(typename T::CopyAtom{}, mma);
        auto copyK      = make_tiled_copy_B(typename T::CopyAtom{}, mma);
        auto copyV      = make_tiled_copy_B(typename T::CopyAtomT{}, mma);
        auto threadQ    = copyQ.get_thread_slice(tid);
        auto threadK    = copyK.get_thread_slice(tid);
        auto threadV    = copyV.get_thread_slice(tid);
        auto tSsQ       = threadQ.partition_S(sQ);
        auto tSsK       = threadK.partition_S(sK);
        auto tOsV       = threadV.partition_S(sVt);
        auto score_id   = thread_mma.partition_C(make_identity_tensor(Shape<Int<BM>, Int<BN>>{}));
        // Process the partial K/V tile first; subsequent tiles need no score masking.
        int block = (p.nk - 1) / BN;
        copy_rows<false, true, FullDim>(global_copy, tQgQ, tQsQ, tQcQ, p.nq - m_block * BM, p.dim);
        auto last_k = tKgK(_, _, _, block);
        copy_rows<false, true, FullDim>(global_copy, last_k, tKsK, tKVcKV, p.nk - block * BN, p.dim);
        cute::cp_async_fence();
        clear(acc_o);
        Softmax<2 * size<1>(acc_o)> softmax;
        auto step = [&](auto first_flag, int n) {
            constexpr bool First = decltype(first_flag)::value;
            auto acc_s           = partition_fragment_C(mma, Shape<Int<BM>, Int<BN>>{});
            clear(acc_s);
            cute::cp_async_wait<0>();
            __syncthreads();
            auto v_tile = tVgV(_, _, _, n);
            copy_rows<!First, First, FullDim>(global_copy, v_tile, tVsV, tKVcKV, p.nk - n * BN, p.dim);
            cute::cp_async_fence();
            gemm(acc_s, rQ, rK, tSsQ, tSsK, mma, copyQ, copyK, threadQ, threadK);
            if constexpr (First) {
#pragma unroll
                for (int i = 0; i < size(acc_s); ++i) {
                    if (get<1>(score_id(i)) >= p.nk - n * BN)
                        acc_s(i) = -INFINITY;
                }
            }
            cute::cp_async_wait<0>();
            __syncthreads();
            // Do not commit an empty async group: the next iteration waits for this K tile.
            if (n > 0) {
                auto next_k = tKgK(_, _, _, n - 1);
                copy_rows<true, false, FullDim>(global_copy, next_k, tKsK, tKVcKV, 0, p.dim);
                cute::cp_async_fence();
            }
            softmax.template update<First>(acc_s, acc_o);
            auto rP   = to_half(acc_s);
            auto tOrP = make_tensor(rP.data(), convert_layout_acc_Aregs<typename T::Mma>(rP.layout()));
            gemm_rs(acc_o, tOrP, rV, tOsV, mma, copyV, threadV);
        };
        step(std::true_type{}, block);
        for (--block; block >= 0; --block)
            step(std::false_type{}, block);
        softmax.normalize(acc_o);
        auto rO      = to_half(acc_o);
        auto sO      = make_tensor(sQ.data(), typename T::QLayout{});
        auto copyO   = make_tiled_copy_C(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{}, mma);
        auto threadO = copyO.get_thread_slice(tid);
        auto tOrO    = threadO.retile_S(rO);
        auto tOsO    = threadO.partition_D(sO);
        cute::copy(copyO, tOrO, tOsO);
        auto mO = make_tensor(make_gmem_ptr(p.o + int64_t(batch) * p.nq * p.heads * p.dim + int64_t(head) * p.dim),
                              make_shape(p.nq, p.dim), make_stride(p.heads * p.dim, _1{}));
        auto gO = local_tile(mO, Shape<Int<BM>, Int<D>>{}, make_coord(m_block, 0));
        typename T::OutputCopy output_copy;
        auto global_threadO = output_copy.get_thread_slice(tid);
        auto from_sO        = global_threadO.partition_S(sO);
        auto to_gO          = global_threadO.partition_D(gO);
        auto output_id      = global_threadO.partition_D(make_identity_tensor(Shape<Int<BM>, Int<D>>{}));
        __syncthreads();
        auto output_regs = make_tensor<Element>(shape(to_gO));
        cute::copy(output_copy, from_sO, output_regs);
        copy_rows<false, false, FullDim>(output_copy, output_regs, to_gO, output_id, p.nq - m_block * BM, p.dim);
    }
}

template <bool contiguous>
static __global__ void fattn_cute_scale_q(const float* x, half* dst, int64_t count, float scale, int64_t dim, int64_t nq, int64_t heads,
        size_t nb1, size_t nb2, size_t nb3) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        const int64_t row = i / dim;
        const float value = contiguous ? x[i] : x[(row % nq) * nb1 + ((row / nq) % heads) * nb2 + (row / (nq * heads)) * nb3 + i % dim];
        dst[i] = __float2half_rn(value * scale);
    }
}

static __global__ void fattn_cute_copy_half(
    const char* src,
    half* dst,
    int64_t count,
    int64_t ne0,
    int64_t ne1,
    int64_t ne2,
    size_t nb1,
    size_t nb2,
    size_t nb3) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        const int64_t row = i / ne0;
        dst[i]            = ((const half*)(src + (row % ne1) * nb1 + ((row / ne1) % ne2) * nb2 +
                                (row / (ne1 * ne2)) * nb3))[i % ne0];
    }
}

static int64_t fattn_cute_padded_dim(int64_t dim) {
    return dim == 80 || dim == 96 || dim == 112 ? dim : GGML_PAD(dim, 64);
}

static size_t fattn_cute_shared_bytes(int64_t dim) {
    return (64 + 2 * 64) * fattn_cute_padded_dim(dim) * sizeof(half);
}

template <int D, bool FullDim>
static void fattn_cute_launch(ggml_backend_cuda_context& ctx, const ggml_fattn_cute::Params& p, int64_t batches) {
    auto kernel = ggml_fattn_cute::kernel<64, 64, D, FullDim>;
    constexpr int shared_bytes = ggml_fattn_cute::Traits<64, 64, D>::SharedBytes;
    static std::once_flag initialized[GGML_CUDA_MAX_DEVICES];
    std::call_once(initialized[ctx.device], [&] {
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
    });
    const dim3 grid((p.nq + 63) / 64, batches, p.heads);
    kernel<<<grid, 128, shared_bytes, ctx.stream()>>>(p);
}

template <int D>
static void fattn_cute_launch(ggml_backend_cuda_context& ctx, const ggml_fattn_cute::Params& p, int64_t batches) {
    if (p.dim == D) {
        fattn_cute_launch<D, true>(ctx, p, batches);
    } else {
        fattn_cute_launch<D, false>(ctx, p, batches);
    }
}

bool ggml_cuda_fattn_cute_supported(int device, const ggml_tensor* dst) {
    static const bool disabled = std::getenv("GGML_CUDA_DISABLE_CUTE_FATTN") != nullptr ||
                                 std::getenv("GGML_CUDA_DISABLE_CUTLASS_FATTN") != nullptr;
    const ggml_tensor* q       = dst->src[0];
    const ggml_tensor* k       = dst->src[1];
    const ggml_tensor* v       = dst->src[2];
    if (disabled || !ampere_mma_available(ggml_cuda_info().devices[device].cc) ||
        fattn_cute_shared_bytes(q->ne[0]) > ggml_cuda_info().devices[device].smpbo ||
        q->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || q->ne[0] < 32 || q->ne[0] > 256 || q->ne[0] % 8 != 0 ||
        q->ne[1] < 1 || k->ne[1] < 1 || dst->src[3] != nullptr || dst->src[4] != nullptr ||
        ggml_get_op_params_f32(dst, 1) != 0.0f || ggml_get_op_params_f32(dst, 2) != 0.0f ||
        k->ne[2] < 1 || q->ne[2] % k->ne[2] != 0 || q->ne[3] != k->ne[3] || !ggml_are_same_shape(k, v) ||
        q->nb[0] != sizeof(float) || !ggml_is_contiguous(dst) ||
        (k->type != GGML_TYPE_F16 && k->type != GGML_TYPE_F32) ||
        (v->type != GGML_TYPE_F16 && v->type != GGML_TYPE_F32) ||
        k->nb[0] != ggml_type_size(k->type) || v->nb[0] != ggml_type_size(v->type)) {
        return false;
    }

    // Bound grid dimensions and 32-bit indexing inside the specialized kernel.
    return q->ne[1] <= INT_MAX / q->ne[0] && k->ne[1] <= INT_MAX / q->ne[0] &&
           q->ne[2] <= 65535 && q->ne[3] <= 65535 && k->ne[0] == q->ne[0] &&
           ggml_nelements(q) <= INT_MAX && ggml_nelements(k) <= INT_MAX;
}

bool ggml_cuda_try_fattn_cute(ggml_backend_cuda_context& ctx, ggml_tensor* dst, half* output, const half* query) {
    if (!ggml_cuda_fattn_cute_supported(ctx.device, dst)) {
        return false;
    }
    const ggml_tensor* q = dst->src[0];
    const ggml_tensor* k = dst->src[1];
    const ggml_tensor* v = dst->src[2];
    ggml_cuda_pool_alloc<half> q_half(ctx.pool());
    if (query == nullptr) {
        q_half.alloc(ggml_nelements(q));
    }
    ggml_cuda_pool_alloc<half> k_half(ctx.pool());
    ggml_cuda_pool_alloc<half> v_half(ctx.pool());
    ggml_cuda_pool_alloc<half> o_half(ctx.pool());
    auto as_half = [&](const ggml_tensor* src, ggml_cuda_pool_alloc<half>& storage) -> const half* {
        if (src->type == GGML_TYPE_F16 && ggml_is_contiguous(src) && uintptr_t(src->data) % 16 == 0) {
            return (const half*)src->data;
        }
        half* data = storage.alloc(ggml_nelements(src));
        if (src->type == GGML_TYPE_F16) {
            fattn_cute_copy_half<<<(ggml_nelements(src) + 255) / 256, 256, 0, ctx.stream()>>>(
                (const char*)src->data, data, ggml_nelements(src), src->ne[0], src->ne[1], src->ne[2],
                src->nb[1], src->nb[2], src->nb[3]);
            CUDA_CHECK(cudaGetLastError());
            return data;
        }
        const size_t size = ggml_type_size(src->type);
        ggml_get_to_fp16_nc_cuda(src->type)(src->data, data, src->ne[0], src->ne[1], src->ne[2], src->ne[3],
                                            src->nb[1] / size, src->nb[2] / size, src->nb[3] / size, ctx.stream());
        return data;
    };
    const half* k_data = as_half(k, k_half);
    const half* v_data = as_half(v, v_half);
    half* o_data       = output != nullptr ? output : o_half.alloc(ggml_nelements(dst));
    if (query == nullptr) {
        auto scale_q = ggml_is_contiguous(q) ? fattn_cute_scale_q<true> : fattn_cute_scale_q<false>;
        scale_q<<<(ggml_nelements(q) + 255) / 256, 256, 0, ctx.stream()>>>(
            (const float*)q->data, q_half.get(), ggml_nelements(q), ggml_get_op_params_f32(dst, 0),
            q->ne[0], q->ne[1], q->ne[2], q->nb[1] / sizeof(float), q->nb[2] / sizeof(float), q->nb[3] / sizeof(float));
        CUDA_CHECK(cudaGetLastError());
        query = q_half.get();
    }
    ggml_fattn_cute::Params p = {
        (const cutlass::half_t*)query, (const cutlass::half_t*)k_data, (const cutlass::half_t*)v_data,
        (cutlass::half_t*)o_data, int(q->ne[1]), int(k->ne[1]), int(q->ne[2]), int(k->ne[2]), int(q->ne[0])};
    switch (fattn_cute_padded_dim(p.dim)) {
        case 64:  fattn_cute_launch<64>(ctx, p, q->ne[3]); break;
        case 80:  fattn_cute_launch<80, true>(ctx, p, q->ne[3]); break;
        case 96:  fattn_cute_launch<96, true>(ctx, p, q->ne[3]); break;
        case 112: fattn_cute_launch<112, true>(ctx, p, q->ne[3]); break;
        case 128: fattn_cute_launch<128>(ctx, p, q->ne[3]); break;
        case 192: fattn_cute_launch<192>(ctx, p, q->ne[3]); break;
        case 256: fattn_cute_launch<256>(ctx, p, q->ne[3]); break;
        default: GGML_ABORT("unsupported CuTe attention dimension");
    }
    CUDA_CHECK(cudaGetLastError());
    if (output == nullptr) {
        ggml_get_to_fp32_cuda(GGML_TYPE_F16)(o_data, (float*)dst->data, ggml_nelements(dst), ctx.stream());
    }
    return true;
}
#endif
