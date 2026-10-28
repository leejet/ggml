#include "mul-mat-i8.cuh"

#ifdef GGML_CUDA_USE_CUTLASS
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/default_gemm_universal_with_visitor.h>
#include <cutlass/epilogue/threadblock/fusion/visitors.hpp>

template <int M, int N, int WM>
struct ggml_cuda_i8_tile_config {
    using Tile = cutlass::gemm::GemmShape<M, N, 64>;
    using Warp = cutlass::gemm::GemmShape<WM, 64, 64>;
};

enum ggml_cuda_i8_tile {
    GGML_CUDA_I8_TILE_128_256,
    GGML_CUDA_I8_TILE_128_128,
    GGML_CUDA_I8_TILE_64_128,
    GGML_CUDA_I8_TILE_16_128,
};

static ggml_cuda_i8_tile ggml_cuda_mul_mat_i8_get_tile(int64_t rows, int64_t n, int nsm) {
    const int64_t columns = (n + 127) / 128;
    if (((rows + 15) / 16) * columns <= nsm) {
        return GGML_CUDA_I8_TILE_16_128;
    }
    if (((rows + 63) / 64) * columns <= nsm) {
        return GGML_CUDA_I8_TILE_64_128;
    }
    const int64_t wide_tiles = ((rows + 127) / 128) * ((n + 255) / 256);
    const int64_t square_tiles = ((rows + 127) / 128) * columns;
    // A narrower tile costs more per output element, but can waste less of the last wave.
    const double wide_cost = double((wide_tiles + nsm - 1) / nsm) / wide_tiles;
    const double square_cost = 1.08 * double((square_tiles + nsm - 1) / nsm) / square_tiles;
    return wide_cost <= square_cost ? GGML_CUDA_I8_TILE_128_256 : GGML_CUDA_I8_TILE_128_128;
}

template <class Launch>
static bool ggml_cuda_mul_mat_i8_dispatch(ggml_backend_cuda_context& ctx, int64_t rows, int64_t n, Launch launch) {
    const ggml_cuda_i8_tile tile = ggml_cuda_mul_mat_i8_get_tile(rows, n, ggml_cuda_info().devices[ctx.device].nsm);
    auto launch_tile = [&](ggml_cuda_i8_tile tile) {
        switch (tile) {
            case GGML_CUDA_I8_TILE_128_256: return launch(ggml_cuda_i8_tile_config<128, 256, 64>{});
            case GGML_CUDA_I8_TILE_128_128: return launch(ggml_cuda_i8_tile_config<128, 128, 64>{});
            case GGML_CUDA_I8_TILE_64_128:  return launch(ggml_cuda_i8_tile_config< 64, 128, 32>{});
            case GGML_CUDA_I8_TILE_16_128:  return launch(ggml_cuda_i8_tile_config< 16, 128, 16>{});
            default: GGML_ABORT("invalid INT8 GEMM tile");
        }
    };
    if (launch_tile(tile)) {
        return true;
    }
    // A tile can exceed a device's shared-memory limit even when its ISA is supported.
    for (ggml_cuda_i8_tile fallback : {GGML_CUDA_I8_TILE_128_256, GGML_CUDA_I8_TILE_128_128,
                                      GGML_CUDA_I8_TILE_64_128, GGML_CUDA_I8_TILE_16_128}) {
        if (fallback != tile && launch_tile(fallback)) {
            return true;
        }
    }
    return false;
}

template <class T>
struct ggml_cuda_i8_mul_round {
    CUTLASS_DEVICE T operator()(const T& a, const T& b) const {
        T result;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < T::kElements; ++i) {
            result[i] = __fmul_rn(a[i], b[i]);
        }
        return result;
    }
};

template <class T>
struct ggml_cuda_i8_add_round {
    CUTLASS_DEVICE T operator()(const T& a, const T& b) const {
        T result;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < T::kElements; ++i) {
            result[i] = __fadd_rn(a[i], b[i]);
        }
        return result;
    }
};

template <bool gated, bool scaled, class Config>
static bool ggml_cuda_mul_mat_i8_cutlass_impl(
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
    float output_scale) {
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!ampere_mma_available(cc) ||
        rows < 128 || rows > INT_MAX || n < 1 || n > INT_MAX || k < 1 || k > INT_MAX ||
        n % 128 != 0 || k % 64 != 0 || w_scales == nullptr ||
        ((uintptr_t)x | (uintptr_t)w | (uintptr_t)x_scales | (uintptr_t)w_scales | (uintptr_t)dst | (uintptr_t)gate | (uintptr_t)residual) % 16 != 0) {
        return false;
    }

    using Tile         = typename Config::Tile;
    using Warp         = typename Config::Warp;
    using Map          = cutlass::epilogue::threadblock::OutputTileThreadLayout<Tile, Warp, float, 4, 1>;
    using Col          = cutlass::epilogue::threadblock::VisitorColBroadcast<Map, float, cute::Stride<cute::_1, cute::_0, int>>;
    using Row          = cutlass::epilogue::threadblock::VisitorRowBroadcast<Map, float, cute::Stride<cute::_0, cute::_1, int>>;
    using Mul          = cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
    using ScaleX       = cutlass::epilogue::threadblock::Sm80EVT<Mul, cutlass::epilogue::threadblock::VisitorAccFetch, Col>;
    using ScaleW       = cutlass::epilogue::threadblock::Sm80EVT<Mul, ScaleX, Row>;
    using Residual     = cutlass::epilogue::threadblock::VisitorAuxLoad<Map, float, cute::Stride<int64_t, cute::_1, int64_t>>;
    using GateMul      = cutlass::epilogue::threadblock::VisitorCompute<ggml_cuda_i8_mul_round, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
    using Add          = cutlass::epilogue::threadblock::VisitorCompute<ggml_cuda_i8_add_round, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
    using Scalar       = cutlass::epilogue::threadblock::VisitorScalarBroadcast<float>;
    using ScaleOut     = cutlass::epilogue::threadblock::Sm80EVT<GateMul, ScaleW, Scalar>;
    using Scaled       = std::conditional_t<scaled, ScaleOut, ScaleW>;
    using Gated        = cutlass::epilogue::threadblock::Sm80EVT<GateMul, Scaled, Row>;
    using WithResidual = cutlass::epilogue::threadblock::Sm80EVT<Add, Gated, Residual>;
    using Output       = std::conditional_t<gated, WithResidual, ScaleW>;
    using Store        = cutlass::epilogue::threadblock::VisitorAuxStore<Map, float, cutlass::FloatRoundStyle::round_to_nearest,
                                                                  cute::Stride<int64_t, cute::_1, int64_t>>;
    using Epilogue     = cutlass::epilogue::threadblock::Sm80EVT<Store, Output>;
    using Kernel       = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
        int8_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 16,
        int8_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 16,
        float, cutlass::layout::RowMajor, 4, int, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        Tile, Warp, cutlass::gemm::GemmShape<16, 8, 32>, Epilogue,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4, cutlass::arch::OpMultiplyAddSaturate>::GemmKernel;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

    typename ScaleW::Arguments scales = {{{}, {x_scales, 0.0f, {}}, {}}, {w_scales, 0.0f, {}}, {}};
    typename Epilogue::Arguments epilogue;
    if constexpr (gated) {
        typename Scaled::Arguments scaled_args;
        if constexpr (scaled) {
            scaled_args = {scales, {{output_scale}}, {}};
        } else {
            scaled_args = scales;
        }
        epilogue = {{{scaled_args, {gate, 0.0f, {}}, {}}, {const_cast<float*>(residual), 0.0f, {n, cute::_1{}, rows * n}}, {}},
                    {dst, {n, cute::_1{}, rows * n}}};
    } else {
        epilogue = {scales, {dst, {n, cute::_1{}, rows * n}}};
    }
    typename Gemm::Arguments args(cutlass::gemm::GemmUniversalMode::kGemm, {(int)rows, (int)n, (int)k}, 1, epilogue,
                                  x, w, nullptr, nullptr, rows * k, n * k, 0, 0, k, k, 0, 0);
    Gemm gemm;
    if (sizeof(typename Kernel::SharedStorage) > ggml_cuda_info().devices[ctx.device].smpbo ||
        gemm.can_implement(args) != cutlass::Status::kSuccess) {
        return false;
    }
    GGML_ASSERT(gemm.initialize(args, nullptr, ctx.stream()) == cutlass::Status::kSuccess);
    GGML_ASSERT(gemm(ctx.stream()) == cutlass::Status::kSuccess);
    return true;
}
bool ggml_cuda_mul_mat_i8_cutlass(
    ggml_backend_cuda_context& ctx,
    const int8_t* x,
    const int8_t* w,
    const float* x_scales,
    const float* w_scales,
    float* dst,
    int64_t rows,
    int64_t n,
    int64_t k) {
    if (rows < 128 || n <= 0 || rows > INT_MAX || n > INT_MAX) {
        return false;
    }
    return ggml_cuda_mul_mat_i8_dispatch(ctx, rows, n, [&](auto config) {
        return ggml_cuda_mul_mat_i8_cutlass_impl<false, false, decltype(config)>(ctx, x, w, x_scales, w_scales, dst, rows, n, k, nullptr, nullptr, 1.0f);
    });
}

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
    float output_scale) {
    if (rows < 128 || n <= 0 || rows > INT_MAX || n > INT_MAX) {
        return false;
    }
    return ggml_cuda_mul_mat_i8_dispatch(ctx, rows, n, [&](auto config) {
        if (output_scale != 1.0f) {
            return ggml_cuda_mul_mat_i8_cutlass_impl<true, true, decltype(config)>(ctx, x, w, x_scales, w_scales, dst, rows, n, k, gate, residual, output_scale);
        }
        return ggml_cuda_mul_mat_i8_cutlass_impl<true, false, decltype(config)>(ctx, x, w, x_scales, w_scales, dst, rows, n, k, gate, residual, output_scale);
    });
}
// The iterator pairs gate/up columns while reading the original checkpoint layout.
struct ggml_cuda_i8_swiglu_weight_layout : cutlass::layout::PermuteBase {
    int stride, half_n;
    CUTLASS_HOST_DEVICE ggml_cuda_i8_swiglu_weight_layout(cutlass::PitchLinearCoord extent, int64_t ld)
        : stride(static_cast<int>(ld)), half_n(extent.strided() / 2) {}
    CUTLASS_HOST_DEVICE ggml_cuda_i8_swiglu_weight_layout(cutlass::MatrixCoord extent, int64_t ld)
        : stride(static_cast<int>(ld)), half_n(extent.column() / 2) {}
    CUTLASS_HOST_DEVICE int64_t operator()(cutlass::PitchLinearCoord c) const { return c.contiguous() + int64_t(c.strided() / 2 + (c.strided() % 2) * half_n) * stride; }
    CUTLASS_HOST_DEVICE int64_t operator()(cutlass::MatrixCoord c) const { return c.row() + int64_t(c.column() / 2 + (c.column() % 2) * half_n) * stride; }
};

template <class Map>
struct ggml_cuda_i8_swiglu_store : cutlass::epilogue::threadblock::VisitorAuxStore<Map, float, cutlass::FloatRoundStyle::round_to_nearest, cute::Stride<int64_t, cute::_1, int64_t>> {
    using Base = cutlass::epilogue::threadblock::VisitorAuxStore<Map, float, cutlass::FloatRoundStyle::round_to_nearest, cute::Stride<int64_t, cute::_1, int64_t>>;
    using Base::Base;
    template <class CB>
    struct Callbacks : CB {
        CUTLASS_DEVICE Callbacks(CB const& base)
            : CB(base) {}
        CUTLASS_DEVICE void end_step(int step) {
            auto src    = cute::filter(this->tC_rAux);
            auto coords = cute::filter(this->tC_cAux(cute::_, cute::_, cute::_, step));
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < cute::size(src); ++i) {
                auto coord = coords(i);
                if (cute::elem_less(coord, this->problem_shape)) {
                    // Each vector contains two gate/up pairs; only their SwiGLU results are stored.
                    const float4 a = *reinterpret_cast<const float4*>(&src(i));
                    float2 value   = make_float2(__fmul_rn(a.x / (1.f + expf(-a.x)), a.y), __fmul_rn(a.z / (1.f + expf(-a.z)), a.w));
                    int64_t n = cute::get<1>(this->problem_shape), m = cute::get<0>(this->problem_shape);
                    int64_t offset                                                 = cute::get<0>(coord) * (n / 2) + cute::get<1>(coord) / 2 + cute::get<2>(coord) * m * (n / 2);
                    *reinterpret_cast<float2*>(this->params_ptr->ptr_aux + offset) = value;
                }
            }
        }
    };
    template <class Shape>
    CUTLASS_DEVICE auto get_callbacks(cutlass::gemm::GemmCoord tile, int tid, Shape shape) {
        auto cb = Base::get_callbacks(tile, tid, shape);
        return Callbacks<decltype(cb)>(cb);
    }
};

static __global__ void ggml_cuda_i8_pair_scales(const float* src, float* dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        dst[i] = src[i / 2 + (i % 2) * (n / 2)];
}

template <class Config>
static bool ggml_cuda_mul_mat_i8_swiglu_impl(
    ggml_backend_cuda_context& ctx,
    const int8_t* x,
    const int8_t* w,
    const float* xs,
    const float* ws,
    float* out,
    int64_t rows,
    int64_t n,
    int64_t k) {
    const cudaStream_t stream = ctx.stream();
    using Tile                  = typename Config::Tile;
    using Warp                  = typename Config::Warp;
    using Swizzle               = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
    using Map                   = cutlass::epilogue::threadblock::OutputTileThreadLayout<Tile, Warp, float, 4, 1>;
    using Col                   = cutlass::epilogue::threadblock::VisitorColBroadcast<Map, float, cute::Stride<cute::_1, cute::_0, int>>;
    using Row                   = cutlass::epilogue::threadblock::VisitorRowBroadcast<Map, float, cute::Stride<cute::_0, cute::_1, int>>;
    using Mul                   = cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
    using ScaleX                = cutlass::epilogue::threadblock::Sm80EVT<Mul, cutlass::epilogue::threadblock::VisitorAccFetch, Col>;
    using ScaleW                = cutlass::epilogue::threadblock::Sm80EVT<Mul, ScaleX, Row>;
    using Store                 = ggml_cuda_i8_swiglu_store<Map>;
    using EVT                   = cutlass::epilogue::threadblock::Sm80EVT<Store, ScaleW>;
    using GemmBase              = typename cutlass::gemm::kernel::DefaultGemmUniversal<int8_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 16, int8_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 16, float, cutlass::layout::RowMajor, int, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, Tile, Warp, cutlass::gemm::GemmShape<16, 8, 32>, cutlass::epilogue::thread::LinearCombination<float, 4, int, float>, Swizzle, 4, cutlass::arch::OpMultiplyAddSaturate, cutlass::gemm::SharedMemoryClearOption::kNone, false, false, false, cutlass::layout::NoPermute, cutlass::layout::NoPermute, ggml_cuda_i8_swiglu_weight_layout>::GemmKernel;
    using Epilogue              = cutlass::epilogue::threadblock::EpilogueWithVisitorCallbacks<typename GemmBase::Epilogue, EVT, 1>;
    using Kernel                = cutlass::gemm::kernel::GemmWithEpilogueVisitor<typename GemmBase::Mma, Epilogue, Swizzle>;
    using Gemm                  = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
    typename EVT::Arguments evt = {{{{}, {xs, 0.f, {}}, {}}, {ws, 0.f, {}}, {}}, {out, {n, cute::_1{}, int64_t(rows) * n}}};
    typename Gemm::Arguments args(cutlass::gemm::GemmUniversalMode::kGemm, {(int) rows, (int) n, (int) k}, 1, evt, x, w, nullptr, nullptr, int64_t(rows) * k, int64_t(n) * k, 0, 0, k, k, 0, 0);
    Gemm gemm;
    if (sizeof(typename Kernel::SharedStorage) > ggml_cuda_info().devices[ctx.device].smpbo ||
        gemm.can_implement(args) != cutlass::Status::kSuccess)
        return false;
    GGML_ASSERT(gemm.initialize(args, nullptr, stream) == cutlass::Status::kSuccess);
    GGML_ASSERT(gemm(stream) == cutlass::Status::kSuccess);
    return true;
}
bool ggml_cuda_mul_mat_i8_swiglu_cutlass(
    ggml_backend_cuda_context& ctx,
    const int8_t* x,
    const int8_t* w,
    const float* xs,
    const float* ws,
    float* out,
    int64_t rows,
    int64_t n,
    int64_t k) {
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!ampere_mma_available(cc) ||
        rows < 128 || rows > INT_MAX || n < 1 || n > INT_MAX || k < 1 || k > INT_MAX || n % 256 != 0 || k % 64 != 0 || ws == nullptr ||
        ((uintptr_t)x | (uintptr_t)w | (uintptr_t)xs | (uintptr_t)ws | (uintptr_t)out) % 16 != 0)
        return false;
    const cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<float> paired_scales(ctx.pool(), n);
    ggml_cuda_i8_pair_scales<<<(n + 255) / 256, 256, 0, stream>>>(ws, paired_scales.get(), (int) n);
    CUDA_CHECK(cudaGetLastError());
    return ggml_cuda_mul_mat_i8_dispatch(ctx, rows, n, [&](auto config) {
        return ggml_cuda_mul_mat_i8_swiglu_impl<decltype(config)>(ctx, x, w, xs, paired_scales.get(), out, rows, n, k);
    });
}
template <class Map, class Out, bool rms = true>
struct ggml_cuda_i8_rms_rope_store : cutlass::epilogue::threadblock::VisitorAuxStore<Map, float, cutlass::FloatRoundStyle::round_to_nearest, cute::Stride<int64_t, cute::_1, int64_t>> {
    using Base = cutlass::epilogue::threadblock::VisitorAuxStore<Map, float, cutlass::FloatRoundStyle::round_to_nearest, cute::Stride<int64_t, cute::_1, int64_t>>;
    struct Arguments : Base::Arguments {
        const float* weight;
        const float* pe;
        float eps;
        float scale;
        int64_t tokens;
        half* half_output;
        int64_t prefix_tokens;
    };
    using Params        = Arguments;
    using SharedStorage = typename Base::SharedStorage;
    const Params* extra;
    template <class Shape>
    static constexpr Params to_underlying_arguments(Shape const&, Arguments const& args, void*) { return args; }
    CUTLASS_HOST_DEVICE ggml_cuda_i8_rms_rope_store(Params const& p, SharedStorage const& s)
        : Base(p, s), extra(&p) {}
    template <class CB>
    struct Callbacks : CB {
        const Params* extra;
        CUTLASS_DEVICE Callbacks(CB const& base, const Params* p)
            : CB(base), extra(p) {}
        CUTLASS_DEVICE float sum32(float4 v) {
            float a = warp_reduce_sum<8>(__fmul_rn(v.x, v.x));
            float b = warp_reduce_sum<8>(__fmul_rn(v.y, v.y));
            float c = warp_reduce_sum<8>(__fmul_rn(v.z, v.z));
            float d = warp_reduce_sum<8>(__fmul_rn(v.w, v.w));
            return __fadd_rn(__fadd_rn(a, c), __fadd_rn(b, d));
        }
        CUTLASS_DEVICE void end_step(int step) {
            static_assert(Map::Detail::kAccessWidth == 16);
            auto src    = cute::filter(this->tC_rAux);
            auto coords = cute::filter(this->tC_cAux(cute::_, cute::_, cute::_, step));
            static_assert(cute::size(src) == 4);
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < cute::size(src); i += 2) {
                float4 va = *reinterpret_cast<const float4*>(&src(i)), vb = *reinterpret_cast<const float4*>(&src(i + 1));
                float norm = 1.0f;
                if constexpr (rms) {
                    // Preserve the standalone RMSNorm reduction tree across four 32-element groups.
                    float sum = __fadd_rn(sum32(va), sum32(vb));
                    sum       = __fadd_rn(sum, __shfl_xor_sync(0xffffffff, sum, 8));
                    norm      = rsqrtf(sum / 128.f + extra->eps);
                }
                CUTLASS_PRAGMA_UNROLL
                for (int j = 0; j < 2; ++j) {
                    auto coord = coords(i + j);
                    int row = cute::get<0>(coord), col = cute::get<1>(coord);
                    int heads = cute::get<1>(this->problem_shape) / 128;
                    if (cute::elem_less(coord, this->problem_shape)) {
                        float4 value  = j ? vb : va;
                        int local     = col % 128;
                        int64_t token = row % extra->tokens, batch = row / extra->tokens, head = col / 128;
                        float4 r = value;
                        if constexpr (rms) {
                            float4 weight = ((const float4*)extra->weight)[local / 4];
                            value.x       = __fmul_rn(__fmul_rn(norm, value.x), weight.x);
                            value.y       = __fmul_rn(__fmul_rn(norm, value.y), weight.y);
                            value.z       = __fmul_rn(__fmul_rn(norm, value.z), weight.z);
                            value.w       = __fmul_rn(__fmul_rn(norm, value.w), weight.w);
                            float4 p0 = ((const float4*)extra->pe)[token * 64 + local / 2], p1 = ((const float4*)extra->pe)[token * 64 + local / 2 + 1];
                            r.x = __fadd_rn(__fmul_rn(value.x, p0.x), __fmul_rn(value.y, p0.y));
                            r.y = __fadd_rn(__fmul_rn(value.x, p0.z), __fmul_rn(value.y, p0.w));
                            r.z = __fadd_rn(__fmul_rn(value.z, p1.x), __fmul_rn(value.w, p1.y));
                            r.w = __fadd_rn(__fmul_rn(value.z, p1.z), __fmul_rn(value.w, p1.w));
                        }
                        int64_t offset = ((batch * heads + head) * (extra->tokens + extra->prefix_tokens) + token + extra->prefix_tokens) * 128 + local;
                        if constexpr (std::is_same<Out, half>::value) {
                            half2 a    = __floats2half2_rn(__fmul_rn(r.x, extra->scale), __fmul_rn(r.y, extra->scale));
                            half2 b    = __floats2half2_rn(__fmul_rn(r.z, extra->scale), __fmul_rn(r.w, extra->scale));
                            half2* dst = (half2*)(extra->half_output + offset);
                            dst[0]     = a;
                            dst[1]     = b;
                        } else {
                            *((float4*)(this->params_ptr->ptr_aux + offset)) = r;
                        }
                    }
                }
            }
        }
    };
    template <class Shape>
    CUTLASS_DEVICE auto get_callbacks(cutlass::gemm::GemmCoord tile, int tid, Shape shape) {
        auto cb = Base::get_callbacks(tile, tid, shape);
        return Callbacks<decltype(cb)>(cb, extra);
    }
};
template <class Out, bool rms = true>
static bool ggml_cuda_mul_mat_i8_rms_rope_impl(ggml_backend_cuda_context& ctx, const int8_t* x, const int8_t* w, const float* xs, const float* ws, float* out, int rows, int n, int k, const float* nw, const float* pe, int64_t tokens, float eps, float scale, half* half_output = nullptr, int64_t prefix_tokens = 0) {
    const cudaStream_t stream   = ctx.stream();
    using Tile                  = cutlass::gemm::GemmShape<128, 256, 64>;
    using Warp                  = cutlass::gemm::GemmShape<64, 64, 64>;
    using Swizzle               = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
    using Map                   = cutlass::epilogue::threadblock::OutputTileThreadLayout<Tile, Warp, float, 4, 1>;
    using Col                   = cutlass::epilogue::threadblock::VisitorColBroadcast<Map, float, cute::Stride<cute::_1, cute::_0, int>>;
    using Row                   = cutlass::epilogue::threadblock::VisitorRowBroadcast<Map, float, cute::Stride<cute::_0, cute::_1, int>>;
    using Mul                   = cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
    using ScaleX                = cutlass::epilogue::threadblock::Sm80EVT<Mul, cutlass::epilogue::threadblock::VisitorAccFetch, Col>;
    using ScaleW                = cutlass::epilogue::threadblock::Sm80EVT<Mul, ScaleX, Row>;
    using Store                 = ggml_cuda_i8_rms_rope_store<Map, Out, rms>;
    using EVT                   = cutlass::epilogue::threadblock::Sm80EVT<Store, ScaleW>;
    using Kernel                = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<int8_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 16, int8_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 16, float, cutlass::layout::RowMajor, 4, int, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, Tile, Warp, cutlass::gemm::GemmShape<16, 8, 32>, EVT, Swizzle, 4, cutlass::arch::OpMultiplyAddSaturate>::GemmKernel;
    using Gemm                  = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
    typename EVT::Arguments evt = {{{{}, {xs, 0.f, {}}, {}}, {ws, 0.f, {}}, {}}, {{out, {n, cute::_1{}, int64_t(rows) * n}}, nw, pe, eps, scale, tokens, half_output ? half_output : (half*)out, prefix_tokens}};
    typename Gemm::Arguments args(cutlass::gemm::GemmUniversalMode::kGemm, {rows, n, k}, 1, evt, x, w, nullptr, nullptr, int64_t(rows) * k, int64_t(n) * k, 0, 0, k, k, 0, 0);
    Gemm gemm;
    if (sizeof(typename Kernel::SharedStorage) > ggml_cuda_info().devices[ctx.device].smpbo ||
        gemm.can_implement(args) != cutlass::Status::kSuccess)
        return false;
    GGML_ASSERT(gemm.initialize(args, nullptr, stream) == cutlass::Status::kSuccess);
    GGML_ASSERT(gemm(stream) == cutlass::Status::kSuccess);
    return true;
}

bool ggml_cuda_mul_mat_i8_rms_rope_cutlass(
    ggml_backend_cuda_context& ctx,
    const int8_t* x,
    const int8_t* w,
    const float* xs,
    const float* ws,
    float* output,
    int64_t rows,
    int64_t n,
    int64_t k,
    const float* norm_weight,
    const float* pe,
    int64_t tokens,
    float eps,
    bool f16,
    float scale) {
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!ampere_mma_available(cc) ||
        rows < 4096 || rows > INT_MAX || n < 4096 || n > INT_MAX || k < 4096 || k > INT_MAX ||
        n % 256 != 0 || k % 64 != 0 || tokens <= 0 || rows % tokens != 0 || ws == nullptr ||
        ((uintptr_t)x | (uintptr_t)w | (uintptr_t)xs | (uintptr_t)ws | (uintptr_t)output |
         (uintptr_t)norm_weight | (uintptr_t)pe) %
                16 !=
            0)
        return false;
    if (f16)
        return ggml_cuda_mul_mat_i8_rms_rope_impl<half>(ctx, x, w, xs, ws, output, (int)rows, (int)n, (int)k, norm_weight, pe, tokens, eps, scale);
    return ggml_cuda_mul_mat_i8_rms_rope_impl<float>(ctx, x, w, xs, ws, output, (int)rows, (int)n, (int)k, norm_weight, pe, tokens, eps, scale);
}

bool ggml_cuda_mul_mat_i8_kv_cutlass(
    ggml_backend_cuda_context& ctx,
    const int8_t* x,
    const int8_t* w,
    const float* xs,
    const float* ws,
    float* workspace,
    half* output,
    int64_t rows,
    int64_t n,
    int64_t k,
    const float* norm_weight,
    const float* pe,
    int64_t tokens,
    int64_t prefix_tokens,
    float eps) {
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!ampere_mma_available(cc) ||
        rows < 4096 || rows > INT_MAX || n < 4096 || n > INT_MAX || k < 4096 || k > INT_MAX ||
        n % 256 != 0 || k % 64 != 0 || tokens <= 0 || rows % tokens != 0 || ws == nullptr ||
        workspace == nullptr || prefix_tokens < 0 || prefix_tokens > INT_MAX - tokens ||
        (norm_weight != nullptr && pe == nullptr) ||
        ((uintptr_t)x | (uintptr_t)w | (uintptr_t)xs | (uintptr_t)ws | (uintptr_t)workspace |
         (uintptr_t)norm_weight | (uintptr_t)pe) %
                16 !=
            0 ||
        (uintptr_t)output % 8 != 0) {
        return false;
    }
    if (norm_weight != nullptr) {
        return ggml_cuda_mul_mat_i8_rms_rope_impl<half>(ctx, x, w, xs, ws, workspace,
                                                        (int)rows, (int)n, (int)k, norm_weight, pe, tokens, eps, 1.0f, output, prefix_tokens);
    }
    return ggml_cuda_mul_mat_i8_rms_rope_impl<half, false>(ctx, x, w, xs, ws, workspace,
                                                           (int)rows, (int)n, (int)k, nullptr, nullptr, tokens, eps, 1.0f, output, prefix_tokens);
}

#endif
