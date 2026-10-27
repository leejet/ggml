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

template <class Launch>
static bool ggml_cuda_mul_mat_i8_dispatch(ggml_backend_cuda_context& ctx, int64_t rows, int64_t n, Launch launch) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(rows);
    GGML_UNUSED(n);
    return launch(ggml_cuda_i8_tile_config<128, 256, 64>{});
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
        n % 256 != 0 || k % 64 != 0 || w_scales == nullptr ||
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

#endif
