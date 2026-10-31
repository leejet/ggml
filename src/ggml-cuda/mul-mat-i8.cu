#include "mul-mat-i8.cuh"

#ifdef GGML_CUDA_USE_CUTLASS
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/default_gemm_universal_with_visitor.h>
#include <cutlass/epilogue/threadblock/fusion/visitors.hpp>
#include <cutlass/gemm/threadblock/threadblock_swizzle_streamk.h>

struct ggml_cuda_i8_banded_streamk : cutlass::gemm::threadblock::ThreadblockSwizzleStreamK {
    using Base = cutlass::gemm::threadblock::ThreadblockSwizzleStreamK;
    using Base::Base;

    CUTLASS_DEVICE cutlass::gemm::GemmCoord get_tile_offset(int index) const {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 890
        const auto shape = tiled_shape();
        if (problem_size.m() >= 4096 && problem_size.n() >= 16384 && shape.m() >= shape.n()) {
            // Column bands retain weight tiles in Ada's L2 across successive row tiles.
            constexpr int band_width = 16;
            if (index >= shape.m() * shape.n()) {
                return {shape.m(), shape.n(), get_batch_idx()};
            }
            const int band   = index / (shape.m() * band_width);
            const int width  = min(band_width, shape.n() - band * band_width);
            const int within = index - band * shape.m() * band_width;
            return {within / width, band * band_width + within % width, get_batch_idx()};
        }
#endif
        return Base::get_tile_offset(index);
    }
};

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

static bool ggml_cuda_i8_use_streamk(int64_t rows) {
    // Scratch initialization and split-K reduction outweigh the benefit on shorter matrices.
    return rows >= 8192;
}

static bool ggml_cuda_i8_small_rows_enabled(int64_t rows, int64_t n, int64_t k) {
    static const char * setting = std::getenv("GGML_CUDA_I8_SMALL_ROWS");
    return setting ? std::atoi(setting) != 0 : rows < 32 && n >= 4096 && k >= 4096;
}

static ggml_cuda_i8_tile ggml_cuda_mul_mat_i8_get_tile(int64_t rows, int64_t n, int nsm) {
    if (rows < 32) {
        return GGML_CUDA_I8_TILE_16_128;
    }
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

template <class Config, bool stream_k>
static bool ggml_cuda_mul_mat_i8_cutlass_impl(
    ggml_backend_cuda_context& ctx,
    const int8_t* x,
    const int8_t* w,
    const float* x_scales,
    const float* w_scales,
    float* dst,
    int64_t rows,
    int64_t n,
    int64_t k) {
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!ampere_mma_available(cc) ||
        rows < (ggml_cuda_i8_small_rows_enabled(rows, n, k) ? 1 : 128) || rows > INT_MAX || n < 1 || n > INT_MAX || k < 1 || k > INT_MAX ||
        n % 128 != 0 || k % 64 != 0 || w_scales == nullptr ||
        ((uintptr_t)x | (uintptr_t)w | (uintptr_t)x_scales | (uintptr_t)w_scales | (uintptr_t)dst) % 16 != 0) {
        return false;
    }

    using Tile         = typename Config::Tile;
    using Warp         = typename Config::Warp;
    using Swizzle      = std::conditional_t<stream_k, ggml_cuda_i8_banded_streamk,
                                           cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>>;
    using Map          = cutlass::epilogue::threadblock::OutputTileThreadLayout<Tile, Warp, float, 4, 1>;
    using Col          = cutlass::epilogue::threadblock::VisitorColBroadcast<Map, float, cute::Stride<cute::_1, cute::_0, int>>;
    using Row          = cutlass::epilogue::threadblock::VisitorRowBroadcast<Map, float, cute::Stride<cute::_0, cute::_1, int>>;
    using Mul          = cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
    using ScaleX       = cutlass::epilogue::threadblock::Sm80EVT<Mul, cutlass::epilogue::threadblock::VisitorAccFetch, Col>;
    using ScaleW       = cutlass::epilogue::threadblock::Sm80EVT<Mul, ScaleX, Row>;
    using Store        = cutlass::epilogue::threadblock::VisitorAuxStore<Map, float, cutlass::FloatRoundStyle::round_to_nearest,
                                                                  cute::Stride<int64_t, cute::_1, int64_t>>;
    using Epilogue     = cutlass::epilogue::threadblock::Sm80EVT<Store, ScaleW>;
    using Kernel       = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
        int8_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 16,
        int8_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 16,
        float, cutlass::layout::RowMajor, 4, int, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        Tile, Warp, cutlass::gemm::GemmShape<16, 8, 32>, Epilogue,
        Swizzle, 4, cutlass::arch::OpMultiplyAddSaturate>::GemmKernel;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

    typename ScaleW::Arguments scales = {{{}, {x_scales, 0.0f, {}}, {}}, {w_scales, 0.0f, {}}, {}};
    typename Epilogue::Arguments epilogue = {scales, {dst, {n, cute::_1{}, rows * n}}};
    typename Gemm::Arguments args(cutlass::gemm::GemmUniversalMode::kGemm, {(int)rows, (int)n, (int)k}, 1, epilogue,
                                  x, w, nullptr, nullptr, rows * k, n * k, 0, 0, k, k, 0, 0);
    Gemm gemm;
    if (sizeof(typename Kernel::SharedStorage) > ggml_cuda_info().devices[ctx.device].smpbo ||
        gemm.can_implement(args) != cutlass::Status::kSuccess) {
        return false;
    }
    ggml_cuda_pool_alloc<char> workspace(ctx.pool());
    const size_t workspace_size = gemm.get_workspace_size(args);
    if (workspace_size > 0) {
        workspace.alloc(workspace_size);
    }
    GGML_ASSERT(gemm.initialize(args, workspace.get(), ctx.stream()) == cutlass::Status::kSuccess);
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
    if (rows < (ggml_cuda_i8_small_rows_enabled(rows, n, k) ? 1 : 128) || n <= 0 || rows > INT_MAX || n > INT_MAX) {
        return false;
    }
    return ggml_cuda_mul_mat_i8_dispatch(ctx, rows, n, [&](auto config) {
        if (ggml_cuda_i8_use_streamk(rows)) {
            return ggml_cuda_mul_mat_i8_cutlass_impl<decltype(config), true>(ctx, x, w, x_scales, w_scales, dst, rows, n, k);
        }
        return ggml_cuda_mul_mat_i8_cutlass_impl<decltype(config), false>(ctx, x, w, x_scales, w_scales, dst, rows, n, k);
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

template <class Config, bool stream_k>
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
    using Swizzle               = std::conditional_t<stream_k, ggml_cuda_i8_banded_streamk,
                                                     cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>>;
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
    using Kernel                = std::conditional_t<stream_k,
                                                     cutlass::gemm::kernel::GemmWithEpilogueVisitorStreamk<typename GemmBase::Mma, Epilogue, Swizzle>,
                                                     cutlass::gemm::kernel::GemmWithEpilogueVisitor<typename GemmBase::Mma, Epilogue, Swizzle>>;
    using Gemm                  = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
    typename EVT::Arguments evt = {{{{}, {xs, 0.f, {}}, {}}, {ws, 0.f, {}}, {}}, {out, {n, cute::_1{}, int64_t(rows) * n}}};
    typename Gemm::Arguments args(cutlass::gemm::GemmUniversalMode::kGemm, {(int) rows, (int) n, (int) k}, 1, evt, x, w, nullptr, nullptr, int64_t(rows) * k, int64_t(n) * k, 0, 0, k, k, 0, 0);
    Gemm gemm;
    if (sizeof(typename Kernel::SharedStorage) > ggml_cuda_info().devices[ctx.device].smpbo ||
        gemm.can_implement(args) != cutlass::Status::kSuccess)
        return false;
    ggml_cuda_pool_alloc<char> workspace(ctx.pool());
    const size_t workspace_size = gemm.get_workspace_size(args);
    if (workspace_size > 0) {
        workspace.alloc(workspace_size);
    }
    GGML_ASSERT(gemm.initialize(args, workspace.get(), stream) == cutlass::Status::kSuccess);
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
        rows < (ggml_cuda_i8_small_rows_enabled(rows, n, k) ? 1 : 128) || rows > INT_MAX || n < 1 || n > INT_MAX || k < 1 || k > INT_MAX || n % 256 != 0 || k % 64 != 0 || ws == nullptr ||
        ((uintptr_t)x | (uintptr_t)w | (uintptr_t)xs | (uintptr_t)ws | (uintptr_t)out) % 16 != 0)
        return false;
    const cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<float> paired_scales(ctx.pool(), n);
    ggml_cuda_i8_pair_scales<<<(n + 255) / 256, 256, 0, stream>>>(ws, paired_scales.get(), (int) n);
    CUDA_CHECK(cudaGetLastError());
    return ggml_cuda_mul_mat_i8_dispatch(ctx, rows, n, [&](auto config) {
        if (ggml_cuda_i8_use_streamk(rows)) {
            return ggml_cuda_mul_mat_i8_swiglu_impl<decltype(config), true>(ctx, x, w, xs, paired_scales.get(), out, rows, n, k);
        }
        return ggml_cuda_mul_mat_i8_swiglu_impl<decltype(config), false>(ctx, x, w, xs, paired_scales.get(), out, rows, n, k);
    });
}

#endif
