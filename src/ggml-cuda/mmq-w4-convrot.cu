#include "ggml-cuda.h"
#include "ggml-impl.h"
#include "mmq-w4-convrot.cuh"

#include "common.cuh"
#include "fp8.cuh"

// Byte-permute helper: selects 4 result bytes from the 8-byte window
// {lo = bytes 0-3, hi = bytes 4-7}. The backends encode the selectors
// differently: __byte_perm (CUDA) takes one 4-bit selector nibble per
// result byte, amdgcn_perm (HIP) takes one selector byte per result byte
// with only the low 3 bits defined (values > 7 read out of range). The
// PERM_SEL_* constants below carry the same logical selection per backend.
static __device__ __forceinline__ uint32_t perm_bytes(uint32_t lo, uint32_t hi, uint32_t sel) {
#if defined(GGML_USE_HIP)
    return __builtin_amdgcn_perm(hi, lo, sel);
#elif !defined(GGML_USE_MUSA)
    return __byte_perm(lo, hi, sel);
#else
    uint32_t r = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t b = (sel >> (4 * i)) & 0xF;
        const uint32_t byte = (b < 4 ? lo : hi) >> (8 * (b & 3)) & 0xFF;
        r |= byte << (8 * i);
    }
    return r;
#endif
}

#if defined(GGML_USE_HIP)
#define PERM_SEL_EVEN 0x06040200u // window bytes 0,2,4,6
#define PERM_SEL_ODD  0x07050301u // window bytes 1,3,5,7
#else
#define PERM_SEL_EVEN 0x6420u
#define PERM_SEL_ODD  0x7531u
#endif

// Stages 8 activation k values as 2 interleaved words so nibble lanes pair
// with activation lanes: word 0 = k 0,2,4,6, word 1 = k 1,3,5,7.
// lo/hi = sequential int8x4 words covering k 8*g .. 8*g+7.
static __device__ __forceinline__ void interleave_act_pair(
        const int lo, const int hi, int * __restrict__ dst) {
    dst[0] = (int) perm_bytes((uint32_t) lo, (uint32_t) hi, PERM_SEL_EVEN);
    dst[1] = (int) perm_bytes((uint32_t) lo, (uint32_t) hi, PERM_SEL_ODD);
}

// W4A4 MMQ for the packed-w4 convrot mulmat (no BLAS route: cublasGemmEx
// cannot consume packed nibbles, so this kernel is always the path).
// Computes y[r*n + o] = sum_k W[o, k] * X[r, k] exactly on the int grid.
// W: [n, packed_k] row-major I8 byte-packed nibbles, low nibble = even k.
// X: [rows, k] row-major I8, padded rows zeroed. packed_k % 8 == 0
// (guaranteed by the supports_op gate, so each staged K step is full).

#define MUL_MAT_W4_BLOCK  16
#define MUL_MAT_W4_ROWS (2 * MUL_MAT_W4_BLOCK) // row tile per CTA: w decode shared by two 16-row tiles
#define MUL_MAT_W4_GPS 4 // groups (16 k) staged per sync round (64 k)
#define MUL_MAT_W4_SLOTS (4 * MUL_MAT_W4_GPS) // decoded int8x4 words per output per stage
#define MUL_MAT_W4_OUT (MUL_MAT_W4_BLOCK * 4) // outputs covered per block (4 per thread)

// Decoded-weight structure: the packed nibbles are expanded into shared
// int8x4 words once per stage (the decode is identical for all 32 row
// threads), so the compute loop is pure dp4a. Nibble bits n are biased by
// XOR 8 into [0,15] (two's-complement value + 8), which moves the sign
// correction out of the dot: sum(w*a) = sum(m*a) - 8*sum(a); the per-row
// byte sums accumulate in the fill threads' registers and reduce by a
// width-8/16 butterfly at epilogue (integer sums are order-invariant).
// Each thread owns 4 outputs (lx + 16*j) x 2 rows (the two 16-row tiles
// share one w decode, halving the w LDS traffic per dp4a); the 8 fill
// lanes per tile stage one row's interleaved activation pairs while the
// other 8 lanes stage the second tile's, so no lane idles in fill. The
// two packed weight words a thread decodes sit at consecutive addresses
// (w_words is even), so they load as one int2. Stages are double-buffered
// so the next stage is decoded while the current one computes (one barrier
// per stage). Out-of-range slots decode to biased 8s, but their activation
// bytes are zero-filled, so the dot stays exact.
static __global__ void __launch_bounds__(MUL_MAT_W4_BLOCK * MUL_MAT_W4_BLOCK) mul_mat_w4a4_convrot_dp4a(
        const char * __restrict__ w, const char * __restrict__ x, int * __restrict__ y,
        const uint32_t n, const uint32_t rows, const uint32_t packed_k) {

    const uint32_t o_base = blockIdx.x * MUL_MAT_W4_OUT;
    const uint32_t r0 = blockIdx.y * MUL_MAT_W4_ROWS + threadIdx.y;
    const uint32_t r1 = r0 + MUL_MAT_W4_BLOCK;
    const uint32_t lx = threadIdx.x;
    const uint32_t ly = threadIdx.y;
    const uint32_t idx = ly * MUL_MAT_W4_BLOCK + lx;
    const uint32_t dc = idx & (MUL_MAT_W4_OUT - 1); // decode output column
    const uint32_t dw = idx >> 6;                   // packed-word pair (0..3)

    __shared__ int w_dec[2][MUL_MAT_W4_SLOTS * MUL_MAT_W4_OUT];
    __shared__ int x_tile[2][MUL_MAT_W4_ROWS * MUL_MAT_W4_SLOTS];

    const uint32_t w_words = packed_k / 4;       // 8 nibbles per word
    const uint32_t x_words = (packed_k * 2) / 4; // 4 int8 lanes per word
    const uint32_t groups  = packed_k / 8;       // 16 k per group
    const int * wi = (const int *) w;
    const int * xi = (const int *) x;

    // fill mapping: lanes lx 0..7 stage tile row ly (r0), lanes 8..15 stage
    // the second tile's row (r1); each lane covers one word pair per stage
    const uint32_t fl = lx & 7;
    const uint32_t fr = (lx >> 3) ? r1 : r0;
    const uint32_t tr = (lx >> 3) * MUL_MAT_W4_BLOCK + ly;
    int my_sum = 0; // byte sum of this thread's staged pairs (row fr)

    // prefill stage 0; the decode phase is spread over all threads (2 packed
    // words each) and w_dec is [slot][output] so warp lanes read consecutive
    // banks in the compute loop
    {
        const uint32_t ocol = o_base + dc;
        int2 wv = make_int2(0, 0);
        if (ocol < n && dw * 2 < w_words) {
            wv = ((const int2 *) wi)[(size_t) ocol * (w_words / 2) + dw];
        }
        w_dec[0][((dw * 2 + 0) * 2 + 0) * MUL_MAT_W4_OUT + dc] = ( wv.x      & 0x0F0F0F0F) ^ 0x08080808;
        w_dec[0][((dw * 2 + 0) * 2 + 1) * MUL_MAT_W4_OUT + dc] = ((wv.x >> 4) & 0x0F0F0F0F) ^ 0x08080808;
        w_dec[0][((dw * 2 + 1) * 2 + 0) * MUL_MAT_W4_OUT + dc] = ( wv.y      & 0x0F0F0F0F) ^ 0x08080808;
        w_dec[0][((dw * 2 + 1) * 2 + 1) * MUL_MAT_W4_OUT + dc] = ((wv.y >> 4) & 0x0F0F0F0F) ^ 0x08080808;
        // two sequential words (8 k) become the interleaved even/odd pair.
        // The byte sum is permutation-invariant, so the raw words feed it.
        const uint32_t gi = fl >> 1;
        const uint32_t xb = gi * 4 + (fl & 1) * 2;
        const int a = (fr < rows && xb < x_words) ? (int) xi[(size_t) fr * x_words + xb] : 0;
        const int b = (fr < rows && xb + 1 < x_words) ? (int) xi[(size_t) fr * x_words + xb + 1] : 0;
        interleave_act_pair(a, b, &x_tile[0][tr * MUL_MAT_W4_SLOTS + (fl >> 1) * 4 + (fl & 1) * 2]);
        my_sum += ggml_cuda_dp4a(a, 0x01010101, 0) + ggml_cuda_dp4a(b, 0x01010101, 0);
    }
    __syncthreads();

    // two partial chains per output and row: accNa sums the even-slot dots,
    // accNb the odd-slot dots; 16 independent dot4 chains hide the dot
    // latency. Integer addition is associative, so accNa + accNb is bitwise
    // identical to a single chained accumulator.
    int acc0a = 0, acc0b = 0;
    int acc1a = 0, acc1b = 0;
    int acc2a = 0, acc2b = 0;
    int acc3a = 0, acc3b = 0;
    int acc4a = 0, acc4b = 0;
    int acc5a = 0, acc5b = 0;
    int acc6a = 0, acc6b = 0;
    int acc7a = 0, acc7b = 0;

    for (uint32_t gs = 0; gs < groups; gs += MUL_MAT_W4_GPS) {
        const uint32_t buf = (gs >> 2) & 1;
        // prefetch the next stage FIRST so its global-load + LDS-write latency
        // overlaps this stage's compute; the end-of-iteration barrier then
        // finds the loads already landed (one barrier per stage)
        if (gs + MUL_MAT_W4_GPS < groups) {
            const uint32_t nbuf = buf ^ 1;
            const uint32_t ngs = gs + MUL_MAT_W4_GPS;
            {
                const uint32_t ocol = o_base + dc;
                const uint32_t widx = ngs * 2 + dw * 2;
                int2 wv = make_int2(0, 0);
                if (ocol < n && widx < w_words) {
                    wv = ((const int2 *) wi)[(size_t) ocol * (w_words / 2) + ngs + dw];
                }
                w_dec[nbuf][((dw * 2 + 0) * 2 + 0) * MUL_MAT_W4_OUT + dc] = ( wv.x      & 0x0F0F0F0F) ^ 0x08080808;
                w_dec[nbuf][((dw * 2 + 0) * 2 + 1) * MUL_MAT_W4_OUT + dc] = ((wv.x >> 4) & 0x0F0F0F0F) ^ 0x08080808;
                w_dec[nbuf][((dw * 2 + 1) * 2 + 0) * MUL_MAT_W4_OUT + dc] = ( wv.y      & 0x0F0F0F0F) ^ 0x08080808;
                w_dec[nbuf][((dw * 2 + 1) * 2 + 1) * MUL_MAT_W4_OUT + dc] = ((wv.y >> 4) & 0x0F0F0F0F) ^ 0x08080808;
                const uint32_t gi = ngs + (fl >> 1);
                const uint32_t xb = gi * 4 + (fl & 1) * 2;
                const int a = (fr < rows && xb < x_words) ? (int) xi[(size_t) fr * x_words + xb] : 0;
                const int b = (fr < rows && xb + 1 < x_words) ? (int) xi[(size_t) fr * x_words + xb + 1] : 0;
                interleave_act_pair(a, b, &x_tile[nbuf][tr * MUL_MAT_W4_SLOTS + (fl >> 1) * 4 + (fl & 1) * 2]);
                my_sum += ggml_cuda_dp4a(a, 0x01010101, 0) + ggml_cuda_dp4a(b, 0x01010101, 0);
            }
        }
#pragma unroll
        for (uint32_t gl = 0; gl < MUL_MAT_W4_GPS; ++gl) {
            const int u0[4] = {
                x_tile[buf][ly * MUL_MAT_W4_SLOTS + gl * 4 + 0],
                x_tile[buf][ly * MUL_MAT_W4_SLOTS + gl * 4 + 1],
                x_tile[buf][ly * MUL_MAT_W4_SLOTS + gl * 4 + 2],
                x_tile[buf][ly * MUL_MAT_W4_SLOTS + gl * 4 + 3],
            };
            const int u1[4] = {
                x_tile[buf][(MUL_MAT_W4_BLOCK + ly) * MUL_MAT_W4_SLOTS + gl * 4 + 0],
                x_tile[buf][(MUL_MAT_W4_BLOCK + ly) * MUL_MAT_W4_SLOTS + gl * 4 + 1],
                x_tile[buf][(MUL_MAT_W4_BLOCK + ly) * MUL_MAT_W4_SLOTS + gl * 4 + 2],
                x_tile[buf][(MUL_MAT_W4_BLOCK + ly) * MUL_MAT_W4_SLOTS + gl * 4 + 3],
            };
#pragma unroll
            for (uint32_t jp = 0; jp < 2; ++jp) {
                const int ue0 = jp == 0 ? u0[0] : u0[2];
                const int uo0 = jp == 0 ? u0[1] : u0[3];
                const int ue1 = jp == 0 ? u1[0] : u1[2];
                const int uo1 = jp == 0 ? u1[1] : u1[3];
                const uint32_t we = (gl * 4 + jp * 2) * MUL_MAT_W4_OUT; // even-lane slot base
                const uint32_t wo = we + MUL_MAT_W4_OUT;                // odd-lane slot base
                // both rows' dots share the w_dec reads; per output and row
                // two partial chains (even/odd slots) keep 16 independent
                // dot4 chains alive to hide the dot latency; integer addition
                // is associative, so each combined result is bitwise
                // identical to a single chained accumulator
                acc0a = ggml_cuda_dp4a(w_dec[buf][we +        lx], ue0, acc0a);
                acc1a = ggml_cuda_dp4a(w_dec[buf][we + 16 +   lx], ue0, acc1a);
                acc2a = ggml_cuda_dp4a(w_dec[buf][we + 32 +   lx], ue0, acc2a);
                acc3a = ggml_cuda_dp4a(w_dec[buf][we + 48 +   lx], ue0, acc3a);
                acc0b = ggml_cuda_dp4a(w_dec[buf][wo +        lx], uo0, acc0b);
                acc1b = ggml_cuda_dp4a(w_dec[buf][wo + 16 +   lx], uo0, acc1b);
                acc2b = ggml_cuda_dp4a(w_dec[buf][wo + 32 +   lx], uo0, acc2b);
                acc3b = ggml_cuda_dp4a(w_dec[buf][wo + 48 +   lx], uo0, acc3b);
                acc4a = ggml_cuda_dp4a(w_dec[buf][we +        lx], ue1, acc4a);
                acc5a = ggml_cuda_dp4a(w_dec[buf][we + 16 +   lx], ue1, acc5a);
                acc6a = ggml_cuda_dp4a(w_dec[buf][we + 32 +   lx], ue1, acc6a);
                acc7a = ggml_cuda_dp4a(w_dec[buf][we + 48 +   lx], ue1, acc7a);
                acc4b = ggml_cuda_dp4a(w_dec[buf][wo +        lx], uo1, acc4b);
                acc5b = ggml_cuda_dp4a(w_dec[buf][wo + 16 +   lx], uo1, acc5b);
                acc6b = ggml_cuda_dp4a(w_dec[buf][wo + 32 +   lx], uo1, acc6b);
                acc7b = ggml_cuda_dp4a(w_dec[buf][wo + 48 +   lx], uo1, acc7b);
            }
        }
        __syncthreads();
    }

    // per-thread byte-sum partials reduce inside each 16-lane segment:
    // lanes 0..7 hold r0's partials, lanes 8..15 hold r1's (same ly), so a
    // width-8 butterfly totals each row and the width-16 broadcasts land the
    // row totals on every lane that writes them. Must run before the row
    // guards so all lanes of a warp stay converged for the shuffles.
    int v = my_sum;
    v += __shfl_down_sync(0xffffffffu, v, 4, 8);
    v += __shfl_down_sync(0xffffffffu, v, 2, 8);
    v += __shfl_down_sync(0xffffffffu, v, 1, 8);
    const int sum0 = __shfl_sync(0xffffffffu, v, 0, 16);
    const int sum1 = __shfl_sync(0xffffffffu, v, 8, 16);

    const int accv[2][4] = {
        { acc0a + acc0b, acc1a + acc1b, acc2a + acc2b, acc3a + acc3b },
        { acc4a + acc4b, acc5a + acc5b, acc6a + acc6b, acc7a + acc7b },
    };
    const int sumv[2] = { sum0, sum1 };
#pragma unroll
    for (uint32_t t = 0; t < 2; ++t) {
        const uint32_t rt = t == 0 ? r0 : r1;
        if (rt < rows) {
#pragma unroll
            for (uint32_t j = 0; j < 4; ++j) {
                const uint32_t oj = o_base + j * 16 + lx;
                if (oj < n) {
                    y[(size_t) rt * n + oj] = accv[t][j] - 8 * sumv[t];
                }
            }
        }
    }
}

// W4A8: weight nibbles index a 16-entry codebook; each (output, 16-k group)
// carries an F8_E4M3 scale s_rel folded with s_channel. The codebook is
// requantized to int8 in-CTA (amax/127 scale), so the dot runs on the int
// grid via table16 + dp4a with one FFMA per group; act scale + bias are
// applied in-kernel (no epilogue pass). k % 16 == 0 (guaranteed by the builder).
#define MUL_MAT_W4A8_BLOCK  16
#define MUL_MAT_W4A8_ROWS (2 * MUL_MAT_W4A8_BLOCK) // row tile per CTA: w decode shared by two 16-row tiles
#define MUL_MAT_W4A8_GPS 4 // groups (16 k) staged per sync round (64 k)
#define MUL_MAT_W4A8_SLOTS (4 * MUL_MAT_W4A8_GPS) // decoded int8x4 words per output per stage
#define MUL_MAT_W4A8_OUT (MUL_MAT_W4A8_BLOCK * 4) // outputs covered per block (4 per thread)

// W4A8 restructure (see the W4A4 kernel): the table decode and the per-group
// F8 scale decode run once per stage into shared, the compute loop is pure
// dp4a + one FFMA per group, each thread owns 4 outputs x 2 rows (w decode
// and group scales are shared between the two row tiles, halving the LDS
// traffic per dp4a), and stages are double-buffered (one barrier per stage).
static __global__ void __launch_bounds__(MUL_MAT_W4A8_BLOCK * MUL_MAT_W4A8_BLOCK) mul_mat_w4a8_convrot_dp4a(
        const char * __restrict__ w, const char * __restrict__ x,
        const float * __restrict__ codebook, const float * __restrict__ s_channel,
        const uint8_t * __restrict__ s_rel, const float * __restrict__ act_scales,
        const float * __restrict__ bias, float * __restrict__ y,
        const uint32_t n, const uint32_t rows, const uint32_t k) {

    const uint32_t o_base = blockIdx.x * MUL_MAT_W4A8_OUT;
    const uint32_t r0 = blockIdx.y * MUL_MAT_W4A8_ROWS + threadIdx.y;
    const uint32_t r1 = r0 + MUL_MAT_W4A8_BLOCK;
    const uint32_t lx = threadIdx.x;
    const uint32_t ly = threadIdx.y;
    const uint32_t idx = ly * MUL_MAT_W4A8_BLOCK + lx;
    const uint32_t dc = idx & (MUL_MAT_W4A8_OUT - 1); // decode output column
    const uint32_t dw = idx >> 6;                     // packed-word pair / group slot

    __shared__ int w_dec[2][MUL_MAT_W4A8_SLOTS * MUL_MAT_W4A8_OUT];
    __shared__ int x_tile[2][MUL_MAT_W4A8_ROWS * MUL_MAT_W4A8_SLOTS];
    __shared__ float s_dec[2][MUL_MAT_W4A8_GPS * MUL_MAT_W4A8_OUT]; // s_cb * (s_channel * s_rel)
    __shared__ __align__(4) int8_t cb8[16]; // int8-requantized codebook
    __shared__ float s_cb;                  // requant scale: cb = cb8 * s_cb
    __shared__ int lut[256];                // per-byte decode: cb8[b & 0xF] | cb8[b >> 4] << 8 (2 k per byte)

    const uint32_t w_words = k / 8;
    const uint32_t x_words = k / 4;
    const uint32_t groups  = k / 16;
    const int * wi = (const int *) w;
    const int * xi = (const int *) x;

    if (ly == 0 && lx == 0) {
        float amax = 0.0f;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            amax = fmaxf(amax, fabsf(codebook[i]));
        }
        // s_cb = amax/127 puts the max entry at +-127, halving the rounding
        // error vs a power-of-2 scale; same expression order as the Vulkan
        // shader so cb8 is bit-identical across backends
        s_cb = amax / 127.0f;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int v = lrintf(codebook[i] / s_cb);
            cb8[i] = (int8_t) (v < -127 ? -127 : (v > 127 ? 127 : v));
        }
    }
    __syncthreads();
    // per-byte pair-LUT: byte b holds the decoded values of both nibbles
    // (low nibble = even k in the low byte, high nibble = odd k in the
    // high byte), so one packed weight word decodes with 4 gathers.
    // One entry per thread: the old 256-iteration loop on lane 0 stalled
    // the whole CTA at the barrier on every launch.
    lut[idx] = (cb8[idx & 0xF] & 0xFF) | ((cb8[idx >> 4] & 0xFF) << 8);
    __syncthreads();

    // prefill stage 0; the decode phase is spread over all threads (2 packed
    // words + 1 group scale each). Each byte holds 2 sequential k; two lut
    // gathers build one int8x4 word of decoded weights, so w_dec slots carry
    // SEQUENTIAL k bytes (no even/odd split, no act interleave on this path).
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
        const uint32_t pw = dw * 2 + j;
        const uint32_t ocol = o_base + dc;
        const int wv = (ocol < n && pw < w_words) ? wi[(size_t) ocol * w_words + pw] : 0;
        w_dec[0][(pw * 2 + 0) * MUL_MAT_W4A8_OUT + dc] = (lut[wv & 0xFF] & 0xFFFF)
                                                       | ((lut[(wv >> 8) & 0xFF] & 0xFFFF) << 16);
        w_dec[0][(pw * 2 + 1) * MUL_MAT_W4A8_OUT + dc] = (lut[(wv >> 16) & 0xFF] & 0xFFFF)
                                                       | ((lut[(wv >> 24) & 0xFF] & 0xFFFF) << 16);
    }
    {
        const uint32_t ocol = o_base + dc;
        float s = 0.0f;
        if (ocol < n && dw < groups) {
            ggml_fp8_e4m3_cuda srel;
            srel.value = s_rel[(size_t) ocol * groups + dw];
            s = s_channel[ocol] * (float) srel;
        }
        s_dec[0][dw * MUL_MAT_W4A8_OUT + dc] = s_cb * s;
    }
    if (lx < 16) {
        // sequential act staging: one word per thread covers the 16 words of
        // the stage (4 groups x 4 words) for both row tiles
        const uint32_t aidx = lx;
        x_tile[0][ly * MUL_MAT_W4A8_SLOTS + lx] =
            (r0 < rows && aidx < x_words) ? xi[(size_t) r0 * x_words + aidx] : 0;
        x_tile[0][(MUL_MAT_W4A8_BLOCK + ly) * MUL_MAT_W4A8_SLOTS + lx] =
            (r1 < rows && aidx < x_words) ? xi[(size_t) r1 * x_words + aidx] : 0;
    }
    __syncthreads();

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    float acc4 = 0.0f;
    float acc5 = 0.0f;
    float acc6 = 0.0f;
    float acc7 = 0.0f;

    for (uint32_t gs = 0; gs < groups; gs += MUL_MAT_W4A8_GPS) {
        const uint32_t buf = (gs >> 2) & 1;
        // prefetch the next stage FIRST so its global-load + LDS-write latency
        // overlaps this stage's compute (one barrier per stage); no numeric
        // change - the per-group float accumulation order is untouched
        if (gs + MUL_MAT_W4A8_GPS < groups) {
            const uint32_t nbuf = buf ^ 1;
            const uint32_t ngs = gs + MUL_MAT_W4A8_GPS;
#pragma unroll
            for (uint32_t j = 0; j < 2; ++j) {
                const uint32_t pw = dw * 2 + j;
                const uint32_t ocol = o_base + dc;
                const uint32_t widx = ngs * 2 + pw;
                const int wv = (ocol < n && widx < w_words) ? wi[(size_t) ocol * w_words + widx] : 0;
                w_dec[nbuf][(pw * 2 + 0) * MUL_MAT_W4A8_OUT + dc] = (lut[wv & 0xFF] & 0xFFFF)
                                                                  | ((lut[(wv >> 8) & 0xFF] & 0xFFFF) << 16);
                w_dec[nbuf][(pw * 2 + 1) * MUL_MAT_W4A8_OUT + dc] = (lut[(wv >> 16) & 0xFF] & 0xFFFF)
                                                                  | ((lut[(wv >> 24) & 0xFF] & 0xFFFF) << 16);
            }
            {
                const uint32_t ocol = o_base + dc;
                float s = 0.0f;
                if (ocol < n && ngs + dw < groups) {
                    ggml_fp8_e4m3_cuda srel;
                    srel.value = s_rel[(size_t) ocol * groups + ngs + dw];
                    s = s_channel[ocol] * (float) srel;
                }
                s_dec[nbuf][dw * MUL_MAT_W4A8_OUT + dc] = s_cb * s;
            }
            if (lx < 16) {
                const uint32_t aidx = ngs * 4 + lx;
                x_tile[nbuf][ly * MUL_MAT_W4A8_SLOTS + lx] =
                    (r0 < rows && aidx < x_words) ? xi[(size_t) r0 * x_words + aidx] : 0;
                x_tile[nbuf][(MUL_MAT_W4A8_BLOCK + ly) * MUL_MAT_W4A8_SLOTS + lx] =
                    (r1 < rows && aidx < x_words) ? xi[(size_t) r1 * x_words + aidx] : 0;
            }
        }
        // unguarded compute: out-of-range groups decode to a cb8[0] pattern,
        // but their hoisted scale in s_dec is zero, so the FFMA adds exactly
        // 0; the int8-requantized codebook decode is already materialized in
        // shared, so the group dot is pure dp4a and only one FFMA per group
        // touches the float grid
#pragma unroll
        for (uint32_t gl = 0; gl < MUL_MAT_W4A8_GPS; ++gl) {
            const int u0[4] = {
                x_tile[buf][ly * MUL_MAT_W4A8_SLOTS + gl * 4 + 0],
                x_tile[buf][ly * MUL_MAT_W4A8_SLOTS + gl * 4 + 1],
                x_tile[buf][ly * MUL_MAT_W4A8_SLOTS + gl * 4 + 2],
                x_tile[buf][ly * MUL_MAT_W4A8_SLOTS + gl * 4 + 3],
            };
            const int u1[4] = {
                x_tile[buf][(MUL_MAT_W4A8_BLOCK + ly) * MUL_MAT_W4A8_SLOTS + gl * 4 + 0],
                x_tile[buf][(MUL_MAT_W4A8_BLOCK + ly) * MUL_MAT_W4A8_SLOTS + gl * 4 + 1],
                x_tile[buf][(MUL_MAT_W4A8_BLOCK + ly) * MUL_MAT_W4A8_SLOTS + gl * 4 + 2],
                x_tile[buf][(MUL_MAT_W4A8_BLOCK + ly) * MUL_MAT_W4A8_SLOTS + gl * 4 + 3],
            };
            int accg0 = ggml_cuda_dp4a(w_dec[buf][(gl * 4 + 0) * MUL_MAT_W4A8_OUT +        lx], u0[0], 0);
            accg0 = ggml_cuda_dp4a(w_dec[buf][(gl * 4 + 1) * MUL_MAT_W4A8_OUT +        lx], u0[1], accg0);
            accg0 = ggml_cuda_dp4a(w_dec[buf][(gl * 4 + 2) * MUL_MAT_W4A8_OUT +        lx], u0[2], accg0);
            accg0 = ggml_cuda_dp4a(w_dec[buf][(gl * 4 + 3) * MUL_MAT_W4A8_OUT +        lx], u0[3], accg0);
            int accg1 = ggml_cuda_dp4a(w_dec[buf][(gl * 4 + 0) * MUL_MAT_W4A8_OUT + 16 + lx], u0[0], 0);
            accg1 = ggml_cuda_dp4a(w_dec[buf][(gl * 4 + 1) * MUL_MAT_W4A8_OUT + 16 + lx], u0[1], accg1);
            accg1 = ggml_cuda_dp4a(w_dec[buf][(gl * 4 + 2) * MUL_MAT_W4A8_OUT + 16 + lx], u0[2], accg1);
            accg1 = ggml_cuda_dp4a(w_dec[buf][(gl * 4 + 3) * MUL_MAT_W4A8_OUT + 16 + lx], u0[3], accg1);
            int accg2 = ggml_cuda_dp4a(w_dec[buf][(gl * 4 + 0) * MUL_MAT_W4A8_OUT + 32 + lx], u0[0], 0);
            accg2 = ggml_cuda_dp4a(w_dec[buf][(gl * 4 + 1) * MUL_MAT_W4A8_OUT + 32 + lx], u0[1], accg2);
            accg2 = ggml_cuda_dp4a(w_dec[buf][(gl * 4 + 2) * MUL_MAT_W4A8_OUT + 32 + lx], u0[2], accg2);
            accg2 = ggml_cuda_dp4a(w_dec[buf][(gl * 4 + 3) * MUL_MAT_W4A8_OUT + 32 + lx], u0[3], accg2);
            int accg3 = ggml_cuda_dp4a(w_dec[buf][(gl * 4 + 0) * MUL_MAT_W4A8_OUT + 48 + lx], u0[0], 0);
            accg3 = ggml_cuda_dp4a(w_dec[buf][(gl * 4 + 1) * MUL_MAT_W4A8_OUT + 48 + lx], u0[1], accg3);
            accg3 = ggml_cuda_dp4a(w_dec[buf][(gl * 4 + 2) * MUL_MAT_W4A8_OUT + 48 + lx], u0[2], accg3);
            accg3 = ggml_cuda_dp4a(w_dec[buf][(gl * 4 + 3) * MUL_MAT_W4A8_OUT + 48 + lx], u0[3], accg3);
            int accg4 = ggml_cuda_dp4a(w_dec[buf][(gl * 4 + 0) * MUL_MAT_W4A8_OUT +        lx], u1[0], 0);
            accg4 = ggml_cuda_dp4a(w_dec[buf][(gl * 4 + 1) * MUL_MAT_W4A8_OUT +        lx], u1[1], accg4);
            accg4 = ggml_cuda_dp4a(w_dec[buf][(gl * 4 + 2) * MUL_MAT_W4A8_OUT +        lx], u1[2], accg4);
            accg4 = ggml_cuda_dp4a(w_dec[buf][(gl * 4 + 3) * MUL_MAT_W4A8_OUT +        lx], u1[3], accg4);
            int accg5 = ggml_cuda_dp4a(w_dec[buf][(gl * 4 + 0) * MUL_MAT_W4A8_OUT + 16 + lx], u1[0], 0);
            accg5 = ggml_cuda_dp4a(w_dec[buf][(gl * 4 + 1) * MUL_MAT_W4A8_OUT + 16 + lx], u1[1], accg5);
            accg5 = ggml_cuda_dp4a(w_dec[buf][(gl * 4 + 2) * MUL_MAT_W4A8_OUT + 16 + lx], u1[2], accg5);
            accg5 = ggml_cuda_dp4a(w_dec[buf][(gl * 4 + 3) * MUL_MAT_W4A8_OUT + 16 + lx], u1[3], accg5);
            int accg6 = ggml_cuda_dp4a(w_dec[buf][(gl * 4 + 0) * MUL_MAT_W4A8_OUT + 32 + lx], u1[0], 0);
            accg6 = ggml_cuda_dp4a(w_dec[buf][(gl * 4 + 1) * MUL_MAT_W4A8_OUT + 32 + lx], u1[1], accg6);
            accg6 = ggml_cuda_dp4a(w_dec[buf][(gl * 4 + 2) * MUL_MAT_W4A8_OUT + 32 + lx], u1[2], accg6);
            accg6 = ggml_cuda_dp4a(w_dec[buf][(gl * 4 + 3) * MUL_MAT_W4A8_OUT + 32 + lx], u1[3], accg6);
            int accg7 = ggml_cuda_dp4a(w_dec[buf][(gl * 4 + 0) * MUL_MAT_W4A8_OUT + 48 + lx], u1[0], 0);
            accg7 = ggml_cuda_dp4a(w_dec[buf][(gl * 4 + 1) * MUL_MAT_W4A8_OUT + 48 + lx], u1[1], accg7);
            accg7 = ggml_cuda_dp4a(w_dec[buf][(gl * 4 + 2) * MUL_MAT_W4A8_OUT + 48 + lx], u1[2], accg7);
            accg7 = ggml_cuda_dp4a(w_dec[buf][(gl * 4 + 3) * MUL_MAT_W4A8_OUT + 48 + lx], u1[3], accg7);
            acc0 = fmaf((float) accg0, s_dec[buf][gl * MUL_MAT_W4A8_OUT +        lx], acc0);
            acc1 = fmaf((float) accg1, s_dec[buf][gl * MUL_MAT_W4A8_OUT + 16 + lx], acc1);
            acc2 = fmaf((float) accg2, s_dec[buf][gl * MUL_MAT_W4A8_OUT + 32 + lx], acc2);
            acc3 = fmaf((float) accg3, s_dec[buf][gl * MUL_MAT_W4A8_OUT + 48 + lx], acc3);
            acc4 = fmaf((float) accg4, s_dec[buf][gl * MUL_MAT_W4A8_OUT +        lx], acc4);
            acc5 = fmaf((float) accg5, s_dec[buf][gl * MUL_MAT_W4A8_OUT + 16 + lx], acc5);
            acc6 = fmaf((float) accg6, s_dec[buf][gl * MUL_MAT_W4A8_OUT + 32 + lx], acc6);
            acc7 = fmaf((float) accg7, s_dec[buf][gl * MUL_MAT_W4A8_OUT + 48 + lx], acc7);
        }
        __syncthreads();
    }

    const float accv[8] = { acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7 };
#pragma unroll
    for (uint32_t t = 0; t < 2; ++t) {
        const uint32_t rt = t == 0 ? r0 : r1;
        if (rt < rows) {
#pragma unroll
            for (uint32_t j = 0; j < 4; ++j) {
                const uint32_t oj = o_base + j * 16 + lx;
                if (oj < n) {
                    float value = accv[t * 4 + j] * act_scales[rt];
                    if (bias != nullptr) {
                        value += bias[oj];
                    }
                    y[(size_t) rt * n + oj] = value;
                }
            }
        }
    }
}

// Route 2: RDNA3 WMMA path (gfx11 only). One V_WMMA_I32_16X16X16_IU8 per
// wave covers a whole 16x16x16 group: A = activations (lane j = tile row, 16
// sequential k bytes), B = decoded weights (lane j = output column, the same
// 256-entry pair-LUT gathers as the dp4a path but kept in registers), D =
// int32 partials (lane j = column, rows 4*dword + lane/16). No w_dec/x_tile/
// s_dec staging and no per-stage barriers - the k loop is barrier-free and
// LDS holds only the LUT, so occupancy is no longer LDS-bound. Raw w/x/s_rel
// loads are prefetched two groups ahead in registers so the global-load ->
// LUT-gather -> wmma chain never exposes L2 latency. Integer accumulation
// is exact and the per-group FFMA order matches the dp4a kernel, so outputs
// are bit-identical to it (and to Vulkan).
//
// The __gfx11xx__ macros exist only in the device pass; the host pass (where
// the <<<>>> launch lives) never sees them. GGML_W4A8_WMMA_GFX11 (set by the
// ggml-hip CMake when the target list contains gfx11) therefore also makes
// this block visible to the host pass so the launch stub is generated; the
// launch itself still checks the runtime cc for multi-arch binaries.
#if defined(__HIP_PLATFORM_AMD__) && \
    (defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || \
     defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__) || \
     (defined(GGML_W4A8_WMMA_GFX11) && !defined(__HIP_DEVICE_COMPILE__)))
typedef int v4si32_w4a8 __attribute__((ext_vector_type(4)));

static __global__ void __launch_bounds__(MUL_MAT_W4A8_BLOCK * MUL_MAT_W4A8_BLOCK) mul_mat_w4a8_convrot_wmma(
        const char * __restrict__ w, const char * __restrict__ x,
        const float * __restrict__ codebook, const float * __restrict__ s_channel,
        const uint8_t * __restrict__ s_rel, const float * __restrict__ act_scales,
        const float * __restrict__ bias, float * __restrict__ y,
        const uint32_t n, const uint32_t rows, const uint32_t k) {

    const uint32_t o_base = blockIdx.x * MUL_MAT_W4A8_OUT;
    const uint32_t r_base = blockIdx.y * MUL_MAT_W4A8_BLOCK;
    const uint32_t idx  = threadIdx.y * MUL_MAT_W4A8_BLOCK + threadIdx.x;
    const uint32_t wv   = idx >> 6;          // wave id: owns a 16-output slab
    const uint32_t lane = idx & 63;
    const uint32_t j    = lane & 15;         // fragment column == A row
    const uint32_t q    = lane >> 4;         // C rows are 4*dword + q
    const uint32_t ocol = o_base + wv * 16 + j;
    const uint32_t ra   = r_base + j;

    __shared__ __align__(4) int8_t cb8[16];
    __shared__ float s_cb;
    __shared__ int lut[256];
    __shared__ float s_lut[256]; // fp8_e4m3 -> float, decoded once (same
                                // conversion as ggml_fp8_e4m3_cuda per byte)

    const uint32_t w_words = k / 8;
    const uint32_t x_words = k / 4;
    const uint32_t groups  = k / 16;
    const int * wi = (const int *) w;
    const int * xi = (const int *) x;

    if (idx == 0) {
        float amax = 0.0f;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            amax = fmaxf(amax, fabsf(codebook[i]));
        }
        s_cb = amax / 127.0f;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int v = lrintf(codebook[i] / s_cb);
            cb8[i] = (int8_t) (v < -127 ? -127 : (v > 127 ? 127 : v));
        }
#pragma unroll
        for (int b = 0; b < 256; ++b) {
            lut[b] = (cb8[b & 0xF] & 0xFF) | ((cb8[b >> 4] & 0xFF) << 8);
        }
#pragma unroll
        for (int b = 0; b < 256; ++b) {
            ggml_fp8_e4m3_cuda f;
            f.value = (uint8_t) b;
            s_lut[b] = (float) f;
        }
    }
    __syncthreads();

    const bool  o_valid = ocol < n;
    const float s_chan  = o_valid ? s_channel[ocol] : 0.0f;
    const float scb     = s_cb;

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const v4si32_w4a8 zero = {};

    // prefetch double-buffer as scalars (runtime-indexed local arrays would
    // spill to scratch): cur is decoded this body, nxt is loaded two bodies
    // ahead so global-load latency never reaches the LUT-gather -> wmma chain
    int w0c = 0, w1c = 0, xc0 = 0, xc1 = 0, xc2 = 0, xc3 = 0, sc = 0;
    int w0n = 0, w1n = 0, xn0 = 0, xn1 = 0, xn2 = 0, xn3 = 0, sn = 0;
    {
        const int * xr = xi + (size_t) ra * x_words;
        xc0 = ra < rows ? xr[0] : 0;
        xc1 = ra < rows ? xr[1] : 0;
        xc2 = ra < rows ? xr[2] : 0;
        xc3 = ra < rows ? xr[3] : 0;
        if (o_valid) {
            const int * wp = wi + (size_t) ocol * w_words;
            w0c = wp[0];
            w1c = wp[1];
            sc  = s_rel[(size_t) ocol * groups];
        }
    }
    if (groups > 1) {
        const int * xr = xi + (size_t) ra * x_words + 4;
        xn0 = ra < rows ? xr[0] : 0;
        xn1 = ra < rows ? xr[1] : 0;
        xn2 = ra < rows ? xr[2] : 0;
        xn3 = ra < rows ? xr[3] : 0;
        if (o_valid) {
            const int * wp = wi + (size_t) ocol * w_words + 2;
            w0n = wp[0];
            w1n = wp[1];
            sn  = s_rel[(size_t) ocol * groups + 1];
        }
    }

    for (uint32_t g = 0; g < groups; ++g) {
        v4si32_w4a8 bfrag = {};
        float s = 0.0f;
        if (o_valid) {
            bfrag = {
                (lut[w0c        & 0xFF] & 0xFFFF) | ((lut[(w0c >>  8) & 0xFF] & 0xFFFF) << 16),
                (lut[(w0c >> 16) & 0xFF] & 0xFFFF) | ((lut[(w0c >> 24) & 0xFF] & 0xFFFF) << 16),
                (lut[w1c        & 0xFF] & 0xFFFF) | ((lut[(w1c >>  8) & 0xFF] & 0xFFFF) << 16),
                (lut[(w1c >> 16) & 0xFF] & 0xFFFF) | ((lut[(w1c >> 24) & 0xFF] & 0xFFFF) << 16),
            };
            // same value and multiply order as the dp4a kernel's
            // s_cb * (s_channel * (float)srel): bit-identical scales
            s = scb * (s_chan * s_lut[sc & 0xFF]);
        }
        const v4si32_w4a8 afrag = {xc0, xc1, xc2, xc3};
        const v4si32_w4a8 dfrag =
            __builtin_amdgcn_wmma_i32_16x16x16_iu8_w64(true, afrag, true, bfrag, zero, false);

        w0c = w0n; w1c = w1n; sc = sn;
        xc0 = xn0; xc1 = xn1; xc2 = xn2; xc3 = xn3;
        if (g + 2 < groups) {
            const int * xr = xi + (size_t) ra * x_words + (g + 2) * 4;
            xn0 = ra < rows ? xr[0] : 0;
            xn1 = ra < rows ? xr[1] : 0;
            xn2 = ra < rows ? xr[2] : 0;
            xn3 = ra < rows ? xr[3] : 0;
            if (o_valid) {
                const int * wp = wi + (size_t) ocol * w_words + (g + 2) * 2;
                w0n = wp[0];
                w1n = wp[1];
                sn  = s_rel[(size_t) ocol * groups + g + 2];
            }
        }

        acc[0] = fmaf((float) dfrag.x, s, acc[0]);
        acc[1] = fmaf((float) dfrag.y, s, acc[1]);
        acc[2] = fmaf((float) dfrag.z, s, acc[2]);
        acc[3] = fmaf((float) dfrag.w, s, acc[3]);
    }

    if (o_valid) {
#pragma unroll
        for (uint32_t d = 0; d < 4; ++d) {
            const uint32_t r = r_base + 4 * d + q;
            if (r < rows) {
                float value = acc[d] * act_scales[r];
                if (bias != nullptr) {
                    value += bias[ocol];
                }
                y[(size_t) r * n + ocol] = value;
            }
        }
    }
}
#endif // gfx11 wmma

// Route 4: RDNA4 WMMA path (gfx12 only). Port of the gfx11 kernel above to
// the gfx12 wave64 intrinsic; same register-direct structure (no w_dec/
// x_tile/s_dec staging, barrier-free k loop), same pair-LUT decode, same
// FFMA drain order, so outputs stay bit-identical to the dp4a path. The
// gfx12 wave64 fragment layout is leaner than gfx11's: each lane supplies
// exactly one 4-byte A and B slice (lane = 32*((k/4)%2) + 16*(k/8) +
// row/col, GPR byte k%4; AMD matrix calculator, RDNA4), so no replicated
// rows and 4x less decode work per wave. In lane terms (j = lane & 15,
// h = lane >> 4) the four h quarters hold k 0-3, 8-11, 4-7, 12-15 - a
// bit-swap kh of h. That one kh indexes the x dword (g*4 + kh), the weight
// gather pair (kh bit 1 picks w0/w1 word, kh bit 0 the half), and the D
// rows (4*kh + dword). The w64 intrinsic compiles under this file's
// -mwavefrontsize64 (verified on gfx1201: emits v_wmma_i32_16x16x16_iu8,
// neg_lo [1,1] for signed x signed), unlike the w32 variant which needs
// wavefrontsize32 and would fight the gfx11 wave64 tuning above.
// Compile-verified only - no RDNA4 hardware was available for the fixture
// gate or a bench; Route 4 research expects 1.3-1.7x over dp4a.
#if defined(__HIP_PLATFORM_AMD__) && \
    (defined(__gfx1200__) || defined(__gfx1201__) || \
     (defined(GGML_W4A8_WMMA_GFX12) && !defined(__HIP_DEVICE_COMPILE__)))
typedef int v4si32_w4a8_g12 __attribute__((ext_vector_type(4)));

static __global__ void __launch_bounds__(MUL_MAT_W4A8_BLOCK * MUL_MAT_W4A8_BLOCK) mul_mat_w4a8_convrot_wmma_gfx12(
        const char * __restrict__ w, const char * __restrict__ x,
        const float * __restrict__ codebook, const float * __restrict__ s_channel,
        const uint8_t * __restrict__ s_rel, const float * __restrict__ act_scales,
        const float * __restrict__ bias, float * __restrict__ y,
        const uint32_t n, const uint32_t rows, const uint32_t k) {

    const uint32_t o_base = blockIdx.x * MUL_MAT_W4A8_OUT;
    const uint32_t r_base = blockIdx.y * MUL_MAT_W4A8_BLOCK;
    const uint32_t idx  = threadIdx.y * MUL_MAT_W4A8_BLOCK + threadIdx.x;
    const uint32_t wv   = idx >> 6;          // wave id: owns a 16-output slab
    const uint32_t lane = idx & 63;
    const uint32_t j    = lane & 15;         // fragment column == A row
    const uint32_t h    = lane >> 4;         // k quarter this lane supplies
    const uint32_t kh   = ((h & 1) << 1) | (h >> 1); // quarter in k order: {0,2,1,3}
    const uint32_t ocol = o_base + wv * 16 + j;
    const uint32_t ra   = r_base + j;

    __shared__ __align__(4) int8_t cb8[16];
    __shared__ float s_cb;
    __shared__ int lut[256];
    __shared__ float s_lut[256]; // fp8_e4m3 -> float, decoded once (same
                                 // conversion as ggml_fp8_e4m3_cuda per byte)

    const uint32_t w_words = k / 8;
    const uint32_t x_words = k / 4;
    const uint32_t groups  = k / 16;
    const int * wi = (const int *) w;
    const int * xi = (const int *) x;

    if (idx == 0) {
        float amax = 0.0f;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            amax = fmaxf(amax, fabsf(codebook[i]));
        }
        s_cb = amax / 127.0f;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int v = lrintf(codebook[i] / s_cb);
            cb8[i] = (int8_t) (v < -127 ? -127 : (v > 127 ? 127 : v));
        }
#pragma unroll
        for (int b = 0; b < 256; ++b) {
            lut[b] = (cb8[b & 0xF] & 0xFF) | ((cb8[b >> 4] & 0xFF) << 8);
        }
#pragma unroll
        for (int b = 0; b < 256; ++b) {
            ggml_fp8_e4m3_cuda f;
            f.value = (uint8_t) b;
            s_lut[b] = (float) f;
        }
    }
    __syncthreads();

    const bool  o_valid = ocol < n;
    const float s_chan  = o_valid ? s_channel[ocol] : 0.0f;
    const float scb     = s_cb;

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const v4si32_w4a8_g12 zero = {};

    // prefetch double-buffer as scalars (runtime-indexed local arrays would
    // spill to scratch): cur is decoded this body, nxt is loaded two bodies
    // ahead so global-load latency never reaches the gather -> wmma chain
    int w0c = 0, w1c = 0, xc = 0, sc = 0;
    int w0n = 0, w1n = 0, xn = 0, sn = 0;
    {
        const int * xr = xi + (size_t) ra * x_words;
        xc = ra < rows ? xr[kh] : 0;
        if (o_valid) {
            const int * wp = wi + (size_t) ocol * w_words;
            w0c = wp[0];
            w1c = wp[1];
            sc  = s_rel[(size_t) ocol * groups];
        }
    }
    if (groups > 1) {
        const int * xr = xi + (size_t) ra * x_words;
        xn = ra < rows ? xr[4 + kh] : 0;
        if (o_valid) {
            const int * wp = wi + (size_t) ocol * w_words + 2;
            w0n = wp[0];
            w1n = wp[1];
            sn  = s_rel[(size_t) ocol * groups + 1];
        }
    }

    for (uint32_t g = 0; g < groups; ++g) {
        int bfrag = 0;
        float s = 0.0f;
        if (o_valid) {
            // this lane's k quarter picks the packed word (kh bit 1) and
            // the packed-byte pair inside it (kh bit 0); low nibble = even k
            const int bwd = (kh & 2) ? w1c : w0c;
            const int bsh = (kh & 1) * 16;
            bfrag = (lut[(bwd >> bsh) & 0xFF] & 0xFFFF) |
                    ((lut[(bwd >> (bsh + 8)) & 0xFF] & 0xFFFF) << 16);
            // same value and multiply order as the dp4a kernel's
            // s_cb * (s_channel * (float)srel): bit-identical scales
            s = scb * (s_chan * s_lut[sc & 0xFF]);
        }
        const v4si32_w4a8_g12 dfrag =
            __builtin_amdgcn_wmma_i32_16x16x16_iu8_w64_gfx12(true, xc, true, bfrag, zero, false);

        w0c = w0n; w1c = w1n; sc = sn;
        xc = xn;
        if (g + 2 < groups) {
            const int * xr = xi + (size_t) ra * x_words;
            xn = ra < rows ? xr[(g + 2) * 4 + kh] : 0;
            if (o_valid) {
                const int * wp = wi + (size_t) ocol * w_words + (g + 2) * 2;
                w0n = wp[0];
                w1n = wp[1];
                sn  = s_rel[(size_t) ocol * groups + g + 2];
            }
        }

        acc[0] = fmaf((float) dfrag.x, s, acc[0]);
        acc[1] = fmaf((float) dfrag.y, s, acc[1]);
        acc[2] = fmaf((float) dfrag.z, s, acc[2]);
        acc[3] = fmaf((float) dfrag.w, s, acc[3]);
    }

    if (o_valid) {
#pragma unroll
        for (uint32_t d = 0; d < 4; ++d) {
            const uint32_t r = r_base + 4 * kh + d;
            if (r < rows) {
                float value = acc[d] * act_scales[r];
                if (bias != nullptr) {
                    value += bias[ocol];
                }
                y[(size_t) r * n + ocol] = value;
            }
        }
    }
}
#endif // gfx12 wmma

void ggml_cuda_mul_mat_w4a4_convrot_dp4a(ggml_backend_cuda_context & ctx, const int8_t * w, const int8_t * x,
        int32_t * y, int64_t n, int64_t rows, int64_t packed_k, cudaStream_t stream) {

    GGML_UNUSED(ctx);

    const dim3 block_nums((n + MUL_MAT_W4_OUT - 1) / MUL_MAT_W4_OUT,
                          (rows + MUL_MAT_W4_ROWS - 1) / MUL_MAT_W4_ROWS);
    const dim3 block_dims(MUL_MAT_W4_BLOCK, MUL_MAT_W4_BLOCK, 1);

    mul_mat_w4a4_convrot_dp4a<<<block_nums, block_dims, 0, stream>>>(
        (const char *) w, (const char *) x, y, (uint32_t) n, (uint32_t) rows, (uint32_t) packed_k);
}

void ggml_cuda_mul_mat_w4a8_convrot_dp4a(ggml_backend_cuda_context & ctx, const int8_t * w, const int8_t * x,
        const float * codebook, const float * s_channel, const uint8_t * s_rel, const float * act_scales,
        const float * bias, float * y, int64_t n, int64_t rows, int64_t k, cudaStream_t stream) {

    GGML_UNUSED(ctx);

    const dim3 block_nums((n + MUL_MAT_W4A8_OUT - 1) / MUL_MAT_W4A8_OUT,
                          (rows + MUL_MAT_W4A8_BLOCK - 1) / MUL_MAT_W4A8_BLOCK);
    const dim3 block_dims(MUL_MAT_W4A8_BLOCK, MUL_MAT_W4A8_BLOCK, 1);

#if defined(GGML_USE_HIP) && (defined(GGML_W4A8_WMMA_GFX11) || defined(GGML_W4A8_WMMA_GFX12))
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
#endif
#if defined(GGML_USE_HIP) && defined(GGML_W4A8_WMMA_GFX11)
    // gfx11 selection: the compile-time define says this build carries gfx11
    // device images; the runtime cc check keeps multi-arch binaries on dp4a
    // for non-gfx11 GPUs. Logged once so every run's log shows the kernel.
    const bool wmma11 = GGML_CUDA_CC_IS_RDNA3_0(cc) || GGML_CUDA_CC_IS_RDNA3_5(cc);
#else
    const bool wmma11 = false;
#endif
#if defined(GGML_USE_HIP) && defined(GGML_W4A8_WMMA_GFX12)
    // gfx12 selection: RDNA4 gets the wave64 gfx12 kernel. Compile-verified
    // only; the runtime cc guard keeps every other GPU on dp4a.
    const bool wmma12 = GGML_CUDA_CC_IS_RDNA4(cc);
#else
    const bool wmma12 = false;
#endif
    static bool logged = false;
    if (!logged) {
        logged = true;
        GGML_LOG_DEBUG("w4a8 convrot kernel: %s\n",
                       wmma12 ? "wmma (gfx12)" : (wmma11 ? "wmma (gfx11)" : "dp4a"));
    }
#if defined(GGML_USE_HIP) && defined(GGML_W4A8_WMMA_GFX12)
    if (wmma12) {
        mul_mat_w4a8_convrot_wmma_gfx12<<<block_nums, block_dims, 0, stream>>>(
            (const char *) w, (const char *) x, codebook, s_channel, s_rel, act_scales, bias, y,
            (uint32_t) n, (uint32_t) rows, (uint32_t) k);
        return;
    }
#endif
#if defined(GGML_USE_HIP) && defined(GGML_W4A8_WMMA_GFX11)
    if (wmma11) {
        mul_mat_w4a8_convrot_wmma<<<block_nums, block_dims, 0, stream>>>(
            (const char *) w, (const char *) x, codebook, s_channel, s_rel, act_scales, bias, y,
            (uint32_t) n, (uint32_t) rows, (uint32_t) k);
        return;
    }
#endif
    // dp4a path only: the CTA covers a 32-row tile (two 16-row tiles share
    // the w decode); the wmma kernels above keep the 16-row grid
    const dim3 block_nums_dp4a((n + MUL_MAT_W4A8_OUT - 1) / MUL_MAT_W4A8_OUT,
                               (rows + MUL_MAT_W4A8_ROWS - 1) / MUL_MAT_W4A8_ROWS);
    mul_mat_w4a8_convrot_dp4a<<<block_nums_dp4a, block_dims, 0, stream>>>(
        (const char *) w, (const char *) x, codebook, s_channel, s_rel, act_scales, bias, y,
        (uint32_t) n, (uint32_t) rows, (uint32_t) k);
}
