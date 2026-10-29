#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include "fattn-mma-f16.cuh"

DECL_FATTN_MMA_F16_CASE(128, 128, 128, 1);
#endif
