#ifndef GGML_EXT_GPU_H
#define GGML_EXT_GPU_H

#include "ggml-ext.h"

enum ggml_ext_gpu_kind {
    GGML_EXT_GPU_CUDA,
    GGML_EXT_GPU_HIP,
    GGML_EXT_GPU_MUSA,
};

struct ggml_ext_gpu_device_info {
    enum ggml_ext_gpu_kind kind;
    int device;
    int warp_size;
    size_t shared_memory;
    bool int8_blas;
    int cc;
    int nsm;
    int max_threads_per_sm;
};
// Query "ggml_ext_gpu_device_v1" for metadata without creating execution state.
typedef bool (*ggml_ext_gpu_device_fn)(ggml_backend_dev_t device, struct ggml_ext_gpu_device_info * info);

struct ggml_ext_gpu_context {
    enum ggml_ext_gpu_kind kind;
    int device;
    void * stream;
};
// Query "ggml_ext_gpu_context_v1" during execution. The borrowed stream belongs
// to the reported runtime; application kernels and runtime calls must use it.
typedef bool (*ggml_ext_gpu_context_fn)(ggml_backend_t backend, struct ggml_ext_gpu_context * context);
// Query "ggml_ext_gpu_blas_v1" to borrow the matching cuBLAS/hipBLAS/muBLAS
// handle on that stream with host scalar pointers. Do not retain/destroy it or
// change its stream, pointer mode, math mode, or workspace.
typedef void * (*ggml_ext_gpu_blas_fn)(ggml_backend_t backend);

#endif
