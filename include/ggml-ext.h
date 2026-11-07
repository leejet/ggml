#ifndef GGML_EXT_H
#define GGML_EXT_H

#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_EXT_ABI_VERSION 1

typedef const struct ggml_ext_op * ggml_ext_op_t;

struct ggml_ext_tensor_desc {
    enum ggml_type type;
    int64_t ne[GGML_MAX_DIMS];
};

// Registrations and callback code must outlive all graphs and backends using them.
// Operators are pure. Attributes are copied into op_params; tensor dependencies
// must be explicit inputs. infer must fill all four positive output dimensions.
struct ggml_ext_op_desc {
    int abi_version;
    const char * name;
    size_t attributes_size;
    bool (*infer)(struct ggml_tensor * const * inputs, int n_inputs, const void * attributes,
                  struct ggml_ext_tensor_desc * output);
};

GGML_API ggml_ext_op_t ggml_ext_register_op(const struct ggml_ext_op_desc * desc);
GGML_API struct ggml_tensor * ggml_ext_op_create(struct ggml_context * ctx, ggml_ext_op_t op,
        struct ggml_tensor * const * inputs, int n_inputs, const void * attributes);
GGML_API ggml_ext_op_t ggml_ext_op_get(const struct ggml_tensor * tensor);
GGML_API const void * ggml_ext_op_attributes(const struct ggml_tensor * tensor);

struct ggml_ext_execution_context {
    int abi_version;
    ggml_backend_t backend;
    ggml_backend_buffer_t workspace;
};

// Single-node supports/preferred and fusion match inspect metadata only. They must
// not call ggml_backend_dev_supports_op recursively. A match consumes consecutive,
// pure operators and writes only the final node, which may be a CPY. In-place
// intermediates are allowed only when their storage is created within the match.
// An in-place final node must overwrite that storage's complete layout without
// external consumers of intermediate values.
// Fused intermediates may have no data or buffer; execute must not access their
// storage. Storage backing the final output is retained, including internal views.
// A terminal CPY retains its write semantics; execute must handle input/output
// storage overlap (including allocator reuse), using workspace when needed.
// Returning zero declines.
struct ggml_ext_kernel_desc {
    int abi_version;
    const char * name;
    enum ggml_op op;
    ggml_ext_op_t custom_op;
    int priority;
    bool (*supports)(ggml_backend_dev_t device, const struct ggml_tensor * node, void * user_data);
    bool (*preferred)(ggml_backend_dev_t device, const struct ggml_tensor * node, void * user_data);
    int (*match)(ggml_backend_dev_t device, const struct ggml_cgraph * graph, int first, void * user_data);
    // Optional pre-allocation reordering. Preserve graph edges and tensor metadata;
    // use ggml_ext_graph_delay to move independent producers safely.
    void (*optimize)(ggml_backend_dev_t device, struct ggml_cgraph * graph, void * user_data);
    // prepare may run during measurement: host state only, no execution/device allocation.
    enum ggml_status (*prepare)(ggml_backend_t backend, struct ggml_tensor * const * nodes, int n_nodes,
                               void ** state, size_t * workspace_size, void * user_data);
    enum ggml_status (*execute)(const struct ggml_ext_execution_context * context,
                               struct ggml_tensor * const * nodes, int n_nodes, void * state, void * user_data);
    // Paired with every prepare call, including failures and null state.
    void (*destroy)(void * state, void * user_data);
    void * user_data;
};

// Register before querying device capabilities or allocating graphs. No unloading
// or replacing registrations while graphs are alive. Higher priorities run first.
GGML_API bool ggml_ext_register_kernel(ggml_backend_dev_t device, const struct ggml_ext_kernel_desc * desc);
enum ggml_ext_kernel_flags {
    // execute enqueues replayable device work. After warmup, recording/replay may
    // omit host calls. No required per-execution host side effects or decisions
    // based on tensor contents; private streams must rejoin the borrowed stream.
    GGML_EXT_KERNEL_CAPTURE_SAFE = 1,
    // Requires CAPTURE_SAFE. Replay may outlive prepare/destroy and be reused by
    // an equivalent rebuilt plan. Device work depends only on tensor metadata,
    // addresses, the device and immutable registration data. Captured resources
    // must belong to the backend or supplied workspace, never per-plan state.
    // Workspace is scratch initialized by recorded work on every execution.
    GGML_EXT_KERNEL_REPLAY_REUSABLE = 2,
};
// Existing registrations retain per-execution callbacks (flags = 0).
GGML_API bool ggml_ext_register_kernel_ex(ggml_backend_dev_t device,
        const struct ggml_ext_kernel_desc * desc, uint32_t flags);
GGML_API bool ggml_ext_dev_supports_native_op(ggml_backend_dev_t device, const struct ggml_tensor * node);
// Delay [first, first + count) until immediately before `before`. Refuses aliasing writes,
// dependencies on the moved nodes, and unknown side effects in the crossed range.
GGML_API bool ggml_ext_graph_delay(struct ggml_cgraph * graph, int first, int count, int before);

// Query "ggml_ext_cuda_context_v1" through the CUDA registry's get_proc_address.
// The returned stream is borrowed, ordered with native work, and must not be freed.
struct ggml_ext_cuda_context {
    // Physical runtime ordinal, including when ggml exposes virtual devices.
    int device;
    void * stream;
};
typedef bool (*ggml_ext_cuda_context_fn)(ggml_backend_t backend, struct ggml_ext_cuda_context * context);
// Query "ggml_ext_cuda_device_v1" during capability checks, before backend creation.
// Returns the physical runtime ordinal, or -1 for a non-CUDA device.
typedef int (*ggml_ext_cuda_device_fn)(ggml_backend_dev_t device);

// Query "ggml_ext_cuda_blas_v1" during execute to borrow the current stream's
// cuBLAS handle, with host scalar pointers. Do not retain/destroy it or change
// its stream, pointer mode, math mode, or workspace. CUDA only.
typedef void * (*ggml_ext_cuda_blas_fn)(ggml_backend_t backend);

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


// Query "ggml_ext_cpu_parallel_v1" to use the backend's configured thread pool.
// The callback receives a temporary output descriptor; userdata can reference the
// original nodes. Execution is synchronous and honors the CPU abort callback.
typedef enum ggml_status (*ggml_ext_cpu_parallel_fn)(ggml_backend_t backend, struct ggml_tensor * output,
                                                   ggml_custom_op_t callback, void * user_data);

#ifdef __cplusplus
}
#endif

#endif
