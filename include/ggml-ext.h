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


#ifdef __cplusplus
}
#endif

#endif
