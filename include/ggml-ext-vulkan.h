#ifndef GGML_EXT_VULKAN_H
#define GGML_EXT_VULKAN_H

#include "ggml-ext.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_EXT_VULKAN_ABI_VERSION 1

struct ggml_ext_vulkan_device_info {
    uint32_t max_group_count[3];
    uint32_t max_group_size[3];
    uint32_t max_group_invocations;
    uint32_t max_push_constants;
    uint64_t max_buffer_range;
    uint64_t buffer_alignment;
    uint32_t subgroup_size;
    uint32_t subgroup_min_size;
    uint32_t subgroup_max_size;
    bool subgroup_size_control;
    bool full_subgroups;
    bool subgroup_shuffle;
    bool float16;
    bool integer_dot_product;
    bool fusion_disabled;
};

// Metadata-only services queried as "ggml_ext_vulkan_compute_v1".
struct ggml_ext_vulkan_compute_info {
    uint32_t shared_memory;
    bool subgroup_arithmetic;
    bool graph_optimization_disabled;
};

struct ggml_ext_vulkan_matrix_properties {
    enum ggml_type a, b, accumulator, result;
    // Exact dimensions for subgroup matrices, granularities for flexible matrices.
    uint32_t m, n, k;
    // Workgroup invocation count, or the device subgroup size for subgroup scope.
    uint32_t invocations;
    bool workgroup_scope;
    bool flexible;
    bool saturating;
};

struct ggml_ext_vulkan_compute_api {
    bool (*properties)(ggml_backend_dev_t device, struct ggml_ext_vulkan_compute_info * info);
    // Enumerates enabled, representable matrix types; false marks the end or an invalid request.
    bool (*matrix_properties)(ggml_backend_dev_t device, uint32_t index,
                               struct ggml_ext_vulkan_matrix_properties * properties);
};
typedef const struct ggml_ext_vulkan_compute_api * (*ggml_ext_vulkan_compute_api_fn)(void);

// SPIR-V uses entry point "main", contiguous storage-buffer bindings starting
// at zero, and uint32 specialization constants with consecutive constant IDs.
// The backend caches a copy of the complete shader/configuration until device
// destruction. No Vulkan object or callback-owned resource is retained.
struct ggml_ext_vulkan_pipeline {
    const char * name;
    const uint32_t * spirv;
    size_t spirv_size;
    uint32_t binding_count;
    uint32_t push_constant_size;
    const uint32_t * constants;
    uint32_t constant_count;
    uint32_t required_subgroup_size;
    bool full_subgroups;
};

// Specify exactly one of tensor or buffer. offset/size are bytes relative to
// that object; size == 0 selects the remainder. A buffer may be plan workspace.
struct ggml_ext_vulkan_binding {
    const struct ggml_tensor * tensor;
    ggml_backend_buffer_t buffer;
    size_t offset;
    size_t size;
};

struct ggml_ext_vulkan_buffer_info {
    // Opaque identity for overlap checks only; never dereference or retain it.
    const void * storage;
    uint64_t offset;
    uint64_t size;
    // Descriptor offsets are rounded down. Add this byte offset in the shader.
    uint32_t shader_offset;
};

// Query "ggml_ext_vulkan_v1" through the Vulkan registry's get_proc_address.
// device_info is metadata-only. Other methods are execution-only, ordered with
// native work on this backend, and require the normal backend synchronization
// before host reads or resource release. Backends are not concurrently callable.
// Invalid requests fail before recording work; dispatch errors never fall back
// after partially executing a kernel. Shader bounds and device capabilities
// remain the extension's responsibility.
struct ggml_ext_vulkan_api {
    int abi_version;
    bool (*device_info)(ggml_backend_dev_t device, struct ggml_ext_vulkan_device_info * info);
    bool (*buffer_info)(ggml_backend_t backend, const struct ggml_ext_vulkan_binding * binding,
                        struct ggml_ext_vulkan_buffer_info * info);
    enum ggml_status (*dispatch)(ggml_backend_t backend, const struct ggml_ext_vulkan_pipeline * pipeline,
                                const struct ggml_ext_vulkan_binding * bindings, const void * push_constants,
                                const uint32_t group_count[3]);
    // Copies equal-size nonoverlapping ranges; Vulkan requires four-byte alignment.
    enum ggml_status (*copy)(ggml_backend_t backend, const struct ggml_ext_vulkan_binding * src,
                            const struct ggml_ext_vulkan_binding * dst);
};

typedef const struct ggml_ext_vulkan_api * (*ggml_ext_vulkan_api_fn)(void);

#ifdef __cplusplus
}
#endif

#endif
