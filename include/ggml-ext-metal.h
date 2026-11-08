#ifndef GGML_EXT_METAL_H
#define GGML_EXT_METAL_H

#include "ggml-ext.h"

#ifdef __cplusplus
extern "C" {
#endif

// Query these services through the Metal registry's get_proc_address. Objective-C
// objects are borrowed and must not be released by the caller.
typedef void * (*ggml_ext_metal_device_fn)(ggml_backend_dev_t device); // id<MTLDevice>

struct ggml_ext_metal_buffer {
    void * buffer; // id<MTLBuffer>
    size_t offset;
};
typedef bool (*ggml_ext_metal_buffer_fn)(ggml_backend_buffer_t buffer, size_t offset, size_t size,
                                       struct ggml_ext_metal_buffer * result);

// Encode only: do not enqueue, commit, wait for, or retain the command buffer.
// Successful submissions are ordered with native work and tracked by backend
// synchronization, including error reporting. Failed encoding is not submitted.
typedef enum ggml_status (*ggml_ext_metal_encode_fn)(void * command_buffer, void * user_data);
typedef enum ggml_status (*ggml_ext_metal_submit_fn)(ggml_backend_t backend,
                                                   ggml_ext_metal_encode_fn encode, void * user_data);

#ifdef __cplusplus
}
#endif
#endif
