#ifndef GGML_EXT_IMPL_H
#define GGML_EXT_IMPL_H

#include "ggml-ext.h"
#include "ggml-backend-impl.h"

struct ggml_ext_plan;

// Optional backend-owned executor for plans whose kernels permit device replay.
// Query "ggml_ext_graph_executor_v1"; reusable state can outlive a plan and must
// not retain graph pointers. Destruction follows synchronization. compute must
// propagate callback errors without retrying execution.
// Query "ggml_ext_graph_batch_v1" for submission batching without replay. Its
// compute must invoke the callback exactly once per call; state is never reused
// across plans. Batching does not require capture-safe extension kernels.
typedef ggml_status (*ggml_ext_graph_callback)(void * user_data);
struct ggml_ext_graph_executor {
    void * (*create)(ggml_backend_t backend);
    ggml_status (*compute)(void * state, ggml_cgraph * graph, ggml_ext_graph_callback callback, void * user_data);
    void (*destroy)(void * state);
};
typedef const ggml_ext_graph_executor * (*ggml_ext_graph_executor_fn)(ggml_backend_t backend);

bool ggml_ext_dev_supports_op(ggml_backend_dev_t device, const ggml_tensor * node);
ggml_ext_plan * ggml_ext_plan_create(ggml_backend_t backend, ggml_cgraph * graph,
        ggml_backend_graph_optimize_params * params, bool allow_fusion);
void ggml_ext_plan_free(ggml_ext_plan * plan);
void ggml_ext_backend_free(ggml_backend_t backend);
size_t ggml_ext_plan_workspace_size(const ggml_ext_plan * plan);
ggml_status ggml_ext_plan_compute(ggml_ext_plan * plan, ggml_cgraph * graph);

#endif
