#ifndef GGML_EXT_IMPL_H
#define GGML_EXT_IMPL_H

#include "ggml-ext.h"
#include "ggml-backend-impl.h"

struct ggml_ext_plan;

bool ggml_ext_dev_supports_op(ggml_backend_dev_t device, const ggml_tensor * node);
ggml_ext_plan * ggml_ext_plan_create(ggml_backend_t backend, ggml_cgraph * graph,
        ggml_backend_graph_optimize_params * params, bool allow_fusion);
void ggml_ext_plan_free(ggml_ext_plan * plan);
size_t ggml_ext_plan_workspace_size(const ggml_ext_plan * plan);
ggml_status ggml_ext_plan_compute(ggml_ext_plan * plan, ggml_cgraph * graph);

#endif
