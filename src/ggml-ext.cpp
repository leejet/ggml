#include "ggml-ext-impl.h"
#include "ggml-impl.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

struct ggml_ext_op {
    std::string name;
    ggml_ext_op_desc desc;
};

struct ggml_ext_kernel {
    std::string name;
    ggml_ext_kernel_desc desc;
    uint32_t flags;
};

struct ggml_ext_registry {
    std::mutex mutex;
    std::vector<std::unique_ptr<ggml_ext_op>> ops;
    std::unordered_map<ggml_backend_dev_t, std::vector<std::unique_ptr<ggml_ext_kernel>>> kernels;
};

static ggml_ext_registry & ggml_ext_registry_get() {
    static ggml_ext_registry registry;
    return registry;
}

static void ggml_ext_unhandled(ggml_tensor *, int, int, void *) {
    GGML_ABORT("registered operations require the backend graph execution API");
}

ggml_ext_op_t ggml_ext_register_op(const ggml_ext_op_desc * desc) {
    if (!desc || desc->abi_version != GGML_EXT_ABI_VERSION || !desc->name || !desc->name[0] || !desc->infer ||
        desc->attributes_size > GGML_MAX_OP_PARAMS - sizeof(ggml_custom_op_params)) {
        return nullptr;
    }
    auto & registry = ggml_ext_registry_get();
    std::lock_guard<std::mutex> lock(registry.mutex);
    for (const auto & op : registry.ops) {
        if (op->name == desc->name) {
            return nullptr;
        }
    }
    std::unique_ptr<ggml_ext_op> op(new ggml_ext_op);
    op->name = desc->name;
    op->desc = *desc;
    op->desc.name = op->name.c_str();
    auto result = op.get();
    registry.ops.push_back(std::move(op));
    return result;
}

ggml_tensor * ggml_ext_op_create(ggml_context * ctx, ggml_ext_op_t op, ggml_tensor * const * inputs,
                                int n_inputs, const void * attributes) {
    if (!ctx || !op || n_inputs < 0 || n_inputs >= GGML_MAX_SRC ||
        (n_inputs && !inputs) || (op->desc.attributes_size && !attributes)) {
        return nullptr;
    }
    for (int i = 0; i < n_inputs; ++i) {
        if (!inputs[i]) {
            return nullptr;
        }
    }
    ggml_ext_tensor_desc output = {};
    if (!op->desc.infer(inputs, n_inputs, attributes, &output) ||
        output.type < 0 || output.type >= GGML_TYPE_COUNT) {
        return nullptr;
    }
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (output.ne[i] <= 0) {
            return nullptr;
        }
    }
    ggml_tensor * result = ggml_custom_4d(ctx, output.type, output.ne[0], output.ne[1], output.ne[2], output.ne[3],
            const_cast<ggml_tensor **>(inputs), n_inputs, ggml_ext_unhandled, 1, const_cast<ggml_ext_op *>(op));
    if (op->desc.attributes_size) {
        memcpy((char *) result->op_params + sizeof(ggml_custom_op_params), attributes, op->desc.attributes_size);
    }
    return result;
}

ggml_ext_op_t ggml_ext_op_get(const ggml_tensor * tensor) {
    if (!tensor || tensor->op != GGML_OP_CUSTOM) {
        return nullptr;
    }
    ggml_custom_op_params params;
    memcpy(&params, tensor->op_params, sizeof(params));
    return params.fun == ggml_ext_unhandled ? (ggml_ext_op_t) params.userdata : nullptr;
}

const void * ggml_ext_op_attributes(const ggml_tensor * tensor) {
    return ggml_ext_op_get(tensor) ? (const char *) tensor->op_params + sizeof(ggml_custom_op_params) : nullptr;
}

bool ggml_ext_register_kernel(ggml_backend_dev_t device, const ggml_ext_kernel_desc * desc) {
    return ggml_ext_register_kernel_ex(device, desc, 0);
}

bool ggml_ext_register_kernel_ex(ggml_backend_dev_t device, const ggml_ext_kernel_desc * desc, uint32_t flags) {
    if (!device || !desc || desc->abi_version != GGML_EXT_ABI_VERSION || !desc->name || !desc->name[0] ||
        !desc->execute || (!desc->supports && !desc->match) || desc->op <= GGML_OP_NONE || desc->op >= GGML_OP_COUNT ||
        (desc->custom_op && desc->op != GGML_OP_CUSTOM) || (desc->prepare && !desc->destroy) ||
        (flags & ~uint32_t(GGML_EXT_KERNEL_CAPTURE_SAFE | GGML_EXT_KERNEL_REPLAY_REUSABLE)) ||
        ((flags & GGML_EXT_KERNEL_REPLAY_REUSABLE) && !(flags & GGML_EXT_KERNEL_CAPTURE_SAFE))) {
        return false;
    }
    auto & registry = ggml_ext_registry_get();
    std::lock_guard<std::mutex> lock(registry.mutex);
    auto & kernels = registry.kernels[device];
    for (const auto & kernel : kernels) {
        if (kernel->name == desc->name) {
            return false;
        }
    }
    std::unique_ptr<ggml_ext_kernel> kernel(new ggml_ext_kernel);
    kernel->name = desc->name;
    kernel->desc = *desc;
    kernel->flags = flags;
    kernel->desc.name = kernel->name.c_str();
    kernels.push_back(std::move(kernel));
    std::stable_sort(kernels.begin(), kernels.end(), [](const std::unique_ptr<ggml_ext_kernel> & a,
                                                       const std::unique_ptr<ggml_ext_kernel> & b) {
        return a->desc.priority > b->desc.priority;
    });
    return true;
}

static std::vector<const ggml_ext_kernel *> ggml_ext_kernels(ggml_backend_dev_t device,
                                                          ggml_op op = GGML_OP_COUNT, bool single_node = false) {
    auto & registry = ggml_ext_registry_get();
    std::lock_guard<std::mutex> lock(registry.mutex);
    std::vector<const ggml_ext_kernel *> result;
    auto it = registry.kernels.find(device);
    if (it != registry.kernels.end()) {
        for (const auto & kernel : it->second) {
            if ((op == GGML_OP_COUNT || kernel->desc.op == op) && (!single_node || kernel->desc.supports)) {
                result.push_back(kernel.get());
            }
        }
    }
    return result;
}

static bool ggml_ext_kernel_matches(const ggml_ext_kernel_desc & desc, const ggml_tensor * node) {
    return node->op == desc.op && !node->view_src &&
        (node->op != GGML_OP_CUSTOM || ggml_ext_op_get(node) == desc.custom_op);
}

bool ggml_ext_dev_supports_op(ggml_backend_dev_t device, const ggml_tensor * node) {
    if (node->op == GGML_OP_NONE || node->view_src) {
        return false;
    }
    for (const auto * kernel : ggml_ext_kernels(device, node->op, true)) {
        const auto & desc = kernel->desc;
        if (ggml_ext_kernel_matches(desc, node) && desc.supports && desc.supports(device, node, desc.user_data)) {
            return true;
        }
    }
    return false;
}

struct ggml_ext_step {
    int first;
    uint64_t native_uid;
    std::vector<ggml_tensor *> nodes;
    const ggml_ext_kernel * kernel;
    void * state = nullptr;
    bool prepared = false;
    size_t workspace_size = 0;
};

struct ggml_ext_plan {
    ggml_backend_buffer_t workspace = nullptr;
    ggml_backend_t backend;
    ggml_status status = GGML_STATUS_SUCCESS;
    std::vector<ggml_ext_step> steps;
    uint64_t tail_uid;
    const ggml_ext_graph_executor * executor = nullptr;
    void * executor_state = nullptr;
    bool reusable = false;
    bool executed = false;
    std::vector<char> replay_key;
};

struct ggml_ext_replay {
    std::vector<char> key;
    const ggml_ext_graph_executor * executor;
    void * state;
    ggml_backend_buffer_t workspace;
    size_t workspace_size;
};

struct ggml_ext_replay_cache {
    std::mutex mutex;
    std::unordered_map<ggml_backend_t, std::vector<ggml_ext_replay>> entries;
};

static ggml_ext_replay_cache & ggml_ext_replay_cache_get() {
    static ggml_ext_replay_cache cache;
    return cache;
}

static void ggml_ext_replay_free(ggml_ext_replay & replay) {
    replay.executor->destroy(replay.state);
    ggml_backend_buffer_free(replay.workspace);
}

void ggml_ext_backend_free(ggml_backend_t backend) {
    auto & cache = ggml_ext_replay_cache_get();
    std::vector<ggml_ext_replay> entries;
    {
        std::lock_guard<std::mutex> lock(cache.mutex);
        auto it = cache.entries.find(backend);
        if (it != cache.entries.end()) {
            entries = std::move(it->second);
            cache.entries.erase(it);
        }
    }
    if (!entries.empty()) {
        ggml_backend_synchronize(backend);
        for (auto & replay : entries) {
            ggml_ext_replay_free(replay);
        }
    }
}

template<typename T>
static void ggml_ext_key_append(std::vector<char> & key, const T & value) {
    const auto * bytes = reinterpret_cast<const char *>(&value);
    key.insert(key.end(), bytes, bytes + sizeof(value));
}

static std::vector<char> ggml_ext_replay_key(const ggml_ext_plan * plan, const ggml_cgraph * graph) {
    std::vector<char> key;
    std::vector<const ggml_tensor *> tensors;
    std::unordered_map<const ggml_tensor *, size_t> indices;
    key.reserve(size_t(graph->n_nodes) * sizeof(ggml_tensor));
    tensors.reserve(graph->n_nodes);
    indices.reserve(graph->n_nodes);
    const auto index = [&](const ggml_tensor * tensor) {
        if (!tensor) {
            return SIZE_MAX;
        }
        auto result = indices.emplace(tensor, tensors.size());
        if (result.second) {
            tensors.push_back(tensor);
        }
        return result.first->second;
    };
    ggml_ext_key_append(key, graph->n_nodes);
    for (int i = 0; i < graph->n_nodes; ++i) {
        ggml_ext_key_append(key, index(graph->nodes[i]));
    }
    for (size_t i = 0; i < tensors.size(); ++i) {
        const auto * tensor = tensors[i];
        ggml_ext_key_append(key, tensor->type);
        ggml_ext_key_append(key, tensor->ne);
        ggml_ext_key_append(key, tensor->nb);
        ggml_ext_key_append(key, tensor->op);
        if (const auto * op = ggml_ext_op_get(tensor)) {
            ggml_ext_key_append(key, op);
            const auto * attrs = static_cast<const char *>(ggml_ext_op_attributes(tensor));
            key.insert(key.end(), attrs, attrs + op->desc.attributes_size);
        } else {
            ggml_ext_key_append(key, tensor->op_params);
        }
        ggml_ext_key_append(key, tensor->flags);
        ggml_ext_key_append(key, tensor->name);
        ggml_ext_key_append(key, tensor->data);
        ggml_ext_key_append(key, tensor->buffer);
        ggml_ext_key_append(key, tensor->extra);
        ggml_ext_key_append(key, tensor->view_offs);
        ggml_ext_key_append(key, index(tensor->view_src));
        for (const auto * src : tensor->src) {
            ggml_ext_key_append(key, index(src));
        }
    }
    ggml_ext_key_append(key, plan->steps.size());
    for (const auto & step : plan->steps) {
        ggml_ext_key_append(key, step.first);
        ggml_ext_key_append(key, step.nodes.size());
        ggml_ext_key_append(key, step.kernel);
        ggml_ext_key_append(key, step.workspace_size);
    }
    return key;
}

static void ggml_ext_replay_restore(ggml_ext_plan * plan) {
    auto & cache = ggml_ext_replay_cache_get();
    ggml_ext_replay replay = {};
    {
        std::lock_guard<std::mutex> lock(cache.mutex);
        auto found = cache.entries.find(plan->backend);
        if (found != cache.entries.end()) {
            auto & entries = found->second;
            for (auto it = entries.begin(); it != entries.end(); ++it) {
                if (it->executor == plan->executor && it->key == plan->replay_key) {
                    replay = std::move(*it);
                    entries.erase(it);
                    break;
                }
            }
        }
    }
    if (replay.state) {
        plan->executor->destroy(plan->executor_state);
        plan->executor_state = replay.state;
        plan->workspace = replay.workspace;
    }
}

static void ggml_ext_replay_store(ggml_ext_plan * plan) {
    const size_t budget = 256 * 1024 * 1024;
    const size_t size = plan->workspace ? ggml_backend_buffer_get_size(plan->workspace) : 0;
    if (size > budget) {
        return;
    }
    auto & cache = ggml_ext_replay_cache_get();
    std::vector<ggml_ext_replay> evicted;
    {
        std::lock_guard<std::mutex> lock(cache.mutex);
        auto & entries = cache.entries[plan->backend];
        size_t bytes = size;
        for (const auto & entry : entries) {
            bytes += entry.workspace_size;
        }
        while (!entries.empty() && (entries.size() >= 16 || bytes > budget)) {
            bytes -= entries.front().workspace_size;
            evicted.push_back(std::move(entries.front()));
            entries.erase(entries.begin());
        }
        entries.push_back({std::move(plan->replay_key), plan->executor, plan->executor_state, plan->workspace, size});
    }
    for (auto & replay : evicted) {
        ggml_ext_replay_free(replay);
    }
    plan->executor_state = nullptr;
    plan->workspace = nullptr;
}

bool ggml_ext_dev_supports_native_op(ggml_backend_dev_t device, const ggml_tensor * node) {
    return device && node && !ggml_ext_op_get(node) && device->iface.supports_op(device, node);
}

static bool ggml_ext_is_view(const ggml_tensor * node) {
    return node->op == GGML_OP_VIEW || node->op == GGML_OP_RESHAPE ||
           node->op == GGML_OP_PERMUTE || node->op == GGML_OP_TRANSPOSE;
}

static bool ggml_ext_is_pure(const ggml_tensor * node, bool allow_inplace = false) {
    if (node->view_src && !ggml_ext_is_view(node) && !allow_inplace) {
        return false;
    }
    switch (node->op) {
        case GGML_OP_ADD: case GGML_OP_MUL: case GGML_OP_SCALE:
        case GGML_OP_NORM: case GGML_OP_RMS_NORM: case GGML_OP_MUL_MAT:
        case GGML_OP_UNARY: case GGML_OP_GLU:
        case GGML_OP_FLASH_ATTN_EXT:
        case GGML_OP_VIEW: case GGML_OP_RESHAPE: case GGML_OP_PERMUTE: case GGML_OP_TRANSPOSE:
        case GGML_OP_CONT: case GGML_OP_CONCAT: case GGML_OP_REPEAT:
            return true;
        case GGML_OP_CUSTOM:
            return ggml_ext_op_get(node) != nullptr;
        case GGML_OP_CPY:
            return !node->view_src && node->src[1] == node;
        default:
            return false;
    }
}

static const ggml_tensor * ggml_ext_storage(const ggml_tensor * node) {
    while (node && node->view_src) {
        node = node->view_src;
    }
    return node;
}

static bool ggml_ext_is_internal_write(const ggml_cgraph * graph, int first, int index) {
    const auto * node = graph->nodes[index];
    if (!node->view_src || ggml_ext_is_view(node) || !ggml_ext_is_pure(node, true)) {
        return false;
    }
    const auto * owner = ggml_ext_storage(node);
    return std::find(graph->nodes + first, graph->nodes + index, owner) != graph->nodes + index;
}

bool ggml_ext_graph_delay(ggml_cgraph * graph, int first, int count, int before) {
    if (!graph || first < 0 || count < 1 || first > graph->n_nodes - count ||
        before < first + count || before > graph->n_nodes) {
        return false;
    }
    for (int i = first; i < before; ++i) {
        const auto * node = graph->nodes[i];
        const bool write = i >= first + count && node->view_src && !ggml_ext_is_view(node) &&
            (node->op == GGML_OP_CPY || ggml_ext_is_pure(node, true));
        const bool internal_write = i < first + count && ggml_ext_is_internal_write(graph, first, i);
        if (!ggml_ext_is_pure(node) && !internal_write && !write) {
            return false;
        }
        if (write) {
            const auto * destination = ggml_ext_storage(node);
            for (int j = first; j < first + count; ++j) {
                if (destination == ggml_ext_storage(graph->nodes[j])) {
                    return false;
                }
                for (const auto * src : graph->nodes[j]->src) {
                    if (src && destination == ggml_ext_storage(src)) {
                        return false;
                    }
                }
            }
        }
        if (i >= first + count) {
            for (const auto * src : node->src) {
                while (src) {
                    if (std::find(graph->nodes + first, graph->nodes + first + count, src) != graph->nodes + first + count) {
                        return false;
                    }
                    src = src->view_src;
                }
            }
        }
    }
    std::rotate(graph->nodes + first, graph->nodes + first + count, graph->nodes + before);
    return true;
}

static bool ggml_ext_can_fuse(const ggml_cgraph * graph, int first, int count) {
    if (count < 2 || count > graph->n_nodes - first) {
        return false;
    }
    std::vector<int> indices;
    std::vector<ggml_op> ops;
    for (int i = first; i < first + count; ++i) {
        const auto * node = graph->nodes[i];
        const bool output = i == first + count - 1;
        const bool copy = output && node->op == GGML_OP_CPY;
        const bool internal_write = ggml_ext_is_internal_write(graph, first, i);
        if ((!copy && !ggml_ext_is_pure(node) && !internal_write) ||
            (output && node->view_src && !copy && !internal_write)) {
            return false;
        }
        indices.push_back(i);
        ops.push_back(node->op);
    }
    int outputs[2] = { first + count - 1, -1 };
    int n_outputs = 1;
    const auto * output = graph->nodes[outputs[0]];
    if (output->op != GGML_OP_CPY && ggml_ext_is_internal_write(graph, first, outputs[0])) {
        const auto * owner = ggml_ext_storage(output);
        if (output->type != owner->type || !ggml_are_same_shape(output, owner) ||
            std::memcmp(output->nb, owner->nb, sizeof(output->nb)) != 0) {
            return false;
        }
        for (const auto * view = output; view != owner; view = view->view_src) {
            if (view->view_offs != 0) {
                return false;
            }
        }
        const int owner_idx = int(std::find(graph->nodes + first, graph->nodes + outputs[0], owner) - graph->nodes);
        int uses = 0;
        for (int i = owner_idx + 1; i <= outputs[0]; ++i) {
            for (const auto * src : graph->nodes[i]->src) {
                uses += src == owner;
            }
        }
        if (uses != ggml_node_get_use_count(graph, owner_idx)) {
            return false;
        }
        // Output flags propagate to view owners. The final write must replace their entire layout.
        outputs[n_outputs++] = owner_idx;
    }
    return ggml_can_fuse_subgraph_ext(graph, indices.data(), count, ops.data(), outputs, n_outputs);
}

static void ggml_ext_optimize_native(ggml_backend_t backend, ggml_cgraph * graph, int first, int end,
                                    ggml_backend_graph_optimize_params * params) {
    if (first < end && params && backend->iface.graph_optimize) {
        auto view = ggml_graph_view(graph, first, end);
        backend->iface.graph_optimize(backend, &view, params);
    }
}

ggml_ext_plan * ggml_ext_plan_create(ggml_backend_t backend, ggml_cgraph * graph,
                                    ggml_backend_graph_optimize_params * params, bool allow_fusion) {
    auto kernels = ggml_ext_kernels(backend->device);
    if (kernels.empty()) {
        ggml_ext_optimize_native(backend, graph, 0, graph->n_nodes, params);
        return nullptr;
    }
    if (params && allow_fusion) {
        for (const auto * kernel : kernels) {
            if (kernel->desc.optimize) {
                kernel->desc.optimize(backend->device, graph, kernel->desc.user_data);
            }
        }
    }
    std::unique_ptr<ggml_ext_plan> plan(new ggml_ext_plan);
    plan->backend = backend;
    plan->tail_uid = ggml_graph_next_uid();
    int native_start = 0;
    for (int i = 0; i < graph->n_nodes; ++i) {
        auto * node = graph->nodes[i];
        if (!(node->flags & GGML_TENSOR_FLAG_COMPUTE)) {
            continue;
        }
        for (const auto * kernel : kernels) {
            const auto & desc = kernel->desc;
            if (!ggml_ext_kernel_matches(desc, node)) {
                continue;
            }
            int count = 0;
            if (params && allow_fusion && desc.match) {
                count = desc.match(backend->device, graph, i, desc.user_data);
                if (!ggml_ext_can_fuse(graph, i, count)) {
                    count = 0;
                }
            }
            if (!count && desc.supports && desc.supports(backend->device, node, desc.user_data)) {
                const bool native = ggml_ext_dev_supports_native_op(backend->device, node);
                if (!native || !desc.preferred || desc.preferred(backend->device, node, desc.user_data)) {
                    count = 1;
                }
            }
            if (!count) {
                continue;
            }
            ggml_ext_optimize_native(backend, graph, native_start, i, params);
            ggml_ext_step step;
            step.first = i;
            step.native_uid = ggml_graph_next_uid();
            step.kernel = kernel;
            step.nodes.assign(graph->nodes + i, graph->nodes + i + count);
            if (count > 1) {
                for (auto * member : step.nodes) {
                    for (auto * src : member->src) {
                        if (src && std::find(step.nodes.begin(), step.nodes.end(), src) == step.nodes.end()) {
                            params->add_alloc_dep(params->user_data, src, step.nodes.back());
                        }
                    }
                    // An in-place final output still needs its owning tensor and views.
                    bool output_storage = false;
                    for (auto * output = step.nodes.back(); output; output = output->view_src) {
                        output_storage |= member == output;
                    }
                    if (params->skip_alloc && !output_storage && !member->data && !member->buffer &&
                        !(member->flags & (GGML_TENSOR_FLAG_INPUT | GGML_TENSOR_FLAG_OUTPUT))) {
                        params->skip_alloc(params->user_data, member);
                    }
                }
            }
            plan->steps.push_back(std::move(step));
            i += count - 1;
            native_start = i + 1;
            break;
        }
    }
    ggml_ext_optimize_native(backend, graph, native_start, graph->n_nodes, params);
    if (plan->steps.empty()) {
        return nullptr;
    }
    for (auto & step : plan->steps) {
        const auto & desc = step.kernel->desc;
        size_t workspace_size = 0;
        if (desc.prepare) {
            step.prepared = true;
            plan->status = desc.prepare(backend, step.nodes.data(), (int) step.nodes.size(),
                                       &step.state, &workspace_size, desc.user_data);
            if (plan->status != GGML_STATUS_SUCCESS) {
                break;
            }
        }
        step.workspace_size = GGML_PAD(workspace_size, ggml_backend_get_alignment(backend));
    }
    if (params && allow_fusion && plan->status == GGML_STATUS_SUCCESS) {
        const bool capture_safe = std::all_of(plan->steps.begin(), plan->steps.end(), [](const ggml_ext_step & step) {
            return (step.kernel->flags & GGML_EXT_KERNEL_CAPTURE_SAFE) != 0;
        });
        auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
        auto query = reinterpret_cast<ggml_ext_graph_executor_fn>(ggml_backend_reg_get_proc_address(
                reg, "ggml_ext_graph_executor_v1"));
        plan->executor = capture_safe && query ? query(backend) : nullptr;
        const bool replay = plan->executor != nullptr;
        if (!plan->executor) {
            query = reinterpret_cast<ggml_ext_graph_executor_fn>(
                    ggml_backend_reg_get_proc_address(reg, "ggml_ext_graph_batch_v1"));
            plan->executor = query ? query(backend) : nullptr;
        }
        if (plan->executor) {
            plan->executor_state = plan->executor->create(backend);
            plan->reusable = replay && std::all_of(plan->steps.begin(), plan->steps.end(), [](const ggml_ext_step & step) {
                return (step.kernel->flags & GGML_EXT_KERNEL_REPLAY_REUSABLE) != 0;
            });
        }
    }
    return plan.release();
}

void ggml_ext_plan_free(ggml_ext_plan * plan) {
    if (!plan) {
        return;
    }
    ggml_backend_synchronize(plan->backend);
    if (plan->reusable && plan->executed && plan->executor_state) {
        ggml_ext_replay_store(plan);
    }
    if (plan->executor_state) {
        plan->executor->destroy(plan->executor_state);
    }
    for (auto & step : plan->steps) {
        const auto & desc = step.kernel->desc;
        if (desc.destroy && step.prepared) {
            desc.destroy(step.state, desc.user_data);
        }
    }
    ggml_backend_buffer_free(plan->workspace);
    delete plan;
}

size_t ggml_ext_plan_workspace_size(const ggml_ext_plan * plan) {
    size_t size = 0;
    if (plan) {
        for (const auto & step : plan->steps) {
            size = std::max(size, step.workspace_size);
        }
    }
    return size;
}

static ggml_status ggml_ext_plan_execute(ggml_ext_plan * plan, ggml_cgraph * graph) {
    int first = 0;
    for (auto & step : plan->steps) {
        if (first < step.first) {
            auto view = ggml_graph_view(graph, first, step.first);
            view.uid = step.native_uid;
            auto status = plan->backend->iface.graph_compute(plan->backend, &view);
            if (status != GGML_STATUS_SUCCESS) {
                return status;
            }
        }
        const auto & desc = step.kernel->desc;
        ggml_ext_execution_context context = { GGML_EXT_ABI_VERSION, plan->backend,
                                              step.workspace_size ? plan->workspace : nullptr };
        auto status = desc.execute(&context, step.nodes.data(), (int) step.nodes.size(), step.state, desc.user_data);
        if (status != GGML_STATUS_SUCCESS) {
            return status;
        }
        first = step.first + (int) step.nodes.size();
    }
    if (first < graph->n_nodes) {
        auto view = ggml_graph_view(graph, first, graph->n_nodes);
        view.uid = plan->tail_uid;
        return plan->backend->iface.graph_compute(plan->backend, &view);
    }
    return GGML_STATUS_SUCCESS;
}

ggml_status ggml_ext_plan_compute(ggml_ext_plan * plan, ggml_cgraph * graph) {
    if (plan->status != GGML_STATUS_SUCCESS) {
        return plan->status;
    }
    if (plan->reusable) {
        plan->replay_key = ggml_ext_replay_key(plan, graph);
        if (!plan->executed && !plan->workspace) {
            ggml_ext_replay_restore(plan);
        }
    }
    const size_t workspace_size = ggml_ext_plan_workspace_size(plan);
    if (workspace_size && !plan->workspace) {
        plan->workspace = ggml_backend_alloc_buffer(plan->backend, workspace_size);
        if (!plan->workspace) {
            return GGML_STATUS_ALLOC_FAILED;
        }
    }
    if (plan->executor_state) {
        struct invocation { ggml_ext_plan * plan; ggml_cgraph * graph; } call = {plan, graph};
        auto status = plan->executor->compute(plan->executor_state, graph, [](void * opaque) {
            auto * call = static_cast<invocation *>(opaque);
            return ggml_ext_plan_execute(call->plan, call->graph);
        }, &call);
        plan->executed = status == GGML_STATUS_SUCCESS;
        return status;
    }
    return ggml_ext_plan_execute(plan, graph);
}
