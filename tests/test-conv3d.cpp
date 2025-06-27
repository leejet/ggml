#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

float ggml_tensor_get_f32(const ggml_tensor* tensor, int l, int k = 0, int j = 0, int i = 0) {
    if (tensor->buffer != NULL) {
        float value;
        ggml_backend_tensor_get(tensor, &value, i * tensor->nb[3] + j * tensor->nb[2] + k * tensor->nb[1] + l * tensor->nb[0], sizeof(float));
        return value;
    }
    GGML_ASSERT(tensor->nb[0] == sizeof(float));
    return *(float*)((char*)(tensor->data) + i * tensor->nb[3] + j * tensor->nb[2] + k * tensor->nb[1] + l * tensor->nb[0]);
}

ggml_fp16_t ggml_tensor_get_f16(const ggml_tensor* tensor, int l, int k = 0, int j = 0, int i = 0) {
    GGML_ASSERT(tensor->nb[0] == sizeof(ggml_fp16_t));
    return *(ggml_fp16_t*)((char*)(tensor->data) + i * tensor->nb[3] + j * tensor->nb[2] + k * tensor->nb[1] + l * tensor->nb[0]);
}

int ggml_tensor_get_i32(const ggml_tensor* tensor, int l, int k = 0, int j = 0, int i = 0) {
    if (tensor->buffer != NULL) {
        float value;
        ggml_backend_tensor_get(tensor, &value, i * tensor->nb[3] + j * tensor->nb[2] + k * tensor->nb[1] + l * tensor->nb[0], sizeof(int));
        return value;
    }
    GGML_ASSERT(tensor->nb[0] == sizeof(int));
    return *(int*)((char*)(tensor->data) + i * tensor->nb[3] + j * tensor->nb[2] + k * tensor->nb[1] + l * tensor->nb[0]);
}

void print_ggml_tensor(struct ggml_tensor* tensor, bool shape_only = false, const char* mark = "") {
    printf("%s (%s): shape(%zu, %zu, %zu, %zu)\n", mark, ggml_type_name(tensor->type), tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
    fflush(stdout);
    if (shape_only) {
        return;
    }
    int range = 19;
    for (int i = 0; i < tensor->ne[3]; i++) {
        if (i >= range && i + range < tensor->ne[3]) {
            continue;
        }
        for (int j = 0; j < tensor->ne[2]; j++) {
            if (j >= range && j + range < tensor->ne[2]) {
                continue;
            }
            for (int k = 0; k < tensor->ne[1]; k++) {
                if (k >= range && k + range < tensor->ne[1]) {
                    continue;
                }
                for (int l = 0; l < tensor->ne[0]; l++) {
                    if (l >= range && l + range < tensor->ne[0]) {
                        continue;
                    }
                    if (tensor->type == GGML_TYPE_F32) {
                        printf("  [%d, %d, %d, %d] = %f\n", i, j, k, l, ggml_tensor_get_f32(tensor, l, k, j, i));
                    } else if (tensor->type == GGML_TYPE_F16) {
                        printf("  [%d, %d, %d, %d] = %f\n", i, j, k, l, ggml_fp16_to_fp32(ggml_tensor_get_f16(tensor, l, k, j, i)));
                    } else if (tensor->type == GGML_TYPE_I32) {
                        printf("  [%d, %d, %d, %d] = %i\n", i, j, k, l, ggml_tensor_get_i32(tensor, l, k, j, i));
                    }
                    fflush(stdout);
                }
            }
        }
    }
}

static void ggml_log_callback_default(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    fputs(text, stderr);
    fflush(stderr);
}

struct test_model {
    struct ggml_tensor * a;
    struct ggml_tensor * b;
    ggml_backend_t backend = NULL;
    ggml_backend_buffer_t buffer;
    struct ggml_context * ctx;
};

int KW = 3, KH = 3, KD = 3, IC = 10, OC = 10;

void load_model(test_model & model, bool use_gpu = false) {
    // create data
    int IW = 8, IH = 6, ID = 8, N = 1;

    // Initialize adata
    std::vector<float> adata(KW * KH * KD * IC * OC);
    for (int i = 0; i < KW * KH * KD * IC * OC; i++) {
        adata[i] = 1.f;
    }

    // Convert adata to fp16 format
    std::vector<ggml_fp16_t> hadata(KW * KH * KD * IC * OC);
    ggml_fp32_to_fp16_row(adata.data(), hadata.data(), KW * KH * KD * IC * OC);

    // Initialize bdata
    std::vector<float> bdata(IW * IH * ID * IC * N);
    for (int i = 0; i < IW * IH * ID * IC * N; i++) {
        bdata[i] = 1.f;
    }

    size_t buffer_size = 0;
    {
        buffer_size += KW * KH * KD * IC * OC * ggml_type_size(GGML_TYPE_F16); // tensor a
        buffer_size += IW * IH * ID * IC * N  * ggml_type_size(GGML_TYPE_F32); // tensor b
        buffer_size += 10240; // overhead
    }

    printf("%s: ggml tensor size    = %d bytes\n", __func__, (int) sizeof(ggml_tensor));
    printf("%s: backend buffer size = %0.2f MB\n", __func__, (buffer_size/ 1024.f/ 1024.f));

    int num_tensors = 2;
    struct ggml_init_params params {
            /*.mem_size   =*/ ggml_tensor_overhead() * num_tensors,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
    };

    ggml_log_set(ggml_log_callback_default, nullptr);

    // initialize the backend
#ifdef GGML_USE_CUDA
    if (use_gpu) {
        fprintf(stderr, "%s: using CUDA backend\n", __func__);
        model.backend = ggml_backend_cuda_init(0);
        if (!model.backend) {
            fprintf(stderr, "%s: ggml_backend_cuda_init() failed\n", __func__);
        }
    }
#endif

#ifdef GGML_USE_METAL
    if (use_gpu) {
        fprintf(stderr, "%s: using Metal backend\n", __func__);
        model.backend = ggml_backend_metal_init();
        if (!model.backend) {
            fprintf(stderr, "%s: ggml_backend_metal_init() failed\n", __func__);
        }
    }
#endif

    if(!model.backend) {
        // fallback to CPU backend
        model.backend = ggml_backend_cpu_init();
    }

    model.buffer = ggml_backend_alloc_buffer(model.backend, buffer_size);

    // create context
    model.ctx = ggml_init(params);

    // create tensors
    model.a = ggml_new_tensor_4d(model.ctx, GGML_TYPE_F16,  KW, KH, KD, IC*OC);
    model.b = ggml_new_tensor_4d(model.ctx, GGML_TYPE_F32, IW, IH, ID, IC*N);

    // create a allocator
    struct ggml_tallocr alloc = ggml_tallocr_new(model.buffer);

    // alloc memory
    ggml_tallocr_alloc(&alloc, model.a);

    // load data to buffer
    if(ggml_backend_is_cpu(model.backend)) {
        memcpy(model.a->data, hadata.data(), ggml_nbytes(model.a));
    } else {
        ggml_backend_tensor_set(model.a, hadata.data(), 0, ggml_nbytes(model.a));
    }

    // alloc memory
    ggml_tallocr_alloc(&alloc, model.b);

    if(ggml_backend_is_cpu(model.backend)
#ifdef GGML_USE_METAL
                || ggml_backend_is_metal(model.backend)
#endif
    ) {
        memcpy(model.b->data, bdata.data(), ggml_nbytes(model.b));
    } else {
        ggml_backend_tensor_set(model.b, bdata.data(), 0, ggml_nbytes(model.b));
    }
}

struct ggml_cgraph * build_graph(const test_model& model) {
    static size_t buf_size = ggml_tensor_overhead()*GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
    static std::vector<uint8_t> buf(buf_size);

    struct ggml_init_params params0 = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf.data(),
        /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_gallocr_alloc_graph()
    };

    // create a temporally context to build the graph
    struct ggml_context * ctx0 = ggml_init(params0);

    struct ggml_cgraph  * gf = ggml_new_graph(ctx0);

    int s0 = 1;
    int s1 = 1;
    int s2 = 1;
    int p0 = 1;
    int p1 = 1;
    int p2 = 1;
    int d0 = 1;
    int d1 = 1;
    int d2 = 1;

    struct ggml_tensor* conv3d_res = ggml_conv_3d(ctx0, model.a, model.b, IC, s0, s1, s2, p0, p1, p2, d0, d1, d2);
    // struct ggml_tensor* conv3d_res = ggml_im2col_3d(ctx0, model.a, model.b, IC, s0, s1, s2, p0, p1, p2, d0, d1, d2, GGML_TYPE_F16);

    print_ggml_tensor(conv3d_res, true);
    ggml_set_name(conv3d_res, "conv3d_res");
    ggml_build_forward_expand(gf, conv3d_res);

    ggml_free(ctx0);
    return gf;
}

struct ggml_cgraph * compute_graph(const test_model & model, ggml_gallocr_t allocr) {
    struct ggml_cgraph * gf = build_graph(model);
    

    // allocate tensors
    ggml_gallocr_alloc_graph(allocr, gf);
    int n_threads = 8;

    if (ggml_backend_is_cpu(model.backend)) {
        ggml_backend_cpu_set_n_threads(model.backend, n_threads);
    }

    ggml_backend_graph_compute(model.backend, gf);

    //ggml_graph_print(gf);

    return gf;
}

int main(void)
{
    ggml_time_init();

    test_model model;
    load_model(model, false);

    ggml_gallocr_t allocr = NULL;

    {
        allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
        //create the worst case graph for memory usage estimation
        struct ggml_cgraph * gf = build_graph(model);
        // compute the required memory
        ggml_gallocr_reserve(allocr, gf);
        size_t mem_size = ggml_gallocr_get_buffer_size(allocr, 0);
        fprintf(stderr, "%s: compute buffer size: %.2f MB\n", __func__, mem_size/1024.0f/1024.0f);
    }

    struct ggml_cgraph * gf_res = compute_graph(model, allocr);

    struct ggml_tensor * conv3d_res = NULL;

    for(int i = 0; i < ggml_graph_n_nodes(gf_res); ++i) {
        if(strcmp(ggml_get_name(ggml_graph_node(gf_res, i)), "conv3d_res") == 0) {
            conv3d_res = ggml_graph_node(gf_res, i);
        }
    }

    std::vector<float> conv3d_data(ggml_nelements(conv3d_res));

    ggml_backend_tensor_get(conv3d_res, conv3d_data.data(), 0, ggml_nbytes(conv3d_res));
    conv3d_res->data = conv3d_data.data();
    print_ggml_tensor(conv3d_res, false);

    const int n_conv3d_test = 480;

    
    bool passed = true;
    

    printf("ggml_conv3d x (%d): %s\n", (int) ggml_nelements(conv3d_res), passed && (ggml_nelements(conv3d_res) == n_conv3d_test) ? "\033[32mPASSED\033[0m" : "\033[31mFAILED\033[0m");

    ggml_free(model.ctx);

    ggml_backend_buffer_free(model.buffer);
    ggml_backend_free(model.backend);
    ggml_gallocr_free(allocr);
    return 0;
}
