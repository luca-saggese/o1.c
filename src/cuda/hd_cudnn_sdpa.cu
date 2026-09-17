/*
 * cuDNN SDPA attention backend (M2 pre-baseline).
 *
 * C++/CUDA wrapper over the cuDNN C++ Frontend (cudnn_frontend v1.22.1,
 * header-only, linked against libcudnn.so). The graph is built ONCE in
 * hd_sdpa_create() and reused for every hd_sdpa_execute().
 *
 * Layout: q [B,Hq,Sq,D], k/v [B,Hkv,Skv,D], out [B,Hq,Sq,D], all bf16
 * contiguous (B=1: [H,S,D] head-major, matching the engine's head-split).
 * The attention mask is passed as an additive bias [1,1,Sq,Skv] bf16.
 *
 * No cudnn_frontend types leak into the rest of the engine (C ABI only).
 */

/* cuda.h (driver API) must precede cudnn_frontend.h: the frontend's
 * experimental attention_utils.h references CUtensorMap/CUresult which are
 * defined in the CUDA driver header, not in cudnn.h. The project's own
 * header is hd_cuda.h (renamed) so <cuda.h> always resolves to the NVIDIA
 * driver API header. */
#include <cuda.h>
#include <cudnn_frontend.h>

#include <cstdio>
#include <cstring>
#include <memory>
#include <string>

#include <cuda_runtime.h>  /* real cudaStream_t for set_stream */
#include "cuda_internal.h"
#include "hd_cudnn_sdpa.h"

using namespace cudnn_frontend::graph;
using cudnn_frontend::detail::create_handle;
using cudnn_frontend::detail::destroy_handle;
using cudnn_frontend::detail::set_stream;
using cudnn_frontend::DataType_t;
/* NOTE: cudnn_frontend::error_t collides with glibc's error_t (bits/types/error_t.h),
 * so it is always referenced fully-qualified below. */

struct hd_sdpa_plan {
    std::shared_ptr<Graph> graph;
    std::shared_ptr<Tensor_attributes> q;
    std::shared_ptr<Tensor_attributes> k;
    std::shared_ptr<Tensor_attributes> v;
    std::shared_ptr<Tensor_attributes> o;
    std::shared_ptr<Tensor_attributes> bias;
    cudnnHandle_t handle;
    void *workspace;
    size_t workspace_bytes;
    int B, Hq, Hkv, Sq, Skv, D;

    /* Destroy the graph (which may reference the cuDNN handle) BEFORE the
     * handle itself, then free the workspace. NOTE: cudnnDestroy on the
     * handle is NOT called here — cuDNN 9.20 double-frees internally when
     * the graph (which shares the handle) is destroyed first. The handle is
     * a small per-generation leak; the OS reclaims it at process exit. */
    ~hd_sdpa_plan() {
        graph.reset();
        q.reset(); k.reset(); v.reset(); o.reset(); bias.reset();
        if (workspace) cudaFree(workspace);
    }
};

static int hd_sdpa_set_error(const char *msg) {
    snprintf(hd_cuda_errbuf(), 512, "hd_sdpa: %s", msg);
    return 1;
}

int hd_sdpa_create(hd_sdpa_plan **out,
                   int batch, int q_heads, int kv_heads,
                   int seq_q, int seq_kv, int head_dim,
                   float scale) {
    if (!out) return hd_sdpa_set_error("null out");
    *out = NULL;
    if (batch != 1) return hd_sdpa_set_error("batch must be 1");
    if (q_heads < 1 || kv_heads < 1 || q_heads % kv_heads != 0)
        return hd_sdpa_set_error("invalid head counts (q_heads % kv_heads != 0)");
    if (seq_q < 1 || seq_kv < 1 || head_dim < 1)
        return hd_sdpa_set_error("invalid dims");

    hd_sdpa_plan *p = new hd_sdpa_plan();
    if (!p) return hd_sdpa_set_error("calloc failed");
    p->B = batch; p->Hq = q_heads; p->Hkv = kv_heads;
    p->Sq = seq_q; p->Skv = seq_kv; p->D = head_dim;

    try {
        /* cuDNN handle (cudnnCreate) + stream binding happens at execute. */
        cudnnStatus_t st = create_handle(&p->handle);
        if (st != CUDNN_STATUS_SUCCESS) {
            delete p;
            return hd_sdpa_set_error("cudnnCreate failed");
        }

        p->graph = std::make_shared<Graph>();
        p->graph->set_io_data_type(DataType_t::BFLOAT16)
                .set_intermediate_data_type(DataType_t::FLOAT)
                .set_compute_data_type(DataType_t::FLOAT);

        /* Q/K/V in [B,H,S,D] contiguous (B=1: [H,S,D]). */
        p->q = p->graph->tensor(
            Tensor_attributes()
                .set_name("Q")
                .set_dim({batch, q_heads, seq_q, head_dim})
                .set_stride({q_heads * seq_q * head_dim,
                             seq_q * head_dim, head_dim, 1}));
        p->k = p->graph->tensor(
            Tensor_attributes()
                .set_name("K")
                .set_dim({batch, kv_heads, seq_kv, head_dim})
                .set_stride({kv_heads * seq_kv * head_dim,
                             seq_kv * head_dim, head_dim, 1}));
        p->v = p->graph->tensor(
            Tensor_attributes()
                .set_name("V")
                .set_dim({batch, kv_heads, seq_kv, head_dim})
                .set_stride({kv_heads * seq_kv * head_dim,
                             seq_kv * head_dim, head_dim, 1}));

        /* Additive bias [1,1,Sq,Skv] bf16 (broadcast over heads). */
        p->bias = p->graph->tensor(
            Tensor_attributes()
                .set_name("Bias")
                .set_dim({1, 1, seq_q, seq_kv})
                .set_stride({seq_q * seq_kv, seq_q * seq_kv, seq_kv, 1}));

        SDPA_attributes opts =
            SDPA_attributes()
                .set_name("hidream_sdpa")
                .set_attn_scale(scale)
                .set_generate_stats(false)
                .set_bias(p->bias);

        auto [O, Stats] = p->graph->sdpa(p->q, p->k, p->v, opts);
        p->o = O;
        p->o->set_output(true)
            .set_dim({batch, q_heads, seq_q, head_dim})
            .set_stride({q_heads * seq_q * head_dim,
                         seq_q * head_dim, head_dim, 1});

        /* Build the operation graph and size the workspace. */
        cudnn_frontend::error_t err = p->graph->validate();
        if (err.get_code() != cudnn_frontend::error_code_t::OK) {
            std::string m = err.get_message();
            destroy_handle(p->handle);
            p->handle = nullptr;
            delete p;
            return hd_sdpa_set_error(("validate failed: " + m).c_str());
        }

        err = p->graph->build_operation_graph(p->handle);
        if (err.get_code() != cudnn_frontend::error_code_t::OK) {
            std::string m = err.get_message();
            destroy_handle(p->handle);
            p->handle = nullptr;
            delete p;
            return hd_sdpa_set_error(("build_operation_graph failed: " + m).c_str());
        }

        err = p->graph->create_execution_plans({cudnn_frontend::HeurMode_t::A});
        if (err.get_code() != cudnn_frontend::error_code_t::OK) {
            std::string m = err.get_message();
            destroy_handle(p->handle);
            p->handle = nullptr;
            delete p;
            return hd_sdpa_set_error(("create_execution_plans failed: " + m).c_str());
        }

        err = p->graph->check_support(p->handle);
        if (err.get_code() != cudnn_frontend::error_code_t::OK) {
            std::string m = err.get_message();
            destroy_handle(p->handle);
            p->handle = nullptr;
            delete p;
            return hd_sdpa_set_error(("check_support failed: " + m).c_str());
        }

        err = p->graph->build_plans(p->handle);
        if (err.get_code() != cudnn_frontend::error_code_t::OK) {
            std::string m = err.get_message();
            destroy_handle(p->handle);
            p->handle = nullptr;
            delete p;
            return hd_sdpa_set_error(("build_plans failed: " + m).c_str());
        }

        int64_t ws = 0;
        err = p->graph->get_workspace_size(ws);
        if (err.get_code() != cudnn_frontend::error_code_t::OK) {
            std::string m = err.get_message();
            destroy_handle(p->handle);
            p->handle = nullptr;
            delete p;
            return hd_sdpa_set_error(("get_workspace_size failed: " + m).c_str());
        }
        p->workspace_bytes = (size_t)ws;
        if (p->workspace_bytes > 0) {
            cudaError_t ce = cudaMalloc(&p->workspace, p->workspace_bytes);
            if (ce != cudaSuccess) {
                destroy_handle(p->handle);
                p->handle = nullptr;
                delete p;
                return hd_sdpa_set_error("cudaMalloc workspace failed");
            }
        }
    } catch (std::exception &e) {
        if (p->handle) { destroy_handle(p->handle); p->handle = nullptr; }
        if (p->workspace) { cudaFree(p->workspace); p->workspace = nullptr; }
        delete p;
        return hd_sdpa_set_error(e.what());
    }

    *out = p;
    return 0;
}

int hd_sdpa_execute(hd_sdpa_plan *plan,
                    const void *q, const void *k, const void *v,
                    const void *mask, void *out,
                    void *stream) {
    if (!plan || !q || !k || !v || !out)
        return hd_sdpa_set_error("null arg in execute");

    /* Bind the stream to the cuDNN handle (idempotent). */
    cudnnStatus_t st = set_stream(plan->handle, (cudaStream_t)stream);
    if (st != CUDNN_STATUS_SUCCESS)
        return hd_sdpa_set_error("cudnnSetStream failed");

    std::unordered_map<std::shared_ptr<Tensor_attributes>, void *> ptrs = {
        {plan->q, (void *)q},
        {plan->k, (void *)k},
        {plan->v, (void *)v},
        {plan->o, out},
    };
    if (mask) ptrs[plan->bias] = (void *)mask;

    cudnn_frontend::error_t err = plan->graph->execute(plan->handle, ptrs, plan->workspace);
    if (err.get_code() != cudnn_frontend::error_code_t::OK)
        return hd_sdpa_set_error("graph execute failed");
    return 0;
}

void hd_sdpa_destroy(hd_sdpa_plan *plan) {
    if (!plan) return;
    delete plan;
}
