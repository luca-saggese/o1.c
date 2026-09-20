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

#include <unordered_map>

using namespace cudnn_frontend::graph;
using cudnn_frontend::detail::create_handle;
using cudnn_frontend::detail::destroy_handle;
using cudnn_frontend::detail::set_stream;
using cudnn_frontend::DataType_t;
/* NOTE: cudnn_frontend::error_t collides with glibc's error_t (bits/types/error_t.h),
 * so it is always referenced fully-qualified below. */

/* One built cuDNN SDPA graph plus its own workspace. A plan holds either one
 * of these (legacy masked path) or two (two-pass split path). */
struct hd_sdpa_pass {
    std::shared_ptr<Graph> graph;
    std::shared_ptr<Tensor_attributes> q;
    std::shared_ptr<Tensor_attributes> k;
    std::shared_ptr<Tensor_attributes> v;
    std::shared_ptr<Tensor_attributes> o;
    void *workspace = nullptr;
    size_t workspace_bytes = 0;
};

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

    /* Two-pass path. ar_len == 0 means the legacy single-graph masked path.
     * ar_len > 0 means: pass `ar` runs causal over the first ar_len queries
     * and keys; pass `gen` runs full attention for the remaining queries
     * against all keys. Both graphs are built once in hd_sdpa_create_split()
     * and only executed afterwards. */
    int ar_len = 0;
    hd_sdpa_pass ar;
    hd_sdpa_pass gen;

    /* Compact staging for the causal head pass: the AR slice [0, ar_len) is
     * copied out of each head of the [H,S,D] buffers into packed [H,ar_len,D]
     * buffers, because cuDNN requires dense head-major tensors. Sized once at
     * create time; never reallocated in the hot path. */
    void *q_stage = nullptr;
    void *k_stage = nullptr;
    void *v_stage = nullptr;
    void *o_stage = nullptr;
    size_t stage_bytes = 0;

    /* Destroy the graphs (which may reference the cuDNN handle) BEFORE the
     * handle itself, then free the workspaces. NOTE: cudnnDestroy on the
     * handle is NOT called here — cuDNN 9.20 double-frees internally when
     * the graph (which shares the handle) is destroyed first. The handle is
     * a small per-generation leak; the OS reclaims it at process exit. */
    ~hd_sdpa_plan() {
        graph.reset();
        q.reset(); k.reset(); v.reset(); o.reset(); bias.reset();
        ar.graph.reset();
        ar.q.reset(); ar.k.reset(); ar.v.reset(); ar.o.reset();
        gen.graph.reset();
        gen.q.reset(); gen.k.reset(); gen.v.reset(); gen.o.reset();
        if (workspace) cudaFree(workspace);
        if (ar.workspace) cudaFree(ar.workspace);
        if (gen.workspace) cudaFree(gen.workspace);
        if (q_stage) cudaFree(q_stage);
        if (k_stage) cudaFree(k_stage);
        if (v_stage) cudaFree(v_stage);
        if (o_stage) cudaFree(o_stage);
    }
};

static int hd_sdpa_set_error(const char *msg) {
    snprintf(hd_cuda_errbuf(), 512, "hd_sdpa: %s", msg);
    return 1;
}

/* Builds one SDPA graph (optionally causal, optionally taking an additive
 * bias tensor) and sizes its workspace. Called only at create time. */
static int hd_sdpa_build_pass(hd_sdpa_pass &p, cudnnHandle_t handle,
                              int batch, int q_heads, int kv_heads,
                              int seq_q, int seq_kv, int head_dim,
                              float scale, bool causal, bool with_bias,
                              std::shared_ptr<Tensor_attributes> &bias_out) {
    p.graph = std::make_shared<Graph>();
    p.graph->set_io_data_type(DataType_t::BFLOAT16)
            .set_intermediate_data_type(DataType_t::FLOAT)
            .set_compute_data_type(DataType_t::FLOAT);

    /* Q/K/V in [B,H,S,D] packed (B=1: [H,S,D]). cuDNN requires a dense
     * head-major layout for these tensors, so a partial token range is
     * handled by compact staging buffers (see hd_sdpa_execute_split), not
     * by non-compact strides. */
    p.q = p.graph->tensor(
        Tensor_attributes()
            .set_name("Q")
            .set_dim({batch, q_heads, seq_q, head_dim})
            .set_stride({q_heads * seq_q * head_dim,
                         seq_q * head_dim, head_dim, 1}));
    p.k = p.graph->tensor(
        Tensor_attributes()
            .set_name("K")
            .set_dim({batch, kv_heads, seq_kv, head_dim})
            .set_stride({kv_heads * seq_kv * head_dim,
                         seq_kv * head_dim, head_dim, 1}));
    p.v = p.graph->tensor(
        Tensor_attributes()
            .set_name("V")
            .set_dim({batch, kv_heads, seq_kv, head_dim})
            .set_stride({kv_heads * seq_kv * head_dim,
                         seq_kv * head_dim, head_dim, 1}));

    SDPA_attributes opts =
        SDPA_attributes()
            .set_name("hidream_sdpa")
            .set_attn_scale(scale)
            .set_generate_stats(false);

    if (with_bias) {
        /* Additive bias [1,1,Sq,Skv] bf16 (broadcast over heads). */
        bias_out = p.graph->tensor(
            Tensor_attributes()
                .set_name("Bias")
                .set_dim({1, 1, seq_q, seq_kv})
                .set_stride({seq_q * seq_kv, seq_q * seq_kv, seq_kv, 1}));
        opts.set_bias(bias_out);
    }
    if (causal) opts.set_causal_mask(true);

    auto [O, Stats] = p.graph->sdpa(p.q, p.k, p.v, opts);
    p.o = O;
    p.o->set_output(true)
        .set_dim({batch, q_heads, seq_q, head_dim})
        .set_stride({q_heads * seq_q * head_dim,
                     seq_q * head_dim, head_dim, 1});

    cudnn_frontend::error_t err = p.graph->validate();
    if (err.get_code() != cudnn_frontend::error_code_t::OK) {
        std::string m = err.get_message();
        return hd_sdpa_set_error(("validate failed: " + m).c_str());
    }
    (void)p.graph->build_operation_graph(handle);
    if (err.get_code() != cudnn_frontend::error_code_t::OK) {
        std::string m = err.get_message();
        return hd_sdpa_set_error(("build_operation_graph failed: " + m).c_str());
    }
    (void)p.graph->create_execution_plans({cudnn_frontend::HeurMode_t::A});
    if (err.get_code() != cudnn_frontend::error_code_t::OK) {
        std::string m = err.get_message();
        return hd_sdpa_set_error(("create_execution_plans failed: " + m).c_str());
    }
    (void)p.graph->check_support(handle);
    if (err.get_code() != cudnn_frontend::error_code_t::OK) {
        std::string m = err.get_message();
        return hd_sdpa_set_error(("check_support failed: " + m).c_str());
    }
    (void)p.graph->build_plans(handle);
    if (err.get_code() != cudnn_frontend::error_code_t::OK) {
        std::string m = err.get_message();
        return hd_sdpa_set_error(("build_plans failed: " + m).c_str());
    }

    int64_t ws = 0;
    (void)p.graph->get_workspace_size(ws);
    p.workspace_bytes = (size_t)ws;
    if (p.workspace_bytes > 0) {
        cudaError_t ce = cudaMalloc(&p.workspace, p.workspace_bytes);
        if (ce != cudaSuccess)
            return hd_sdpa_set_error("cudaMalloc workspace failed");
    }
    return 0;
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
        int rc = hd_sdpa_build_pass(p->ar, p->handle, batch, q_heads, kv_heads,
                                    seq_q, seq_kv, head_dim,
                                    scale,
                                    /*causal=*/false, /*with_bias=*/true,
                                    p->bias);
        if (rc != 0) {
            destroy_handle(p->handle);
            p->handle = nullptr;
            delete p;
            return rc;
        }
        /* Keep the single-graph members pointing at the built pass so the
         * legacy hd_sdpa_execute() path is unchanged. */
        p->graph = p->ar.graph;
        p->q = p->ar.q; p->k = p->ar.k; p->v = p->ar.v; p->o = p->ar.o;
        p->workspace = p->ar.workspace;
        p->workspace_bytes = p->ar.workspace_bytes;
        p->ar.workspace = nullptr;   /* owned by the legacy members now */
        p->ar.workspace_bytes = 0;
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

    /* Split plans carry their own two graphs and need no additive bias;
     * the mask argument is ignored for them. */
    if (plan->ar_len > 0)
        return hd_sdpa_execute_split(plan, q, k, v, out, stream);

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

/* ------------------------------------------------------------------ */
/* Two-pass SDPA                                                       */
/*                                                                     */
/* The single-graph masked path pays for a mixed [Sq,Skv] additive bias */
/* over the whole sequence, where only the first (text_len-1) rows are  */
/* actually causal and the rest are fully unmasked. Splitting the query  */
/* range into a causal head and a full tail lets cuDNN use its native    */
/* causal kernels for the head and a bias-free full kernel for the tail. */
/* ------------------------------------------------------------------ */
int hd_sdpa_create_split(hd_sdpa_plan **out,
                         int batch, int q_heads, int kv_heads,
                         int seq, int ar_len, int head_dim,
                         float scale) {
    if (!out) return hd_sdpa_set_error("null out");
    *out = NULL;
    if (batch != 1) return hd_sdpa_set_error("batch must be 1");
    if (q_heads < 1 || kv_heads < 1 || q_heads % kv_heads != 0)
        return hd_sdpa_set_error("invalid head counts (q_heads % kv_heads != 0)");
    if (seq < 1 || head_dim < 1)
        return hd_sdpa_set_error("invalid dims");
    if (ar_len < 0 || ar_len >= seq)
        return hd_sdpa_set_error("invalid ar_len (must be 0 < ar_len < seq)");

    hd_sdpa_plan *p = new hd_sdpa_plan();
    if (!p) return hd_sdpa_set_error("calloc failed");
    p->B = batch; p->Hq = q_heads; p->Hkv = kv_heads;
    p->Sq = seq; p->Skv = seq; p->D = head_dim;
    p->ar_len = ar_len;

    try {
        cudnnStatus_t st = create_handle(&p->handle);
        if (st != CUDNN_STATUS_SUCCESS) {
            delete p;
            return hd_sdpa_set_error("cudnnCreate failed");
        }

        /* Pass 1 (causal head): Q/K/V[0, ar_len) against each other, causal.
         * Runs on packed staging buffers of [H, ar_len, D]. */
        int rc = hd_sdpa_build_pass(p->ar, p->handle, batch, q_heads, kv_heads,
                                    ar_len, ar_len, head_dim,
                                    scale,
                                    /*causal=*/true, /*with_bias=*/false,
                                    p->bias);
        if (rc != 0) { destroy_handle(p->handle); p->handle = nullptr; delete p; return rc; }

        /* Pass 2 (generation tail): every row at or beyond ar_len is fully
         * unmasked, so the whole tail is exactly plain non-causal attention
         * of all S queries over all S keys. This needs no mask, no bias and
         * no staging: it runs directly on the contiguous [H,S,D] buffers. */
        rc = hd_sdpa_build_pass(p->gen, p->handle, batch, q_heads, kv_heads,
                                seq, seq, head_dim,
                                scale,
                                /*causal=*/false, /*with_bias=*/false,
                                p->bias);
        if (rc != 0) { destroy_handle(p->handle); p->handle = nullptr; delete p; return rc; }

        /* Compact staging for the causal head pass (allocate once). */
        size_t qe = (size_t)q_heads * (size_t)ar_len * (size_t)head_dim;
        size_t kve = (size_t)kv_heads * (size_t)ar_len * (size_t)head_dim;
        p->stage_bytes = qe * 2;
        if (cudaMalloc(&p->q_stage, qe * 2) != cudaSuccess ||
            cudaMalloc(&p->k_stage, kve * 2) != cudaSuccess ||
            cudaMalloc(&p->v_stage, kve * 2) != cudaSuccess ||
            cudaMalloc(&p->o_stage, qe * 2) != cudaSuccess) {
            destroy_handle(p->handle); p->handle = nullptr;
            delete p;
            return hd_sdpa_set_error("cudaMalloc staging failed");
        }
    } catch (std::exception &e) {
        if (p->handle) { destroy_handle(p->handle); p->handle = nullptr; }
        delete p;
        return hd_sdpa_set_error(e.what());
    }

    *out = p;
    return 0;
}

int hd_sdpa_execute_split(hd_sdpa_plan *plan,
                          const void *q, const void *k, const void *v,
                          void *out, void *stream) {
    if (!plan || !q || !k || !v || !out)
        return hd_sdpa_set_error("null arg in execute");
    if (plan->ar_len <= 0)
        return hd_sdpa_set_error("execute_split on a non-split plan");

    cudaStream_t s = (cudaStream_t)stream;
    cudnnStatus_t st = set_stream(plan->handle, s);
    if (st != CUDNN_STATUS_SUCCESS)
        return hd_sdpa_set_error("cudnnSetStream failed");

    const int D = plan->D;
    const int ar = plan->ar_len;
    const size_t row = (size_t)D * 2;                 /* one token, one head */
    const size_t src_pitch = (size_t)plan->Sq * row;  /* head stride in [H,S,D] */
    const size_t dst_pitch = (size_t)ar * row;        /* head stride in [H,ar,D] */

    /* Gather the causal head slice into packed staging buffers. */
    if (cudaMemcpy2DAsync(plan->q_stage, dst_pitch, q, src_pitch, dst_pitch,
                          (size_t)plan->Hq, cudaMemcpyDeviceToDevice, s) != cudaSuccess ||
        cudaMemcpy2DAsync(plan->k_stage, dst_pitch, k, src_pitch, dst_pitch,
                          (size_t)plan->Hkv, cudaMemcpyDeviceToDevice, s) != cudaSuccess ||
        cudaMemcpy2DAsync(plan->v_stage, dst_pitch, v, src_pitch, dst_pitch,
                          (size_t)plan->Hkv, cudaMemcpyDeviceToDevice, s) != cudaSuccess)
        return hd_sdpa_set_error("causal-head staging copy failed");

    /* Pass 2 first: writes every output row directly (contiguous, no bias). */
    {
        std::unordered_map<std::shared_ptr<Tensor_attributes>, void *> ptrs = {
            {plan->gen.q, (void *)q},
            {plan->gen.k, (void *)k},
            {plan->gen.v, (void *)v},
            {plan->gen.o, out},
        };
        cudnn_frontend::error_t err =
            plan->gen.graph->execute(plan->handle, ptrs, plan->gen.workspace);
        if (err.get_code() != cudnn_frontend::error_code_t::OK) {
            std::string m = err.get_message();
            return hd_sdpa_set_error(("pass2 (full) execute failed: " + m).c_str());
        }
    }

    /* Pass 1: causal attention on the staged head, output to staging. */
    {
        std::unordered_map<std::shared_ptr<Tensor_attributes>, void *> ptrs = {
            {plan->ar.q, plan->q_stage},
            {plan->ar.k, plan->k_stage},
            {plan->ar.v, plan->v_stage},
            {plan->ar.o, plan->o_stage},
        };
        cudnn_frontend::error_t err =
            plan->ar.graph->execute(plan->handle, ptrs, plan->ar.workspace);
        if (err.get_code() != cudnn_frontend::error_code_t::OK) {
            std::string m = err.get_message();
            return hd_sdpa_set_error(("pass1 (causal) execute failed: " + m).c_str());
        }
    }

    /* Scatter the causal head back over the pass-2 output. */
    if (cudaMemcpy2DAsync(out, src_pitch, plan->o_stage, dst_pitch, dst_pitch,
                          (size_t)plan->Hq, cudaMemcpyDeviceToDevice, s) != cudaSuccess)
        return hd_sdpa_set_error("causal-head scatter failed");
    return 0;
}

/* Executes only one pass (0 = causal head, 1 = full tail). Used by the A/B
 * harness to time the passes separately; the tail pass writes the whole
 * output, the head pass writes only the staged [H,ar_len,D] result. */
int hd_sdpa_execute_split_pass(hd_sdpa_plan *plan, int which,
                               const void *q, const void *k, const void *v,
                               void *out, void *stream) {
    if (!plan || !q || !k || !v || !out)
        return hd_sdpa_set_error("null arg in execute");
    if (plan->ar_len <= 0)
        return hd_sdpa_set_error("execute_split_pass on a non-split plan");

    cudaStream_t s = (cudaStream_t)stream;
    cudnnStatus_t st = set_stream(plan->handle, s);
    if (st != CUDNN_STATUS_SUCCESS)
        return hd_sdpa_set_error("cudnnSetStream failed");

    const size_t row = (size_t)plan->D * 2;
    const size_t src_pitch = (size_t)plan->Sq * row;
    const size_t dst_pitch = (size_t)plan->ar_len * row;
    std::unordered_map<std::shared_ptr<Tensor_attributes>, void *> ptrs;
    cudnn_frontend::error_t err;

    if (which == 0) {
        if (cudaMemcpy2DAsync(plan->q_stage, dst_pitch, q, src_pitch, dst_pitch,
                              (size_t)plan->Hq, cudaMemcpyDeviceToDevice, s) != cudaSuccess ||
            cudaMemcpy2DAsync(plan->k_stage, dst_pitch, k, src_pitch, dst_pitch,
                              (size_t)plan->Hkv, cudaMemcpyDeviceToDevice, s) != cudaSuccess ||
            cudaMemcpy2DAsync(plan->v_stage, dst_pitch, v, src_pitch, dst_pitch,
                              (size_t)plan->Hkv, cudaMemcpyDeviceToDevice, s) != cudaSuccess)
            return hd_sdpa_set_error("causal-head staging copy failed");
        ptrs = {{plan->ar.q, plan->q_stage}, {plan->ar.k, plan->k_stage},
                {plan->ar.v, plan->v_stage}, {plan->ar.o, plan->o_stage}};
        err = plan->ar.graph->execute(plan->handle, ptrs, plan->ar.workspace);
    } else {
        ptrs = {{plan->gen.q, (void *)q}, {plan->gen.k, (void *)k},
                {plan->gen.v, (void *)v}, {plan->gen.o, out}};
        err = plan->gen.graph->execute(plan->handle, ptrs, plan->gen.workspace);
    }
    if (err.get_code() != cudnn_frontend::error_code_t::OK) {
        std::string m = err.get_message();
        return hd_sdpa_set_error(("split pass execute failed: " + m).c_str());
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* Production entry point                                              */
/*                                                                     */
/* Builds the split (two-pass) plan, which is the production attention  */
/* path for the t2i/ref sequence layouts whose mask is exactly "causal  */
/* for the first text_len-1 rows, fully unmasked afterwards". `ar_len`  */
/* is that split point. On any build failure it falls back to the       */
/* single-graph masked plan so callers never lose attention.            */
/* ------------------------------------------------------------------ */
extern "C" int hd_sdpa_create_prod(hd_sdpa_plan **out,
                        int batch, int q_heads, int kv_heads,
                        int seq, int ar_len, int head_dim,
                        float scale) {
    if (!out) return hd_sdpa_set_error("null out");
    *out = NULL;
    if (ar_len > 0 && ar_len < seq)
        return hd_sdpa_create_split(out, batch, q_heads, kv_heads,
                                    seq, ar_len, head_dim, scale);
    /* ar_len <= 0 (or degenerate) => no causal head, keep the masked path */
    return hd_sdpa_create(out, batch, q_heads, kv_heads,
                          seq, seq, head_dim, scale);
}
