/*
 * M2 B0: production GEMM microbenchmark at REAL 2048 sequence shapes.
 *
 * Loads the real Dev weights (which also initialises the persistent cuBLAS
 * runtime), then times each production GEMM shape with CUDA events over many
 * iterations. Reports M N K, dtype, backend, ms/call and achieved TFLOP/s.
 *
 * Usage: bench_gemm [S] [iters]   (default S=4115, iters=50)
 *
 * Shapes (row-major y[M,N] = x[M,K] . w[N,K]^T, bf16 in/out, fp32 accum):
 *   q_proj    M=S N=NH*HD=4096  K=H=4096
 *   k_proj    M=S N=NKV*HD=1024 K=H=4096
 *   v_proj    M=S N=NKV*HD=1024 K=H=4096
 *   o_proj    M=S N=H=4096      K=NH*HD=4096
 *   gate_proj M=S N=I=12288     K=H=4096
 *   up_proj   M=S N=I=12288     K=H=4096
 *   down_proj M=S N=H=4096      K=I=12288
 *   final_head M=S N=FF=3072    K=H=4096
 */

#define _POSIX_C_SOURCE 200809L

#include "hidream.h"
#include "safetensors.h"
#include "block.h"
#include "forward.h"
#include "gemm.h"

#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DEV_DIR "models/dev"
#define H  4096
#define I  12288
#define NH 32
#define NKV 8
#define HD 128
#define FF 3072

static void *dev_alloc(size_t bytes) {
    void *p = NULL;
    if (cudaMalloc(&p, bytes) != cudaSuccess) return NULL;
    return p;
}

typedef struct {
    const char *name;
    int M, N, K;
    const void *w;
} gemm_case;

static double bench_backend(const gemm_case *g, const void *x, void *y,
                            int iters, int prod_backend) {
    hd_gemm_set_prod_backend(prod_backend);
    /* warmup (also triggers Lt plan creation + tuning on first call) */
    for (int i = 0; i < 3; i++)
        hd_linear(x, g->w, NULL, y, g->M, g->N, g->K, 1);
    cudaDeviceSynchronize();

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0, 0);
    for (int i = 0; i < iters; i++)
        hd_linear(x, g->w, NULL, y, g->M, g->N, g->K, 1);
    cudaEventRecord(t1, 0);
    cudaEventSynchronize(t1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    return ms / (double)iters;
}

static void bench_one(const gemm_case *g, const void *x, void *y, int iters) {
    double lt_ms = bench_backend(g, x, y, iters, 0);
    double ex_ms = bench_backend(g, x, y, iters, 1);
    hd_gemm_set_prod_backend(0);

    double flops = 2.0 * (double)g->M * g->N * g->K;
    double lt_tf = flops / (lt_ms * 1e-3) / 1e12;
    double ex_tf = flops / (ex_ms * 1e-3) / 1e12;
    printf("%-12s M=%-6d N=%-6d K=%-6d  Lt %8.3f ms %7.2f TF  |  "
           "GemmEx %8.3f ms %7.2f TF  |  %5.2fx\n",
           g->name, g->M, g->N, g->K, lt_ms, lt_tf, ex_ms, ex_tf,
           ex_ms / lt_ms);
}

int main(int argc, char **argv) {
    int S = (argc > 1) ? atoi(argv[1]) : 4115;
    int iters = (argc > 2) ? atoi(argv[2]) : 50;

    printf("M2 B0 GEMM microbench: S=%d iters=%d H=%d I=%d NH=%d NKV=%d HD=%d FF=%d\n",
           S, iters, H, I, NH, NKV, HD, FF);

    hd_st_index idx;
    if (hd_st_index_load(DEV_DIR, &idx) != HD_OK) {
        printf("FAIL: hd_st_index_load: %s\n", hd_st_last_error());
        return 1;
    }
    hd_weight_store store;
    if (hd_weights_to_device(DEV_DIR, &idx, 0, &store) != HD_OK) {
        printf("FAIL: hd_weights_to_device: %s\n", hd_weights_last_error());
        hd_st_index_free(&idx);
        return 1;
    }
    hd_st_index_free(&idx);

    hd_block_binding bw;
    if (hd_block_resolve(&store, 0, &bw) != HD_OK) {
        printf("FAIL: hd_block_resolve: %s\n", hd_last_error());
        hd_weight_store_free(&store);
        return 1;
    }

    /* final head weight: resolve via the forward binding (same path as the
     * production forward) */
    hd_forward_binding fb;
    memset(&fb, 0, sizeof(fb));
    const void *fl_w = NULL;
    if (hd_forward_resolve(&store, 36, &fb) == HD_OK) {
        fl_w = fb.fl_w;
    }

    size_t hid = (size_t)S * H * 2;
    size_t qkv = (size_t)S * NH * HD * 2;
    size_t kv  = (size_t)S * NKV * HD * 2;
    size_t ff  = (size_t)S * I * 2;
    size_t fh  = (size_t)S * FF * 2;

    void *x_hid = dev_alloc(hid);
    void *x_qkv = dev_alloc(qkv);
    void *x_ff  = dev_alloc(ff);
    void *y_hid = dev_alloc(hid);
    void *y_qkv = dev_alloc(qkv);
    void *y_kv  = dev_alloc(kv);
    void *y_ff  = dev_alloc(ff);
    void *y_fh  = dev_alloc(fh);
    if (!x_hid || !x_qkv || !x_ff || !y_hid || !y_qkv || !y_kv || !y_ff || !y_fh) {
        printf("FAIL: device alloc\n");
        return 1;
    }
    cudaMemset(x_hid, 0, hid);
    cudaMemset(x_qkv, 0, qkv);
    cudaMemset(x_ff, 0, ff);

    printf("\n=== B0 GEMM (real 2048 shapes) ===\n");
    gemm_case cases[] = {
        {"q_proj",    S, NH * HD, H,  bw.q_proj},
        {"k_proj",    S, NKV * HD, H, bw.k_proj},
        {"v_proj",    S, NKV * HD, H, bw.v_proj},
        {"o_proj",    S, H, NH * HD,  bw.o_proj},
        {"gate_proj", S, I, H,        bw.gate_proj},
        {"up_proj",   S, I, H,        bw.up_proj},
        {"down_proj", S, H, I,        bw.down_proj},
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        const void *x = x_hid;
        void *y = y_hid;
        if (!strcmp(cases[i].name, "o_proj")) { x = x_qkv; y = y_hid; }
        if (!strcmp(cases[i].name, "gate_proj") ||
            !strcmp(cases[i].name, "up_proj")) { x = x_hid; y = y_ff; }
        if (!strcmp(cases[i].name, "down_proj")) { x = x_ff; y = y_hid; }
        bench_one(&cases[i], x, y, iters);
    }
    if (fl_w) {
        gemm_case fh = {"final_head", S, FF, H, fl_w};
        bench_one(&fh, x_hid, y_fh, iters);
    } else {
        printf("final_head   (weight not found; skipped)\n");
    }

    /* M=1 timestep-embedder GEMMs (t_embedder1.mlp.0 / mlp.2) */
    if (fb.te0_w) {
        gemm_case t0 = {"te0(M=1)", 1, H, 256, fb.te0_w};
        bench_one(&t0, x_hid, y_hid, iters);
    }
    if (fb.te2_w) {
        gemm_case t2 = {"te2(M=1)", 1, H, H, fb.te2_w};
        bench_one(&t2, x_hid, y_hid, iters);
    }

    cudaFree(x_hid); cudaFree(x_qkv); cudaFree(x_ff);
    cudaFree(y_hid); cudaFree(y_qkv); cudaFree(y_kv); cudaFree(y_ff); cudaFree(y_fh);
    hd_forward_binding_free(&fb);
    hd_weight_store_free(&store);
    return 0;
}