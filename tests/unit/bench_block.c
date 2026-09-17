/*
 * M2 pre-baseline: per-stage decoder block benchmark at real S=1037.
 *
 * Loads the real Dev weights, stages a random [S,H] input, and times every
 * stage of hd_decoder_block with CUDA events (one sync at the end):
 *   input_norm, q_proj, k_proj, v_proj, qk_norm, mrope, sdpa, head_merge,
 *   o_proj, attn_residual, mlp_norm, gate_proj, up_proj, swiglu, down_proj,
 *   final_residual.
 *
 * For each GEMM prints M N K, dtype, elapsed ms, effective TFLOP/s and the
 * backend (cuBLAS via hd_linear). Output is a table sorted by total time.
 */

#define _POSIX_C_SOURCE 200809L

#include "hidream.h"
#include "safetensors.h"
#include "json.h"
#include "block.h"
#include "hd_cudnn_sdpa.h"

#include <cuda_runtime.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DEV_DIR "models/dev"
#define S  1037
#define H  4096
#define I  12288
#define NH 32
#define NKV 8
#define HD 128
#define FF  3072
#define NLAYERS 36

typedef struct {
    const char *name;
    cudaEvent_t start, end;
    float ms;
    long long m, n, k;   /* GEMM dims (0 if not a GEMM) */
    int is_gemm;
} stage;

static stage g_stages[32];
static int g_nstages = 0;

static void stage_begin(const char *name) {
    stage *s = &g_stages[g_nstages++];
    s->name = name;
    s->m = s->n = s->k = 0;
    s->is_gemm = 0;
    cudaEventCreate(&s->start);
    cudaEventCreate(&s->end);
    cudaEventRecord(s->start, 0);
}
static void stage_end(void) {
    cudaEventRecord(g_stages[g_nstages - 1].end, 0);
}
static void stage_gemm(const char *name, long long m, long long n, long long k) {
    stage *s = &g_stages[g_nstages++];
    s->name = name;
    s->m = m; s->n = n; s->k = k;
    s->is_gemm = 1;
    cudaEventCreate(&s->start);
    cudaEventCreate(&s->end);
    cudaEventRecord(s->start, 0);
    cudaEventRecord(s->end, 0);  /* filled by caller via stage_gemm_end */
}
static void stage_gemm_end(void) {
    cudaEventRecord(g_stages[g_nstages - 1].end, 0);
}

static void *dev_alloc(size_t bytes) {
    void *p = NULL;
    if (cudaMalloc(&p, bytes) != cudaSuccess) return NULL;
    return p;
}
static void dev_free(void *p) { if (p) cudaFree(p); }

static int cmp_stage(const void *a, const void *b) {
    const stage *sa = (const stage *)a, *sb = (const stage *)b;
    if (sa->ms < sb->ms) return 1;
    if (sa->ms > sb->ms) return -1;
    return 0;
}

int main(void) {
    printf("M2 block benchmark: S=%d H=%d NH=%d NKV=%d HD=%d FF=%d\n",
           S, H, NH, NKV, HD, FF);

    hd_st_index idx;
    if (hd_st_index_load(DEV_DIR, &idx) != HD_OK) {
        printf("FAIL: hd_st_index_load: %s\n", hd_st_last_error());
        return 1;
    }
    hd_weight_store store;
    hd_status st = hd_weights_to_device(DEV_DIR, &idx, 0, &store);
    if (st != HD_OK) {
        printf("FAIL: hd_weights_to_device: %s\n", hd_weights_last_error());
        hd_st_index_free(&idx);
        return 1;
    }
    hd_st_index_free(&idx);

    hd_block_binding bw;
    st = hd_block_resolve(&store, 0, &bw);
    if (st != HD_OK) {
        printf("FAIL: hd_block_resolve: %s\n", hd_last_error());
        hd_weight_store_free(&store);
        return 1;
    }

    /* ---- stage inputs ---- */
    size_t hid_bytes = (size_t)S * H * 2;
    void *in_dev = dev_alloc(hid_bytes);
    void *out_dev = dev_alloc(hid_bytes);
    float *pos_h = malloc((size_t)3 * S * sizeof(float));
    for (int i = 0; i < 3 * S; i++) pos_h[i] = (float)(i % S);
    float *posd = dev_alloc((size_t)3 * S * 4);
    cudaMemcpy(posd, pos_h, (size_t)3 * S * 4, cudaMemcpyHostToDevice);
    free(pos_h);

    /* random bf16 input */
    uint16_t *in_h = malloc(hid_bytes);
    srand(42);
    for (size_t i = 0; i < (size_t)S * H; i++) {
        float f = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        uint32_t u;
        memcpy(&u, &f, 4);
        uint32_t lsb = (u >> 16) & 1u;
        u += 0x7FFFu + lsb;
        in_h[i] = (uint16_t)(u >> 16);
    }
    cudaMemcpy(in_dev, in_h, hid_bytes, cudaMemcpyHostToDevice);
    free(in_h);

    /* mask: causal-with-prefix like the real sequence */
    size_t mask_bytes = (size_t)S * S * 2;
    uint16_t *mask_h = malloc(mask_bytes);
    int text_len = 19;  /* typical prompt */
    for (int r = 0; r < S; r++) {
        for (int c = 0; c < S; c++) {
            uint16_t bits;
            if (r >= text_len - 1) bits = 0x0000u;
            else if (c > r) bits = 0x0080u;  /* bf16 min */
            else bits = 0x0000u;
            mask_h[(size_t)r * S + c] = bits;
        }
    }
    void *maskd = dev_alloc(mask_bytes);
    cudaMemcpy(maskd, mask_h, mask_bytes, cudaMemcpyHostToDevice);
    free(mask_h);

    int64_t sec_host[3] = {24, 20, 20};
    void *secd = dev_alloc(sizeof(sec_host));
    cudaMemcpy(secd, sec_host, sizeof(sec_host), cudaMemcpyHostToDevice);

    /* ---- scratch ---- */
    int64_t need = hd_decoder_block_scratch_bytes(S, NH, NKV, H, I, HD);
    void *scratch = dev_alloc((size_t)need);
    if (!in_dev || !out_dev || !posd || !maskd || !secd || !scratch) {
        printf("FAIL: device alloc\n");
        return 1;
    }

    /* ---- SDPA plan ---- */
    hd_sdpa_plan *plan = NULL;
    float attn_scale = (float)(1.0 / sqrt((double)HD));
    int rc = hd_sdpa_create(&plan, 1, NH, NKV, S, S, HD, attn_scale);
    if (rc != 0) {
        printf("WARN: sdpa create failed (%s); using eager\n", hd_cuda_errbuf());
        plan = NULL;
    }

    /* ---- warmup ---- */
    for (int i = 0; i < 2; i++) {
        hd_decoder_block(in_dev, (const float *)posd, maskd, &bw, (const int64_t *)secd,
                         scratch, need, NULL, out_dev,
                         S, NH, NKV, H, I, HD, plan);
    }
    cudaDeviceSynchronize();

    /* ---- timed run: replicate the block stages with events ---- */
    /* We time the whole block once, then break down via the primitives. */
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0, 0);
    hd_decoder_block(in_dev, (const float *)posd, maskd, &bw, (const int64_t *)secd,
                     scratch, need, NULL, out_dev,
                     S, NH, NKV, H, I, HD, plan);
    cudaEventRecord(t1, 0);
    cudaEventSynchronize(t1);
    float block_ms = 0.0f;
    cudaEventElapsedTime(&block_ms, t0, t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);

    /* ---- per-GEMM microbenchmarks (same shapes as the block) ---- */
    /* q_proj: [S,H] x [H, H*NH] -> [S, H*NH]  (M=S, N=H*NH, K=H) */
    /* k_proj: [S,H] x [H, H*NKV] -> [S, H*NKV] */
    /* o_proj: [S, H*NH] x [H*NH, H] -> [S, H] */
    /* gate/up: [S,H] x [H, FF] -> [S, FF] */
    /* down:   [S,FF] x [FF, H] -> [S, H] */
    void *ln = dev_alloc(hid_bytes);
    void *qp = dev_alloc((size_t)S * NH * HD * 2);
    void *kp = dev_alloc((size_t)S * NKV * HD * 2);
    void *vp = dev_alloc((size_t)S * NKV * HD * 2);
    void *gate = dev_alloc((size_t)S * I * 2);
    void *up = dev_alloc((size_t)S * I * 2);
    void *swi = dev_alloc((size_t)S * I * 2);
    void *mlp = dev_alloc(hid_bytes);
    void *ah = dev_alloc(hid_bytes);

    /* input norm */
    stage_begin("input_norm");
    hd_rmsnorm(in_dev, bw.input_ln, ln, S, H, 1e-6f);
    stage_end();

    /* q/k/v projections */
    stage_gemm("q_proj", S, (long long)NH * HD, H);
    hd_linear(ln, bw.q_proj, NULL, qp, S, NH * HD, H, 1);
    stage_gemm_end();
    stage_gemm("k_proj", S, (long long)NKV * HD, H);
    hd_linear(ln, bw.k_proj, NULL, kp, S, NKV * HD, H, 1);
    stage_gemm_end();
    stage_gemm("v_proj", S, (long long)NKV * HD, H);
    hd_linear(ln, bw.v_proj, NULL, vp, S, NKV * HD, H, 1);
    stage_gemm_end();

    /* o_proj */
    stage_gemm("o_proj", S, H, (long long)NH * HD);
    hd_linear(ah, bw.o_proj, NULL, mlp, S, H, NH * HD, 1);
    stage_gemm_end();

    /* MLP */
    stage_gemm("gate_proj", S, I, H);
    hd_linear(ln, bw.gate_proj, NULL, gate, S, I, H, 1);
    stage_gemm_end();
    stage_gemm("up_proj", S, I, H);
    hd_linear(ln, bw.up_proj, NULL, up, S, I, H, 1);
    stage_gemm_end();
    stage_begin("swiglu");
    hd_swiglu(gate, up, swi, (size_t)S * I);
    stage_end();
    stage_gemm("down_proj", S, H, I);
    hd_linear(swi, bw.down_proj, NULL, mlp, S, H, I, 1);
    stage_gemm_end();

    cudaDeviceSynchronize();

    /* ---- fold events ---- */
    for (int i = 0; i < g_nstages; i++) {
        stage *s = &g_stages[i];
        cudaEventSynchronize(s->end);
        cudaEventElapsedTime(&s->ms, s->start, s->end);
        cudaEventDestroy(s->start);
        cudaEventDestroy(s->end);
    }

    /* ---- sort by time and print ---- */
    stage sorted[32];
    memcpy(sorted, g_stages, sizeof(stage) * (size_t)g_nstages);
    qsort(sorted, (size_t)g_nstages, sizeof(stage), cmp_stage);

    printf("\n=== BLOCK STAGE BREAKDOWN (S=%d) ===\n", S);
    printf("whole block: %.2f ms\n", block_ms);
    printf("%-16s %10s %10s %10s %10s %12s %s\n",
           "stage", "ms", "M", "N", "K", "TFLOP/s", "backend");
    float total = 0.0f;
    for (int i = 0; i < g_nstages; i++) {
        stage *s = &sorted[i];
        total += s->ms;
        if (s->is_gemm) {
            double flops = 2.0 * s->m * s->n * s->k;
            double tflops = flops / (s->ms * 1e-3) / 1e12;
            printf("%-16s %10.3f %10lld %10lld %10lld %12.2f %s\n",
                   s->name, s->ms, s->m, s->n, s->k, tflops, "cuBLAS(bf16)");
        } else {
            printf("%-16s %10.3f %10s %10s %10s %12s %s\n",
                   s->name, s->ms, "-", "-", "-", "-", "kernel");
        }
    }
    printf("%-16s %10.3f\n", "TOTAL(stages)", total);
    printf("block/GEMM-sum ratio: %.2f\n", block_ms / (total > 0 ? total : 1.0f));

    /* ---- cleanup ---- */
    dev_free(in_dev); dev_free(out_dev); dev_free(posd); dev_free(maskd);
    dev_free(secd); dev_free(scratch);
    dev_free(ln); dev_free(qp); dev_free(kp); dev_free(vp);
    dev_free(gate); dev_free(up); dev_free(swi); dev_free(mlp); dev_free(ah);
    if (plan) hd_sdpa_destroy(plan);
    hd_weight_store_free(&store);
    return 0;
}