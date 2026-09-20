/*
 * M2 C4: two-pass SDPA A/B benchmark and correctness gate.
 *
 * Builds the real production attention operands at the frozen 2048 geometry
 * (grid 64x64 -> image_len 4096, text_len 19, SEQLEN = 4115, H = 4096, NH = 32,
 * NKV = 8, HD = 128), then compares the current single-graph mixed-mask SDPA
 * against the two-pass split (causal head + full tail) on:
 *
 *   - correctness: max_abs / NRMSE / cosine of the attention output
 *   - steady-state time with CUDA events, both paths warmed first
 *
 * The mask is the real production mask produced by hd_seq_t2i, so the
 * correctness comparison uses exactly the semantics the engine feeds cuDNN.
 */

#define _POSIX_C_SOURCE 200809L

#include "hidream.h"
#include "sequence.h"
#include "hd_cudnn_sdpa.h"
#include "hd_cuda.h"
#include "cuda_internal.h"

#include <cuda_runtime.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GRID_H   64
#define GRID_W   64
#define IMAGE_LEN (GRID_H * GRID_W)
#define TEXT_LEN 19
#define SEQLEN   (TEXT_LEN + IMAGE_LEN)   /* 4115 */
/* C4.1 edit/ref geometry (measured): S=8278, text_len=150, ar_len=149 */
#define EDIT_TEXT_LEN 150
#define EDIT_SEQLEN   8278
#define NH       32
#define NKV      8
#define HD       128
#define PATCH    32
#define FIXPOINT 4096
#define TMS_ID   151673
#define IMG_ID   151655
#define VID_ID   151656
#define VIS_START 151652
#define TMS_NUM  1
#define MERGE    2
#define WARM     5
#define REPEAT   20

static int fails = 0;

static void dev_free(void *p) { if (p) cudaFree(p); }

static void *dev_alloc(size_t n) {
    void *p = NULL;
    if (cudaMalloc(&p, n) != cudaSuccess) { printf("FAIL: cudaMalloc %zu\n", n); exit(1); }
    return p;
}

/* deterministic pseudo-random bf16 fill */
static void fill_bf16(uint16_t *h, size_t n, unsigned seed) {
    for (size_t i = 0; i < n; i++) {
        uint32_t x = (uint32_t)(i * 1103515245u + seed * 12345u + 12345u);
        float f = ((float)((x >> 8) & 0xFFFFu) / 65535.0f - 0.5f) * 0.5f;
        uint32_t bits;
        memcpy(&bits, &f, 4);
        h[i] = (uint16_t)(bits >> 16);
    }
}

static void stats(const float *a, const float *b, size_t n,
                  double *max_abs, double *nrmse, double *cosine) {
    double ma = 0.0, se = 0.0, na = 0.0, nb = 0.0, ab = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs((double)a[i] - (double)b[i]);
        if (d > ma) ma = d;
        se += d * d;
        na += (double)a[i] * (double)a[i];
        nb += (double)b[i] * (double)b[i];
        ab += (double)a[i] * (double)b[i];
    }
    *max_abs = ma;
    *nrmse = (na > 0.0) ? sqrt(se / na) : 0.0;
    *cosine = (na > 0.0 && nb > 0.0) ? ab / (sqrt(na) * sqrt(nb)) : 1.0;
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    /* C4.1: an "edit" argument switches to the real edit/ref geometry
     * (S = 8278, text_len = 150, ar_len = 149). The edit mask was verified
     * empirically to be exactly "causal for rows [0,149), fully unmasked for
     * rows [149,8278)" (0 violations, causal_bad = 0, full_bad = 0), so the
     * canonical mask below reproduces it element for element. */
    int edit_mode = (argc > 1 && strcmp(argv[1], "edit") == 0);

    int    g_S      = edit_mode ? EDIT_SEQLEN  : SEQLEN;
    int    g_text   = edit_mode ? EDIT_TEXT_LEN : TEXT_LEN;
    int    g_ar_len = g_text - 1;

    printf("=== C4 two-pass SDPA A/B [%s] (SEQLEN=%d text=%d ar_len=%d) ===\n",
           edit_mode ? "edit" : "t2i", g_S, g_text, g_ar_len);

    /* ---- real production sequence + mask ---- */
    hd_sequence seq;
    memset(&seq, 0, sizeof(seq));
    unsigned char *edit_mask = NULL;
    if (edit_mode) {
        edit_mask = (unsigned char *)malloc((size_t)g_S * (size_t)g_S * 2);
        if (!edit_mask) { printf("FAIL: mask alloc\n"); return 1; }
        for (int r = 0; r < g_S; r++)
            for (int c = 0; c < g_S; c++) {
                uint16_t bits;
                if (r >= g_ar_len)      bits = 0x0000u;   /* full attention */
                else if (c > r)         bits = 0xFF00u;   /* bf16 -inf */
                else                    bits = 0x0000u;
                size_t off = ((size_t)r * (size_t)g_S + c) * 2;
                edit_mask[off] = (unsigned char)(bits & 0xFF);
                edit_mask[off + 1] = (unsigned char)(bits >> 8);
            }
        seq.S = g_S; seq.text_len = g_text;
        seq.image_len = g_S - g_text; seq.img_begin = g_text;
        seq.mask_bf16 = edit_mask;
    } else {
        int64_t *ids = (int64_t *)malloc((size_t)TEXT_LEN * sizeof(int64_t));
        for (int i = 0; i < TEXT_LEN; i++) ids[i] = 1000 + i;
        hd_status st = hd_seq_t2i(ids, TEXT_LEN, GRID_H * PATCH, GRID_W * PATCH,
                                  PATCH, IMG_ID, VID_ID, VIS_START, TMS_ID,
                                  TMS_NUM, MERGE, FIXPOINT, &seq);
        free(ids);
        if (st != HD_OK) { printf("FAIL: hd_seq_t2i: %s\n", hd_last_error()); return 1; }
    }
    printf("seq: SEQLEN=%d text_len=%d image_len=%d img_begin=%d\n",
           seq.S, seq.text_len, seq.image_len, seq.img_begin);
    if (seq.S != g_S) { printf("FAIL: expected SEQLEN=%d\n", g_S); return 1; }

    /* ar_len per the C4 spec: text_len - 1 */
    int ar_len = g_ar_len;
    printf("ar_len (text_len-1) = %d ; gen_len = %d\n", ar_len, g_S - ar_len);

    /* ---- device operands ---- */
    size_t q_bytes = (size_t)NH * g_S * HD * 2;
    size_t kv_bytes = (size_t)NKV * g_S * HD * 2;
    uint16_t *qh = (uint16_t *)malloc(q_bytes);
    uint16_t *kh = (uint16_t *)malloc(kv_bytes);
    uint16_t *vh = (uint16_t *)malloc(kv_bytes);
    fill_bf16(qh, (size_t)NH * g_S * HD, 1);
    fill_bf16(kh, (size_t)NKV * g_S * HD, 2);
    fill_bf16(vh, (size_t)NKV * g_S * HD, 3);
    void *qd = dev_alloc(q_bytes), *kd = dev_alloc(kv_bytes), *vd = dev_alloc(kv_bytes);
    cudaMemcpy(qd, qh, q_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(kd, kh, kv_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(vd, vh, kv_bytes, cudaMemcpyHostToDevice);
    free(qh); free(kh); free(vh);

    void *maskd = dev_alloc((size_t)g_S * (size_t)g_S * 2);
    cudaMemcpy(maskd, seq.mask_bf16, (size_t)g_S * (size_t)g_S * 2, cudaMemcpyHostToDevice);

    void *out_ref = dev_alloc(q_bytes);
    void *out_two = dev_alloc(q_bytes);
    float *fr = (float *)malloc((size_t)NH * g_S * HD * 4);
    float *ft = (float *)malloc((size_t)NH * g_S * HD * 4);

    float scale = (float)(1.0 / sqrt((double)HD));

    /* ---- plans (built once) ---- */
    double t0 = now_s();
    hd_sdpa_plan *p_ref = NULL;
    int rc = hd_sdpa_create(&p_ref, 1, NH, NKV, g_S, g_S, HD, scale);
    if (rc != 0) { printf("FAIL: hd_sdpa_create: %s\n", hd_cuda_errbuf()); return 1; }
    double t_ref_create = now_s() - t0;

    t0 = now_s();
    hd_sdpa_plan *p_two = NULL;
    rc = hd_sdpa_create_split(&p_two, 1, NH, NKV, g_S, ar_len, HD, scale);
    if (rc != 0) { printf("FAIL: hd_sdpa_create_split: %s\n", hd_cuda_errbuf()); return 1; }
    double t_two_create = now_s() - t0;
    printf("plan build: mixed=%.3f s  two-pass=%.3f s\n", t_ref_create, t_two_create);

    /* ---- correctness ---- */
    rc = hd_sdpa_execute(p_ref, qd, kd, vd, maskd, out_ref, 0);
    if (rc != 0) { printf("FAIL: mixed execute: %s\n", hd_cuda_errbuf()); return 1; }
    rc = hd_sdpa_execute_split(p_two, qd, kd, vd, out_two, 0);
    if (rc != 0) { printf("FAIL: two-pass execute: %s\n", hd_cuda_errbuf()); return 1; }
    cudaDeviceSynchronize();

    cudaMemcpy(fr, out_ref, q_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(ft, out_two, q_bytes, cudaMemcpyDeviceToHost);
    hd_bf16_buf_to_f32(fr, fr, (size_t)NH * g_S * HD);
    hd_bf16_buf_to_f32(ft, ft, (size_t)NH * g_S * HD);

    double max_abs, nrmse, cos;
    stats(fr, ft, (size_t)NH * g_S * HD, &max_abs, &nrmse, &cos);
    printf("\n--- CORRECTNESS (mixed-mask vs two-pass, all %d heads) ---\n", NH);
    printf("  max_abs = %.6g\n  NRMSE   = %.6g\n  cosine  = %.9f\n", max_abs, nrmse, cos);

    /* per-region split: the two paths must agree on both halves */
    size_t head_off, region;
    double ma_ar, nr_ar, cs_ar, ma_gen, nr_gen, cs_gen;
    for (int h = 0; h < NH; h++) {
        head_off = (size_t)h * g_S * HD;
        stats(fr + head_off, ft + head_off, (size_t)ar_len * HD, &ma_ar, &nr_ar, &cs_ar);
        stats(fr + head_off + (size_t)ar_len * HD, ft + head_off + (size_t)ar_len * HD,
              (size_t)(g_S - ar_len) * HD, &ma_gen, &nr_gen, &cs_gen);
        if (h == 0)
            printf("  head0: causal-head NRMSE=%.6g cos=%.9f | full-tail NRMSE=%.6g cos=%.9f\n",
                   nr_ar, cs_ar, nr_gen, cs_gen);
    }
    (void)region;

    /* ---- steady-state timing ---- */
    cudaEvent_t e0, e1, ea0, ea1, eg0, eg1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventCreate(&ea0); cudaEventCreate(&ea1);
    cudaEventCreate(&eg0); cudaEventCreate(&eg1);
    float ms_ref = 0.0f, ms_two = 0.0f, ms_ar = 0.0f, ms_gen = 0.0f;

    for (int i = 0; i < WARM; i++) {
        hd_sdpa_execute(p_ref, qd, kd, vd, maskd, out_ref, 0);
        hd_sdpa_execute_split(p_two, qd, kd, vd, out_two, 0);
        hd_sdpa_execute_split_pass(p_two, 0, qd, kd, vd, out_two, 0);
        hd_sdpa_execute_split_pass(p_two, 1, qd, kd, vd, out_two, 0);
    }
    cudaDeviceSynchronize();

    for (int i = 0; i < REPEAT; i++) {
        float m;
        cudaEventRecord(e0, 0);
        hd_sdpa_execute(p_ref, qd, kd, vd, maskd, out_ref, 0);
        cudaEventRecord(e1, 0);
        cudaEventSynchronize(e1);
        cudaEventElapsedTime(&m, e0, e1); ms_ref += m;

        cudaEventRecord(e0, 0);
        hd_sdpa_execute_split(p_two, qd, kd, vd, out_two, 0);
        cudaEventRecord(e1, 0);
        cudaEventSynchronize(e1);
        cudaEventElapsedTime(&m, e0, e1); ms_two += m;

        cudaEventRecord(ea0, 0);
        hd_sdpa_execute_split_pass(p_two, 0, qd, kd, vd, out_two, 0);
        cudaEventRecord(ea1, 0);
        cudaEventSynchronize(ea1);
        cudaEventElapsedTime(&m, ea0, ea1); ms_ar += m;

        cudaEventRecord(eg0, 0);
        hd_sdpa_execute_split_pass(p_two, 1, qd, kd, vd, out_two, 0);
        cudaEventRecord(eg1, 0);
        cudaEventSynchronize(eg1);
        cudaEventElapsedTime(&m, eg0, eg1); ms_gen += m;
    }
    ms_ref /= REPEAT; ms_two /= REPEAT; ms_ar /= REPEAT; ms_gen /= REPEAT;

    printf("\n--- TIMING (mean of %d, %d warmups) ---\n", REPEAT, WARM);
    printf("  current mixed-mask SDPA : %.4f ms\n", ms_ref);
    printf("  two-pass pass1 (causal) : %.4f ms\n", ms_ar);
    printf("  two-pass pass2 (full)   : %.4f ms\n", ms_gen);
    printf("  two-pass total          : %.4f ms\n", ms_two);
    double speedup = (double)ms_ref / (double)ms_two;
    printf("  speedup                 : %.4fx  (%.2f%% faster)\n",
           speedup, (1.0 - (double)ms_two / (double)ms_ref) * 100.0);
    printf("  pass1+pass2 sum         : %.4f ms (launch overhead %.4f ms)\n",
           ms_ar + ms_gen, ms_two - (ms_ar + ms_gen));

    /* per-call GPU throughput for context */
    double flops_full = 2.0 * 2.0 * (double)NH * (double)g_S * (double)g_S * HD;  /* qk + av */
    printf("  mixed effective         : %.2f TFLOP/s\n",
           flops_full / (ms_ref * 1e-3) / 1e12);
    double flops_two = 2.0 * 2.0 * (double)NH *
        ((double)ar_len * ar_len + (double)(g_S - ar_len) * (double)g_S) * HD;
    printf("  two-pass effective      : %.2f TFLOP/s (%.2f TFLOP/s on real work)\n",
           flops_full / (ms_two * 1e-3) / 1e12, flops_two / (ms_two * 1e-3) / 1e12);

    /* ---- decision ---- */
    int correct = (nrmse <= 1e-2 && cos >= 0.999);
    int fast = ((1.0 - (double)ms_two / (double)ms_ref) >= 0.15);
    printf("\n--- DECISION ---\n");
    printf("  correctness gate (nrmse<=1e-2 && cos>=0.999): %s\n", correct ? "PASS" : "FAIL");
    printf("  performance gate (>=15%% faster)             : %s\n", fast ? "PASS" : "FAIL");
    printf("  => %s\n", (correct && fast) ? "ACCEPT" : "REJECT");

    cudaEventDestroy(e0); cudaEventDestroy(e1);
    cudaEventDestroy(ea0); cudaEventDestroy(ea1);
    cudaEventDestroy(eg0); cudaEventDestroy(eg1);
    hd_sdpa_destroy(p_ref);
    hd_sdpa_destroy(p_two);
    dev_free(qd); dev_free(kd); dev_free(vd); dev_free(maskd);
    dev_free(out_ref); dev_free(out_two);
    free(fr); free(ft);
    if (edit_mask) { free(edit_mask); seq.mask_bf16 = NULL; }
    hd_sequence_free(&seq);
    return (correct && fast) ? 0 : 2;
}
