/*
 * M2: final-head GEMM decisive test.
 *
 * Same buffers, 5 configurations:
 *   A. reference GEMM (hd_linear_reference)
 *   B. cuBLAS COMPUTE_32F + GEMM_DEFAULT
 *   C. cuBLAS COMPUTE_32F + GEMM_DEFAULT + DISALLOW_REDUCED_PRECISION_REDUCTION
 *   D. cuBLAS COMPUTE_32F_PEDANTIC + GEMM_DEFAULT
 *   E. cuBLAS BF16 in / FP32 compute / FP32 out, then explicit BF16 convert
 *
 * Golden input: 43_final_head_input [23,4096] bf16
 * Weights:      fl_w [3072,4096] bf16, fl_b [3072] bf16
 * Golden out:   44_final_head_output [23,3072] bf16
 *
 * Reports NRMSE/cos/max_abs vs golden, elapsed ms, TFLOP/s, plus
 * W/X pointer + layout diagnostics.
 */

#define _POSIX_C_SOURCE 200809L

#include "hidream.h"
#include "safetensors.h"
#include "json.h"
#include "forward.h"
#include "gemm.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define GOLDEN_DIR "artifacts/m1/golden/M1_V3_BASE_BLOCKS_0"
#define DEV_DIR    "models/dev"

#define S 23
#define H 4096
#define N_OUT 3072

static double rms(const float *a, size_t n) {
    double s = 0.0;
    for (size_t i = 0; i < n; i++) s += (double)a[i] * a[i];
    return sqrt(s / (double)n);
}
static double rms_diff(const float *a, const float *b, size_t n) {
    double s = 0.0;
    for (size_t i = 0; i < n; i++) { double d = (double)a[i] - b[i]; s += d * d; }
    return sqrt(s / (double)n);
}
static double cosine(const float *a, const float *b, size_t n) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < n; i++) {
        dot += (double)a[i] * b[i];
        na += (double)a[i] * a[i];
        nb += (double)b[i] * b[i];
    }
    if (na == 0.0 || nb == 0.0) return na == nb ? 1.0 : 0.0;
    return dot / (sqrt(na) * sqrt(nb));
}

static void *read_file_bytes(const char *path, size_t *out) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    void *buf = malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return NULL; }
    fseek(f, 0, SEEK_SET);
    fread(buf, 1, (size_t)sz, f);
    fclose(f);
    ((char *)buf)[sz] = '\0';
    *out = (size_t)sz;
    return buf;
}

static void *dev_alloc(size_t bytes) {
    void *p = NULL;
    if (cudaMalloc(&p, bytes) != cudaSuccess) return NULL;
    return p;
}
static void dev_free(void *p) { if (p) cudaFree(p); }

typedef struct { size_t offset; size_t nbytes; } ftensor;

static int find_tensor(const hd_json *meta, const char *name, ftensor *out) {
    const hd_json *arr = hd_json_get(meta, "tensors");
    if (!arr || arr->type != HD_JSON_ARRAY) return -1;
    size_t n = hd_json_array_len(arr);
    for (size_t i = 0; i < n; i++) {
        const hd_json *t = hd_json_array_at(arr, i);
        const char *nm = hd_json_string(hd_json_get(t, "name"));
        if (nm && strcmp(nm, name) == 0) {
            out->offset = (size_t)hd_json_int(hd_json_get(t, "offset"), 0);
            out->nbytes = (size_t)hd_json_int(hd_json_get(t, "nbytes"), 0);
            return 0;
        }
    }
    return -1;
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

/* Host-side bias add in fp32, then bf16 round. y_bf16[M,N] += bias[N]. */
static void add_bias_bf16(uint16_t *y, const uint16_t *bias, int M, int N) {
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float v = hd_bf16_to_f32(y[(size_t)m * N + n]) + hd_bf16_to_f32(bias[n]);
            y[(size_t)m * N + n] = hd_f32_to_bf16(v);
        }
}

static void report(const char *tag, const uint16_t *y, const void *golden,
                   const ftensor *ref, double ms, double tflops) {
    size_t n = ref->nbytes / 2;
    float *cf = malloc(n * sizeof(float));
    float *rf = malloc(n * sizeof(float));
    hd_bf16_buf_to_f32(y, cf, n);
    hd_bf16_buf_to_f32((const char *)golden + ref->offset, rf, n);
    double nr = rms(rf, n) > 0.0 ? rms_diff(cf, rf, n) / rms(rf, n) : 0.0;
    double cs = cosine(cf, rf, n);
    double maxa = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs((double)cf[i] - rf[i]);
        if (d > maxa) maxa = d;
    }
    printf("%-8s nrmse=%.5g cos=%.8g max_abs=%.5g  %.3f ms  %.1f TFLOP/s\n",
           tag, nr, cs, maxa, ms, tflops);
    free(cf); free(rf);
}

int main(void) {
    char mpath[512], bpath[512];
    snprintf(mpath, sizeof(mpath), GOLDEN_DIR "/M1_V3_BASE_BLOCKS_0.json");
    snprintf(bpath, sizeof(bpath), GOLDEN_DIR "/M1_V3_BASE_BLOCKS_0.bin");
    size_t msz = 0, bsz = 0;
    void *mb = read_file_bytes(mpath, &msz);
    void *bin = read_file_bytes(bpath, &bsz);
    if (!mb || !bin) { printf("FAIL: fixture not found\n"); return 1; }
    const char *err = NULL;
    hd_json *meta = hd_json_parse((const char *)mb, &err);
    free(mb);
    if (!meta) { printf("FAIL: parse meta\n"); free(bin); return 1; }

    ftensor fhi, fho;
    if (find_tensor(meta, "43_final_head_input", &fhi) ||
        find_tensor(meta, "44_final_head_output", &fho)) {
        printf("FAIL: fixture missing final head tensors\n");
        return 1;
    }

    hd_st_index idx;
    if (hd_st_index_load(DEV_DIR, &idx) != HD_OK) {
        printf("FAIL: hd_st_index_load: %s\n", hd_st_last_error());
        return 1;
    }
    hd_weight_store store;
    hd_status st = hd_weights_to_device(DEV_DIR, &idx, 0, &store);
    if (st != HD_OK) {
        printf("FAIL: hd_weights_to_device: %s\n", hd_weights_last_error());
        return 1;
    }
    hd_st_index_free(&idx);

    hd_forward_binding bw;
    st = hd_forward_resolve(&store, 36, &bw);
    if (st != HD_OK) { printf("FAIL: hd_forward_resolve: %s\n", hd_last_error()); return 1; }

    /* ---- device buffers ---- */
    void *xd = dev_alloc(fhi.nbytes);
    void *wd = dev_alloc((size_t)N_OUT * H * 2);
    void *bd = dev_alloc((size_t)N_OUT * 2);
    void *yd = dev_alloc(fho.nbytes);
    void *yd_f32 = dev_alloc((size_t)S * N_OUT * 4);
    if (!xd || !wd || !bd || !yd || !yd_f32) { printf("FAIL: alloc\n"); return 1; }

    cudaMemcpy(xd, (const char *)bin + fhi.offset, fhi.nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(wd, bw.fl_w, (size_t)N_OUT * H * 2, cudaMemcpyDeviceToDevice);
    cudaMemcpy(bd, bw.fl_b, (size_t)N_OUT * 2, cudaMemcpyDeviceToDevice);

    /* ---- diagnostics: W/X pointers, layout, first values ---- */
    printf("final head: M=%d N=%d K=%d  X=%p W=%p B=%p\n", S, N_OUT, H, xd, wd, bd);
    printf("  transA=OP_T transB=OP_N m=N=%d n=M=%d k=K=%d lda=K=%d ldb=K=%d ldc=N=%d\n",
           N_OUT, S, H, H, H, N_OUT);
    {
        uint16_t w8[8], x8[8];
        cudaMemcpy(w8, wd, 16, cudaMemcpyDeviceToHost);
        cudaMemcpy(x8, xd, 16, cudaMemcpyDeviceToHost);
        printf("  first 8 W: ");
        for (int i = 0; i < 8; i++) printf("%.4f ", hd_bf16_to_f32(w8[i]));
        printf("\n  first 8 X: ");
        for (int i = 0; i < 8; i++) printf("%.4f ", hd_bf16_to_f32(x8[i]));
        printf("\n");
    }

    cublasHandle_t ch;
    cublasCreate(&ch);
    cublasSetStream(ch, 0);
    float alpha = 1.0f, beta = 0.0f;
    double flops = 2.0 * S * N_OUT * H;

    /* ---- A. reference ---- */
    {
        double t0 = now_s();
        hd_gemm_set_backend(0);
        hd_linear(xd, wd, bd, yd, S, N_OUT, H, 1);
        cudaDeviceSynchronize();
        double ms = (now_s() - t0) * 1e3;
        hd_gemm_set_backend(1);
        uint16_t *yh = malloc(fho.nbytes);
        cudaMemcpy(yh, yd, fho.nbytes, cudaMemcpyDeviceToHost);
        report("A.ref", yh, bin, &fho, ms, flops / (ms * 1e-3) / 1e12);
        free(yh);
    }

    /* ---- B. cuBLAS 32F + DEFAULT ---- */
    {
        double t0 = now_s();
        cublasGemmEx(ch, CUBLAS_OP_T, CUBLAS_OP_N, N_OUT, S, H,
                     &alpha, wd, CUDA_R_16BF, H, xd, CUDA_R_16BF, H,
                     &beta, yd, CUDA_R_16BF, N_OUT,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        cudaDeviceSynchronize();
        double ms = (now_s() - t0) * 1e3;
        uint16_t *yh = malloc(fho.nbytes);
        cudaMemcpy(yh, yd, fho.nbytes, cudaMemcpyDeviceToHost);
        add_bias_bf16(yh, (const uint16_t *)bw.fl_b, S, N_OUT);
        report("B.cublas", yh, bin, &fho, ms, flops / (ms * 1e-3) / 1e12);
        free(yh);
    }

    /* ---- C. cuBLAS 32F + DEFAULT + DISALLOW_REDUCED_PRECISION_REDUCTION ---- */
    {
        cublasSetMathMode(ch, CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
        double t0 = now_s();
        cublasGemmEx(ch, CUBLAS_OP_T, CUBLAS_OP_N, N_OUT, S, H,
                     &alpha, wd, CUDA_R_16BF, H, xd, CUDA_R_16BF, H,
                     &beta, yd, CUDA_R_16BF, N_OUT,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        cudaDeviceSynchronize();
        double ms = (now_s() - t0) * 1e3;
        uint16_t *yh = malloc(fho.nbytes);
        cudaMemcpy(yh, yd, fho.nbytes, cudaMemcpyDeviceToHost);
        add_bias_bf16(yh, (const uint16_t *)bw.fl_b, S, N_OUT);
        report("C.disallow", yh, bin, &fho, ms, flops / (ms * 1e-3) / 1e12);
        free(yh);
        cublasSetMathMode(ch, CUBLAS_DEFAULT_MATH);
    }

    /* ---- D. cuBLAS 32F_PEDANTIC + DEFAULT ---- */
    {
        double t0 = now_s();
        cublasGemmEx(ch, CUBLAS_OP_T, CUBLAS_OP_N, N_OUT, S, H,
                     &alpha, wd, CUDA_R_16BF, H, xd, CUDA_R_16BF, H,
                     &beta, yd, CUDA_R_16BF, N_OUT,
                     CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT);
        cudaDeviceSynchronize();
        double ms = (now_s() - t0) * 1e3;
        uint16_t *yh = malloc(fho.nbytes);
        cudaMemcpy(yh, yd, fho.nbytes, cudaMemcpyDeviceToHost);
        add_bias_bf16(yh, (const uint16_t *)bw.fl_b, S, N_OUT);
        report("D.pedantic", yh, bin, &fho, ms, flops / (ms * 1e-3) / 1e12);
        free(yh);
    }

    /* ---- E. cuBLAS BF16 in / FP32 compute / FP32 out ---- */
    {
        double t0 = now_s();
        cublasGemmEx(ch, CUBLAS_OP_T, CUBLAS_OP_N, N_OUT, S, H,
                     &alpha, wd, CUDA_R_16BF, H, xd, CUDA_R_16BF, H,
                     &beta, yd_f32, CUDA_R_32F, N_OUT,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        cudaDeviceSynchronize();
        double ms = (now_s() - t0) * 1e3;
        /* FP32 -> BF16 with the same converter used by the reference. */
        float *yhf = malloc((size_t)S * N_OUT * 4);
        uint16_t *yh = malloc(fho.nbytes);
        cudaMemcpy(yhf, yd_f32, (size_t)S * N_OUT * 4, cudaMemcpyDeviceToHost);
        for (int m = 0; m < S; m++)
            for (int n = 0; n < N_OUT; n++) {
                float v = yhf[(size_t)m * N_OUT + n] + hd_bf16_to_f32(((const uint16_t *)bw.fl_b)[n]);
                yh[(size_t)m * N_OUT + n] = hd_f32_to_bf16(v);
            }
        report("E.f32out", yh, bin, &fho, ms, flops / (ms * 1e-3) / 1e12);
        free(yhf); free(yh);
    }

    cublasDestroy(ch);
    dev_free(xd); dev_free(wd); dev_free(bd); dev_free(yd); dev_free(yd_f32);
    hd_forward_binding_free(&bw);
    hd_weight_store_free(&store);
    hd_json_free(meta); free(bin);
    return 0;
}