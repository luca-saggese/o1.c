/*
 * M2 pre-baseline: cuDNN SDPA attention backend validation (full forward).
 *
 * Runs the FULL transformer forward (36 blocks) twice on the frozen
 * M1_V3_DEV_FORWARD_0 fixture inputs:
 *   - eager reference attention (ws.sdpa = NULL)
 *   - cuDNN SDPA (ws.sdpa = plan)
 * and compares the two complete outputs. The SDPA backend must stay within
 * the composite class D bound (NRMSE <= 1e-2, cos >= 0.999) of the eager
 * reference across the whole 36-layer forward.
 *
 * Also reports per-backend wall time.
 */

#define _POSIX_C_SOURCE 200809L

#include "hidream.h"
#include "safetensors.h"
#include "json.h"
#include "block.h"
#include "forward.h"
#include "hd_cudnn_sdpa.h"

#include <cuda_runtime.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define GOLDEN_DIR "artifacts/m1/golden/M1_V3_DEV_FORWARD_0"
#define DEV_DIR    "models/dev"

#define S  23
#define T  19
#define IMG 4
#define H  4096
#define I  12288
#define NH 32
#define NKV 8
#define HD 128
#define FF  3072
#define NLAYERS 36
#define TMS_ID 151673

static int failures = 0;
static int passes = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        printf("FAIL: %s (at %s:%d)\n", msg, __FILE__, __LINE__); \
        failures++; \
    } else { \
        printf("ok: %s\n", msg); \
        passes++; \
    } \
} while (0)

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

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

int main(void) {
    char inb_path[512];
    snprintf(inb_path, sizeof(inb_path), GOLDEN_DIR "/inputs.bin");
    size_t ibsz = 0;
    void *inputs = read_file_bytes(inb_path, &ibsz);
    if (!inputs) { printf("SKIP: inputs.bin not found\n"); return 0; }

    char inmeta_path[512];
    snprintf(inmeta_path, sizeof(inmeta_path), GOLDEN_DIR "/inputs.json");
    size_t imsz = 0;
    void *imb = read_file_bytes(inmeta_path, &imsz);
    if (!imb) { printf("SKIP: inputs.json not found\n"); free(inputs); return 0; }
    const char *err = NULL;
    hd_json *imeta = hd_json_parse((const char *)imb, &err);
    free(imb);
    if (!imeta) { printf("FAIL: parse inputs.json\n"); free(inputs); return 1; }
    const hd_json *iarr = hd_json_get(imeta, "tensors");
    int64_t off_pos=0, off_mask=0, off_vin=0, off_ts=0;
    size_t n_i = iarr ? hd_json_array_len(iarr) : 0;
    for (size_t i = 0; i < n_i; i++) {
        const hd_json *t = hd_json_array_at(iarr, i);
        const char *nm = hd_json_string(hd_json_get(t, "name"));
        int64_t off = hd_json_int(hd_json_get(t, "offset"), 0);
        if (!nm) continue;
        if (!strcmp(nm, "pos_f32")) off_pos = off;
        else if (!strcmp(nm, "mask")) off_mask = off;
        else if (!strcmp(nm, "vinputs")) off_vin = off;
        else if (!strcmp(nm, "timestep")) off_ts = off;
    }
    hd_json_free(imeta);

    printf("SDPA full-forward validation (fixture M1_V3_DEV_FORWARD_0)\n");
    printf("  text=%d img=%d seq=%d H=%d layers=%d\n", T, IMG, S, H, NLAYERS);

    hd_st_index idx;
    if (hd_st_index_load(DEV_DIR, &idx) != HD_OK) {
        printf("FAIL: hd_st_index_load: %s\n", hd_st_last_error());
        free(inputs); return 1;
    }
    hd_weight_store store;
    hd_status st = hd_weights_to_device(DEV_DIR, &idx, 0, &store);
    if (st != HD_OK) {
        printf("FAIL: hd_weights_to_device: %s\n", hd_weights_last_error());
        hd_st_index_free(&idx); free(inputs); return 1;
    }
    hd_st_index_free(&idx);

    hd_forward_binding bw;
    st = hd_forward_resolve(&store, NLAYERS, &bw);
    if (st != HD_OK) {
        printf("FAIL: hd_forward_resolve: %s\n", hd_last_error());
        hd_weight_store_free(&store); free(inputs); return 1;
    }

    int64_t scratch_bytes = 0;
    int64_t ws_bytes = hd_forward_workspace_bytes(S, IMG, NH, NKV, H, I, HD,
                                                  &scratch_bytes);
    void *wsbase = dev_alloc((size_t)ws_bytes);
    if (!wsbase) {
        printf("FAIL: workspace alloc\n");
        hd_forward_binding_free(&bw); hd_weight_store_free(&store);
        free(inputs); return 1;
    }

    const char *p = (const char *)inputs;
    void *posd  = dev_alloc(276);
    void *maskd = dev_alloc(1058);
    void *vind  = dev_alloc(24576);
    float *tsd  = dev_alloc(4);
    int64_t sec_host[3] = {24, 20, 20};
    void *secd = dev_alloc(sizeof(sec_host));
    cudaMemcpy(posd,  p + off_pos,  276,  cudaMemcpyHostToDevice);
    cudaMemcpy(maskd, p + off_mask, 1058, cudaMemcpyHostToDevice);
    cudaMemcpy(vind,  p + off_vin,  24576,cudaMemcpyHostToDevice);
    cudaMemcpy(tsd,   p + off_ts,   4,    cudaMemcpyHostToDevice);
    cudaMemcpy(secd, sec_host, sizeof(sec_host), cudaMemcpyHostToDevice);

    /* input_ids: text tokens from golden 01_model_input (int64 [1,19]) */
    char meta_path[512], bin_path[512];
    snprintf(meta_path, sizeof(meta_path), GOLDEN_DIR "/M1_V3_DEV_FORWARD_0.json");
    snprintf(bin_path,  sizeof(bin_path),  GOLDEN_DIR "/M1_V3_DEV_FORWARD_0.bin");
    size_t gsz = 0;
    void *golden = read_file_bytes(bin_path, &gsz);
    if (!golden) { printf("FAIL: golden bin not found\n"); return 1; }
    size_t msz = 0;
    void *mb = read_file_bytes(meta_path, &msz);
    if (!mb) { printf("FAIL: golden meta not found\n"); free(golden); return 1; }
    hd_json *meta = hd_json_parse((const char *)mb, &err);
    free(mb);
    if (!meta) { printf("FAIL: parse golden meta\n"); free(golden); return 1; }
    const hd_json *tarr = hd_json_get(meta, "tensors");
    int64_t off_in = -1;
    size_t nt = tarr ? hd_json_array_len(tarr) : 0;
    for (size_t i = 0; i < nt; i++) {
        const hd_json *t = hd_json_array_at(tarr, i);
        const char *nm = hd_json_string(hd_json_get(t, "name"));
        if (nm && !strcmp(nm, "01_model_input")) {
            off_in = hd_json_int(hd_json_get(t, "offset"), 0);
            break;
        }
    }
    hd_json_free(meta);
    if (off_in < 0) { printf("FAIL: golden missing 01_model_input\n"); free(golden); return 1; }
    int64_t *input_ids_h = malloc(T * sizeof(int64_t));
    memcpy(input_ids_h, (const char *)golden + off_in, T * sizeof(int64_t));
    int64_t *idsd = dev_alloc(T * sizeof(int64_t));
    cudaMemcpy(idsd, input_ids_h, T * sizeof(int64_t), cudaMemcpyHostToDevice);
    free(input_ids_h); free(golden);

    size_t out_bytes = (size_t)S * FF * 2;
    void *out_eager = dev_alloc(out_bytes);
    void *out_sdpa  = dev_alloc(out_bytes);

    /* ---- eager reference forward ---- */
    hd_forward_workspace ws;
    memset(&ws, 0, sizeof(ws));
    ws.hidden_a = wsbase;
    ws.block_scratch_bytes = scratch_bytes;
    double t0 = now_s();
    st = hd_forward(&bw, &ws, idsd, T, (const float *)posd, maskd,
                    vind, IMG, tsd, secd, S, NH, NKV, H, I, HD, TMS_ID,
                    NULL, out_eager, NULL);
    cudaDeviceSynchronize();
    double t_eager = now_s() - t0;
    CHECK(st == HD_OK, "eager forward ran without error");
    if (st != HD_OK) { printf("  eager error: %s\n", hd_last_error()); }

    /* ---- cuDNN SDPA forward ---- */
    hd_sdpa_plan *plan = NULL;
    float attn_scale = (float)(1.0 / sqrt((double)HD));
    int rc = hd_sdpa_create(&plan, 1, NH, NKV, S, S, HD, attn_scale);
    CHECK(rc == 0, "hd_sdpa_create succeeded");
    if (rc != 0) {
        const char *e = hd_cuda_errbuf();
        printf("  sdpa create error: %s\n", e && e[0] ? e : "(no message)");
    } else {
        memset(&ws, 0, sizeof(ws));
        ws.hidden_a = wsbase;
        ws.block_scratch_bytes = scratch_bytes;
        ws.sdpa = plan;
        t0 = now_s();
        st = hd_forward(&bw, &ws, idsd, T, (const float *)posd, maskd,
                        vind, IMG, tsd, secd, S, NH, NKV, H, I, HD, TMS_ID,
                        NULL, out_sdpa, NULL);
        cudaDeviceSynchronize();
        double t_sdpa = now_s() - t0;
        CHECK(st == HD_OK, "sdpa forward ran without error");
        if (st != HD_OK) printf("  sdpa error: %s\n", hd_last_error());

        /* ---- compare eager vs sdpa ---- */
        size_t n = (size_t)S * FF;
        float *fe = malloc(n * sizeof(float));
        float *fs = malloc(n * sizeof(float));
        cudaMemcpy(fe, out_eager, out_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(fs, out_sdpa,  out_bytes, cudaMemcpyDeviceToHost);
        hd_bf16_buf_to_f32(fe, fe, n);
        hd_bf16_buf_to_f32(fs, fs, n);
        double nr = rms(fe, n) > 0.0 ? rms_diff(fe, fs, n) / rms(fe, n) : 0.0;
        double cs = cosine(fe, fs, n);
        int ok = (nr <= 1e-2 && cs >= 0.999);
        printf("  eager vs sdpa: nrmse=%.5g cos=%.8g -> %s (lim nrmse<=1e-2 cos>=0.999)\n",
               nr, cs, ok ? "PASS" : "FAIL");
        CHECK(ok, "sdpa matches eager reference (full forward)");
        printf("  forward time: eager=%.3f ms  sdpa=%.3f ms  (%.2fx)\n",
               t_eager * 1e3, t_sdpa * 1e3, t_eager / t_sdpa);
        free(fe); free(fs);
        hd_sdpa_destroy(plan);
    }

    /* ---- cleanup ---- */
    free(inputs);
    dev_free(wsbase); dev_free(posd); dev_free(maskd); dev_free(vind);
    dev_free(tsd); dev_free(secd); dev_free(idsd);
    dev_free(out_eager); dev_free(out_sdpa);
    hd_forward_binding_free(&bw);
    hd_weight_store_free(&store);

    printf("\n%d assertions passed, %d failed\n", passes, failures);
    if (failures) return 1;
    printf("\nSDPA full forward: all checks passed\n");
    return 0;
}