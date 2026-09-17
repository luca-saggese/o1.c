/*
 * M2 pre-baseline: cuDNN SDPA attention backend validation (one block).
 *
 * Runs the SAME decoder block twice on the frozen block_0 fixture input:
 *   - eager reference attention (hd_attention_eager)
 *   - cuDNN SDPA (hd_sdpa_execute + hd_head_merge)
 * and compares the two block outputs. The SDPA backend must be numerically
 * close to the eager reference (composite class D bound: NRMSE <= 1e-2,
 * cos >= 0.999) while using the cuDNN FlashAttention-2 path.
 *
 * Also reports per-backend wall time for the block.
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
#include <time.h>

#define GOLDEN_DIR "artifacts/m1/golden"
#define DEV_DIR    "models/dev"
#define BLOCK_FIXTURE_ID "block_0"

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

typedef struct { size_t offset; size_t nbytes; char dtype[16]; } ftensor;

static int payload_row(const hd_json *meta, const char *name, ftensor *out) {
    const char *lists[2] = { "inputs", "outputs" };
    for (int li = 0; li < 2; li++) {
        const hd_json *arr = hd_json_get(meta, lists[li]);
        if (!arr || arr->type != HD_JSON_ARRAY) continue;
        size_t n = hd_json_array_len(arr);
        for (size_t i = 0; i < n; i++) {
            const hd_json *t = hd_json_array_at(arr, i);
            const char *nm = hd_json_string(hd_json_get(t, "name"));
            if (nm && strcmp(nm, name) == 0) {
                out->offset = (size_t)hd_json_int(hd_json_get(t, "offset"), 0);
                out->nbytes = (size_t)hd_json_int(hd_json_get(t, "nbytes"), 0);
                const char *dt = hd_json_string(hd_json_get(t, "dtype"));
                snprintf(out->dtype, sizeof(out->dtype), "%s", dt ? dt : "?");
                return 0;
            }
        }
    }
    return -1;
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

int main(void) {
    char mpath[512];
    snprintf(mpath, sizeof(mpath), GOLDEN_DIR "/" BLOCK_FIXTURE_ID ".json");
    size_t msz = 0;
    void *mb = read_file_bytes(mpath, &msz);
    if (!mb) {
        printf("SKIP: block fixture metadata %s not found\n", mpath);
        return 0;
    }
    const char *err = NULL;
    hd_json *meta = hd_json_parse((const char *)mb, &err);
    free(mb);
    if (!meta) {
        printf("FAIL: parse block fixture meta: %s\n", err ? err : "parse error");
        return 1;
    }

    char bpath[512];
    snprintf(bpath, sizeof(bpath), GOLDEN_DIR "/" BLOCK_FIXTURE_ID ".bin");
    size_t bsz = 0;
    void *bin = read_file_bytes(bpath, &bsz);
    if (!bin) {
        printf("FAIL: block fixture bin %s not found\n", bpath);
        hd_json_free(meta);
        return 1;
    }

    ftensor tx, tp, tm;
    if (payload_row(meta, "x", &tx) ||
        payload_row(meta, "pos_f32", &tp) ||
        payload_row(meta, "mask", &tm)) {
        printf("FAIL: block fixture missing required inputs\n");
        hd_json_free(meta); free(bin);
        return 1;
    }

    const hd_json *par = hd_json_get(meta, "params");
    int seq  = (int)hd_json_int(hd_json_get(par, "seq"), 23);
    int H    = (int)hd_json_int(hd_json_get(par, "hidden_size"), 4096);
    int I    = (int)hd_json_int(hd_json_get(par, "ff_hidden"), 12288);
    int NH   = (int)hd_json_int(hd_json_get(par, "heads"), 32);
    int NKV  = (int)hd_json_int(hd_json_get(par, "kv_heads"), 8);
    int HD   = (int)hd_json_int(hd_json_get(par, "head_dim"), 128);
    int layer = (int)hd_json_int(hd_json_get(par, "layer"), 0);

    size_t hid_bytes = (size_t)seq * H * sizeof(uint16_t);
    printf("SDPA block validation: seq=%d H=%d NH=%d NKV=%d HD=%d layer=%d\n",
           seq, H, NH, NKV, HD, layer);

    hd_st_index idx;
    if (hd_st_index_load(DEV_DIR, &idx) != HD_OK) {
        printf("FAIL: cannot load weight index (%s)\n", hd_st_last_error());
        hd_json_free(meta); free(bin);
        return 1;
    }
    hd_weight_store store;
    hd_status st = hd_weights_to_device(DEV_DIR, &idx, 0, &store);
    if (st != HD_OK) {
        printf("FAIL: hd_weights_to_device: %s\n", hd_weights_last_error());
        hd_st_index_free(&idx); hd_json_free(meta); free(bin);
        return 1;
    }
    hd_st_index_free(&idx);

    void *xd = dev_alloc(tx.nbytes);
    float *posd = dev_alloc(tp.nbytes);
    void *maskd = dev_alloc(tm.nbytes);
    cudaMemcpy(xd, (const char *)bin + tx.offset, tx.nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(posd, (const char *)bin + tp.offset, tp.nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(maskd, (const char *)bin + tm.offset, tm.nbytes, cudaMemcpyHostToDevice);

    int64_t need = hd_decoder_block_scratch_bytes(seq, NH, NKV, H, I, HD);
    void *scratch = dev_alloc((size_t)need);
    if (!xd || !posd || !maskd || !scratch) {
        printf("FAIL: device allocation failed\n");
        dev_free(xd); dev_free(posd); dev_free(maskd); dev_free(scratch);
        hd_weight_store_free(&store); hd_json_free(meta); free(bin);
        return 1;
    }

    hd_block_binding bw;
    st = hd_block_resolve(&store, layer, &bw);
    if (st != HD_OK) {
        printf("FAIL: hd_block_resolve: %s\n", hd_last_error());
        dev_free(xd); dev_free(posd); dev_free(maskd); dev_free(scratch);
        hd_weight_store_free(&store); hd_json_free(meta); free(bin);
        return 1;
    }

    int64_t sec_host[3] = {24, 20, 20};
    void *secd = dev_alloc(sizeof(sec_host));
    cudaMemcpy(secd, sec_host, sizeof(sec_host), cudaMemcpyHostToDevice);

    void *out_eager = dev_alloc(hid_bytes);
    void *out_sdpa  = dev_alloc(hid_bytes);

    /* ---- eager reference ---- */
    double t0 = now_s();
    st = hd_decoder_block(xd, (const float *)posd, maskd, &bw, (const int64_t *)secd,
                          scratch, need, NULL, out_eager,
                          seq, NH, NKV, H, I, HD, NULL);
    cudaDeviceSynchronize();
    double t_eager = now_s() - t0;
    CHECK(st == HD_OK, "eager block ran without error");
    if (st != HD_OK) { printf("  eager error: %s\n", hd_last_error()); }

    /* ---- cuDNN SDPA ---- */
    hd_sdpa_plan *plan = NULL;
    float attn_scale = (float)(1.0 / sqrt((double)HD));
    int rc = hd_sdpa_create(&plan, 1, NH, NKV, seq, seq, HD, attn_scale);
    CHECK(rc == 0, "hd_sdpa_create succeeded");
    if (rc != 0) {
        const char *e = hd_cuda_errbuf();
        printf("  sdpa create error: %s\n", e && e[0] ? e : "(no message)");
    } else {
        t0 = now_s();
        st = hd_decoder_block(xd, (const float *)posd, maskd, &bw, (const int64_t *)secd,
                              scratch, need, NULL, out_sdpa,
                              seq, NH, NKV, H, I, HD, plan);
        cudaDeviceSynchronize();
        double t_sdpa = now_s() - t0;
        CHECK(st == HD_OK, "sdpa block ran without error");
        if (st != HD_OK) printf("  sdpa error: %s\n", hd_last_error());

        /* ---- compare eager vs sdpa ---- */
        size_t n = (size_t)seq * H;
        float *fe = malloc(n * sizeof(float));
        float *fs = malloc(n * sizeof(float));
        cudaMemcpy(fe, out_eager, hid_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(fs, out_sdpa,  hid_bytes, cudaMemcpyDeviceToHost);
        hd_bf16_buf_to_f32(fe, fe, n);
        hd_bf16_buf_to_f32(fs, fs, n);
        double nr = rms(fe, n) > 0.0 ? rms_diff(fe, fs, n) / rms(fe, n) : 0.0;
        double cs = cosine(fe, fs, n);
        int ok = (nr <= 1e-2 && cs >= 0.999);
        printf("  eager vs sdpa: nrmse=%.5g cos=%.8g -> %s (lim nrmse<=1e-2 cos>=0.999)\n",
               nr, cs, ok ? "PASS" : "FAIL");
        CHECK(ok, "sdpa matches eager reference");
        printf("  block time: eager=%.3f ms  sdpa=%.3f ms  (%.2fx)\n",
               t_eager * 1e3, t_sdpa * 1e3, t_eager / t_sdpa);
        free(fe); free(fs);
        hd_sdpa_destroy(plan);
    }

    /* ---- cleanup ---- */
    dev_free(xd); dev_free(posd); dev_free(maskd); dev_free(scratch);
    dev_free(secd); dev_free(out_eager); dev_free(out_sdpa);
    hd_weight_store_free(&store);
    hd_json_free(meta); free(bin);

    printf("\n%d assertions passed, %d failed\n", passes, failures);
    if (failures) return 1;
    printf("\nSDPA block: all checks passed\n");
    return 0;
}