/*
 * M1.7 Base compatibility one-forward validation harness (V3).
 *
 * Drives hd_forward on the frozen M1_V3_BASE_FORWARD_0 golden fixture and
 * compares every captured checkpoint against the Python whole-model forward
 * golden in strict order: model input -> embedding -> target embedding ->
 * timestep conditioning -> block 0 -> mid block -> last block -> final norm
 * input -> final norm output -> final head -> complete output.
 *
 * This is the Base compatibility gate (MILESTONES.md §M1.7): Base passes
 * configuration + weight loading + one-forward compatibility through the
 * SAME shared implementation path as Dev (no Dev-only topology hardcoding).
 * Base topology is identical to Dev (H=4096, 36 layers, NH=32, NKV=8,
 * HD=128, FF=12288); only the weights differ.
 *
 * TIMESTEP DOMAIN (docs/M1_FORWARD_CONTRACT.md §15.1): the golden was
 * captured at model_timestep = 1 - 999/1000 ≈ 0.001 (embedder input ≈ 1.0).
 * The native run consumes the golden model_timestep bytes and passes them to
 * hd_forward, which multiplies by 1000 internally. Passing scheduler time
 * (999) here is a FAIL (timestep-domain regression).
 *
 * Comparison stops at the first meaningful divergence (strict order per the
 * M1.4 contract). Bounds are the M1.4 whole-forward gate bounds
 * (docs/M1_NUMERICAL_CONTRACT.md §6), frozen in
 * docs/M1_7_BASE_TOLERANCES.md before this gate ran. Do not tune to pass.
 *
 * BF16 compute / FP32 accumulate. A single device-resident forward executes
 * the SAME production path hd_forward; diagnostics observe it in-place.
 */

#include "hidream.h"
#include "safetensors.h"
#include "json.h"
#include "block.h"
#include "forward.h"

#include <cuda_runtime.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GOLDEN_DIR "artifacts/m1/golden/M1_V3_BASE_FORWARD_0"
#define BASE_DIR   "models/base"

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
#define MID 18
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

/* ------------------------------------------------------------------ */
/* Metrics (mirror test_full_forward.c)                                */
/* ------------------------------------------------------------------ */

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

/* Golden binary layout rows from the meta JSON ("tensors" array). */
typedef struct { size_t offset; size_t nbytes; char name[64]; } gtensor;

/* Row-wise diagnostics (mirror test_full_forward.c): per-row RMSE/NRMSE/
 * cosine/RMS_ref/RMS_native/max_abs. Used to localize drift to specific
 * rows (e.g. the TMS row) instead of relying on flattened global metrics. */
static void row_compare(const char *tag, const void *dev, const void *bin,
                        const gtensor *ref, int rows, int Hw) {
    int64_t n = ref->nbytes / 2;
    if (n != (int64_t)rows * Hw) {
        printf("  %-26s (size %lld != %d*%d, skipped)\n", tag, (long long)n, rows, Hw);
        return;
    }
    float *cf = malloc((size_t)n * sizeof(float));
    float *rf = malloc((size_t)n * sizeof(float));
    void *host = malloc((size_t)n * 2);
    cudaMemcpy(host, dev, (size_t)n * 2, cudaMemcpyDeviceToHost);
    hd_bf16_buf_to_f32(host, cf, (size_t)n);
    hd_bf16_buf_to_f32((const char *)bin + ref->offset, rf, (size_t)n);
    printf("  %s (row-wise, every %d rows):\n", tag, rows / 8 + 1);
    for (int r = 0; r < rows; r++) {
        const float *cr = cf + (size_t)r * Hw;
        const float *rr = rf + (size_t)r * Hw;
        double rms_c = 0, rms_r = 0, dot = 0, sd2 = 0, mx = 0;
        for (int k = 0; k < Hw; k++) {
            double a = cr[k], b = rr[k], d = a - b;
            rms_c += a * a; rms_r += b * b; dot += a * b; sd2 += d * d;
            double ad = fabs(d); if (ad > mx) mx = ad;
        }
        rms_c = sqrt(rms_c / Hw); rms_r = sqrt(rms_r / Hw);
        double nr = rms_r > 0 ? sqrt(sd2 / Hw) / rms_r : 0.0;
        double cs = (rms_c > 0 && rms_r > 0) ? dot / (rms_c * rms_r * Hw) : 0.0;
        if (r % (rows / 8 + 1) == 0 || r == rows - 1)
            printf("    row %2d  nrmse=%.5g cos=%.8g RMS_ref=%.4g RMS_na=%.4g max_abs=%.4g\n",
                   r, nr, cs, rms_r, rms_c, mx);
        (void)cs;
    }
    free(cf); free(rf); free(host);
}

static int load_golden_layout(const char *meta_path, gtensor *list, size_t cap) {
    size_t msz = 0;
    void *mb = read_file_bytes(meta_path, &msz);
    if (!mb) return -1;
    const char *err = NULL;
    hd_json *meta = hd_json_parse((const char *)mb, &err);
    free(mb);
    if (!meta) return -1;
    const hd_json *arr = hd_json_get(meta, "tensors");
    if (!arr || arr->type != HD_JSON_ARRAY) { hd_json_free(meta); return -1; }
    size_t n = hd_json_array_len(arr);
    if (n > cap) n = cap;
    for (size_t i = 0; i < n; i++) {
        const hd_json *t = hd_json_array_at(arr, i);
        const char *nm = hd_json_string(hd_json_get(t, "name"));
        if (!nm) continue;
        snprintf(list[i].name, sizeof(list[i].name), "%s", nm);
        list[i].offset = (size_t)hd_json_int(hd_json_get(t, "offset"), 0);
        list[i].nbytes = (size_t)hd_json_int(hd_json_get(t, "nbytes"), 0);
    }
    hd_json_free(meta);
    return (int)n;
}

static gtensor *find_tensor(gtensor *list, int n, const char *name) {
    for (int i = 0; i < n; i++)
        if (strcmp(list[i].name, name) == 0) return &list[i];
    return NULL;
}

/* Compare candidate device bf16 to a golden bf16 slice; class C/D bounds. */
static int compare_ckpt(const char *tag, const void *dev,
                        const void *bin, const gtensor *ref,
                        double nrmse_lim, double cos_lim) {
    size_t n = ref->nbytes / 2;
    float *cf = malloc(n * sizeof(float));
    float *rf = malloc(n * sizeof(float));
    void *host = malloc(n * 2);
    cudaMemcpy(host, dev, n * 2, cudaMemcpyDeviceToHost);
    hd_bf16_buf_to_f32(host, cf, n);
    hd_bf16_buf_to_f32((const char *)bin + ref->offset, rf, n);
    double nr = 0.0, cs = 0.0, maxa = 0.0;
    if (rms(rf, n) > 0.0) nr = rms_diff(cf, rf, n) / rms(rf, n);
    cs = cosine(cf, rf, n);
    long nan = 0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs((double)cf[i] - rf[i]);
        if (d > maxa) maxa = d;
        if (isnan(cf[i]) || isinf(cf[i]) || isnan(rf[i]) || isinf(rf[i])) nan++;
    }
    int ok = (nr <= nrmse_lim) && (cs >= cos_lim) && (nan == 0);
    printf("  %-26s nrmse=%.5g cos=%.8g max_abs=%.5g nan/inf=%ld -> %s\n",
           tag, nr, cs, maxa, nan, ok ? "PASS" : "FAIL");
    CHECK(ok, tag);
    free(cf); free(rf); free(host);
    return ok;
}

int main(void) {
    /* ---- load golden layout ---- */
    char meta_path[512], bin_path[512];
    snprintf(meta_path, sizeof(meta_path), GOLDEN_DIR "/M1_V3_BASE_FORWARD_0.json");
    snprintf(bin_path,  sizeof(bin_path),  GOLDEN_DIR "/M1_V3_BASE_FORWARD_0.bin");
    gtensor glist[16];
    int gn = load_golden_layout(meta_path, glist, 16);
    if (gn < 0) { printf("FAIL: cannot load golden layout %s\n", meta_path); return 1; }

    size_t gsz = 0;
    void *golden = read_file_bytes(bin_path, &gsz);
    if (!golden) { printf("FAIL: golden bin not found\n"); return 1; }

    /* ---- timestep domain: read model_timestep from golden metadata ---- */
    size_t msz = 0;
    void *mb = read_file_bytes(meta_path, &msz);
    if (!mb) { free(golden); return 1; }
    const char *err = NULL;
    hd_json *meta = hd_json_parse((const char *)mb, &err);
    free(mb);
    if (!meta) { free(golden); return 1; }
    double model_timestep = hd_json_double(hd_json_get(meta, "model_timestep"), -1.0);
    double sched_ts = hd_json_double(hd_json_get(meta, "scheduler_timestep"), -1.0);
    double embed_in = hd_json_double(hd_json_get(meta, "timestep_embedder_input"), -1.0);
    hd_json_free(meta);
    CHECK(fabs(model_timestep - 0.001) < 1e-4,
          "golden model_timestep ≈ 0.001 (model domain, NOT scheduler time)");
    CHECK(fabs(sched_ts - 999.0) < 1e-3, "golden scheduler_timestep = 999");
    CHECK(fabs(embed_in - 1.0) < 1e-3, "golden timestep_embedder_input ≈ 1.0");
    if (fabs(model_timestep - 0.001) >= 1e-4) {
        printf("FAIL: golden model_timestep = %g (expected ≈ 0.001)\n",
               model_timestep);
        free(golden); return 1;
    }

    printf("M1.7 Base one-forward validation (V3, fixture M1_V3_BASE_FORWARD_0)\n");
    printf("  text=%d img=%d seq=%d H=%d layers=%d mid=%d FF=%d\n",
           T, IMG, S, H, NLAYERS, MID, FF);
    printf("  scheduler_timestep=%.1f model_timestep=%.6f embedder_input=%.4f\n",
           sched_ts, model_timestep, embed_in);

    /* ---- load device weights (Base profile) ---- */
    hd_st_index idx;
    if (hd_st_index_load(BASE_DIR, &idx) != HD_OK) {
        printf("FAIL: hd_st_index_load(base): %s\n", hd_st_last_error());
        free(golden); return 1;
    }
    hd_weight_store store;
    hd_status st = hd_weights_to_device(BASE_DIR, &idx, 0, &store);
    if (st != HD_OK) {
        printf("FAIL: hd_weights_to_device(base): %s\n", hd_weights_last_error());
        hd_st_index_free(&idx); free(golden); return 1;
    }
    hd_st_index_free(&idx);

    /* ---- resolve forward bindings ---- */
    hd_forward_binding bw;
    st = hd_forward_resolve(&store, NLAYERS, &bw);
    if (st != HD_OK) {
        printf("FAIL: hd_forward_resolve: %s\n", hd_last_error());
        hd_weight_store_free(&store); free(golden); return 1;
    }
    CHECK(hd_forward_num_layers(&bw) == NLAYERS, "forward resolved 36 layers");

    /* ---- allocate persistent workspace ---- */
    int64_t scratch_bytes = 0;
    int64_t ws_bytes = hd_forward_workspace_bytes(S, IMG, NH, NKV, H, I, HD,
                                                  &scratch_bytes);
    void *wsbase = dev_alloc((size_t)ws_bytes);
    hd_forward_workspace ws;
    memset(&ws, 0, sizeof(ws));
    ws.hidden_a = wsbase;
    ws.block_scratch_bytes = scratch_bytes;
    CHECK(wsbase != NULL, "workspace allocated");
    if (!wsbase) { hd_forward_binding_free(&bw); hd_weight_store_free(&store);
                   free(golden); return 1; }

    /* ---- stage device inputs from golden tensors ---- */
    gtensor *gpos = find_tensor(glist, gn, "pos_f32");
    gtensor *gmask = find_tensor(glist, gn, "mask");
    gtensor *gvin = find_tensor(glist, gn, "vinputs");
    gtensor *gts = find_tensor(glist, gn, "timestep");
    if (!gpos || !gmask || !gvin || !gts) {
        printf("FAIL: golden missing input tensors (pos_f32/mask/vinputs/timestep)\n");
        hd_forward_binding_free(&bw); hd_weight_store_free(&store);
        free(golden); return 1;
    }
    void *posd  = dev_alloc(276);
    void *maskd = dev_alloc(1058);
    void *vind  = dev_alloc(24576);
    float *tsd  = dev_alloc(4);
    int64_t sec_host[3] = {24, 20, 20};
    void *secd = dev_alloc(sizeof(sec_host));
    cudaMemcpy(posd,  (const char *)golden + gpos->offset,  276,  cudaMemcpyHostToDevice);
    cudaMemcpy(maskd, (const char *)golden + gmask->offset, 1058, cudaMemcpyHostToDevice);
    cudaMemcpy(vind,  (const char *)golden + gvin->offset,  24576,cudaMemcpyHostToDevice);
    cudaMemcpy(secd, sec_host, sizeof(sec_host), cudaMemcpyHostToDevice);

    /* model_timestep from golden metadata (≈0.001), NOT scheduler time.
     * hd_forward multiplies by 1000 internally (forward.c hd_scale_f32). */
    float ts_host = (float)model_timestep;
    cudaMemcpy(tsd, &ts_host, 4, cudaMemcpyHostToDevice);

    /* input_ids: the text tokens come from golden 01_model_input (int64, [1,19]) */
    gtensor *tin = find_tensor(glist, gn, "01_model_input");
    if (!tin) { printf("FAIL: golden missing 01_model_input\n"); return 1; }
    int64_t *input_ids_h = malloc(T * sizeof(int64_t));
    memcpy(input_ids_h, (const char *)golden + tin->offset, T * sizeof(int64_t));
    int64_t *idsd = dev_alloc(T * sizeof(int64_t));
    cudaMemcpy(idsd, input_ids_h, T * sizeof(int64_t), cudaMemcpyHostToDevice);

    /* ---- diagnostic buffers for every checkpoint ---- */
    size_t hid_bytes = (size_t)S * H * 2;
    void *diag_emb   = dev_alloc((size_t)T * H * 2);
    void *diag_tsc   = dev_alloc((size_t)T * H * 2);
    void *diag_tgt   = dev_alloc((size_t)IMG * H * 2);
    void *diag_b0in  = dev_alloc(hid_bytes);
    void *diag_b0    = dev_alloc(hid_bytes);
    void *diag_bmid  = dev_alloc(hid_bytes);
    void *diag_blast = dev_alloc(hid_bytes);
    void *diag_bnorm = dev_alloc(hid_bytes);
    void *diag_anorm = dev_alloc(hid_bytes);
    void *diag_head  = dev_alloc((size_t)S * FF * 2);
    void *out_dev    = dev_alloc((size_t)S * FF * 2);
    hd_forward_diagnostics diag;
    memset(&diag, 0, sizeof(diag));
    diag.after_embedding            = diag_emb;
    diag.after_timestep_conditioning= diag_tsc;
    diag.after_target_embedding     = diag_tgt;
    diag.after_block_0_input        = diag_b0in;
    diag.after_block_0              = diag_b0;
    diag.after_block_mid            = diag_bmid;
    diag.after_block_last           = diag_blast;
    diag.before_final_norm          = diag_bnorm;
    diag.after_final_norm           = diag_anorm;
    diag.after_final_head           = diag_head;

    /* ---- run the full forward once ---- */
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0, 0);
    st = hd_forward(&bw, &ws, idsd, T, (const float *)posd, maskd,
                    vind, IMG, tsd, secd, S, NH, NKV, H, I, HD, TMS_ID,
                    &diag, out_dev, NULL);
    cudaEventRecord(t1, 0);
    cudaEventSynchronize(t1);
    float wall_ms = 0.0f;
    cudaEventElapsedTime(&wall_ms, t0, t1);
    CHECK(st == HD_OK, "hd_forward ran without error");
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    if (st != HD_OK) { printf("  forward error: %s\n", hd_last_error()); return 1; }
    cudaDeviceSynchronize();
    printf("  native forward wall: %.3f ms\n", wall_ms);

    /* ---- strict-order comparison against golden checkpoints ---- */
    int ok = 1;
    gtensor *g;

    g = find_tensor(glist, gn, "02_embedding_output");
    if (g) ok &= compare_ckpt("embedding", diag_emb, golden, g, 5e-3, 0.9999);
    else { CHECK(0, "golden missing embedding"); ok = 0; }

    g = find_tensor(glist, gn, "03_target_or_image_embedding_output");
    if (g) ok &= compare_ckpt("target_embedding", diag_tgt, golden, g, 5e-3, 0.9999);
    else { CHECK(0, "golden missing target embedding"); ok = 0; }

    g = find_tensor(glist, gn, "04_timestep_conditioning");
    if (g) ok &= compare_ckpt("timestep_conditioning", diag_tsc, golden, g, 5e-3, 0.9999);
    else { CHECK(0, "golden missing timestep conditioning"); ok = 0; }

    g = find_tensor(glist, gn, "05_block_00_output");
    if (g) ok &= compare_ckpt("block_0", diag_b0, golden, g, 1e-2, 0.9999);
    else { CHECK(0, "golden missing block 0"); ok = 0; }

    g = find_tensor(glist, gn, "06_block_mid_output");
    if (g) ok &= compare_ckpt("block_mid", diag_bmid, golden, g, 1e-2, 0.9999);
    else { CHECK(0, "golden missing block mid"); ok = 0; }
    if (g) row_compare("block_mid(native)", diag_bmid, golden, g, S, H);

    g = find_tensor(glist, gn, "07_block_last_output");
    if (g) ok &= compare_ckpt("block_last", diag_blast, golden, g, 3e-2, 0.999);
    else { CHECK(0, "golden missing block last"); ok = 0; }
    if (g) row_compare("block_last(native)", diag_blast, golden, g, S, H);

    g = find_tensor(glist, gn, "08_final_norm_input");
    if (g) ok &= compare_ckpt("final_norm_input", diag_bnorm, golden, g, 3e-2, 0.999);
    else { CHECK(0, "golden missing final norm input"); ok = 0; }

    g = find_tensor(glist, gn, "09_final_norm_output");
    if (g) ok &= compare_ckpt("final_norm", diag_anorm, golden, g, 0.1, 0.99);
    else { CHECK(0, "golden missing final norm"); ok = 0; }

    g = find_tensor(glist, gn, "10_final_head_input");
    if (g) ok &= compare_ckpt("final_head_input", diag_anorm, golden, g, 0.1, 0.99);
    else { CHECK(0, "golden missing final head input"); ok = 0; }

    g = find_tensor(glist, gn, "11_complete_model_output");
    if (g) ok &= compare_ckpt("complete_output", out_dev, golden, g, 0.18, 0.99);
    else { CHECK(0, "golden missing complete output"); ok = 0; }

    /* ---- cleanup ---- */
    free(input_ids_h); free(golden);
    dev_free(posd); dev_free(maskd); dev_free(vind); dev_free(tsd); dev_free(secd);
    dev_free(idsd); dev_free(wsbase);
    dev_free(diag_emb); dev_free(diag_tsc); dev_free(diag_tgt); dev_free(diag_b0in);
    dev_free(diag_b0); dev_free(diag_bmid); dev_free(diag_blast); dev_free(diag_bnorm); dev_free(diag_anorm);
    dev_free(diag_head); dev_free(out_dev);
    hd_forward_binding_free(&bw);
    hd_weight_store_free(&store);

    printf("\n%d assertions passed, %d failed\n", passes, failures);
    printf("strict-order: first divergence gate %s\n", ok ? "PASS" : "FAIL");
    if (failures) return 1;
    return 0;
}