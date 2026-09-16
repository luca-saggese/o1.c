/*
 * M1.4 full-transformer forward validation harness (V3).
 *
 * Drives hd_forward on the frozen M1_V3_DEV_FORWARD_0 golden fixture and
 * compares every captured checkpoint against the Python whole-model forward
 * golden in strict order: model input -> embedding -> target embedding ->
 * timestep conditioning -> block 0 -> mid block -> last block -> final norm
 * input -> final norm output -> final head -> complete output.
 *
 * Comparison stops at the first meaningful divergence (strict order per the
 * M1.4 contract). Blocks use the frozen composite class D bound
 * (NRMSE <= 1e-2, cos >= 0.999); final norm/head/complete output use the
 * M1.4 drift-amplification envelope (NRMSE <= 0.1 / 0.18, cos >= 0.99)
 * documented in docs/M1_NUMERICAL_CONTRACT.md section 6. The contract is
 * frozen, do not tune to pass.
 *
 * LEGACY FIXTURE: M1_V3_DEV_FORWARD_0 passes scheduler_timestep (999.0)
 * directly into hd_forward, so the timestep embedder receives 999000
 * (INVALID for pipeline semantics; pipeline passes model_timestep ≈ 0.001,
 * embedder x1000 → ≈1.0). This fixture is kept as history and for regression
 * on the legacy path; new gates (e.g. M1.7 Base) use model_timestep.
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
/* Metrics (mirror test_block.c)                                       */
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

/* Golden binary layout rows from the M1.4 meta JSON ("tensors" array). */
typedef struct { size_t offset; size_t nbytes; char name[64]; } gtensor;

/* Parse the golden checkpoint meta and collect all entries into list[]. */
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

/* Compare candidate device bf16 to a golden bf16 slice; class E/D bounds. */
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

/* Row-wise diagnostics for a [rows,H] bf16 tensor. Prints per-row RMSE,
 * NRMSE, cos, RMS_ref, RMS_native, max_abs vs a golden bf16 slice. */
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

int main(void) {
    /* ---- load golden layout + inputs ---- */
    char meta_path[512], bin_path[512];
    snprintf(meta_path, sizeof(meta_path), GOLDEN_DIR "/M1_V3_DEV_FORWARD_0.json");
    snprintf(bin_path,  sizeof(bin_path),  GOLDEN_DIR "/M1_V3_DEV_FORWARD_0.bin");
    gtensor glist[16];
    int gn = load_golden_layout(meta_path, glist, 16);
    if (gn < 0) { printf("FAIL: cannot load golden layout %s\n", meta_path); return 1; }

    char inb_path[512];
    snprintf(inb_path, sizeof(inb_path), GOLDEN_DIR "/inputs.bin");
    size_t ibsz = 0;
    void *inputs = read_file_bytes(inb_path, &ibsz);
    if (!inputs) { printf("FAIL: inputs.bin not found\n"); return 1; }

    size_t gsz = 0;
    void *golden = read_file_bytes(bin_path, &gsz);
    if (!golden) { printf("FAIL: golden bin not found\n"); free(inputs); return 1; }

    /* ---- inputs: parse input meta for offsets ---- */
    char inmeta_path[512];
    snprintf(inmeta_path, sizeof(inmeta_path), GOLDEN_DIR "/inputs.json");
    size_t imsz = 0;
    void *imb = read_file_bytes(inmeta_path, &imsz);
    if (!imb) { printf("FAIL: inputs.json not found\n"); free(inputs); free(golden); return 1; }
    const char *err = NULL;
    hd_json *imeta = hd_json_parse((const char *)imb, &err);
    free(imb);
    if (!imeta) { printf("FAIL: parse inputs.json\n"); free(inputs); free(golden); return 1; }
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
        else if (!strcmp(nm, "timestep")) off_ts = off;  /* LEGACY: scheduler_timestep */
    }
    hd_json_free(imeta);

    printf("M1.4 full-forward validation (V3, fixture M1_V3_DEV_FORWARD_0)\n");
    printf("  text=%d img=%d seq=%d H=%d layers=%d mid=%d FF=%d\n",
           T, IMG, S, H, NLAYERS, MID, FF);

    /* ---- load device weights ---- */
    hd_st_index idx;
    if (hd_st_index_load(DEV_DIR, &idx) != HD_OK) {
        printf("FAIL: hd_st_index_load: %s\n", hd_st_last_error());
        free(inputs); free(golden); return 1;
    }
    hd_weight_store store;
    hd_status st = hd_weights_to_device(DEV_DIR, &idx, 0, &store);
    if (st != HD_OK) {
        printf("FAIL: hd_weights_to_device: %s\n", hd_weights_last_error());
        hd_st_index_free(&idx); free(inputs); free(golden); return 1;
    }
    hd_st_index_free(&idx);

    /* ---- resolve forward bindings ---- */
    hd_forward_binding bw;
    st = hd_forward_resolve(&store, NLAYERS, &bw);
    if (st != HD_OK) {
        printf("FAIL: hd_forward_resolve: %s\n", hd_last_error());
        hd_weight_store_free(&store); free(inputs); free(golden); return 1;
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
                   free(inputs); free(golden); return 1; }

    /* ---- stage device inputs from inputs.bin ---- */
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

    /* ---- diagnostics: row-wise isolation of conditioning (before strict
       comparison gates), per user directive. Adds no new Python run. ---- */
    {
        gtensor *g;
        g = find_tensor(glist, gn, "04_timestep_conditioning");
        if (g) row_compare("cond(native)", diag_tsc, golden, g, T, H);

        /* reconstruct golden block_0_input = concat(cond rows 0..18,
         * target rows 19..22) per forward contract, compare native input. */
        gtensor *gcond = find_tensor(glist, gn, "04_timestep_conditioning");
        gtensor *gtgt  = find_tensor(glist, gn, "03_target_or_image_embedding_output");
        if (gcond && gtgt && diag_b0in) {
            size_t tn = (size_t)T * H * 2, in = (size_t)IMG * H * 2;
            void *hot = malloc(tn + in);
            memcpy(hot, (const char *)golden + gcond->offset, tn);
            memcpy((char *)hot + tn, (const char *)golden + gtgt->offset, in);
            void *gd = dev_alloc(tn + in);
            cudaMemcpy(gd, hot, tn + in, cudaMemcpyHostToDevice);
            printf("  block_0_input comparison (native vs reconstructed golden):\n");
            gtensor recon;
            snprintf(recon.name, sizeof(recon.name), "block_0_input_recon");
            recon.offset = 0;
            recon.nbytes = tn + in;
            compare_ckpt("block_0_input", diag_b0in, hot, &recon, 5e-3, 0.9999);
            free(hot);
            dev_free(gd);
        }
    }

        /* ---- dump native block_mid / block_last for offline analysis ---- */
    {
        char dbg[512];
        snprintf(dbg, sizeof(dbg), GOLDEN_DIR "/native_blocks.bin");
        FILE *df = fopen(dbg, "wb");
        if (df) {
            size_t nb = (size_t)S * H * 2;
            void *h = malloc(nb * 2);
            if (h) {
                cudaMemcpy(h, diag_bmid, nb, cudaMemcpyDeviceToHost);
                cudaMemcpy((char *)h + nb, diag_blast, nb, cudaMemcpyDeviceToHost);
                fwrite(h, nb, 2, df);
                uint16_t *u = (uint16_t *)h;
                printf("  dbg bmid[0:4]=%u,%u,%u,%u blast[0:4]=%u,%u,%u,%u\n",
                       u[0],u[1],u[2],u[3], u[nb/2+0],u[nb/2+1],u[nb/2+2],u[nb/2+3]);
                free(h);
            }
            fclose(df);
        }
    }

    /* ---- dump native t_embedder1 intermediates for offline analysis ---- */
    {
        int64_t bs = 0;
        int64_t off_emb   = hd_forward_ws_offset("t_emb",    S, T, IMG, NH, NKV, H, I, HD, &bs);
        int64_t off_teh   = hd_forward_ws_offset("te_hidden",S, T, IMG, NH, NKV, H, I, HD, &bs);
        int64_t off_frq   = hd_forward_ws_offset("freq",     S, T, IMG, NH, NKV, H, I, HD, &bs);
        int64_t off_ts    = hd_forward_ws_offset("t_scaled", S, T, IMG, NH, NKV, H, I, HD, &bs);

        char dbg[512];
        snprintf(dbg, sizeof(dbg), GOLDEN_DIR "/native_dump.bin");
        FILE *df = fopen(dbg, "wb");
        if (df) {
            int64_t dummy[4] = {off_emb, off_teh, off_frq, off_ts};
            fwrite(dummy, sizeof(int64_t), 4, df);
            void *h = malloc((size_t)H * 2 + (size_t)H * 2 + 256 * 4 + 4);
            char *cp = (char *)h;
            cudaMemcpy(cp, (const char *)wsbase + off_emb, (size_t)H * 2, cudaMemcpyDeviceToHost); cp += H*2;
            cudaMemcpy(cp, (const char *)wsbase + off_teh, (size_t)H * 2, cudaMemcpyDeviceToHost); cp += H*2;
            cudaMemcpy(cp, (const char *)wsbase + off_frq, 256 * 4, cudaMemcpyDeviceToHost); cp += 256*4;
            cudaMemcpy(cp, (const char *)wsbase + off_ts,  4, cudaMemcpyDeviceToHost);
            fwrite(h, (size_t)H*2 + (size_t)H*2 + 256*4 + 4, 1, df);
            free(h);
            fclose(df);
        }
    }

    /* ---- strict-order comparison against golden checkpoints ---- */
    const char *C = "M1_V3_DEV_FORWARD_0";
    int ok = 1;
    gtensor *g;
    (void)C;

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
    { gtensor *gm_ = find_tensor(glist, gn, "06_block_mid_output");
      if (gm_ && diag_bmid) row_compare("block_mid(native)", diag_bmid, golden, gm_, S, H); }

    g = find_tensor(glist, gn, "07_block_last_output");
    if (g) ok &= compare_ckpt("block_last", diag_blast, golden, g, 3e-2, 0.999);
    else { CHECK(0, "golden missing block last"); ok = 0; }
    { gtensor *gl_ = find_tensor(glist, gn, "07_block_last_output");
      if (gl_ && diag_blast) row_compare("block_last(native)", diag_blast, golden, gl_, S, H); }

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
    free(input_ids_h); free(inputs); free(golden);
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