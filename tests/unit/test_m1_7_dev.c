/*
 * M1.7 full Dev generation validation harness (V5).
 *
 * Drives the native forward + native scheduler for the COMPLETE 28-step
 * denoising trajectory on the frozen M1_7_DEV_FULLGEN golden fixture and
 * validates, per docs/M1_7_TOLERANCES.md:
 *
 *   1. per-step z_next staircase (NRMSE/cos per step range) + max_abs_err
 *      and no NaN/Inf;
 *   2. final decoded 64x64 image vs the golden decode: pixel MAE <= 4.0,
 *      PSNR >= 32 dB, self-contained SSIM >= 0.90 (8x8 windows, slide 4,
 *      box kernels, no skimage);
 *   3. step0 invariant: z_prev == M1.4 vinputs (byte-equal);
 *   4. final step semantics: manifest sigma[28] = 0.0 -> z_next = denoised
 *      (the oracle's noise-skip rule replicated naturally, no hardcoded
 *      step-index skip).
 *
 * The native run replays the frozen per-step post-clamp noise verbatim
 * (docs/M1_NUMERICAL_CONTRACT.md section 7): any z-chain divergence is
 * attributable to native forward drift, never to RNG/scheduler mismatch.
 *
 * BF16 compute / FP32 accumulate. A single device-resident forward executes
 * the SAME production path hd_forward as the engine.
 */

#include "hidream.h"
#include "safetensors.h"
#include "json.h"
#include "block.h"
#include "forward.h"
#include "scheduler.h"

#include <cuda_runtime.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GOLDEN_DIR "artifacts/m1/golden/M1_7_DEV_FULLGEN"
#define M14_DIR   "artifacts/m1/golden/M1_V3_DEV_FORWARD_0"
#define DEV_DIR   "models/dev"
#define MANIFEST  "config/startup_manifest_dev.json"

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

#define NIMG (IMG * FF)          /* 4 * 3072 = 12288 elements */
#define NUM_STEPS 28
#define S_NOISE 8.0f
#define PATCH 32                /* 64x64 image, 4 tokens, 3 channels */

/* staircase tolerances (docs/M1_7_TOLERANCES.md section 3.2) */
#define TOL_STEPS0 16           /* steps 0..15 */
#define TOL_STEPS1 24           /* steps 16..23 */
/* steps 24..27 use the final pair */
#define NRMSE_T0 5e-3
#define COS_T0   0.9999
#define NRMSE_T1 1e-2
#define COS_T1   0.9995
#define NRMSE_T2 2.5e-2
#define COS_T2   0.999
#define MAX_ABS_ULPS 8.0   /* max_abs_err <= 8 * ULP(max_abs(golden)) */

/* image gate limits (docs/M1_7_TOLERANCES.md section 4.1) */
#define IMG_MAE_LIM  4.0
#define IMG_PSNR_LIM 32.0
#define IMG_SSIM_LIM 0.90

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
/* Metrics                                                             */
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

/* ------------------------------------------------------------------ */
/* Golden layout                                                       */
/* ------------------------------------------------------------------ */

typedef struct { size_t offset; size_t nbytes; char name[64]; } gtensor;

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

/* z_next chain compare: candidate device bf16 vs golden bf16 slice. */
static int compare_znext(const char *tag, const void *dev,
                         const void *bin, const gtensor *ref,
                         double nrmse_lim, double cos_lim) {
    size_t n = ref->nbytes / 2;
    float *cf = malloc(n * sizeof(float));
    float *rf = malloc(n * sizeof(float));
    void *host = malloc(n * 2);
    cudaMemcpy(host, dev, n * 2, cudaMemcpyDeviceToHost);
    hd_bf16_buf_to_f32(host, cf, n);
    hd_bf16_buf_to_f32((const char *)bin + ref->offset, rf, n);
    double nr = 0.0, cs = 0.0, maxa = 0.0, gmax = 0.0;
    if (rms(rf, n) > 0.0) nr = rms_diff(cf, rf, n) / rms(rf, n);
    cs = cosine(cf, rf, n);
    long nan = 0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs((double)cf[i] - rf[i]);
        if (d > maxa) maxa = d;
        double ga = fabs((double)rf[i]);
        if (ga > gmax) gmax = ga;
        if (isnan(cf[i]) || isinf(cf[i]) || isnan(rf[i]) || isinf(rf[i])) nan++;
    }
    /* bf16 ULP at the golden tensor's max magnitude (7 mantissa bits) */
    double ulp = (gmax > 0.0) ? pow(2.0, floor(log2(gmax)) - 7.0) : 0.0;
    double maxa_lim = MAX_ABS_ULPS * ulp;
    int ok = (nr <= nrmse_lim) && (cs >= cos_lim) && (maxa <= maxa_lim) &&
             (nan == 0);
    printf("  %-30s nrmse=%.5g cos=%.8g max_abs=%.5g (lim %.3g = %.0f ulp) "
           "nan/inf=%ld [lims %.2g/%.4g] -> %s\n",
           tag, nr, cs, maxa, maxa_lim, MAX_ABS_ULPS, nan, nrmse_lim, cos_lim,
           ok ? "PASS" : "FAIL");
    CHECK(ok, tag);
    free(cf); free(rf); free(host);
    return ok;
}

/* ------------------------------------------------------------------ */
/* Image decode + metrics (self-contained, no skimage)                 */
/* ------------------------------------------------------------------ */

/* Decode z bf16[4,3072] -> uint8[64,64,3] per docs/M1_7_TOLERANCES.md 4. */
static void decode_z(const void *z_dev, unsigned char *out) {
    size_t n = NIMG;
    float *zf = malloc(n * sizeof(float));
    void *host = malloc(n * 2);
    cudaMemcpy(host, z_dev, n * 2, cudaMemcpyDeviceToHost);
    hd_bf16_buf_to_f32(host, zf, n);
    for (int tok = 0; tok < IMG; tok++) {
        for (int c = 0; c < 3; c++) {
            for (int p1 = 0; p1 < PATCH; p1++) {
                for (int p2 = 0; p2 < PATCH; p2++) {
                    float v = (zf[(size_t)tok * 3072 + (size_t)c * 1024 +
                                  (size_t)p1 * 32 + p2] + 1.0f) / 2.0f;
                    float px = roundf(v * 255.0f);
                    if (px < 0.0f) px = 0.0f;
                    if (px > 255.0f) px = 255.0f;
                    out[((size_t)(tok / 2 * 32 + p1) * 64 +
                         (size_t)(tok % 2 * 32 + p2)) * 3 + c] =
                        (unsigned char)px;
                }
            }
        }
    }
    free(zf); free(host);
}

static double px_mae(const unsigned char *a, const unsigned char *b, size_t n) {
    double s = 0.0;
    for (size_t i = 0; i < n; i++) s += fabs((double)a[i] - b[i]);
    return s / (double)n;
}
static double px_rmse(const unsigned char *a, const unsigned char *b, size_t n) {
    double s = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = (double)a[i] - b[i];
        s += d * d;
    }
    return sqrt(s / (double)n);
}

/* SSIM over 8x8 windows, slide 4, box kernels, per docs/M1_7_TOLERANCES.md. */
static double ssim_3ch(const unsigned char *a, const unsigned char *b,
                       int W, int Hh, int win, int slide) {
    double C1 = pow(0.01 * 255.0, 2.0);
    double C2 = pow(0.03 * 255.0, 2.0);
    long total = 0;
    double acc = 0.0;
    for (int wy = 0; wy + win <= Hh; wy += slide) {
        for (int wx = 0; wx + win <= W; wx += slide) {
            for (int c = 0; c < 3; c++) {
                double ma = 0.0, mb = 0.0, va = 0.0, vb = 0.0, cov = 0.0;
                for (int dy = 0; dy < win; dy++) {
                    for (int dx = 0; dx < win; dx++) {
                        double x = a[((size_t)(wy + dy) * W + wx + dx) * 3 + c];
                        double y = b[((size_t)(wy + dy) * W + wx + dx) * 3 + c];
                        ma += x; mb += y;
                    }
                }
                double inv = 1.0 / (double)(win * win);
                ma *= inv; mb *= inv;
                for (int dy = 0; dy < win; dy++) {
                    for (int dx = 0; dx < win; dx++) {
                        double x = a[((size_t)(wy + dy) * W + wx + dx) * 3 + c];
                        double y = b[((size_t)(wy + dy) * W + wx + dx) * 3 + c];
                        va += (x - ma) * (x - ma);
                        vb += (y - mb) * (y - mb);
                        cov += (x - ma) * (y - mb);
                    }
                }
                va *= inv; vb *= inv; cov *= inv;
                acc += (2.0 * ma * mb + C1) * (2.0 * cov + C2) /
                       ((ma * ma + mb * mb + C1) * (va + vb + C2));
                total++;
            }
        }
    }
    return total ? acc / (double)total : 1.0;
}

/* ------------------------------------------------------------------ */

int main(void) {
    char meta_path[512], bin_path[512];
    snprintf(meta_path, sizeof(meta_path), GOLDEN_DIR "/M1_7_DEV_FULLGEN.json");
    snprintf(bin_path,  sizeof(bin_path),  GOLDEN_DIR "/M1_7_DEV_FULLGEN.bin");
    gtensor glist[512];
    int gn = load_golden_layout(meta_path, glist, 512);
    if (gn < 0) { printf("FAIL: cannot load golden layout %s\n", meta_path); return 1; }

    size_t gsz = 0;
    void *golden = read_file_bytes(bin_path, &gsz);
    if (!golden) { printf("FAIL: golden bin not found\n"); return 1; }

    /* ---- M1.4 inputs: pos_f32 / mask / vinputs (step0 z_prev) ---- */
    char inb_path[512];
    snprintf(inb_path, sizeof(inb_path), M14_DIR "/inputs.bin");
    size_t ibsz = 0;
    void *inputs = read_file_bytes(inb_path, &ibsz);
    if (!inputs) { printf("FAIL: M1.4 inputs.bin not found\n"); free(golden); return 1; }

    char inmeta_path[512];
    snprintf(inmeta_path, sizeof(inmeta_path), M14_DIR "/inputs.json");
    size_t imsz = 0;
    void *imb = read_file_bytes(inmeta_path, &imsz);
    if (!imb) { printf("FAIL: inputs.json not found\n"); free(inputs); free(golden); return 1; }
    const char *err = NULL;
    hd_json *imeta = hd_json_parse((const char *)imb, &err);
    free(imb);
    if (!imeta) { printf("FAIL: parse inputs.json\n"); free(inputs); free(golden); return 1; }
    const hd_json *iarr = hd_json_get(imeta, "tensors");
    int64_t off_pos = 0, off_mask = 0, off_vin = 0;
    size_t n_i = iarr ? hd_json_array_len(iarr) : 0;
    for (size_t i = 0; i < n_i; i++) {
        const hd_json *t = hd_json_array_at(iarr, i);
        const char *nm = hd_json_string(hd_json_get(t, "name"));
        int64_t off = hd_json_int(hd_json_get(t, "offset"), 0);
        if (!nm) continue;
        if (!strcmp(nm, "pos_f32")) off_pos = off;
        else if (!strcmp(nm, "mask")) off_mask = off;
        else if (!strcmp(nm, "vinputs")) off_vin = off;
    }
    hd_json_free(imeta);

    printf("M1.7 full Dev generation validation (V5, fixture M1_7_DEV_FULLGEN)\n");
    printf("  text=%d img=%d seq=%d H=%d layers=%d steps=%d s_noise=%.1f\n",
           T, IMG, S, H, NLAYERS, NUM_STEPS, S_NOISE);

    /* step0 vinputs must be byte-equal to golden step00_z_prev (proven invariant) */
    gtensor *gzp0 = find_tensor(glist, gn, "step00_z_prev");
    if (!gzp0) { printf("FAIL: golden missing step00_z_prev\n"); return 1; }
    int vin_eq = (gzp0->nbytes == 24576) &&
        memcmp((const char *)inputs + off_vin, (const char *)golden + gzp0->offset,
               24576) == 0;
    printf("  step0 vinputs == step00_z_prev byte-equal: %s\n",
           vin_eq ? "PASS" : "FAIL");
    CHECK(vin_eq, "step0 vinputs byte-equal to golden step00_z_prev");

    /* ---- manifest sigmas from config/startup_manifest_dev.json ---- */
    float sigmas[HD_SCHED_MAX_STEPS];
    int n_sigmas = 0;
    {
        size_t msz = 0;
        void *mb = read_file_bytes(MANIFEST, &msz);
        if (!mb) { printf("FAIL: cannot read %s\n", MANIFEST); return 1; }
        const char *jerr = NULL;
        hd_json *root = hd_json_parse((const char *)mb, &jerr);
        free(mb);
        if (!root) { printf("FAIL: parse manifest\n"); return 1; }
        const hd_json *sched = hd_json_get(root, "scheduler");
        const hd_json *sa = sched ? hd_json_get(sched, "sigmas") : NULL;
        if (sa && sa->type == HD_JSON_ARRAY) {
            n_sigmas = (int)hd_json_array_len(sa);
            if (n_sigmas > HD_SCHED_MAX_STEPS) n_sigmas = HD_SCHED_MAX_STEPS;
            for (int i = 0; i < n_sigmas; i++)
                sigmas[i] = (float)hd_json_double(hd_json_array_at(sa, i), 0.0);
        }
        hd_json_free(root);
    }
    if (n_sigmas < NUM_STEPS + 1) {
        printf("FAIL: manifest sigmas too short (%d)\n", n_sigmas);
        return 1;
    }
    printf("  manifest sigmas[0..3] = %.9f %.9f %.9f %.9f\n",
           sigmas[0], sigmas[1], sigmas[2], sigmas[3]);
    printf("  manifest sigma[%d] (final) = %.9f\n", n_sigmas - 1,
           sigmas[n_sigmas - 1]);
    CHECK(sigmas[n_sigmas - 1] == 0.0f,
          "final manifest sigma = 0.0 (noise-skip semantics via schedule)");

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

    hd_forward_binding bw;
    st = hd_forward_resolve(&store, NLAYERS, &bw);
    if (st != HD_OK) {
        printf("FAIL: hd_forward_resolve: %s\n", hd_last_error());
        hd_weight_store_free(&store); free(inputs); free(golden); return 1;
    }

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

    /* ---- stage device inputs from M1.4 inputs.bin ---- */
    const char *p = (const char *)inputs;
    void *posd  = dev_alloc(276);
    void *maskd = dev_alloc(1058);
    void *vind  = dev_alloc(24576);
    int64_t sec_host[3] = {24, 20, 20};
    void *secd = dev_alloc(sizeof(sec_host));
    cudaMemcpy(posd,  p + off_pos,  276,  cudaMemcpyHostToDevice);
    cudaMemcpy(maskd, p + off_mask, 1058, cudaMemcpyHostToDevice);
    cudaMemcpy(vind,  p + off_vin,  24576,cudaMemcpyHostToDevice);
    cudaMemcpy(secd, sec_host, sizeof(sec_host), cudaMemcpyHostToDevice);

    /* input_ids from M1.4 golden 01_model_input (int64, [1,19]) */
    char m14bin_path[512];
    snprintf(m14bin_path, sizeof(m14bin_path), M14_DIR "/M1_V3_DEV_FORWARD_0.bin");
    size_t m14sz = 0;
    void *m14bin = read_file_bytes(m14bin_path, &m14sz);
    if (!m14bin) { printf("FAIL: M1.4 golden bin not found\n"); return 1; }
    char m14meta_path[512];
    snprintf(m14meta_path, sizeof(m14meta_path), M14_DIR "/M1_V3_DEV_FORWARD_0.json");
    gtensor m14list[64];
    int m14n = load_golden_layout(m14meta_path, m14list, 64);
    if (m14n < 0) { printf("FAIL: cannot load M1.4 golden layout\n"); return 1; }
    gtensor *tin = find_tensor(m14list, m14n, "01_model_input");
    if (!tin) { printf("FAIL: M1.4 golden missing 01_model_input\n"); return 1; }
    int64_t *input_ids_h = malloc(T * sizeof(int64_t));
    memcpy(input_ids_h, (const char *)m14bin + tin->offset, T * sizeof(int64_t));
    free(m14bin);
    int64_t *idsd = dev_alloc(T * sizeof(int64_t));
    cudaMemcpy(idsd, input_ids_h, T * sizeof(int64_t), cudaMemcpyHostToDevice);

    /* ---- device buffers for the scheduler chain ---- */
    size_t nimg = (size_t)NIMG;
    void *z_prev_dev = dev_alloc(24576);          /* bf16 [4,3072] */
    void *z_next_dev = dev_alloc(24576);          /* bf16 [4,3072] */
    float *mo_dev    = dev_alloc(nimg * 4);       /* fp32 model_output */
    float *noise_dev = dev_alloc(nimg * 4);      /* fp32 post-clamp noise */
    float *scratch   = dev_alloc(3 * nimg * 4);   /* scheduler scratch 3*n */
    float *tsd       = dev_alloc(4);              /* model_timestep fp32 */
    void *out_dev    = dev_alloc((size_t)S * FF * 2); /* [23,3072] bf16 */
    void *xp_dev     = dev_alloc(24576);          /* x_pred_masked [4,3072] bf16 */
    if (!z_prev_dev || !z_next_dev || !mo_dev || !noise_dev || !scratch ||
        !tsd || !out_dev || !xp_dev) {
        printf("FAIL: device alloc oom\n");
        return 1;
    }

    hd_scheduler sched;
    st = hd_scheduler_init(&sched, sigmas, n_sigmas, S_NOISE);
    CHECK(st == HD_OK, "scheduler init from manifest sigmas (29 incl 0.0)");
    if (st != HD_OK) return 1;

    /* ================================================================ */
    /* Full 28-step chain (the M1.7 gate)                               */
    /* ================================================================ */
    printf("\n28-step z-chain gate (native forward + native scheduler)\n");
    cudaMemcpy(z_prev_dev, (const char *)inputs + off_vin, 24576,
               cudaMemcpyHostToDevice);   /* step0 z_prev == M1.4 vinputs */

    int chain_ok = 1;
    for (int i = 0; i < NUM_STEPS; i++) {
        char nm[64];
        snprintf(nm, sizeof(nm), "step%02d_model_timestep", i);
        gtensor *gmt = find_tensor(glist, gn, nm);
        snprintf(nm, sizeof(nm), "step%02d_sigma", i);
        gtensor *gsig = find_tensor(glist, gn, nm);
        snprintf(nm, sizeof(nm), "step%02d_model_output", i);
        gtensor *gmo = find_tensor(glist, gn, nm);
        snprintf(nm, sizeof(nm), "step%02d_noise_post_clamp", i);
        gtensor *gno = find_tensor(glist, gn, nm);
        snprintf(nm, sizeof(nm), "step%02d_z_next", i);
        gtensor *gzn = find_tensor(glist, gn, nm);
        if (!gmt || !gsig || !gmo || !gno || !gzn) {
            printf("FAIL: golden missing step%02d tensors\n", i);
            chain_ok = 0; break;
        }

        cudaMemcpy(tsd, (const char *)golden + gmt->offset, 4,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(noise_dev, (const char *)golden + gno->offset, nimg * 4,
                   cudaMemcpyHostToDevice);
        st = hd_forward(&bw, &ws, idsd, T, (const float *)posd, maskd,
                        z_prev_dev, IMG, tsd, secd, S, NH, NKV, H, I, HD,
                        TMS_ID, NULL, out_dev, NULL);
        if (st != HD_OK) { printf("FAIL: hd_forward step%d: %s\n",
                                  i, hd_last_error()); chain_ok = 0; break; }
        cudaMemcpy(xp_dev, (const char *)out_dev + (size_t)19 * FF * 2, 24576,
                   cudaMemcpyDeviceToDevice);
        float sigma;
        memcpy(&sigma, (const char *)golden + gsig->offset, 4);
        hd_sched_vcond(z_prev_dev, xp_dev, sigma, mo_dev, (int)nimg);
        st = hd_scheduler_step(&sched, z_prev_dev, mo_dev, noise_dev, S_NOISE,
                               z_next_dev, (int)nimg, scratch);
        if (st != HD_OK) { printf("FAIL: scheduler step%d: %s\n",
                                  i, hd_last_error()); chain_ok = 0; break; }

        double nr_lim, cs_lim;
        if (i < TOL_STEPS0) { nr_lim = NRMSE_T0; cs_lim = COS_T0; }
        else if (i < TOL_STEPS1) { nr_lim = NRMSE_T1; cs_lim = COS_T1; }
        else { nr_lim = NRMSE_T2; cs_lim = COS_T2; }
        char tag[64];
        snprintf(tag, sizeof(tag), "step%02d z_next", i);
        chain_ok &= compare_znext(tag, z_next_dev, golden, gzn, nr_lim, cs_lim);
        cudaMemcpy(z_prev_dev, z_next_dev, 24576, cudaMemcpyDeviceToDevice);
    }
    CHECK(chain_ok, "28-step z_next chain parity gate");

    /* ================================================================ */
    /* Final image decode + metrics (docs/M1_7_TOLERANCES.md section 4) */
    /* ================================================================ */
    printf("\nFinal image gate (native decode vs golden decode, 64x64x3)\n");
    gtensor *gimg = find_tensor(glist, gn, "final_image");
    if (!gimg || gimg->nbytes != 64 * 64 * 3) {
        printf("FAIL: golden missing final_image\n");
        return 1;
    }
    unsigned char *gold_img = malloc(64 * 64 * 3);
    memcpy(gold_img, (const char *)golden + gimg->offset, 64 * 64 * 3);
    unsigned char *nat_img = malloc(64 * 64 * 3);
    decode_z(z_prev_dev, nat_img);   /* z_prev_dev holds step27 z_next */

    double mae = px_mae(nat_img, gold_img, 64 * 64 * 3);
    double rmse = px_rmse(nat_img, gold_img, 64 * 64 * 3);
    double psnr = 20.0 * log10(255.0 / (rmse + 1e-12));
    double ssim = ssim_3ch(nat_img, gold_img, 64, 64, 8, 4);
    printf("  pixel MAE  = %.4f (lim %.2f)\n", mae, IMG_MAE_LIM);
    printf("  PSNR       = %.2f dB (lim >= %.1f)\n", psnr, IMG_PSNR_LIM);
    printf("  SSIM (8x8) = %.5f (lim >= %.2f)\n", ssim, IMG_SSIM_LIM);
    CHECK(mae <= IMG_MAE_LIM, "final image pixel MAE gate");
    CHECK(psnr >= IMG_PSNR_LIM, "final image PSNR gate");
    CHECK(ssim >= IMG_SSIM_LIM, "final image SSIM gate");

    /* ---- cleanup ---- */
    free(gold_img); free(nat_img);
    free(input_ids_h); free(inputs); free(golden);
    dev_free(posd); dev_free(maskd); dev_free(vind); dev_free(secd);
    dev_free(idsd); dev_free(wsbase);
    dev_free(z_prev_dev); dev_free(z_next_dev); dev_free(mo_dev);
    dev_free(noise_dev); dev_free(scratch); dev_free(tsd);
    dev_free(out_dev); dev_free(xp_dev);
    hd_forward_binding_free(&bw);
    hd_weight_store_free(&store);

    printf("\n%d assertions passed, %d failed\n", passes, failures);
    if (failures) return 1;
    return 0;
}
