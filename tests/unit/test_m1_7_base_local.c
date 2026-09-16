/*
 * M1.7 Base closure: local-layer-18 decisive test + first-half drift profile.
 *
 * Answers the M1.7 Base gate question: block_mid (layer 18) NRMSE = 0.010551
 * vs the Class D bound (1e-2, 5.5% over). Is that a genuine layer-18 bug, or
 * legitimate input-amplified accumulated BF16 drift across the preceding 18
 * layers?
 *
 * Two gates, no Python rerun (golden fixture M1_V3_BASE_BLOCKS_0 already
 * captured all 36 block outputs):
 *
 *  A) E_local_18 (decisive): seed native absolute layer 18 with the EXACT
 *     golden layer-17 output, run ONLY native block 18, compare vs golden
 *     layer-18 output.
 *        E_local_18 <= 1e-2  -> layer 18 is internally correct; 0.010551 is
 *                               accumulated upstream drift (do NOT loosen
 *                               Class D)
 *        E_local_18 >  1e-2  -> genuine layer-18 defect, debug internally.
 *
 *  B) First-half drift profile: run native blocks 0..k accumulating from
 *     the exact golden step-5 input (cat of 04_timestep_conditioning +
 *     03_target_or_image_embedding_output), compare vs golden layer-k for
 *     k in {0, 4, 8, 12, 16, 17, 18}. Shows how native-vs-oracle drift
 *     accumulates across the first half of the network.
 *
 * The golden layers are absolute indices 0..35 (05_block_XX_output).
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

#define GOLDEN_DIR "artifacts/m1/golden/M1_V3_BASE_BLOCKS_0"
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

/* First-half drift profile probe layers (accumulated native 0..k). */
static const int PROBE_LAYERS[] = {0, 4, 8, 12, 16, 17, 18};
#define NPROBE ((int)(sizeof(PROBE_LAYERS) / sizeof(PROBE_LAYERS[0])))

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

/* Row-wise NRMSE/cos for the decisive layer-18 comparison. */
static void row_metrics(const char *tag, const float *nf, const float *rf,
                        int rows, int Hw) {
    printf("  %s row-wise (every 4th row):\n", tag);
    for (int r = 0; r < rows; r += 4) {
        const float *cn = nf + (size_t)r * Hw, *cr = rf + (size_t)r * Hw;
        double rc = 0, rr = 0, dot = 0, sd2 = 0;
        for (int k = 0; k < Hw; k++) {
            double a = cn[k], b = cr[k], d = a - b;
            rc += a * a; rr += b * b; dot += a * b; sd2 += d * d;
        }
        rc = sqrt(rc / Hw); rr = sqrt(rr / Hw);
        printf("    row %2d nrmse=%.5g cos=%.8g RMS_ref=%.4g RMS_na=%.4g\n",
               r, rr > 0 ? sqrt(sd2 / Hw) / rr : 0,
               rc > 0 && rr > 0 ? dot / (rc * rr * Hw) : 0, rr, rc);
    }
}

int main(void) {
    /* ---- load golden bin + meta ---- */
    size_t gsz = 0;
    void *golden = read_file_bytes(GOLDEN_DIR "/M1_V3_BASE_BLOCKS_0.bin", &gsz);
    if (!golden) { printf("FAIL: golden bin\n"); return 1; }
    char meta_path[512];
    snprintf(meta_path, sizeof(meta_path), GOLDEN_DIR "/M1_V3_BASE_BLOCKS_0.json");
    size_t msz = 0;
    void *mb = read_file_bytes(meta_path, &msz);
    if (!mb) { printf("FAIL: meta\n"); return 1; }
    const char *err = NULL;
    hd_json *meta = hd_json_parse((const char *)mb, &err);
    free(mb);
    if (!meta) { printf("FAIL: parse meta\n"); return 1; }

    const hd_json *arr = hd_json_get(meta, "tensors");
    /* Map block index -> {offset, nbytes} in the golden bin. */
    size_t blk_off[NLAYERS], blk_nb[NLAYERS];
    size_t off_pos = 0, n_pos = 0, off_mask = 0, n_mask = 0;
    size_t off_tsc = 0, n_tsc = 0, off_tgt = 0, n_tgt = 0;
    memset(blk_off, 0, sizeof(blk_off));
    for (size_t i = 0; i < (size_t)hd_json_array_len(arr); i++) {
        const hd_json *t = hd_json_array_at(arr, i);
        const char *nm = hd_json_string(hd_json_get(t, "name"));
        if (!nm) continue;
        size_t off = (size_t)hd_json_int(hd_json_get(t, "offset"), 0);
        size_t nb  = (size_t)hd_json_int(hd_json_get(t, "nbytes"), 0);
        if (!strncmp(nm, "05_block_", 9)) {
            int li = atoi(nm + 9);
            if (li >= 0 && li < NLAYERS) { blk_off[li] = off; blk_nb[li] = nb; }
        } else if (!strcmp(nm, "pos_f32"))   { off_pos = off; n_pos = nb; }
        else if (!strcmp(nm, "mask"))        { off_mask = off; n_mask = nb; }
        else if (!strcmp(nm, "04_timestep_conditioning")) { off_tsc = off; n_tsc = nb; }
        else if (!strcmp(nm, "03_target_or_image_embedding_output")) { off_tgt = off; n_tgt = nb; }
    }
    double model_timestep = hd_json_double(hd_json_get(meta, "model_timestep"), -1.0);
    double sched_ts = hd_json_double(hd_json_get(meta, "scheduler_timestep"), -1.0);
    hd_json_free(meta);

    int missing = 0;
    for (int i = 0; i < NLAYERS; i++) if (blk_nb[i] == 0) missing++;
    if (missing || !n_pos || !n_mask || !n_tsc || !n_tgt) {
        printf("FAIL: golden missing tensors (blocks missing=%d, pos/mask/tsc/tgt=%d%d%d%d)\n",
               missing, n_pos != 0, n_mask != 0, n_tsc != 0, n_tgt != 0);
        return 1;
    }
    printf("M1.7 Base local-layer-18 test (fixture M1_V3_BASE_BLOCKS_0)\n");
    printf("  scheduler_timestep=%.1f model_timestep=%.6f\n", sched_ts, model_timestep);

    /* ---- device inputs ---- */
    int64_t sec_host[3] = {24, 20, 20};
    void *posd = NULL, *maskd = NULL, *secd = NULL;
    cudaMalloc(&posd, n_pos); cudaMalloc(&maskd, n_mask); cudaMalloc(&secd, 24);
    cudaMemcpy(posd,  (const char *)golden + off_pos,  n_pos,  cudaMemcpyHostToDevice);
    cudaMemcpy(maskd, (const char *)golden + off_mask, n_mask, cudaMemcpyHostToDevice);
    cudaMemcpy(secd, sec_host, 24, cudaMemcpyHostToDevice);

    /* ---- weights ---- */
    hd_st_index idx;
    if (hd_st_index_load(BASE_DIR, &idx) != HD_OK) { printf("FAIL: index\n"); return 1; }
    hd_weight_store store;
    if (hd_weights_to_device(BASE_DIR, &idx, 0, &store) != HD_OK) {
        printf("FAIL: weights_to_device\n"); return 1; }
    hd_st_index_free(&idx);
    hd_forward_binding bw;
    if (hd_forward_resolve(&store, NLAYERS, &bw) != HD_OK) {
        printf("FAIL: resolve\n"); return 1; }

    /* ---- scratch + ping-pong ---- */
    int64_t scr_bytes = hd_decoder_block_scratch_bytes(S, NH, NKV, H, I, HD);
    void *scratch = NULL; cudaMalloc(&scratch, scr_bytes);
    size_t shb = (size_t)S * H * 2;      /* [S,H] bf16 bytes */
    void *bufA = NULL, *bufB = NULL;
    cudaMalloc(&bufA, shb); cudaMalloc(&bufB, shb);

    size_t nf = S * H;
    float *nf_f = malloc(nf * sizeof(float));
    float *rf_f = malloc(nf * sizeof(float));
    void *host = malloc(shb);

    /* ================================================================== */
    /* Gate A (decisive): E_local_18                                      */
    /* golden layer-17 output -> native block 18 -> vs golden layer-18    */
    /* ================================================================== */
    cudaMemcpy(bufA, (const char *)golden + blk_off[17], shb,
               cudaMemcpyHostToDevice);
    hd_status st = hd_decoder_block(bufA, (const float *)posd, maskd,
                                    &bw.blocks[18], secd, scratch, scr_bytes,
                                    NULL, bufB, S, NH, NKV, H, I, HD);
    if (st != HD_OK) { printf("FAIL: block 18: %s\n", hd_last_error()); return 1; }
    cudaDeviceSynchronize();
    cudaMemcpy(host, bufB, shb, cudaMemcpyDeviceToHost);
    hd_bf16_buf_to_f32(host, nf_f, nf);
    hd_bf16_buf_to_f32((const char *)golden + blk_off[18], rf_f, nf);
    double nr18 = rms(rf_f, nf) > 0 ? rms_diff(nf_f, rf_f, nf) / rms(rf_f, nf) : 0.0;
    double cs18 = cosine(nf_f, rf_f, nf);
    long nan18 = 0; double maxa18 = 0;
    for (size_t i = 0; i < nf; i++) {
        double d = fabs((double)nf_f[i] - rf_f[i]); if (d > maxa18) maxa18 = d;
        if (isnan(nf_f[i]) || isinf(nf_f[i]) || isnan(rf_f[i]) || isinf(rf_f[i])) nan18++;
    }
    printf("\n=== GATE A (decisive): E_local_18 ===\n");
    printf("  seed=golden layer-17 output -> native block 18 -> vs golden layer-18\n");
    printf("  nrmse=%.6g cos=%.9g max_abs=%.5g nan/inf=%ld\n",
           nr18, cs18, maxa18, nan18);
    printf("  E_local_18 <= 1e-2 -> %s\n",
           (nr18 <= 1e-2 && cs18 >= 0.9999 && nan18 == 0)
               ? "layer 18 internally CORRECT (0.010551 = accumulated drift)"
               : "layer 18 BUGGY (real defect)");
    row_metrics("E_local_18", nf_f, rf_f, S, H);

    /* ================================================================== */
    /* Gate B: first-half drift profile (native blocks 0..k vs golden)    */
    /* Seed = exact golden step-5 input: cat(04_timestep_conditioning,    */
    /*        03_target_or_image_embedding_output)                        */
    /* ================================================================== */
    printf("\n=== GATE B: first-half drift profile ===\n");
    /* seed bufA = tsc [T,H] + target [IMG,H] */
    size_t tsc_b = (size_t)T * H * 2, tgt_b = (size_t)IMG * H * 2;
    cudaMemcpy(bufA, (const char *)golden + off_tsc, tsc_b, cudaMemcpyHostToDevice);
    cudaMemcpy((uint8_t *)bufA + tsc_b, (const char *)golden + off_tgt, tgt_b,
               cudaMemcpyHostToDevice);

    const void *cur_in = bufA;
    void *cur_out = bufB;
    int pi = 0;
    for (int i = 0; i < NLAYERS; i++) {
        hd_status s = hd_decoder_block(cur_in, (const float *)posd, maskd,
                                       &bw.blocks[i], secd, scratch, scr_bytes,
                                       NULL, cur_out, S, NH, NKV, H, I, HD);
        if (s != HD_OK) { printf("FAIL: block %d: %s\n", i, hd_last_error()); return 1; }
        void *tmp = cur_out; cur_out = (void *)cur_in; cur_in = tmp;
        if (pi < NPROBE && PROBE_LAYERS[pi] == i) {
            /* `cur_in` holds layer i output after swap */
            cudaMemcpy(host, cur_in, shb, cudaMemcpyDeviceToHost);
            hd_bf16_buf_to_f32(host, nf_f, nf);
            hd_bf16_buf_to_f32((const char *)golden + blk_off[i], rf_f, nf);
            double nr = rms(rf_f, nf) > 0 ? rms_diff(nf_f, rf_f, nf) / rms(rf_f, nf) : 0.0;
            double cs = cosine(nf_f, rf_f, nf);
            printf("  layer %2d  accumulated nrmse=%.5g cos=%.8g\n", i, nr, cs);
            pi++;
        }
    }

    /* ---- cleanup ---- */
    free(golden); free(nf_f); free(rf_f); free(host);
    cudaFree(posd); cudaFree(maskd); cudaFree(secd);
    cudaFree(scratch); cudaFree(bufA); cudaFree(bufB);
    hd_forward_binding_free(&bw);
    hd_weight_store_free(&store);

    printf("\nE_local_18=%g -> %s\n", nr18,
           nr18 <= 1e-2 ? "PASS (drift classification)" : "FAIL (bug)");
    return nr18 <= 1e-2 ? 0 : 1;
}
