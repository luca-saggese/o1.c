/*
 * M1.4 decisive block-tail experiment (native-only, no Python).
 *
 * Question: is block_last (layer 35) divergence a genuine bug in layers
 * 19..35, or legitimate input-amplified bf16 roundoff drift?
 *
 * Experiment: run ONLY native layers 19..35 (17 layers), seeding the first
 * input with the FROZEN GOLDEN block_mid (layer 18) output. If the native
 * tail then reproduces the GOLDEN block_last (layer 35) output within the
 * block class-D bound, then native layers 19..35 are CORRECT and the M1.4
 * block_last nrmse=0.0246 divergence is purely amplification of the small
 * block_mid input error (0.00599) through the image-row magnitude explosion
 * (rms 9.5 -> 22k), i.e. legitimate bf16 drift. If native tail diverges from
 * golden last even when started from golden mid, layers 19..35 are buggy.
 *
 * No new Python run: golden block_mid / block_last are already frozen.
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

int main(void) {
    /* ---- load golden bin to locate block_mid / block_last ---- */
    size_t gsz = 0;
    void *golden = read_file_bytes(GOLDEN_DIR "/M1_V3_DEV_FORWARD_0.bin", &gsz);
    if (!golden) { printf("FAIL: golden bin\n"); return 1; }
    char meta_path[512];
    snprintf(meta_path, sizeof(meta_path), GOLDEN_DIR "/M1_V3_DEV_FORWARD_0.json");
    size_t msz = 0;
    void *mb = read_file_bytes(meta_path, &msz);
    if (!mb) { printf("FAIL: meta\n"); return 1; }
    const char *err = NULL;
    hd_json *meta = hd_json_parse((const char *)mb, &err);
    free(mb);
    if (!meta) { printf("FAIL: parse meta\n"); return 1; }
    const hd_json *arr = hd_json_get(meta, "tensors");
    size_t off_mid = 0, off_last = 0, n_mid = 0, n_last = 0;
    for (size_t i = 0; i < hd_json_array_len(arr); i++) {
        const hd_json *t = hd_json_array_at(arr, i);
        const char *nm = hd_json_string(hd_json_get(t, "name"));
        if (!nm) continue;
        if (!strcmp(nm, "06_block_mid_output")) {
            off_mid = (size_t)hd_json_int(hd_json_get(t, "offset"), 0);
            n_mid  = (size_t)hd_json_int(hd_json_get(t, "nbytes"), 0);
        } else if (!strcmp(nm, "07_block_last_output")) {
            off_last = (size_t)hd_json_int(hd_json_get(t, "offset"), 0);
            n_last  = (size_t)hd_json_int(hd_json_get(t, "nbytes"), 0);
        }
    }
    hd_json_free(meta);
    if (!n_mid || !n_last) { printf("FAIL: missing mid/last\n"); return 1; }
    printf("golden block_mid bytes=%zu block_last bytes=%zu\n", n_mid, n_last);

    /* ---- inputs: pos_f32, mask ---- */
    size_t ibsz = 0;
    void *inputs = read_file_bytes(GOLDEN_DIR "/inputs.bin", &ibsz);
    if (!inputs) { printf("FAIL: inputs.bin\n"); return 1; }
    char inmeta_path[512];
    snprintf(inmeta_path, sizeof(inmeta_path), GOLDEN_DIR "/inputs.json");
    size_t imsz = 0;
    void *imb = read_file_bytes(inmeta_path, &imsz);
    if (!imb) { printf("FAIL: inputs.json\n"); return 1; }
    hd_json *imeta = hd_json_parse((const char *)imb, &err);
    free(imb);
    if (!imeta) { printf("FAIL: parse inputs.json\n"); return 1; }
    const hd_json *iarr = hd_json_get(imeta, "tensors");
    int64_t off_pos = 0, off_mask = 0;
    for (size_t i = 0; i < hd_json_array_len(iarr); i++) {
        const hd_json *t = hd_json_array_at(iarr, i);
        const char *nm = hd_json_string(hd_json_get(t, "name"));
        if (!nm) continue;
        if (!strcmp(nm, "pos_f32")) off_pos = hd_json_int(hd_json_get(t, "offset"), 0);
        else if (!strcmp(nm, "mask")) off_mask = hd_json_int(hd_json_get(t, "offset"), 0);
    }
    hd_json_free(imeta);
    int64_t sec_host[3] = {24, 20, 20};
    void *posd  = NULL, *maskd = NULL, *secd = NULL;
    cudaMalloc(&posd, 276);  cudaMalloc(&maskd, 1058); cudaMalloc(&secd, 24);
    cudaMemcpy(posd,  (const char *)inputs + off_pos,  276,  cudaMemcpyHostToDevice);
    cudaMemcpy(maskd, (const char *)inputs + off_mask, 1058, cudaMemcpyHostToDevice);
    cudaMemcpy(secd, sec_host, 24, cudaMemcpyHostToDevice);
    free(inputs);

    /* ---- weights ---- */
    hd_st_index idx;
    if (hd_st_index_load(DEV_DIR, &idx) != HD_OK) { printf("FAIL: index\n"); return 1; }
    hd_weight_store store;
    if (hd_weights_to_device(DEV_DIR, &idx, 0, &store) != HD_OK) {
        printf("FAIL: weights_to_device\n"); return 1; }
    hd_st_index_free(&idx);
    hd_forward_binding bw;
    if (hd_forward_resolve(&store, NLAYERS, &bw) != HD_OK) {
        printf("FAIL: resolve\n"); return 1; }
    printf("tail test bound layers %d (running i=%d..%d)\n",
           bw.n_layers, MID+1, NLAYERS-1);

    /* ---- scratch + ping-pong buffers ---- */
    int64_t scr_bytes = hd_decoder_block_scratch_bytes(S, NH, NKV, H, I, HD);
    void *scratch = NULL; cudaMalloc(&scratch, scr_bytes);
    size_t shb = (size_t)S * H * 2;
    void *bufA = NULL, *bufB = NULL; cudaMalloc(&bufA, shb); cudaMalloc(&bufB, shb);

    /* ---- seed bufA with GOLDEN block_mid ---- */
    cudaMemcpy(bufA, (const char *)golden + off_mid, shb, cudaMemcpyHostToDevice);
    /* Optional: seed from native_blocks.bin native mid instead, to isolate
     * tail-test machinery. Enabled when TEST_SEED_NATIVE=1. */
    const char *seed_native = getenv("TEST_SEED_NATIVE");
    if (seed_native && seed_native[0] == '1') {
        size_t ns = 0;
        void *nb = read_file_bytes(GOLDEN_DIR "/native_blocks.bin", &ns);
        if (nb && ns >= shb) cudaMemcpy(bufA, nb, shb, cudaMemcpyHostToDevice);
        printf("[seed=native mid]\n");
        free(nb);
    }

    /* ---- run native layers 19..35 ping-pong ---- */
    const void *cur_in = bufA;
    void *cur_out = bufB;
    for (int i = MID + 1; i < NLAYERS; i++) {
        hd_status s = hd_decoder_block(cur_in, (const float *)posd, maskd,
                                       &bw.blocks[i], secd, scratch, scr_bytes,
                                       NULL, cur_out, S, NH, NKV, H, I, HD, NULL);
        if (s != HD_OK) { printf("FAIL: block %d: %s\n", i, hd_last_error()); return 1; }
        void *tmp = cur_out; cur_out = (void *)cur_in; cur_in = tmp;
    }
    /* final write landed in cur_out then swap -> cur_in holds final */
    cudaDeviceSynchronize();

    /* ---- compare native tail out vs GOLDEN block_last ---- */
    size_t nf32 = n_last / 2;                 /* number of floats (= S*H) */
    float *nf = malloc(nf32 * sizeof(float));
    float *rf = malloc(nf32 * sizeof(float));
    void *host = malloc(n_last);
    cudaMemcpy(host, cur_in, n_last, cudaMemcpyDeviceToHost);
    hd_bf16_buf_to_f32(host, nf, nf32);
    hd_bf16_buf_to_f32((const char *)golden + off_last, rf, nf32);

    /* Self-consistency gate: does the isolated tail reproduce the real
     * forward's NATIVE last (native_blocks.bin second half)? */
    float *native_last = malloc(nf32 * sizeof(float));
    {
        size_t ns = 0;
        void *nb = read_file_bytes(GOLDEN_DIR "/native_blocks.bin", &ns);
        if (nb && ns >= n_last) {
            hd_bf16_buf_to_f32((const char *)nb + n_last, native_last, nf32);
            double base = rms(native_last, nf32) > 0 ? rms(native_last, nf32) : 1.0;
            double nr2 = rms_diff(nf, native_last, nf32) / base;
            double cs2 = cosine(nf, native_last, nf32);
            printf("  isolated-tail vs forward NATIVE last:\n");
            printf("    nrmse=%.5g cos=%.8g  (should be ~0/1 if machinery matches)\n", nr2, cs2);
            if (nb) { /* also independently: tail vs golden ref */ }
        }
        free(nb);
    }
    double nr = rms(rf, n_last/2) > 0 ? rms_diff(nf, rf, n_last/2)/rms(rf, n_last/2) : 0.0;
    double cs = cosine(nf, rf, n_last/2);
    long nan = 0; double maxa = 0;
    for (size_t i = 0; i < n_last/2; i++) {
        double d = fabs((double)nf[i]-rf[i]); if (d>maxa) maxa=d;
        if (isnan(nf[i])||isinf(nf[i])||isnan(rf[i])||isinf(rf[i])) nan++;
    }
    printf("\n=== DECISIVE RESULT: native layers 19..35 seeded with GOLDEN block_mid ===\n");
    printf("  native-tail vs GOLDEN block_last: nrmse=%.5g cos=%.8g max_abs=%.5g nan/inf=%ld\n",
           nr, cs, maxa, nan);
    printf("  -> %s\n", (nr <= 1e-2 && cs >= 0.9999 && nan == 0)
            ? "layers 19..35 CORRECT (drift is input-amplified)"
            : "layers 19..35 BUGGY (real defect)");

    /* row-wise tail detail (image rows 21/22 are the drift amplifiers) */
    printf("  row-wise (every 4th row):\n");
    for (int r = 0; r < S; r += 4) {
        const float *cn = nf + (size_t)r*H, *cr = rf + (size_t)r*H;
        double rc=0, rr=0, dot=0, sd2=0;
        for (int k=0;k<H;k++){double a=cn[k],b=cr[k],dd=a-b;rc+=a*a;rr+=b*b;dot+=a*b;sd2+=dd*dd;}
        rc=sqrt(rc/H); rr=sqrt(rr/H);
        printf("    row %2d nrmse=%.5g cos=%.8g RMS_ref=%.4g RMS_na=%.4g\n",
               r, rr>0?sqrt(sd2/H)/rr:0, rc>0&&rr>0?dot/(rc*rr*H):0, rr, rc);
    }

    /* ------------------------------------------------------------------ */
    /* Norm + head self-consistency gate.                                  */
    /* Seed RMSNorm with the GOLDEN block_last (no drift input); if the    */
    /* final-norm / head primitives then reproduce golden final_norm and   */
    /* complete_model_output within class E, the downstream M1.4 failures  */
    /* are purely amplification of the legitimate block_last drift, not a  */
    /* bug in norm/head. No Python rerun.                                  */
    /* ------------------------------------------------------------------ */
    {
        float eps = 1e-6f;
        void *gnorm_in  = NULL, *gnorm_out = NULL, *ghead_out = NULL;
        cudaMalloc(&gnorm_in,  (size_t)S * H * 2);
        cudaMalloc(&gnorm_out, (size_t)S * H * 2);
        cudaMalloc(&ghead_out, (size_t)S * FF * 2);
        /* seed with golden block_last output */
        cudaMemcpy(gnorm_in, (const char *)golden + off_last, (size_t)S * H * 2,
                   cudaMemcpyHostToDevice);
        hd_rmsnorm(gnorm_in, bw.fnorm_w, gnorm_out, S, H, eps);
        hd_linear(gnorm_out, bw.fl_w, bw.fl_b, ghead_out, S, FF, H, 1);
        cudaDeviceSynchronize();

        /* golden 09_final_norm_output [S,H] and 11_complete_model_output [S,FF] */
        int64_t off_norm = 0, n_norm = 0, off_out = 0, n_out = 0;
        for (size_t i = 0; i < hd_json_array_len(meta); i++) {} /* meta freed above */
        /* re-parse meta for norm/out offsets */
        {
            size_t m2 = 0;
            void *mb2 = read_file_bytes(meta_path, &m2);
            const char *e2 = NULL;
            hd_json *meta2 = hd_json_parse((const char *)mb2, &e2);
            const hd_json *arr2 = hd_json_get(meta2, "tensors");
            for (size_t i = 0; i < hd_json_array_len(arr2); i++) {
                const hd_json *t = hd_json_array_at(arr2, i);
                const char *nm = hd_json_string(hd_json_get(t, "name"));
                if (!nm) continue;
                if (!strcmp(nm, "09_final_norm_output")) {
                    off_norm = hd_json_int(hd_json_get(t, "offset"), 0);
                    n_norm  = hd_json_int(hd_json_get(t, "nbytes"), 0);
                } else if (!strcmp(nm, "11_complete_model_output")) {
                    off_out = hd_json_int(hd_json_get(t, "offset"), 0);
                    n_out   = hd_json_int(hd_json_get(t, "nbytes"), 0);
                }
            }
            hd_json_free(meta2);
            free(mb2);
        }
        if (n_norm && n_out) {
            size_t nn = n_norm/2, no = n_out/2;
            float *nfh = malloc(nn*4), *rfh = malloc(nn*4);
            void *h = malloc(n_norm);
            cudaMemcpy(h, gnorm_out, n_norm, cudaMemcpyDeviceToHost);
            hd_bf16_buf_to_f32(h, nfh, nn);
            hd_bf16_buf_to_f32((const char*)golden+off_norm, rfh, nn);
            double bn = rms(rfh, nn)>0?rms(rfh,nn):1.0;
            double nrm = rms_diff(nfh,rfh,nn)/bn, csn = cosine(nfh,rfh,nn);
            printf("\n=== NORM+HEAD SELF-CONSISTENCY (seed=golden block_last) ===\n");
            printf("  final_norm      nrmse=%.6g cos=%.9g  (class E: 1e-3 / 0.99999)\n", nrm, csn);
            free(nfh); free(rfh); free(h);
            float *nfo = malloc(no*4), *rfo = malloc(no*4);
            void *ho = malloc(n_out);
            cudaMemcpy(ho, ghead_out, n_out, cudaMemcpyDeviceToHost);
            hd_bf16_buf_to_f32(ho, nfo, no);
            hd_bf16_buf_to_f32((const char*)golden+off_out, rfo, no);
            double bo = rms(rfo,no)>0?rms(rfo,no):1.0;
            double nro = rms_diff(nfo,rfo,no)/bo, cso = cosine(nfo,rfo,no);
            printf("  complete output nrmse=%.6g cos=%.9g  (class E: 1e-3 / 0.99999)\n", nro, cso);
            free(nfo); free(rfo); free(ho);
        }
        cudaFree(gnorm_in); cudaFree(gnorm_out); cudaFree(ghead_out);
    }

    return 0;
}
