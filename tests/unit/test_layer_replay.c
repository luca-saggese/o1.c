/*
 * M2: layer-by-layer replay audit (cuBLAS GEMM backend).
 *
 * Uses the frozen all-blocks fixture M1_V3_BASE_BLOCKS_0:
 *   - for each layer i: feed golden block_{i-1} output into the native
 *     decoder block and compare against golden block_i output.
 *   - layer 0 input is the real concat(text_cond [19,H], img_emb [4,H]).
 *
 * This isolates whether the complete_output drift (NRMSE~0.15) is:
 *   A) distributed accumulation across layers
 *   B) a single defective layer
 *   C) final norm / head
 *
 * Also replays golden layer 17 -> native 18..35 and golden layer 27 ->
 * native 28..35, and measures final_norm + final head separately.
 */

#define _POSIX_C_SOURCE 200809L

#include "hidream.h"
#include "safetensors.h"
#include "json.h"
#include "block.h"
#include "forward.h"
#include "hd_cudnn_sdpa.h"
#include "gemm.h"

extern char *hd_cuda_errbuf(void);

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define GOLDEN_DIR "artifacts/m1/golden/M1_V3_BASE_BLOCKS_0"
#define DEV_DIR    "models/dev"

#define NLAYERS 36
#define T 19
#define I 4
#define S (T + I)
#define H 4096
#define NH 32
#define NKV 8
#define HD 128
#define FF 12288

static int failures = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        printf("FAIL: %s (at %s:%d)\n", msg, __FILE__, __LINE__); \
        failures++; \
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

typedef struct { size_t offset; size_t nbytes; } ftensor;

/* Find tensor by exact name in the flat "tensors" array. */
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

static void compare(const char *tag, const void *dev, const void *bin,
                    const ftensor *ref, double *nr_out, double *cs_out) {
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
    for (size_t i = 0; i < n; i++) {
        double d = fabs((double)cf[i] - rf[i]);
        if (d > maxa) maxa = d;
    }
    printf("  %-28s nrmse=%.5g cos=%.8g max_abs=%.5g\n", tag, nr, cs, maxa);
    if (nr_out) *nr_out = nr;
    if (cs_out) *cs_out = cs;
    free(cf); free(rf); free(host);
}

int main(void) {
    char mpath[512], bpath[512];
    snprintf(mpath, sizeof(mpath), GOLDEN_DIR "/M1_V3_BASE_BLOCKS_0.json");
    snprintf(bpath, sizeof(bpath), GOLDEN_DIR "/M1_V3_BASE_BLOCKS_0.bin");
    size_t msz = 0, bsz = 0;
    void *mb = read_file_bytes(mpath, &msz);
    void *bin = read_file_bytes(bpath, &bsz);
    if (!mb || !bin) {
        printf("FAIL: fixture not found (%s)\n", mpath);
        return 1;
    }
    const char *err = NULL;
    hd_json *meta = hd_json_parse((const char *)mb, &err);
    free(mb);
    if (!meta) { printf("FAIL: parse meta\n"); free(bin); return 1; }

    /* ---- weights ---- */
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
    st = hd_forward_resolve(&store, NLAYERS, &bw);
    if (st != HD_OK) { printf("FAIL: hd_forward_resolve: %s\n", hd_last_error()); return 1; }

    /* ---- device inputs ---- */
    ftensor tp, tm, tcond, timg;
    if (find_tensor(meta, "pos_f32", &tp) ||
        find_tensor(meta, "mask", &tm) ||
        find_tensor(meta, "04_timestep_conditioning", &tcond) ||
        find_tensor(meta, "03_target_or_image_embedding_output", &timg)) {
        printf("FAIL: fixture missing inputs\n");
        return 1;
    }
    void *posd = dev_alloc(tp.nbytes);
    void *maskd = dev_alloc(tm.nbytes);
    cudaMemcpy(posd, (const char *)bin + tp.offset, tp.nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(maskd, (const char *)bin + tm.offset, tm.nbytes, cudaMemcpyHostToDevice);

    int64_t sec_host[3] = {24, 20, 20};
    void *secd = dev_alloc(sizeof(sec_host));
    cudaMemcpy(secd, sec_host, sizeof(sec_host), cudaMemcpyHostToDevice);

    /* ---- SDPA plan (production attention) ---- */
    hd_sdpa_plan *plan = NULL;
    float attn_scale = (float)(1.0 / sqrt((double)HD));
    int rc = hd_sdpa_create(&plan, 1, NH, NKV, S, S, HD, attn_scale);
    if (rc != 0) { printf("FAIL: hd_sdpa_create: %s\n", hd_cuda_errbuf()); return 1; }

    /* ---- buffers ---- */
    size_t hid_bytes = (size_t)S * H * 2;
    int64_t need = hd_decoder_block_scratch_bytes(S, NH, NKV, H, FF, HD);
    void *scratch = dev_alloc((size_t)need);
    void *in_a = dev_alloc(hid_bytes);
    void *in_b = dev_alloc(hid_bytes);
    void *out_a = dev_alloc(hid_bytes);
    void *out_b = dev_alloc(hid_bytes);
    if (!scratch || !in_a || !in_b || !out_a || !out_b) {
        printf("FAIL: device allocation\n");
        return 1;
    }

    /* ---- layer 0 input: concat(text_cond [19,H], img_emb [4,H]) ---- */
    cudaMemcpy(in_a, (const char *)bin + tcond.offset, (size_t)T * H * 2, cudaMemcpyHostToDevice);
    cudaMemcpy((uint8_t *)in_a + (size_t)T * H * 2, (const char *)bin + timg.offset,
               (size_t)I * H * 2, cudaMemcpyHostToDevice);

    printf("Layer-by-layer replay (golden in -> native block -> golden out)\n");
    printf("  seq=%d H=%d NH=%d NKV=%d HD=%d FF=%d layers=%d\n", S, H, NH, NKV, HD, FF, NLAYERS);

    double worst_nr = 0.0;
    int worst_layer = -1;
    for (int i = 0; i < NLAYERS; i++) {
        char nm[64];
        snprintf(nm, sizeof(nm), "05_block_%02d_output", i);
        ftensor ref;
        if (find_tensor(meta, nm, &ref)) {
            printf("FAIL: missing golden %s\n", nm);
            return 1;
        }
        st = hd_decoder_block(in_a, (const float *)posd, maskd, &bw.blocks[i],
                              (const int64_t *)secd, scratch, need, NULL, out_a,
                              S, NH, NKV, H, FF, HD, plan);
        if (st != HD_OK) { printf("FAIL: block %d: %s\n", i, hd_last_error()); return 1; }
        cudaDeviceSynchronize();
        double nr, cs;
        compare(nm, out_a, bin, &ref, &nr, &cs);
        if (nr > worst_nr) { worst_nr = nr; worst_layer = i; }
        /* swap: next input = this output */
        void *tmp = in_a; in_a = out_a; out_a = tmp;
    }
    printf("  worst single-layer nrmse=%.5g at layer %d\n", worst_nr, worst_layer);

    /* ---- reference vs cuBLAS on IDENTICAL inputs, per layer ---- */
    printf("Reference vs cuBLAS per layer (golden in -> block out)\n");
    {
        void *r_in = dev_alloc(hid_bytes);
        void *r_out = dev_alloc(hid_bytes);
        void *c_in = dev_alloc(hid_bytes);
        void *c_out = dev_alloc(hid_bytes);
        cudaMemcpy(r_in, (const char *)bin + tcond.offset, (size_t)T * H * 2, cudaMemcpyHostToDevice);
        cudaMemcpy((uint8_t *)r_in + (size_t)T * H * 2, (const char *)bin + timg.offset,
                   (size_t)I * H * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(c_in, r_in, hid_bytes, cudaMemcpyDeviceToDevice);
        hd_gemm_set_backend(0);
        for (int i = 0; i < NLAYERS; i++) {
            st = hd_decoder_block(r_in, (const float *)posd, maskd, &bw.blocks[i],
                                  (const int64_t *)secd, scratch, need, NULL, r_out,
                                  S, NH, NKV, H, FF, HD, plan);
            if (st != HD_OK) { printf("FAIL: ref block %d: %s\n", i, hd_last_error()); return 1; }
            cudaDeviceSynchronize();
            void *tmp = r_in; r_in = r_out; r_out = tmp;
        }
        hd_gemm_set_backend(1);
        for (int i = 0; i < NLAYERS; i++) {
            st = hd_decoder_block(c_in, (const float *)posd, maskd, &bw.blocks[i],
                                  (const int64_t *)secd, scratch, need, NULL, c_out,
                                  S, NH, NKV, H, FF, HD, plan);
            if (st != HD_OK) { printf("FAIL: cublas block %d: %s\n", i, hd_last_error()); return 1; }
            cudaDeviceSynchronize();
            void *tmp = c_in; c_in = c_out; c_out = tmp;
        }
        size_t n = hid_bytes / 2;
        float *fc = malloc(n * sizeof(float));
        float *fr = malloc(n * sizeof(float));
        void *hc = malloc(n * 2);
        void *hr = malloc(n * 2);
        cudaMemcpy(hc, c_in, n * 2, cudaMemcpyDeviceToHost);
        cudaMemcpy(hr, r_in, n * 2, cudaMemcpyDeviceToHost);
        hd_bf16_buf_to_f32(hc, fc, n);
        hd_bf16_buf_to_f32(hr, fr, n);
        double nr = rms(fr, n) > 0.0 ? rms_diff(fc, fr, n) / rms(fr, n) : 0.0;
        double cs = cosine(fc, fr, n);
        printf("  full-chain ref vs cuBLAS (block 35): nrmse=%.5g cos=%.8g\n", nr, cs);
        free(fc); free(fr); free(hc); free(hr);
        dev_free(r_in); dev_free(r_out); dev_free(c_in); dev_free(c_out);
    }

    /* ---- per-projection drift in the worst layer ---- */
    printf("Per-projection reference vs cuBLAS (worst layer %d, golden input)\n", worst_layer);
    {
        /* input = golden block_{worst-1} output (or concat for layer 0) */
        void *p_in = dev_alloc(hid_bytes);
        void *p_out = dev_alloc(hid_bytes);
        void *p_ref = dev_alloc(hid_bytes);
        if (worst_layer == 0) {
            cudaMemcpy(p_in, (const char *)bin + tcond.offset, (size_t)T * H * 2, cudaMemcpyHostToDevice);
            cudaMemcpy((uint8_t *)p_in + (size_t)T * H * 2, (const char *)bin + timg.offset,
                       (size_t)I * H * 2, cudaMemcpyHostToDevice);
        } else {
            char pnm[64];
            snprintf(pnm, sizeof(pnm), "05_block_%02d_output", worst_layer - 1);
            ftensor prev;
            if (find_tensor(meta, pnm, &prev)) { printf("FAIL: missing %s\n", pnm); return 1; }
            cudaMemcpy(p_in, (const char *)bin + prev.offset, hid_bytes, cudaMemcpyHostToDevice);
        }
        /* ln0 = input_layernorm(p_in) — identical for both backends */
        void *ln0 = dev_alloc(hid_bytes);
        hd_rmsnorm(p_in, bw.blocks[worst_layer].input_ln, ln0, S, H, 1e-6f);
        cudaDeviceSynchronize();

        struct { const char *name; const void *w; int M, N, K; } projs[] = {
            { "q_proj",  bw.blocks[worst_layer].q_proj,  S, NH * HD, H },
            { "k_proj",  bw.blocks[worst_layer].k_proj,  S, NKV * HD, H },
            { "v_proj",  bw.blocks[worst_layer].v_proj,  S, NKV * HD, H },
            { "o_proj",  bw.blocks[worst_layer].o_proj,  S, H, NH * HD },
            { "gate_proj", bw.blocks[worst_layer].gate_proj, S, FF, H },
            { "up_proj", bw.blocks[worst_layer].up_proj, S, FF, H },
            { "down_proj", bw.blocks[worst_layer].down_proj, S, H, FF },
        };
        for (size_t pi = 0; pi < sizeof(projs) / sizeof(projs[0]); pi++) {
            const void *inp = (strcmp(projs[pi].name, "gate_proj") == 0 ||
                               strcmp(projs[pi].name, "up_proj") == 0 ||
                               strcmp(projs[pi].name, "down_proj") == 0) ? ln0 : ln0;
            (void)inp;
            hd_gemm_set_backend(0);
            hd_linear(ln0, projs[pi].w, NULL, p_ref, projs[pi].M, projs[pi].N, projs[pi].K, 1);
            cudaDeviceSynchronize();
            hd_gemm_set_backend(1);
            hd_linear(ln0, projs[pi].w, NULL, p_out, projs[pi].M, projs[pi].N, projs[pi].K, 1);
            cudaDeviceSynchronize();
            size_t pn = (size_t)projs[pi].M * projs[pi].N;
            float *fc = malloc(pn * sizeof(float));
            float *fr = malloc(pn * sizeof(float));
            void *hc = malloc(pn * 2);
            void *hr = malloc(pn * 2);
            cudaMemcpy(hc, p_out, pn * 2, cudaMemcpyDeviceToHost);
            cudaMemcpy(hr, p_ref, pn * 2, cudaMemcpyDeviceToHost);
            hd_bf16_buf_to_f32(hc, fc, pn);
            hd_bf16_buf_to_f32(hr, fr, pn);
            double nr = rms(fr, pn) > 0.0 ? rms_diff(fc, fr, pn) / rms(fr, pn) : 0.0;
            double cs = cosine(fc, fr, pn);
            double maxa = 0.0;
            for (size_t i = 0; i < pn; i++) {
                double d = fabs((double)fc[i] - fr[i]);
                if (d > maxa) maxa = d;
            }
            printf("  %-10s M=%4d N=%5d K=%5d  nrmse=%.5g cos=%.8g max_abs=%.5g\n",
                   projs[pi].name, projs[pi].M, projs[pi].N, projs[pi].K, nr, cs, maxa);
            free(fc); free(fr); free(hc); free(hr);
        }
        dev_free(p_in); dev_free(p_out); dev_free(p_ref); dev_free(ln0);
    }

    /* ---- cuBLAS vs reference, layer by layer (same golden input) ---- */
    printf("cuBLAS vs reference per layer (golden in -> native block)\n");
    {
        void *ref_in = dev_alloc(hid_bytes);
        void *ref_out = dev_alloc(hid_bytes);
        cudaMemcpy(ref_in, (const char *)bin + tcond.offset, (size_t)T * H * 2, cudaMemcpyHostToDevice);
        cudaMemcpy((uint8_t *)ref_in + (size_t)T * H * 2, (const char *)bin + timg.offset,
                   (size_t)I * H * 2, cudaMemcpyHostToDevice);
        hd_gemm_set_backend(0);
        for (int i = 0; i < NLAYERS; i++) {
            st = hd_decoder_block(ref_in, (const float *)posd, maskd, &bw.blocks[i],
                                  (const int64_t *)secd, scratch, need, NULL, ref_out,
                                  S, NH, NKV, H, FF, HD, plan);
            if (st != HD_OK) { printf("FAIL: ref block %d: %s\n", i, hd_last_error()); return 1; }
            cudaDeviceSynchronize();
            void *tmp = ref_in; ref_in = ref_out; ref_out = tmp;
        }
        hd_gemm_set_backend(1);
        /* in_a currently holds the cuBLAS chain output (after the swap loop
         * above, in_a holds block_35 output). Compare against ref chain. */
        size_t n = hid_bytes / 2;
        float *fc = malloc(n * sizeof(float));
        float *fr = malloc(n * sizeof(float));
        void *hc = malloc(n * 2);
        void *hr = malloc(n * 2);
        cudaMemcpy(hc, in_a, n * 2, cudaMemcpyDeviceToHost);
        cudaMemcpy(hr, ref_in, n * 2, cudaMemcpyDeviceToHost);
        hd_bf16_buf_to_f32(hc, fc, n);
        hd_bf16_buf_to_f32(hr, fr, n);
        double nr = rms(fr, n) > 0.0 ? rms_diff(fc, fr, n) / rms(fr, n) : 0.0;
        double cs = cosine(fc, fr, n);
        printf("  cublas-chain vs ref-chain (block 35): nrmse=%.5g cos=%.8g\n", nr, cs);
        free(fc); free(fr); free(hc); free(hr);
        dev_free(ref_in); dev_free(ref_out);
    }

    /* ---- replay golden 17 -> native 18..35 ---- */
    printf("Replay golden layer 17 -> native 18..35\n");
    {
        ftensor g17;
        if (find_tensor(meta, "05_block_17_output", &g17)) return 1;
        cudaMemcpy(in_a, (const char *)bin + g17.offset, hid_bytes, cudaMemcpyHostToDevice);
        for (int i = 18; i < NLAYERS; i++) {
            st = hd_decoder_block(in_a, (const float *)posd, maskd, &bw.blocks[i],
                                  (const int64_t *)secd, scratch, need, NULL, out_a,
                                  S, NH, NKV, H, FF, HD, plan);
            if (st != HD_OK) { printf("FAIL: block %d: %s\n", i, hd_last_error()); return 1; }
            cudaDeviceSynchronize();
            void *tmp = in_a; in_a = out_a; out_a = tmp;
        }
        ftensor g35;
        if (find_tensor(meta, "05_block_35_output", &g35)) return 1;
        compare("native 18..35 vs golden 35", in_a, bin, &g35, NULL, NULL);
    }

    /* ---- replay golden 27 -> native 28..35 ---- */
    printf("Replay golden layer 27 -> native 28..35\n");
    {
        ftensor g27;
        if (find_tensor(meta, "05_block_27_output", &g27)) return 1;
        cudaMemcpy(in_a, (const char *)bin + g27.offset, hid_bytes, cudaMemcpyHostToDevice);
        for (int i = 28; i < NLAYERS; i++) {
            st = hd_decoder_block(in_a, (const float *)posd, maskd, &bw.blocks[i],
                                  (const int64_t *)secd, scratch, need, NULL, out_a,
                                  S, NH, NKV, H, FF, HD, plan);
            if (st != HD_OK) { printf("FAIL: block %d: %s\n", i, hd_last_error()); return 1; }
            cudaDeviceSynchronize();
            void *tmp = in_a; in_a = out_a; out_a = tmp;
        }
        ftensor g35;
        if (find_tensor(meta, "05_block_35_output", &g35)) return 1;
        compare("native 28..35 vs golden 35", in_a, bin, &g35, NULL, NULL);
    }

    /* ---- final norm + head measured separately ---- */
    printf("Final norm + head (golden inputs)\n");
    {
        ftensor fni, fno, fhi, fho;
        if (find_tensor(meta, "41_final_norm_input", &fni) ||
            find_tensor(meta, "42_final_norm_output", &fno) ||
            find_tensor(meta, "43_final_head_input", &fhi) ||
            find_tensor(meta, "44_final_head_output", &fho)) {
            printf("FAIL: fixture missing final tensors\n");
            return 1;
        }
        void *norm_in = dev_alloc(fni.nbytes);
        void *norm_out = dev_alloc(fno.nbytes);
        void *head_out = dev_alloc(fho.nbytes);
        cudaMemcpy(norm_in, (const char *)bin + fni.offset, fni.nbytes, cudaMemcpyHostToDevice);
        hd_rmsnorm(norm_in, bw.fnorm_w, norm_out, S, H, 1e-6f);
        cudaDeviceSynchronize();
        compare("final_norm (golden in)", norm_out, bin, &fno, NULL, NULL);

        cudaMemcpy(norm_in, (const char *)bin + fhi.offset, fhi.nbytes, cudaMemcpyHostToDevice);
        hd_linear(norm_in, bw.fl_w, bw.fl_b, head_out, S, 3072, H, 1);
        cudaDeviceSynchronize();
        compare("final_head (golden in)", head_out, bin, &fho, NULL, NULL);

        /* ---- decisive: same buffers, 5 configs ---- */
        const int M = S, N = 3072, K = H;
        void *head_ref = dev_alloc(fho.nbytes);
        void *head_f32 = dev_alloc((size_t)M * N * 4);
        cublasHandle_t ch;
        cublasCreate(&ch);
        cublasSetStream(ch, 0);
        float alpha = 1.0f, beta = 0.0f;
        double flops = 2.0 * M * N * K;
        size_t n = fho.nbytes / 2;
        /* bias is a DEVICE pointer; copy to host once. */
        uint16_t *bias_h = malloc((size_t)N * 2);
        cudaMemcpy(bias_h, bw.fl_b, (size_t)N * 2, cudaMemcpyDeviceToHost);

        printf("  final-head decisive: M=%d N=%d K=%d X=%p W=%p B=%p\n",
               M, N, K, norm_in, bw.fl_w, bw.fl_b);
        printf("    transA=OP_T transB=OP_N m=N=%d n=M=%d k=K=%d lda=K=%d ldb=K=%d ldc=N=%d\n",
               N, M, K, K, K, N);
        {
            uint16_t w8[8], x8[8];
            cudaMemcpy(w8, bw.fl_w, 16, cudaMemcpyDeviceToHost);
            cudaMemcpy(x8, norm_in, 16, cudaMemcpyDeviceToHost);
            printf("    first 8 W: ");
            for (int i = 0; i < 8; i++) printf("%.4f ", hd_bf16_to_f32(w8[i]));
            printf("\n    first 8 X: ");
            for (int i = 0; i < 8; i++) printf("%.4f ", hd_bf16_to_f32(x8[i]));
            printf("\n");
        }

        /* host-side bias add in fp32, then bf16 round. */
        void add_bias(uint16_t *y) {
            for (int m = 0; m < M; m++)
                for (int nn = 0; nn < N; nn++) {
                    float v = hd_bf16_to_f32(y[(size_t)m * N + nn]) + hd_bf16_to_f32(bias_h[nn]);
                    y[(size_t)m * N + nn] = hd_f32_to_bf16(v);
                }
        }

        /* A. reference */
        {
            double t0 = now_s();
            hd_gemm_set_backend(0);
            hd_linear(norm_in, bw.fl_w, bw.fl_b, head_ref, M, N, K, 1);
            cudaDeviceSynchronize();
            double ms = (now_s() - t0) * 1e3;
            hd_gemm_set_backend(1);
            compare("A.ref", head_ref, bin, &fho, NULL, NULL);
            printf("    A.ref %.3f ms %.1f TFLOP/s\n", ms, flops / (ms * 1e-3) / 1e12);
        }

        /* B. cuBLAS 32F + DEFAULT */
        {
            double t0 = now_s();
            cublasGemmEx(ch, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
                         &alpha, bw.fl_w, CUDA_R_16BF, K,
                                 norm_in, CUDA_R_16BF, K,
                         &beta,  head_out, CUDA_R_16BF, N,
                         CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
            cudaDeviceSynchronize();
            double ms = (now_s() - t0) * 1e3;
            uint16_t *yh = malloc(n * 2);
            cudaMemcpy(yh, head_out, n * 2, cudaMemcpyDeviceToHost);
            add_bias(yh);
            float *cf = malloc(n * sizeof(float));
            float *rf = malloc(n * sizeof(float));
            hd_bf16_buf_to_f32(yh, cf, n);
            hd_bf16_buf_to_f32((const char *)bin + fho.offset, rf, n);
            double nr = rms(rf, n) > 0.0 ? rms_diff(cf, rf, n) / rms(rf, n) : 0.0;
            double cs = cosine(cf, rf, n);
            printf("  B.cublas nrmse=%.5g cos=%.8g  %.3f ms %.1f TFLOP/s\n",
                   nr, cs, ms, flops / (ms * 1e-3) / 1e12);
            free(yh); free(cf); free(rf);
        }

        /* C. cuBLAS 32F + DEFAULT + DISALLOW_REDUCED_PRECISION_REDUCTION */
        {
            cublasSetMathMode(ch, CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
            double t0 = now_s();
            cublasGemmEx(ch, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
                         &alpha, bw.fl_w, CUDA_R_16BF, K,
                                 norm_in, CUDA_R_16BF, K,
                         &beta,  head_out, CUDA_R_16BF, N,
                         CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
            cudaDeviceSynchronize();
            double ms = (now_s() - t0) * 1e3;
            uint16_t *yh = malloc(n * 2);
            cudaMemcpy(yh, head_out, n * 2, cudaMemcpyDeviceToHost);
            add_bias(yh);
            float *cf = malloc(n * sizeof(float));
            float *rf = malloc(n * sizeof(float));
            hd_bf16_buf_to_f32(yh, cf, n);
            hd_bf16_buf_to_f32((const char *)bin + fho.offset, rf, n);
            double nr = rms(rf, n) > 0.0 ? rms_diff(cf, rf, n) / rms(rf, n) : 0.0;
            double cs = cosine(cf, rf, n);
            printf("  C.disallow nrmse=%.5g cos=%.8g  %.3f ms %.1f TFLOP/s\n",
                   nr, cs, ms, flops / (ms * 1e-3) / 1e12);
            free(yh); free(cf); free(rf);
            cublasSetMathMode(ch, CUBLAS_DEFAULT_MATH);
        }

        /* D. cuBLAS 32F_PEDANTIC + DEFAULT */
        {
            double t0 = now_s();
            cublasGemmEx(ch, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
                         &alpha, bw.fl_w, CUDA_R_16BF, K,
                                 norm_in, CUDA_R_16BF, K,
                         &beta,  head_out, CUDA_R_16BF, N,
                         CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT);
            cudaDeviceSynchronize();
            double ms = (now_s() - t0) * 1e3;
            uint16_t *yh = malloc(n * 2);
            cudaMemcpy(yh, head_out, n * 2, cudaMemcpyDeviceToHost);
            add_bias(yh);
            float *cf = malloc(n * sizeof(float));
            float *rf = malloc(n * sizeof(float));
            hd_bf16_buf_to_f32(yh, cf, n);
            hd_bf16_buf_to_f32((const char *)bin + fho.offset, rf, n);
            double nr = rms(rf, n) > 0.0 ? rms_diff(cf, rf, n) / rms(rf, n) : 0.0;
            double cs = cosine(cf, rf, n);
            printf("  D.pedantic nrmse=%.5g cos=%.8g  %.3f ms %.1f TFLOP/s\n",
                   nr, cs, ms, flops / (ms * 1e-3) / 1e12);
            free(yh); free(cf); free(rf);
        }

        /* E. cuBLAS BF16 in / FP32 compute / FP32 out, explicit BF16 convert */
        {
            double t0 = now_s();
            cublasGemmEx(ch, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
                         &alpha, bw.fl_w, CUDA_R_16BF, K,
                                 norm_in, CUDA_R_16BF, K,
                         &beta,  head_f32, CUDA_R_32F, N,
                         CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
            cudaDeviceSynchronize();
            double ms = (now_s() - t0) * 1e3;
            float *yhf = malloc(n * sizeof(float));
            uint16_t *yh = malloc(n * 2);
            cudaMemcpy(yhf, head_f32, n * 4, cudaMemcpyDeviceToHost);
            for (int m = 0; m < M; m++)
                for (int nn = 0; nn < N; nn++) {
                    float v = yhf[(size_t)m * N + nn] + hd_bf16_to_f32(bias_h[nn]);
                    yh[(size_t)m * N + nn] = hd_f32_to_bf16(v);
                }
            float *cf = malloc(n * sizeof(float));
            float *rf = malloc(n * sizeof(float));
            hd_bf16_buf_to_f32(yh, cf, n);
            hd_bf16_buf_to_f32((const char *)bin + fho.offset, rf, n);
            double nr = rms(rf, n) > 0.0 ? rms_diff(cf, rf, n) / rms(rf, n) : 0.0;
            double cs = cosine(cf, rf, n);
            printf("  E.f32out nrmse=%.5g cos=%.8g  %.3f ms %.1f TFLOP/s\n",
                   nr, cs, ms, flops / (ms * 1e-3) / 1e12);
            free(yhf); free(yh); free(cf); free(rf);
        }

        free(bias_h);
        cublasDestroy(ch);
        dev_free(head_ref); dev_free(head_f32);
        dev_free(norm_in); dev_free(norm_out); dev_free(head_out);
    }

    /* ---- cleanup ---- */
    dev_free(posd); dev_free(maskd); dev_free(secd); dev_free(scratch);
    dev_free(in_a); dev_free(in_b); dev_free(out_a); dev_free(out_b);
    hd_sdpa_destroy(plan);
    hd_forward_binding_free(&bw);
    hd_weight_store_free(&store);
    hd_json_free(meta); free(bin);

    printf("\n%s\n", failures ? "REPLAY FAIL" : "REPLAY PASS");
    return failures ? 1 : 0;
}