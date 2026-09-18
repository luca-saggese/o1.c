/*
 * M2 ref-image visual conditioning test.
 *
 * Loads the oracle capture (artifacts/ref_image_audit/oracle_visual_stages.pt)
 * and validates the native vision tower stage-by-stage:
 *   patch_embed -> pos_embed -> rot -> 27 blocks -> merger -> deepstack
 *
 * The oracle .pt is a torch save; we read it via a small Python helper that
 * dumps the tensors to raw .bin files (see tools/ref_image/dump_oracle.py).
 * This test reads those .bin files. If the oracle dump is absent, the test
 * reports SKIP.
 *
 * Numerics: BF16 compute / FP32 accumulate (M1 contract). Comparison uses
 * cosine + NRMSE against the oracle fp32 tensors.
 */

#include "hd_cuda.h"
#include "hidream.h"
#include "vision.h"
#include "vision_kernels.h"
#include "gemm.h"

#include <cuda_runtime.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ORACLE_DIR "artifacts/ref_image_audit/oracle_dump"

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

static float *load_f32(const char *name, size_t *n_out) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s.bin", ORACLE_DIR, name);
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    float *buf = malloc((size_t)sz);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        free(buf); fclose(f); return NULL;
    }
    fclose(f);
    *n_out = (size_t)sz / sizeof(float);
    return buf;
}

static float cosine(const float *a, const float *b, size_t n) {
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < n; i++) {
        dot += (double)a[i] * b[i];
        na += (double)a[i] * a[i];
        nb += (double)b[i] * b[i];
    }
    if (na == 0 || nb == 0) return 0.0f;
    return (float)(dot / (sqrt(na) * sqrt(nb)));
}

static float nrmse(const float *a, const float *b, size_t n) {
    double num = 0, den = 0;
    for (size_t i = 0; i < n; i++) {
        double d = (double)a[i] - b[i];
        num += d * d;
        den += (double)b[i] * b[i];
    }
    if (den == 0) return 0.0f;
    return (float)sqrt(num / den);
}

int main(void) {
    size_t n_pv = 0, n_pe = 0, n_rot = 0, n_emb = 0, n_ds = 0;
    float *pv = load_f32("pixel_values", &n_pv);
    float *pe = load_f32("pos_embed", &n_pe);
    float *rot = load_f32("rot_pos", &n_rot);
    float *emb = load_f32("image_embeds", &n_emb);
    float *ds = load_f32("deepstack_0", &n_ds);

    if (!pv || !pe || !rot || !emb || !ds) {
        printf("SKIP: oracle dump not found in %s (run tools/ref_image/dump_oracle.py)\n",
               ORACLE_DIR);
        return 0;
    }
    printf("oracle: pv=%zu pe=%zu rot=%zu emb=%zu ds=%zu\n",
           n_pv, n_pe, n_rot, n_emb, n_ds);

    int n = 520;          /* 26*20 */
    int grid_h = 26, grid_w = 20;
    int m = 130;          /* n/4 */
    int H = HD_VISION_HIDDEN;

    /* ---- device buffers ---- */
    void *pv_d = NULL, *emb_d = NULL, *ds_d = NULL;
    cudaMalloc(&pv_d, (size_t)n * HD_VISION_PATCH_DIM * 2);
    cudaMalloc(&emb_d, (size_t)m * HD_VISION_OUT_HIDDEN * 2);
    cudaMalloc(&ds_d, (size_t)m * HD_VISION_OUT_HIDDEN * 2);

    /* upload pixel_values as bf16 */
    {
        float *pv_f32 = malloc((size_t)n * HD_VISION_PATCH_DIM * sizeof(float));
        memcpy(pv_f32, pv, (size_t)n * HD_VISION_PATCH_DIM * sizeof(float));
        void *pv_f32_d = NULL;
        cudaMalloc(&pv_f32_d, (size_t)n * HD_VISION_PATCH_DIM * 4);
        cudaMemcpy(pv_f32_d, pv_f32, (size_t)n * HD_VISION_PATCH_DIM * 4,
                   cudaMemcpyHostToDevice);
        hd_f32_convert_bf16(pv_f32_d, pv_d, n * HD_VISION_PATCH_DIM);
        cudaFree(pv_f32_d);
        free(pv_f32);
    }

    /* ---- resolve weights from the real model ---- */
    hd_st_index idx;
    if (hd_st_index_load("models/dev", &idx) != HD_OK) {
        printf("FAIL: index load\n");
        return 1;
    }
    hd_weight_store store;
    if (hd_weights_to_device("models/dev", &idx, 0, &store) != HD_OK) {
        printf("FAIL: weights load: %s\n", hd_weights_last_error());
        return 1;
    }
    hd_st_index_free(&idx);

    hd_vision_binding vb;
    if (hd_vision_resolve(&store, &vb) != HD_OK) {
        printf("FAIL: vision resolve: %s\n", hd_last_error());
        return 1;
    }

    /* ---- workspace ---- */
    int64_t ws_bytes = hd_vision_workspace_bytes(n);
    printf("ws_bytes=%lld\n", (long long)ws_bytes);
    void *wsbase = NULL;
    cudaMalloc(&wsbase, (size_t)ws_bytes);
    hd_vision_workspace ws;
    memset(&ws, 0, sizeof(ws));
    ws.patch_out = wsbase;
    ws.bytes = ws_bytes;

    /* cuDNN SDPA plan for the vision attention */
    {
        hd_sdpa_plan *plan = NULL;
        float scale = (float)(1.0 / sqrt((double)HD_VISION_HEAD_DIM));
        int rc = hd_sdpa_create(&plan, 1, HD_VISION_HEADS, HD_VISION_HEADS,
                                n, n, HD_VISION_HEAD_DIM, scale);
        if (rc == 0) ws.sdpa = plan;
        else printf("note: cuDNN SDPA plan failed (%s); using eager\n",
                    hd_cuda_last_error());
    }

    void *ds_out[HD_VISION_NUM_DS] = {ds_d, ds_d, ds_d};
    void *b0_d = NULL;
    cudaMalloc(&b0_d, (size_t)n * H * 2);
    ws.block0_snap = b0_d;
    cudaError_t pre_err = cudaGetLastError();
    if (pre_err != cudaSuccess)
        printf("pre-forward CUDA error: %s\n", cudaGetErrorString(pre_err));
    hd_status st = hd_vision_forward(&vb, &ws, pv_d, n, grid_h, grid_w,
                                     emb_d, ds_out);
    cudaDeviceSynchronize();
    if (st != HD_OK) printf("vision forward error: %s\n", hd_last_error());
    CHECK(st == HD_OK, "vision forward runs");

    /* ---- compare patch_embed ---- */
    {
        size_t n_pe2 = 0;
        float *pe_oracle = load_f32("patch_embed", &n_pe2);
        if (pe_oracle) {
            /* patch_out is the first region of the workspace */
            uint16_t *pe_bf16 = malloc((size_t)n * H * 2);
            float *pe_f32 = malloc((size_t)n * H * sizeof(float));
            cudaMemcpy(pe_bf16, wsbase, (size_t)n * H * 2, cudaMemcpyDeviceToHost);
            hd_bf16_buf_to_f32(pe_bf16, pe_f32, (size_t)n * H);
            float c = cosine(pe_f32, pe_oracle, (size_t)n * H);
            float r = nrmse(pe_f32, pe_oracle, (size_t)n * H);
            printf("patch_embed: cos=%.6f nrmse=%.6f\n", c, r);
            CHECK(c > 0.99f, "patch_embed cosine > 0.99");
            free(pe_bf16); free(pe_f32); free(pe_oracle);
        }
    }

    /* ---- compare pos_embed (interpolated) ---- */
    {
        size_t n_pos = 0;
        float *pos_oracle = load_f32("pos_embed", &n_pos);
        if (pos_oracle) {
            uint16_t *pos_bf16 = malloc((size_t)n * H * 2);
            float *pos_f32 = malloc((size_t)n * H * sizeof(float));
            cudaMemcpy(pos_bf16, (uint8_t *)wsbase + (int64_t)n * H * 2,
                       (size_t)n * H * 2, cudaMemcpyDeviceToHost);
            hd_bf16_buf_to_f32(pos_bf16, pos_f32, (size_t)n * H);
            float c = cosine(pos_f32, pos_oracle, (size_t)n * H);
            float r = nrmse(pos_f32, pos_oracle, (size_t)n * H);
            printf("pos_embed: cos=%.6f nrmse=%.6f\n", c, r);
            CHECK(c > 0.99f, "pos_embed cosine > 0.99");
            free(pos_bf16); free(pos_f32); free(pos_oracle);
        }
    }

    /* ---- compare block0_in (patch_out + pos_emb) ---- */
    {
        size_t n_b0 = 0;
        float *b0_oracle = load_f32("block0_in", &n_b0);
        if (b0_oracle) {
            uint16_t *b0_bf16 = malloc((size_t)n * H * 2);
            float *b0_f32 = malloc((size_t)n * H * sizeof(float));
            cudaMemcpy(b0_bf16, (uint8_t *)wsbase + (int64_t)n * H * 2 * 3,
                       (size_t)n * H * 2, cudaMemcpyDeviceToHost);
            hd_bf16_buf_to_f32(b0_bf16, b0_f32, (size_t)n * H);
            float c = cosine(b0_f32, b0_oracle, (size_t)n * H);
            float r = nrmse(b0_f32, b0_oracle, (size_t)n * H);
            printf("block0_in: cos=%.6f nrmse=%.6f\n", c, r);
            CHECK(c > 0.99f, "block0_in cosine > 0.99");
            free(b0_bf16); free(b0_f32); free(b0_oracle);
        }
    }

    /* ---- compare h_a (patch+pos) against oracle patch+pos sum ---- */
    {
        size_t n_pe = 0, n_pos = 0;
        float *pe_o = load_f32("patch_embed", &n_pe);
        float *pos_o = load_f32("pos_embed", &n_pos);
        if (pe_o && pos_o && n_pe == n_pos) {
            float *sum_o = malloc((size_t)n_pe * sizeof(float));
            for (size_t i = 0; i < n_pe; i++) sum_o[i] = pe_o[i] + pos_o[i];
            uint16_t *ha_bf16 = malloc((size_t)n * H * 2);
            float *ha_f32 = malloc((size_t)n * H * sizeof(float));
            cudaMemcpy(ha_bf16, (uint8_t *)wsbase + (int64_t)n * H * 2 * 3,
                       (size_t)n * H * 2, cudaMemcpyDeviceToHost);
            hd_bf16_buf_to_f32(ha_bf16, ha_f32, (size_t)n * H);
            float c = cosine(ha_f32, sum_o, (size_t)n * H);
            float r = nrmse(ha_f32, sum_o, (size_t)n * H);
            printf("h_a (patch+pos): cos=%.6f nrmse=%.6f\n", c, r);
            free(sum_o); free(ha_bf16); free(ha_f32);
        }
        free(pe_o); free(pos_o);
    }

    /* ---- compare block0_out (first block output) ---- */
    {
        size_t n_b0o = 0;
        float *b0o_oracle = load_f32("block0_out", &n_b0o);
        if (b0o_oracle) {
            uint16_t *b0o_bf16 = malloc((size_t)n * H * 2);
            float *b0o_f32 = malloc((size_t)n * H * sizeof(float));
            cudaMemcpy(b0o_bf16, b0_d, (size_t)n * H * 2,
                       cudaMemcpyDeviceToHost);
            hd_bf16_buf_to_f32(b0o_bf16, b0o_f32, (size_t)n * H);
            float c = cosine(b0o_f32, b0o_oracle, (size_t)n * H);
            float r = nrmse(b0o_f32, b0o_oracle, (size_t)n * H);
            printf("block0_out: cos=%.6f nrmse=%.6f\n", c, r);
            CHECK(c > 0.99f, "block0_out cosine > 0.99");
            free(b0o_bf16); free(b0o_f32); free(b0o_oracle);
        }
    }

    /* ---- compare block0 internal stages (oracle_block0.pt) ---- */
    {
        const char *stages[] = {"norm1_out", "qkv_out", "attn_out",
                                "o_proj_out", "norm2_out", "fc1_out",
                                "fc2_out"};
        hd_vision_offsets lo;
        hd_vision_layout(n, m, &lo);
        int64_t offs[] = {
            lo.ln1, lo.qkv, lo.attn_out, lo.attn_resid, lo.ln2, lo.fc1, lo.fc2
        };
        for (int s = 0; s < 7; s++) {
            char path[256];
            snprintf(path, sizeof(path), "%s/oracle_block0_%s.bin",
                     ORACLE_DIR, stages[s]);
            FILE *f = fopen(path, "rb");
            if (!f) continue;
            fseek(f, 0, SEEK_END);
            long nbytes = ftell(f);
            fseek(f, 0, SEEK_SET);
            float *oracle = malloc((size_t)nbytes);
            fread(oracle, 1, (size_t)nbytes, f);
            fclose(f);
            size_t nf = (size_t)nbytes / sizeof(float);
            uint16_t *bf16 = malloc(nf * 2);
            float *f32 = malloc(nf * sizeof(float));
            cudaMemcpy(bf16, (uint8_t *)wsbase + offs[s], nf * 2,
                       cudaMemcpyDeviceToHost);
            hd_bf16_buf_to_f32(bf16, f32, nf);
            float c = cosine(f32, oracle, nf);
            float r = nrmse(f32, oracle, nf);
            printf("block0 %s: cos=%.6f nrmse=%.6f\n", stages[s], c, r);
            free(bf16); free(f32); free(oracle);
        }
    }

    /* ---- compare image_embeds ---- */
    {
        uint16_t *emb_bf16 = malloc((size_t)m * HD_VISION_OUT_HIDDEN * 2);
        float *emb_f32 = malloc((size_t)m * HD_VISION_OUT_HIDDEN * sizeof(float));
        cudaMemcpy(emb_bf16, emb_d, (size_t)m * HD_VISION_OUT_HIDDEN * 2,
                   cudaMemcpyDeviceToHost);
        hd_bf16_buf_to_f32(emb_bf16, emb_f32, (size_t)m * HD_VISION_OUT_HIDDEN);
        float c = cosine(emb_f32, emb, (size_t)m * HD_VISION_OUT_HIDDEN);
        float r = nrmse(emb_f32, emb, (size_t)m * HD_VISION_OUT_HIDDEN);
        printf("image_embeds: cos=%.6f nrmse=%.6f\n", c, r);
        CHECK(c > 0.99f, "image_embeds cosine > 0.99");
        free(emb_bf16); free(emb_f32);
    }

    /* ---- compare deepstack 0 ---- */
    {
        uint16_t *ds_bf16 = malloc((size_t)m * HD_VISION_OUT_HIDDEN * 2);
        float *ds_f32 = malloc((size_t)m * HD_VISION_OUT_HIDDEN * sizeof(float));
        cudaMemcpy(ds_bf16, ds_d, (size_t)m * HD_VISION_OUT_HIDDEN * 2,
                   cudaMemcpyDeviceToHost);
        hd_bf16_buf_to_f32(ds_bf16, ds_f32, (size_t)m * HD_VISION_OUT_HIDDEN);
        float c = cosine(ds_f32, ds, (size_t)m * HD_VISION_OUT_HIDDEN);
        float r = nrmse(ds_f32, ds, (size_t)m * HD_VISION_OUT_HIDDEN);
        printf("deepstack_0: cos=%.6f nrmse=%.6f\n", c, r);
        CHECK(c > 0.99f, "deepstack_0 cosine > 0.99");
        free(ds_bf16); free(ds_f32);
    }

    cudaFree(pv_d); cudaFree(emb_d); cudaFree(ds_d); cudaFree(wsbase);
    if (ws.sdpa) hd_sdpa_destroy(ws.sdpa);
    hd_weight_store_free(&store);
    free(pv); free(pe); free(rot); free(emb); free(ds);

    printf("\n%d passed, %d failed\n", passes, failures);
    return failures ? 1 : 0;
}