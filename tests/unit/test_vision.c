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
    const char *model_path = getenv("HD_TEST_MODEL_PATH");
    if (!model_path || !model_path[0]) model_path = "models/dev";
    hd_weight_store store;
    size_t model_len = strlen(model_path);
    if (model_len > 5 &&
        strcmp(model_path + model_len - 5, ".gguf") == 0) {
        if (hd_weights_to_device_gguf(model_path, 0, &store) != HD_OK) {
            printf("FAIL: GGUF weights load: %s\n", hd_weights_last_error());
            return 1;
        }
    } else {
        hd_st_index idx;
        if (hd_st_index_load(model_path, &idx) != HD_OK) {
            printf("FAIL: index load\n");
            return 1;
        }
        if (hd_weights_to_device(model_path, &idx, 0, &store) != HD_OK) {
            printf("FAIL: weights load: %s\n", hd_weights_last_error());
            hd_st_index_free(&idx);
            return 1;
        }
        hd_st_index_free(&idx);
    }

    hd_vision_binding vb;
    if (hd_vision_resolve(&store, &vb) != HD_OK) {
        printf("FAIL: vision resolve: %s\n", hd_last_error());
        return 1;
    }

    /* ---- workspace ---- */
    int64_t ws_bytes = hd_vision_workspace_bytes(n);
    int64_t tab_bytes = hd_vision_tables_bytes(n);
    printf("ws_bytes=%lld tab_bytes=%lld\n", (long long)ws_bytes,
           (long long)tab_bytes);
    void *wsbase = NULL;
    cudaMalloc(&wsbase, (size_t)(ws_bytes + tab_bytes));
    hd_vision_workspace ws;
    memset(&ws, 0, sizeof(ws));
    ws.patch_out = wsbase;
    ws.tables = (uint8_t *)wsbase + ws_bytes;
    ws.tables_n = -1;
    ws.bytes = ws_bytes + tab_bytes;

    /* cuDNN SDPA plan for the vision attention (HD_VISION_EAGER=1 forces the
             * eager reference backend for A/B comparison). */
            {
                const char *eager = getenv("HD_VISION_EAGER");
                if (!eager || strcmp(eager, "1") != 0) {
                    hd_sdpa_plan *plan = NULL;
                    float scale = (float)(1.0 / sqrt((double)HD_VISION_HEAD_DIM));
                    int rc = hd_sdpa_create(&plan, 1, HD_VISION_HEADS, HD_VISION_HEADS,
                                            n, n, HD_VISION_HEAD_DIM, scale);
                    if (rc == 0) ws.sdpa = plan;
                    else printf("note: cuDNN SDPA plan failed (%s); using eager\n",
                                hd_cuda_last_error());
                } else {
                    printf("note: HD_VISION_EAGER=1, using eager attention\n");
                }
            }
            void *ds_out[HD_VISION_NUM_DS] = {ds_d, ds_d, ds_d};
    /* deepstack outputs must be distinct buffers: each merger writes its
     * own [m,4096] (the test previously aliased all three to ds_d, so
     * deepstack 1/2 overwrote deepstack 0's output). */
    void *ds1_d = NULL, *ds2_d = NULL;
    cudaMalloc(&ds1_d, (size_t)m * HD_VISION_OUT_HIDDEN * 2);
    cudaMalloc(&ds2_d, (size_t)m * HD_VISION_OUT_HIDDEN * 2);
    ds_out[1] = ds1_d;
    ds_out[2] = ds2_d;
    void *b0_d = NULL;
    cudaMalloc(&b0_d, (size_t)n * H * 2);
    ws.block0_snap = b0_d;

    /* Block-0 stage snapshots: dedicated device buffers captured DURING the
     * forward (workspace scratch is recycled by the 27 blocks). */
    void *b0_snaps[HD_B0_SNAP_COUNT];
    memset(b0_snaps, 0, sizeof(b0_snaps));
    size_t nH = (size_t)n * H * 2;
    size_t nQ = (size_t)n * 3456 * 2;
    size_t nHD = (size_t)n * HD_VISION_HEADS * HD_VISION_HEAD_DIM * 2;
    size_t nI = (size_t)n * HD_VISION_INTERMEDIATE * 2;
    cudaMalloc(&b0_snaps[HD_B0_INPUT], nH);
    cudaMalloc(&b0_snaps[HD_B0_NORM1], nH);
    cudaMalloc(&b0_snaps[HD_B0_QKV], nQ);
    cudaMalloc(&b0_snaps[HD_B0_Q], nHD);
    cudaMalloc(&b0_snaps[HD_B0_K], nHD);
    cudaMalloc(&b0_snaps[HD_B0_V], nHD);
    cudaMalloc(&b0_snaps[HD_B0_Q_ROT], nHD);
    cudaMalloc(&b0_snaps[HD_B0_K_ROT], nHD);
    cudaMalloc(&b0_snaps[HD_B0_ATTN_HEADS], nHD);
    cudaMalloc(&b0_snaps[HD_B0_ATTN_MERGED], nH);
    cudaMalloc(&b0_snaps[HD_B0_PROJ], nH);
    cudaMalloc(&b0_snaps[HD_B0_ATTN_RESID], nH);
    cudaMalloc(&b0_snaps[HD_B0_NORM2], nH);
    cudaMalloc(&b0_snaps[HD_B0_FC1], nI);
    cudaMalloc(&b0_snaps[HD_B0_FC2], nH);
    cudaMalloc(&b0_snaps[HD_B0_OUTPUT], nH);
    for (int si = 0; si < HD_B0_SNAP_COUNT; si++)
        ws.block0_snaps[si] = b0_snaps[si];

    /* Block-output snapshots at layers 0,1,2,4,8,16,24,26 */
    int snap_layers[] = {0, 1, 2, 4, 8, 16, 24, 26};
    void *block_snaps[HD_VISION_DEPTH];
    memset(block_snaps, 0, sizeof(block_snaps));
    for (int si = 0; si < 8; si++) {
        cudaMalloc(&block_snaps[snap_layers[si]], nH);
        ws.block_out_snaps[snap_layers[si]] = block_snaps[snap_layers[si]];
    }

    /* Merger stage snapshots */
    void *merger_snaps[5];
    memset(merger_snaps, 0, sizeof(merger_snaps));
    cudaMalloc(&merger_snaps[0], nH);                    /* norm [n,1152] */
    cudaMalloc(&merger_snaps[1], (size_t)m * 4608 * 2);  /* merge [m,4608] */
    cudaMalloc(&merger_snaps[2], (size_t)m * 4608 * 2);  /* fc1 [m,4608] */
    cudaMalloc(&merger_snaps[3], (size_t)m * 4608 * 2);  /* gelu [m,4608] */
    cudaMalloc(&merger_snaps[4], (size_t)m * HD_VISION_OUT_HIDDEN * 2); /* fc2 */
    for (int si = 0; si < 5; si++) ws.merger_snaps[si] = merger_snaps[si];

    /* Deepstack merger 0 snapshots */
    void *ds_merger_snaps[HD_VISION_NUM_DS][4];
    memset(ds_merger_snaps, 0, sizeof(ds_merger_snaps));
    cudaMalloc(&ds_merger_snaps[0][0], (size_t)m * 4608 * 2);
    cudaMalloc(&ds_merger_snaps[0][1], (size_t)m * 4608 * 2);
    cudaMalloc(&ds_merger_snaps[0][2], (size_t)m * 4608 * 2);
    cudaMalloc(&ds_merger_snaps[0][3], (size_t)m * HD_VISION_OUT_HIDDEN * 2);
    for (int si = 0; si < 4; si++)
        ws.ds_merger_snaps[0][si] = ds_merger_snaps[0][si];

    cudaError_t pre_err = cudaGetLastError();
    if (pre_err != cudaSuccess)
        printf("pre-forward CUDA error: %s\n", cudaGetErrorString(pre_err));
    hd_status st = hd_vision_prepare_tables(&ws, n, grid_h, grid_w);
    if (st != HD_OK) printf("prepare tables error: %s\n", hd_last_error());
    CHECK(st == HD_OK, "vision tables prepare");
    st = hd_vision_forward(&vb, &ws, pv_d, n, grid_h, grid_w,
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

    /* ---- compare block0 internal stages (oracle_block0.pt) ----
     * Snapshots were captured DURING the forward at i==0 into dedicated
     * buffers, so these are the true block-0 values. */
    {
        struct { const char *oracle; int slot; size_t nf; } stages[] = {
            {"block0_input",  HD_B0_INPUT,       (size_t)n * H},
            {"block0_norm1",  HD_B0_NORM1,       (size_t)n * H},
            {"block0_qkv",    HD_B0_QKV,         (size_t)n * 3456},
            /* oracle attn_out is AFTER proj (upstream returns proj output);
             * compare against our proj stage (attn_resid pre-residual). */
            {"block0_proj",   HD_B0_PROJ,        (size_t)n * H},
            {"block0_attn_resid", HD_B0_ATTN_RESID, (size_t)n * H},
            {"block0_norm2",  HD_B0_NORM2,       (size_t)n * H},
            {"block0_fc1",    HD_B0_FC1,         (size_t)n * HD_VISION_INTERMEDIATE},
            {"block0_fc2",    HD_B0_FC2,         (size_t)n * H},
            {"block0_output", HD_B0_OUTPUT,      (size_t)n * H},
        };
        for (size_t s = 0; s < sizeof(stages) / sizeof(stages[0]); s++) {
            char path[256];
            snprintf(path, sizeof(path), "%s/oracle_block0_%s.bin",
                     ORACLE_DIR, stages[s].oracle);
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
            cudaMemcpy(bf16, b0_snaps[stages[s].slot], nf * 2,
                       cudaMemcpyDeviceToHost);
            hd_bf16_buf_to_f32(bf16, f32, nf);
            float c = cosine(f32, oracle, nf);
            float r = nrmse(f32, oracle, nf);
            printf("block0 %-14s: cos=%.6f nrmse=%.6f\n",
                   stages[s].oracle, c, r);
            free(bf16); free(f32); free(oracle);
        }
    }

    /* ---- compare q/k/v pre-rope and post-rope (block 0) ----
     * Native layout: [Hd, n, D] head-major. Oracle layout: [seq, heads, hd].
     * Compare elementwise with the transpose. */
    {
        struct { const char *oracle; int slot; } qkv_stages[] = {
            {"q_pre",  HD_B0_Q},
            {"k_pre",  HD_B0_K},
            {"v",      HD_B0_V},
            {"q_post", HD_B0_Q_ROT},
            {"k_post", HD_B0_K_ROT},
        };
        int Hd = HD_VISION_HEADS, D = HD_VISION_HEAD_DIM;
        for (size_t s = 0; s < sizeof(qkv_stages) / sizeof(qkv_stages[0]); s++) {
            char path[256];
            snprintf(path, sizeof(path), "%s/oracle_block0_%s.bin",
                     ORACLE_DIR, qkv_stages[s].oracle);
            FILE *f = fopen(path, "rb");
            if (!f) continue;
            fseek(f, 0, SEEK_END);
            long nbytes = ftell(f);
            fseek(f, 0, SEEK_SET);
            float *oracle = malloc((size_t)nbytes);
            fread(oracle, 1, (size_t)nbytes, f);
            fclose(f);
            size_t nf = (size_t)nbytes / sizeof(float); /* n*Hd*D */
            uint16_t *bf16 = malloc(nf * 2);
            float *f32 = malloc(nf * sizeof(float));
            cudaMemcpy(bf16, b0_snaps[qkv_stages[s].slot], nf * 2,
                       cudaMemcpyDeviceToHost);
            hd_bf16_buf_to_f32(bf16, f32, nf);
            /* native [Hd,n,D] -> oracle [n,Hd,D]: transpose h<->s */
            float *t32 = malloc(nf * sizeof(float));
            for (int h = 0; h < Hd; h++)
                for (int ss = 0; ss < n; ss++)
                    for (int dd = 0; dd < D; dd++)
                        t32[((size_t)ss * Hd + h) * D + dd] =
                            f32[((size_t)h * n + ss) * D + dd];
            float c = cosine(t32, oracle, nf);
            float r = nrmse(t32, oracle, nf);
            printf("block0 %-8s: cos=%.6f nrmse=%.6f\n",
                   qkv_stages[s].oracle, c, r);
            free(bf16); free(f32); free(t32); free(oracle);
        }
    }

    /* ---- compare block outputs at layers 0,1,2,4,8,16,24,26 ---- */
    {
        int snap_layers[] = {0, 1, 2, 4, 8, 16, 24, 26};
        for (int si = 0; si < 8; si++) {
            int layer = snap_layers[si];
            char path[256];
            snprintf(path, sizeof(path), "%s/oracle_block%d_out.bin",
                     ORACLE_DIR, layer);
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
            cudaMemcpy(bf16, block_snaps[layer], nf * 2,
                       cudaMemcpyDeviceToHost);
            hd_bf16_buf_to_f32(bf16, f32, nf);
            float c = cosine(f32, oracle, nf);
            float r = nrmse(f32, oracle, nf);
            printf("block%-2d_out: cos=%.6f nrmse=%.6f\n", layer, c, r);
            free(bf16); free(f32); free(oracle);
        }
    }

    /* ---- compare final merger stages ---- */
    {
        /* oracle: merger_norm_out [n,1152], merger_fc1_out [m,4608],
         * merger_fc2_out [m,4096] (== image_embeds). */
        struct { const char *oracle; int slot; size_t nf; } mst[] = {
            {"merger_norm_out", 0, (size_t)n * H},
            {"merger_fc1_out",  2, (size_t)m * 4608},
            {"merger_fc2_out",  4, (size_t)m * HD_VISION_OUT_HIDDEN},
        };
        for (size_t s = 0; s < sizeof(mst) / sizeof(mst[0]); s++) {
            char path[256];
            snprintf(path, sizeof(path), "%s/oracle_%s.bin",
                     ORACLE_DIR, mst[s].oracle);
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
            cudaMemcpy(bf16, merger_snaps[mst[s].slot], nf * 2,
                       cudaMemcpyDeviceToHost);
            hd_bf16_buf_to_f32(bf16, f32, nf);
            float c = cosine(f32, oracle, nf);
            float r = nrmse(f32, oracle, nf);
            printf("merger %-14s: cos=%.6f nrmse=%.6f\n", mst[s].oracle, c, r);
            free(bf16); free(f32); free(oracle);
        }
    }

    /* ---- compare deepstack merger 0 stages ---- */
    {
        struct { const char *oracle; int slot; size_t nf; } dst[] = {
            {"ds0_norm_out", 0, (size_t)m * 4608},
            {"ds0_fc1_out",  1, (size_t)m * 4608},
            {"ds0_fc2_out",  3, (size_t)m * HD_VISION_OUT_HIDDEN},
        };
        for (size_t s = 0; s < sizeof(dst) / sizeof(dst[0]); s++) {
            char path[256];
            snprintf(path, sizeof(path), "%s/oracle_%s.bin",
                     ORACLE_DIR, dst[s].oracle);
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
            cudaMemcpy(bf16, ds_merger_snaps[0][dst[s].slot], nf * 2,
                       cudaMemcpyDeviceToHost);
            hd_bf16_buf_to_f32(bf16, f32, nf);
            float c = cosine(f32, oracle, nf);
            float r = nrmse(f32, oracle, nf);
            printf("ds0 %-14s: cos=%.6f nrmse=%.6f\n", dst[s].oracle, c, r);
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

    /* ---- compare all DeepStack outputs ---- */
    {
        const char *names[HD_VISION_NUM_DS] = {
            "deepstack_0", "deepstack_1", "deepstack_2"
        };
        void *native_ds[HD_VISION_NUM_DS] = {ds_d, ds1_d, ds2_d};
        uint16_t *ds_bf16 = malloc((size_t)m * HD_VISION_OUT_HIDDEN * 2);
        float *ds_f32 = malloc((size_t)m * HD_VISION_OUT_HIDDEN * sizeof(float));
        for (int i = 0; i < HD_VISION_NUM_DS; i++) {
            size_t n_ref = 0;
            float *ref = i == 0 ? ds : load_f32(names[i], &n_ref);
            if (!ref || (i > 0 && n_ref != (size_t)m * HD_VISION_OUT_HIDDEN)) {
                CHECK(0, "load DeepStack oracle");
                free(i == 0 ? NULL : ref);
                continue;
            }
            cudaMemcpy(ds_bf16, native_ds[i],
                       (size_t)m * HD_VISION_OUT_HIDDEN * 2,
                       cudaMemcpyDeviceToHost);
            hd_bf16_buf_to_f32(ds_bf16, ds_f32,
                               (size_t)m * HD_VISION_OUT_HIDDEN);
            float c = cosine(ds_f32, ref,
                             (size_t)m * HD_VISION_OUT_HIDDEN);
            float r = nrmse(ds_f32, ref,
                            (size_t)m * HD_VISION_OUT_HIDDEN);
            printf("%s: cos=%.6f nrmse=%.6f\n", names[i], c, r);
            char msg[64];
            snprintf(msg, sizeof(msg), "%s cosine > 0.99", names[i]);
            CHECK(c > 0.99f, msg);
            if (i > 0) free(ref);
        }
        free(ds_bf16); free(ds_f32);
    }

    cudaFree(pv_d); cudaFree(emb_d); cudaFree(ds_d);
    cudaFree(ds1_d); cudaFree(ds2_d); cudaFree(wsbase);
    if (ws.sdpa) hd_sdpa_destroy(ws.sdpa);
    hd_weight_store_free(&store);
    free(pv); free(pe); free(rot); free(emb); free(ds);

    printf("\n%d passed, %d failed\n", passes, failures);
    return failures ? 1 : 0;
}