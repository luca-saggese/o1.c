/*
 * M2 ref-image visual conditioning -- Qwen3-VL vision tower.
 *
 * Reproduces the upstream Qwen3VLVisionModel (transformers 4.57.1) in
 * native CUDA, matching the oracle stage-by-stage:
 *
 *   pixel_values [n, 1536] bf16 (patchified, processor layout)
 *     -> patch_embed (Conv3d as matmul)          [n, 1152]
 *     -> + pos_embeds (4-corner bilinear interp) [n, 1152]
 *     -> 27 x Qwen3VLVisionBlock                 [n, 1152]
 *     -> spatial merge (2x2 unshuffle)           [m, 4608]
 *     -> merger (norm + fc1 + GELU + fc2)        [m, 4096]  = image_embeds
 *     -> deepstack mergers at blocks 8/16/24     [m, 4096]  x3
 *
 * GEMMs use the production cuBLAS backend (hd_linear). Attention uses the
 * cuDNN SDPA backend (hd_sdpa_*) when a plan is provided, else the eager
 * reference. Numerics: BF16 compute with FP32 accumulation (M1 contract).
 * LayerNorm (eps 1e-6) and GELU (pytorch_tanh) per the vision config.
 *
 * Reference: docs/REF_IMAGE_NATIVE_IMPLEMENTATION.md,
 * artifacts/ref_image_audit/vision_tower_graph.md.
 */

#include "vision.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>

#include "cuda_internal.h"

/* Temporary stage-check macro for the vision tower bring-up. */
#define CUDA_STAGE_CHECK(name) do { \
    cudaError_t e1 = cudaGetLastError(); \
    if (e1 != cudaSuccess) { \
        fprintf(stderr, "%s launch: %s\n", name, cudaGetErrorString(e1)); \
        abort(); \
    } \
    cudaError_t e2 = cudaDeviceSynchronize(); \
    if (e2 != cudaSuccess) { \
        fprintf(stderr, "%s sync: %s\n", name, cudaGetErrorString(e2)); \
        abort(); \
    } \
} while (0)
#include "gemm.h"
#include "vision_kernels.h"

/* ------------------------------------------------------------------ */
/* Binding resolution (stage B)                                        */
/* ------------------------------------------------------------------ */

static const void *resolve_one(const hd_weight_store *ws, const char *name) {
    for (int64_t i = 0; i < ws->n_allocs; i++) {
        if (ws->allocs[i].name[0] &&
            strcmp(ws->allocs[i].name, name) == 0)
            return ws->allocs[i].dev_ptr;
    }
    return NULL;
}

static hd_status resolve_block(const hd_weight_store *ws, int i,
                               hd_vision_block_binding *b) {
    char n[256];
    const char *suffix[12] = {
        "norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias",
        "attn.qkv.weight", "attn.qkv.bias", "attn.proj.weight",
        "attn.proj.bias", "mlp.linear_fc1.weight", "mlp.linear_fc1.bias",
        "mlp.linear_fc2.weight", "mlp.linear_fc2.bias",
    };
    const void **dst[12] = {
        &b->norm1_w, &b->norm1_b, &b->norm2_w, &b->norm2_b,
        &b->qkv_w, &b->qkv_b, &b->proj_w, &b->proj_b,
        &b->fc1_w, &b->fc1_b, &b->fc2_w, &b->fc2_b,
    };
    for (int k = 0; k < 12; k++) {
        snprintf(n, sizeof(n), "model.visual.blocks.%d.%s", i, suffix[k]);
        *dst[k] = resolve_one(ws, n);
        if (!*dst[k]) {
            hd_set_error("vision: missing %s", n);
            return HD_ERR_MISSING;
        }
    }
    return HD_OK;
}

static hd_status resolve_merger(const hd_weight_store *ws, const char *prefix,
                                hd_vision_merger_binding *b) {
    char n[256];
    const char *suffix[6] = {
        "norm.weight", "norm.bias", "linear_fc1.weight", "linear_fc1.bias",
        "linear_fc2.weight", "linear_fc2.bias",
    };
    const void **dst[6] = {
        &b->norm_w, &b->norm_b, &b->fc1_w, &b->fc1_b, &b->fc2_w, &b->fc2_b,
    };
    for (int k = 0; k < 6; k++) {
        snprintf(n, sizeof(n), "%s.%s", prefix, suffix[k]);
        *dst[k] = resolve_one(ws, n);
        if (!*dst[k]) {
            hd_set_error("vision: missing %s", n);
            return HD_ERR_MISSING;
        }
    }
    return HD_OK;
}

hd_status hd_vision_resolve(const hd_weight_store *wstore,
                            hd_vision_binding *out) {
    if (!wstore || !out) {
        hd_set_error("vision: null argument in resolve");
        return HD_ERR_MISSING;
    }
    memset(out, 0, sizeof(*out));

    out->patch_proj_w = resolve_one(wstore, "model.visual.patch_embed.proj.weight");
    out->patch_proj_b = resolve_one(wstore, "model.visual.patch_embed.proj.bias");
    out->pos_embed    = resolve_one(wstore, "model.visual.pos_embed.weight");
    if (!out->patch_proj_w || !out->patch_proj_b || !out->pos_embed) {
        hd_set_error("vision: missing patch_embed/pos_embed weights");
        return HD_ERR_MISSING;
    }

    for (int i = 0; i < HD_VISION_DEPTH; i++) {
        hd_status s = resolve_block(wstore, i, &out->blocks[i]);
        if (s != HD_OK) return s;
    }
    hd_status s = resolve_merger(wstore, "model.visual.merger", &out->merger);
    if (s != HD_OK) return s;
    for (int j = 0; j < HD_VISION_NUM_DS; j++) {
        char prefix[128];
        snprintf(prefix, sizeof(prefix), "model.visual.deepstack_merger_list.%d", j);
        s = resolve_merger(wstore, prefix, &out->deepstack[j]);
        if (s != HD_OK) return s;
    }
    return HD_OK;
}

/* ------------------------------------------------------------------ */
/* Workspace layout and sizing                                         */
/* ------------------------------------------------------------------ */

/* Byte offsets of every workspace region, in allocation order. The forward
 * carves identical pointers every call (no rebinding). */
void hd_vision_layout(int64_t n, int64_t m, hd_vision_offsets *o) {
    int64_t H = HD_VISION_HIDDEN, D = HD_VISION_HEAD_DIM;
    int64_t Hd = HD_VISION_HEADS, I = HD_VISION_INTERMEDIATE;
    int64_t O = HD_VISION_OUT_HIDDEN;
    int64_t bf16 = 2;
    int64_t b = 0;
    o->patch_out = b; b += n * H * bf16;
    o->pos_emb   = b; b += n * H * bf16;
    o->rot       = b; b += n * H * bf16;
    o->h_a       = b; b += n * H * bf16;
    o->h_b       = b; b += n * H * bf16;
    o->ln1       = b; b += n * H * bf16;
    o->attn_resid= b; b += n * H * bf16;
    o->ln2       = b; b += n * H * bf16;
    o->fc2       = b; b += n * H * bf16;
    o->mlp_resid = b; b += n * H * bf16;
    o->q         = b; b += n * Hd * D * bf16;
    o->k         = b; b += n * Hd * D * bf16;
    o->v         = b; b += n * Hd * D * bf16;
    o->qr        = b; b += n * Hd * D * bf16;
    o->kr        = b; b += n * Hd * D * bf16;
    o->qkv       = b; b += n * 3456 * bf16;
    o->scores    = b; b += n * n * Hd * bf16;
    o->probs     = b; b += n * n * Hd * bf16;
    o->attn_out  = b; b += n * H * bf16;
    o->fc1       = b; b += n * I * bf16;
    o->cosf      = b; b += n * H * 4;
    o->sinf      = b; b += n * H * 4;
    o->merged    = b; b += m * 4608 * bf16;
    o->merge_norm= b; b += m * 4608 * bf16;
    o->merge_fc1 = b; b += m * 4608 * bf16;
    o->merge_fc2 = b; b += m * O * bf16;
    o->ds_merged = b; b += m * 4608 * bf16;
    o->ds_fc1    = b; b += m * 4608 * bf16;
    o->ds_fc2    = b; b += m * O * bf16;
    o->total_bytes = b;
}

int64_t hd_vision_workspace_bytes(int64_t n) {
    int64_t m = n / (HD_VISION_MERGE_SIZE * HD_VISION_MERGE_SIZE);
    if (m < 1) m = 1;
    hd_vision_offsets o;
    hd_vision_layout(n, m, &o);
    fprintf(stderr,
            "VISION_LAYOUT_BUILD %s %s n=%lld rot=%lld ds_fc2=%lld total=%lld\n",
            __DATE__, __TIME__,
            (long long)n,
            (long long)o.rot,
            (long long)o.ds_fc2,
            (long long)o.total_bytes);
    return o.total_bytes;
}

/* ------------------------------------------------------------------ */
/* Vision kernels (stage D/E/F)                                        */
/* ------------------------------------------------------------------ */

/* ------------------------------------------------------------------ */
/* Host helpers for pos_embed / rot_pos_emb coordinate tables          */
/* ------------------------------------------------------------------ */

/*
 * Build the 4-corner bilinear interpolation tables for fast_pos_embed_
 * interpolate. grid_h x grid_w patches; pos_embed has num_grid_per_side
 * rows per side (48). Returns malloc'd idx[4*n] and wgt[4*n] int/float.
 *
 * The oracle applies the spatial-merge permute AFTER interpolation:
 *   view(t, h//m, m, w//m, m, -1).permute(0,1,3,2,4,5).flatten(0,4)
 * so the output rows are in (bh, bw, mh, mw) order, NOT (h, w) row-major.
 * We fold that ordering into the table: output row (bh,bw,mh,mw) samples
 * the pos_embed grid at h = bh*m+mh, w = bw*m+mw.
 */
static void build_pos_interp(int grid_h, int grid_w, int num_grid_per_side,
                             int merge_size, int **idx_out, float **wgt_out) {
    int m = merge_size;
    int mh = grid_h / m, mw = grid_w / m;
    int n = mh * mw * m * m;
    int *idx = malloc((size_t)4 * n * sizeof(int));
    float *wgt = malloc((size_t)4 * n * sizeof(float));
    int row = 0;
    for (int bh = 0; bh < mh; bh++) {
        for (int bw = 0; bw < mw; bw++) {
            for (int mh_i = 0; mh_i < m; mh_i++) {
                for (int mw_i = 0; mw_i < m; mw_i++) {
                    int h = bh * m + mh_i;
                    int w = bw * m + mw_i;
                    float hf = (float)h * (float)(num_grid_per_side - 1) / (float)(grid_h - 1);
                    int hf0 = (int)hf;
                    int hf1 = hf0 + 1 < num_grid_per_side ? hf0 + 1 : num_grid_per_side - 1;
                    float dh = hf - (float)hf0;
                    float wf = (float)w * (float)(num_grid_per_side - 1) / (float)(grid_w - 1);
                    int wf0 = (int)wf;
                    int wf1 = wf0 + 1 < num_grid_per_side ? wf0 + 1 : num_grid_per_side - 1;
                    float dw = wf - (float)wf0;
                    idx[row] = hf0 * num_grid_per_side + wf0;
                    idx[n + row] = hf0 * num_grid_per_side + wf1;
                    idx[2 * n + row] = hf1 * num_grid_per_side + wf0;
                    idx[3 * n + row] = hf1 * num_grid_per_side + wf1;
                    wgt[row] = (1.0f - dh) * (1.0f - dw);
                    wgt[n + row] = (1.0f - dh) * dw;
                    wgt[2 * n + row] = dh * (1.0f - dw);
                    wgt[3 * n + row] = dh * dw;
                    row++;
                }
            }
        }
    }
    *idx_out = idx;
    *wgt_out = wgt;
}

/*
 * Build the (row, col) coordinate table for rot_pos_emb. For each merged
 * block position, the coords are the full-resolution row/col of the block's
 * top-left patch. For grid_h x grid_w with merge_size m: merged grid is
 * (grid_h/m) x (grid_w/m), each merged token covers m x m patches.
 * Returns malloc'd coords[2*n] int.
 */
static void build_rot_coords(int grid_h, int grid_w, int merge_size,
                             int **coords_out) {
    int m = merge_size;
    int mh = grid_h / m, mw = grid_w / m;
    int n = mh * mw * m * m;
    int *coords = malloc((size_t)2 * n * sizeof(int));
    int o = 0;
    for (int bh = 0; bh < mh; bh++) {
        for (int bw = 0; bw < mw; bw++) {
            for (int ih = 0; ih < m; ih++) {
                for (int iw = 0; iw < m; iw++) {
                    coords[o] = bh * m + ih;
                    coords[n + o] = bw * m + iw;
                    o++;
                }
            }
        }
    }
    *coords_out = coords;
}

/* ------------------------------------------------------------------ */
/* Vision forward                                                      */
/* ------------------------------------------------------------------ */

hd_status hd_vision_forward(const hd_vision_binding *bw,
                            hd_vision_workspace *ws,
                            const void *pixel_values, int n,
                            int grid_h, int grid_w,
                            void *image_embeds_out,
                            void *deepstack_out[HD_VISION_NUM_DS]) {
    if (!bw || !ws || !pixel_values || n <= 0 || grid_h <= 0 || grid_w <= 0 ||
        !image_embeds_out) {
        hd_set_error("vision: null argument");
        return HD_ERR_MISSING;
    }
    if (grid_h % 2 || grid_w % 2) {
        hd_set_error("vision: grid %dx%d not divisible by merge size 2",
                     grid_h, grid_w);
        return HD_ERR_MISSING;
    }
    int H = HD_VISION_HIDDEN;
    int D = HD_VISION_HEAD_DIM;
    int Hd = HD_VISION_HEADS;
    int I = HD_VISION_INTERMEDIATE;
    int O = HD_VISION_OUT_HIDDEN;
    int m = n / (HD_VISION_MERGE_SIZE * HD_VISION_MERGE_SIZE);
    float eps = 1e-6f;
    float scaling = (float)(1.0 / sqrt((double)D));

    hd_vision_offsets o;
    hd_vision_layout(n, m, &o);
    uint8_t *base = (uint8_t *)ws->patch_out;
    void *patch_out = base + o.patch_out;
    void *pos_emb   = base + o.pos_emb;
    void *rot       = base + o.rot;
    void *h_a       = base + o.h_a;
    void *h_b       = base + o.h_b;
    void *ln1       = base + o.ln1;
    void *attn_resid= base + o.attn_resid;
    void *ln2       = base + o.ln2;
    void *fc2       = base + o.fc2;
    void *mlp_resid = base + o.mlp_resid;
    void *q         = base + o.q;
    void *k         = base + o.k;
    void *v         = base + o.v;
    void *qr        = base + o.qr;
    void *kr        = base + o.kr;
    void *qkv       = base + o.qkv;
    void *scores    = base + o.scores;
    void *probs     = base + o.probs;
    void *attn_out  = base + o.attn_out;
    void *fc1       = base + o.fc1;
    float *cosf     = (float *)(base + o.cosf);
    float *sinf     = (float *)(base + o.sinf);
    void *merged    = base + o.merged;
    void *merge_norm= base + o.merge_norm;
    void *merge_fc1 = base + o.merge_fc1;
    void *merge_fc2 = base + o.merge_fc2;
    void *ds_merged = base + o.ds_merged;
    void *ds_fc1    = base + o.ds_fc1;
    void *ds_fc2    = base + o.ds_fc2;

    /* ---- stage D: patch_embed ---- */
    {
        hd_vision_patch(pixel_values, bw->patch_proj_w, bw->patch_proj_b,
                     patch_out, n, HD_VISION_PATCH_DIM, H);
    }
    CUDA_STAGE_CHECK("hd_vision_patch");

    /* ---- pos_embed interpolate + add ---- */
    {
        int *idx = NULL; float *wgt = NULL;
        build_pos_interp(grid_h, grid_w, 48, HD_VISION_MERGE_SIZE, &idx, &wgt);
        int *idx_d = NULL; float *wgt_d = NULL;
        cudaMalloc(&idx_d, (size_t)4 * n * sizeof(int));
        cudaMalloc(&wgt_d, (size_t)4 * n * sizeof(float));
        if (!idx_d || !wgt_d) {
            free(idx); free(wgt);
            hd_set_error("vision: pos interp alloc oom");
            return HD_ERR_OOM;
        }
        cudaMemcpy(idx_d, idx, (size_t)4 * n * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(wgt_d, wgt, (size_t)4 * n * sizeof(float), cudaMemcpyHostToDevice);
        free(idx); free(wgt);
        hd_vision_pos_interp(bw->pos_embed, idx_d, wgt_d, pos_emb, n, H);
        cudaFree(idx_d); cudaFree(wgt_d);
        /* hidden = patch_out + pos_emb -> h_a */
        hd_residual_add(patch_out, pos_emb, h_a, (size_t)n * H);
    }
    CUDA_STAGE_CHECK("hd_vision_pos_interp + residual_add");

    /* ---- rot_pos_emb -> cos/sin ---- */
    {
        int *coords = NULL;
        build_rot_coords(grid_h, grid_w, HD_VISION_MERGE_SIZE, &coords);
        int *coords_d = NULL;
        cudaMalloc(&coords_d, (size_t)2 * n * sizeof(int));
        if (!coords_d) { free(coords); hd_set_error("vision: rot coords oom"); return HD_ERR_OOM; }
        cudaMemcpy(coords_d, coords, (size_t)2 * n * sizeof(int), cudaMemcpyHostToDevice);
        free(coords);
        /* inv_freq: 1/(theta^(arange(0, dim, 2)/dim)), dim = head_dim/2 = 36 */
        float inv_freq[18];
        for (int k = 0; k < 18; k++) {
            float e = (float)(2 * k) / 36.0f;
            inv_freq[k] = 1.0f / powf(10000.0f, e);
        }
        float *inv_d = NULL;
        cudaMalloc(&inv_d, 18 * sizeof(float));
        if (!inv_d) { cudaFree(coords_d); hd_set_error("vision: inv_freq oom"); return HD_ERR_OOM; }
        cudaMemcpy(inv_d, inv_freq, 18 * sizeof(float), cudaMemcpyHostToDevice);
        hd_vision_rot(inv_d, coords_d, rot, n, 18);
        cudaFree(inv_d); cudaFree(coords_d);
        /* cos/sin from rot [n, 36] -> [n, 72] fp32 */
        hd_vision_rot_cos_sin(rot, cosf, sinf, n, 18);
    }

    /* ---- 27 blocks ---- */
    {
        const void *cur_in = h_a;
        void *cur_out = h_b;
        for (int i = 0; i < HD_VISION_DEPTH; i++) {
            const hd_vision_block_binding *blk = &bw->blocks[i];
            /* norm1 */
            hd_vision_layernorm(cur_in, blk->norm1_w, blk->norm1_b, ln1, n, H, eps);
            /* qkv (cuBLAS) */
            hd_linear(ln1, blk->qkv_w, blk->qkv_b, qkv, n, 3456, H, 1);
            /* split q/k/v -> [Hd, n, D] */
            hd_vision_qkv_split(qkv, q, k, v, n, Hd, D);
            /* rotary on q/k (cos/sin [n, 72]) */
            hd_apply_rotary(q, cosf, sinf, qr, Hd, n, D);
            hd_apply_rotary(k, cosf, sinf, kr, Hd, n, D);
            /* attention: cuDNN SDPA (no mask) or eager reference.
             * cuDNN writes head-major [Hd, n, D]; eager writes seq-major
             * [n, Hd*D]. Normalize to seq-major in attn_out. */
            int sdpa_ok = 0;
            if (ws->sdpa) {
                int rc = hd_sdpa_execute(ws->sdpa, qr, kr, v, NULL, qkv, 0);
                if (rc == 0) {
                    /* qkv is free after qkv_split: use it as the head-major
                     * temp buffer, then transpose to seq-major attn_out. */
                    hd_vision_attn_merge(qkv, attn_out, Hd, n, D);
                    sdpa_ok = 1;
                }
            }
            if (!sdpa_ok) {
                hd_attention_eager(qr, kr, v, NULL, scores, probs,
                                   attn_out, Hd, Hd, n, D, scaling);
            }
            /* proj (cuBLAS) over [n, Hd*D] */
            hd_linear(attn_out, blk->proj_w, blk->proj_b, attn_resid, n, H, H, 1);
            /* residual: attn_resid = cur_in + attn_out */
            hd_residual_add(cur_in, attn_resid, attn_resid, (size_t)n * H);
            /* norm2 */
            hd_vision_layernorm(attn_resid, blk->norm2_w, blk->norm2_b, ln2, n, H, eps);
            /* mlp: fc1 -> gelu -> fc2 */
            hd_linear(ln2, blk->fc1_w, blk->fc1_b, fc1, n, I, H, 1);
            hd_vision_gelu(fc1, fc1, (size_t)n * I);
            hd_linear(fc1, blk->fc2_w, blk->fc2_b, fc2, n, H, I, 1);
            /* residual: mlp_resid = attn_resid + fc2 */
            hd_residual_add(attn_resid, fc2, mlp_resid, (size_t)n * H);

            /* Block-0 debug snapshots: capture at the exact execution point
             * into dedicated buffers (the workspace scratch is reused by
             * later blocks, so it cannot be read back after the forward). */
            if (i == 0 && ws->block0_snaps) {
                void **s = ws->block0_snaps;
                size_t nH = (size_t)n * H * 2;
                size_t nQ = (size_t)n * 3456 * 2;
                size_t nHD = (size_t)n * Hd * D * 2;
                if (s[HD_B0_INPUT]) cudaMemcpy(s[HD_B0_INPUT], cur_in, nH, cudaMemcpyDeviceToDevice);
                if (s[HD_B0_NORM1]) cudaMemcpy(s[HD_B0_NORM1], ln1, nH, cudaMemcpyDeviceToDevice);
                if (s[HD_B0_QKV]) cudaMemcpy(s[HD_B0_QKV], qkv, nQ, cudaMemcpyDeviceToDevice);
                if (s[HD_B0_Q]) cudaMemcpy(s[HD_B0_Q], q, nHD, cudaMemcpyDeviceToDevice);
                if (s[HD_B0_K]) cudaMemcpy(s[HD_B0_K], k, nHD, cudaMemcpyDeviceToDevice);
                if (s[HD_B0_V]) cudaMemcpy(s[HD_B0_V], v, nHD, cudaMemcpyDeviceToDevice);
                if (s[HD_B0_Q_ROT]) cudaMemcpy(s[HD_B0_Q_ROT], qr, nHD, cudaMemcpyDeviceToDevice);
                if (s[HD_B0_K_ROT]) cudaMemcpy(s[HD_B0_K_ROT], kr, nHD, cudaMemcpyDeviceToDevice);
                if (s[HD_B0_ATTN_HEADS]) cudaMemcpy(s[HD_B0_ATTN_HEADS], qkv, nHD, cudaMemcpyDeviceToDevice);
                if (s[HD_B0_ATTN_MERGED]) cudaMemcpy(s[HD_B0_ATTN_MERGED], attn_out, nH, cudaMemcpyDeviceToDevice);
                if (s[HD_B0_PROJ]) cudaMemcpy(s[HD_B0_PROJ], attn_resid, nH, cudaMemcpyDeviceToDevice);
                if (s[HD_B0_ATTN_RESID]) cudaMemcpy(s[HD_B0_ATTN_RESID], attn_resid, nH, cudaMemcpyDeviceToDevice);
                if (s[HD_B0_NORM2]) cudaMemcpy(s[HD_B0_NORM2], ln2, nH, cudaMemcpyDeviceToDevice);
                if (s[HD_B0_FC1]) cudaMemcpy(s[HD_B0_FC1], fc1, (size_t)n * I * 2, cudaMemcpyDeviceToDevice);
                if (s[HD_B0_FC2]) cudaMemcpy(s[HD_B0_FC2], fc2, nH, cudaMemcpyDeviceToDevice);
                if (s[HD_B0_OUTPUT]) cudaMemcpy(s[HD_B0_OUTPUT], mlp_resid, nH, cudaMemcpyDeviceToDevice);
            }

            /* deepstack capture at layers 8/16/24 */
            if (deepstack_out) {
                int ds_idx = -1;
                if (i == 8) ds_idx = 0;
                else if (i == 16) ds_idx = 1;
                else if (i == 24) ds_idx = 2;
                if (ds_idx >= 0) {
                    const hd_vision_merger_binding *ds = &bw->deepstack[ds_idx];
                    /* spatial merge -> [m, 4608] */
                    hd_vision_spatial_merge(mlp_resid, ds_merged, grid_h, grid_w, H);
                    /* norm over 4608 (post-shuffle) */
                    hd_vision_layernorm(ds_merged, ds->norm_w, ds->norm_b, ds_fc1, m, 4608, eps);
                    /* fc1 -> gelu -> fc2 */
                    hd_linear(ds_fc1, ds->fc1_w, ds->fc1_b, ds_fc1, m, 4608, 4608, 1);
                    hd_vision_gelu(ds_fc1, ds_fc1, (size_t)m * 4608);
                    hd_linear(ds_fc1, ds->fc2_w, ds->fc2_b, ds_fc2, m, O, 4608, 1);
                    cudaMemcpy(deepstack_out[ds_idx], ds_fc2,
                               (size_t)m * O * 2, cudaMemcpyDeviceToDevice);
                }
            }

            /* copy mlp_resid into cur_out FIRST, then swap so the next
             * iteration reads the fresh output. */
            cudaMemcpy(cur_out, mlp_resid, (size_t)n * H * 2,
                       cudaMemcpyDeviceToDevice);
            if (i == 0 && ws->block0_snap) {
                cudaMemcpy(ws->block0_snap, mlp_resid, (size_t)n * H * 2,
                           cudaMemcpyDeviceToDevice);
            }
            void *tmp = cur_out;
            cur_out = (void *)cur_in;
            cur_in = tmp;
        }
    }

    /* ---- final merger ---- */
    {
        const hd_vision_merger_binding *mg = &bw->merger;
        /* norm over 1152 (use_postshuffle_norm=False): norm applies to the
         * [n, 1152] block output BEFORE the merge view. */
        hd_vision_layernorm(mlp_resid, mg->norm_w, mg->norm_b, merge_norm, n, H, eps);
        /* spatial merge of the normed [n, 1152] -> [m, 4608] */
        hd_vision_spatial_merge(merge_norm, merged, grid_h, grid_w, H);
        /* fc1 -> gelu -> fc2 */
        hd_linear(merged, mg->fc1_w, mg->fc1_b, merge_fc1, m, 4608, 4608, 1);
        hd_vision_gelu(merge_fc1, merge_fc1, (size_t)m * 4608);
        hd_linear(merge_fc1, mg->fc2_w, mg->fc2_b, merge_fc2, m, O, 4608, 1);
        cudaMemcpy(image_embeds_out, merge_fc2, (size_t)m * O * 2,
                   cudaMemcpyDeviceToDevice);
    }

    return HD_OK;
}