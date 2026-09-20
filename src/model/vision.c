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

#include "gemm.h"
#include "vision_kernels.h"
#include "o1_timing.h"

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
    o->mask      = b; b += n * n * bf16;
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
    return o.total_bytes;
}

/*
 * Persistent table region: pos-interp idx [4n] int32 + wgt [4n] float,
 * rotary coords [2n] int32, rotary inv_freq [18] float. Allocated once by
 * the caller and reused for every forward of the same shape.
 */
int64_t hd_vision_tables_bytes(int64_t n) {
    int64_t b = 0;
    b += n * 4 * 4;   /* idx int32 */
    b += n * 4 * 4;   /* wgt float */
    b += n * 2 * 4;   /* rot coords int32 */
    b += 18 * 4;      /* inv_freq float */
    return b;
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

/*
 * Builds the persistent device tables (pos-interp idx/wgt, rotary coords,
 * rotary inv_freq) for (n, grid_h, grid_w) into the caller-provided table
 * region and binds them into the workspace. Idempotent for a repeated
 * shape. `ws->tables` must point at hd_vision_tables_bytes(n) bytes.
 */
hd_status hd_vision_prepare_tables(hd_vision_workspace *ws, int n,
                                   int grid_h, int grid_w) {
    if (!ws || n <= 0 || grid_h <= 0 || grid_w <= 0) {
        hd_set_error("vision: bad prepare_tables args");
        return HD_ERR_MISSING;
    }
    if (ws->tables_n == n && ws->tables_gh == grid_h && ws->tables_gw == grid_w)
        return HD_OK;

    uint8_t *t = (uint8_t *)ws->tables;
    if (t == NULL) {
        hd_set_error("vision: table region not bound");
        return HD_ERR_MISSING;
    }
    int *pos_idx_d = (int *)t;          t += (size_t)n * 4 * 4;
    float *pos_wgt_d = (float *)t;      t += (size_t)n * 4 * 4;
    int *coords_d = (int *)t;           t += (size_t)n * 2 * 4;
    float *inv_d = (float *)t;          t += 18 * 4;

    int *idx = NULL; float *wgt = NULL;
    build_pos_interp(grid_h, grid_w, 48, HD_VISION_MERGE_SIZE, &idx, &wgt);
    int *coords = NULL;
    build_rot_coords(grid_h, grid_w, HD_VISION_MERGE_SIZE, &coords);
    if (!idx || !wgt || !coords) {
        free(idx); free(wgt); free(coords);
        hd_set_error("vision: table build oom");
        return HD_ERR_OOM;
    }
    cudaMemcpy(pos_idx_d, idx, (size_t)4 * n * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(pos_wgt_d, wgt, (size_t)4 * n * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(coords_d, coords, (size_t)2 * n * sizeof(int),
               cudaMemcpyHostToDevice);
    free(idx); free(wgt); free(coords);

    float inv_freq[18];
    for (int k = 0; k < 18; k++) {
        float e = (float)(2 * k) / 36.0f;
        inv_freq[k] = 1.0f / powf(10000.0f, e);
    }
    cudaMemcpy(inv_d, inv_freq, 18 * sizeof(float), cudaMemcpyHostToDevice);

    ws->pos_idx = pos_idx_d;
    ws->pos_wgt = pos_wgt_d;
    ws->rot_coords = coords_d;
    ws->rot_inv = inv_d;
    ws->tables_n = n;
    ws->tables_gh = grid_h;
    ws->tables_gw = grid_w;
    return HD_OK;
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
    void *mask      = base + o.mask;
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

    O1_TIMING_BEGIN_GPU("VISION_TOTAL");
    /* ---- stage D: patch_embed ----
     * Production path: the Conv3d-as-matmul patch embedding runs through the
     * shared production GEMM backend (X[n,1536] x W[1152,1536]^T + bias,
     * bf16 in/out, fp32 accumulate). HD_VISION_PATCH_REF=1 keeps the old
     * scalar reference kernel for A/B validation. */
    O1_TIMING_BEGIN_GPU("VISION_PATCH_EMBED");
    {
        const char *ref = getenv("HD_VISION_PATCH_REF");
        if (ref && strcmp(ref, "1") == 0)
            hd_vision_patch(pixel_values, bw->patch_proj_w, bw->patch_proj_b,
                            patch_out, n, HD_VISION_PATCH_DIM, H);
        else
            hd_linear(pixel_values, bw->patch_proj_w, bw->patch_proj_b,
                      patch_out, n, H, HD_VISION_PATCH_DIM, 1);
    }
    O1_TIMING_END_GPU("VISION_PATCH_EMBED");

    /* ---- pos_embed interpolate + add ---- */
    O1_TIMING_BEGIN_GPU("VISION_PREPROCESS");
    {
        /* Persistent device tables: built once by hd_vision_prepare_tables()
         * for this (n, grid_h, grid_w) and reused for every forward. The
         * hot path therefore performs no malloc / H2D / table rebuild. */
        if (ws->pos_idx == NULL || ws->pos_wgt == NULL ||
            ws->rot_coords == NULL || ws->rot_inv == NULL) {
            hd_set_error("vision: persistent tables not prepared");
            return HD_ERR_MISSING;
        }
        hd_vision_pos_interp(bw->pos_embed, ws->pos_idx, ws->pos_wgt,
                             pos_emb, n, H);
        /* hidden = patch_out + pos_emb -> h_a */
        hd_residual_add(patch_out, pos_emb, h_a, (size_t)n * H);
    }

    /* ---- rot_pos_emb -> cos/sin ---- */
    {
        hd_vision_rot(ws->rot_inv, ws->rot_coords, rot, n, 18);
        /* cos/sin from rot [n, 36] -> [n, 72] fp32 */
        hd_vision_rot_cos_sin(rot, cosf, sinf, n, 18);
    }
    O1_TIMING_END_GPU("VISION_PREPROCESS");

    /* ---- 27 blocks ---- */
    O1_TIMING_BEGIN_GPU("VISION_27_BLOCKS");
    {
        const void *cur_in = h_a;
        void *cur_out = h_b;
        /* Zero additive attention mask [n,n] bf16 (vision attention is
         * non-causal / full attention). Both backends consume it: the
         * eager reference adds it to the scores, and the cuDNN graph has a
         * bound zero bias tensor. Materialized once per forward, outside
         * the block loop. */
        cudaMemset(mask, 0, (size_t)n * n * 2);
        for (int i = 0; i < HD_VISION_DEPTH; i++) {
            const hd_vision_block_binding *blk = &bw->blocks[i];
            /* norm1 */
            O1_TIMING_BEGIN_GPU("V_b_layernorm");
            hd_vision_layernorm(cur_in, blk->norm1_w, blk->norm1_b, ln1, n, H, eps);
            O1_TIMING_END_GPU("V_b_layernorm");
            /* qkv (cuBLAS) */
            O1_TIMING_BEGIN_GPU("V_b_qkv_gemm");
            hd_linear(ln1, blk->qkv_w, blk->qkv_b, qkv, n, 3456, H, 1);
            O1_TIMING_END_GPU("V_b_qkv_gemm");
            /* split q/k/v -> [Hd, n, D] */
            O1_TIMING_BEGIN_GPU("V_b_rope_split_merge");
            hd_vision_qkv_split(qkv, q, k, v, n, Hd, D);
            /* rotary on q/k (cos/sin [n, 72]) */
            hd_apply_rotary_f32(q, cosf, sinf, qr, Hd, n, D);
            hd_apply_rotary_f32(k, cosf, sinf, kr, Hd, n, D);
            O1_TIMING_END_GPU("V_b_rope_split_merge");
            /* attention: cuDNN SDPA (no mask) or eager reference.
             * cuDNN writes head-major [Hd, n, D]; eager writes seq-major
             * [n, Hd*D]. Normalize to seq-major in attn_out. */
            O1_TIMING_BEGIN_GPU("V_b_sdpa");
            int sdpa_ok = 0;
            if (ws->sdpa) {
                int rc = hd_sdpa_execute(ws->sdpa, qr, kr, v, mask, qkv, 0);
                if (rc == 0) {
                    /* qkv is free after qkv_split: use it as the head-major
                     * temp buffer, then transpose to seq-major attn_out. */
                    hd_vision_attn_merge(qkv, attn_out, Hd, n, D);
                    sdpa_ok = 1;
                    O1_TIMING_COUNTER_ADD("VIS_SDPA_CUDNN", 1);
                }
#ifdef O1_DEBUG_TIMING
                else if (i == 0)
                    fprintf(stderr, "[edit-geom] vision sdpa execute FAILED "
                                    "n=%d: %s\n", n, hd_last_error());
#endif
            }
            if (!sdpa_ok) {
                O1_TIMING_COUNTER_ADD("VIS_SDPA_EAGER", 1);
                hd_attention_eager(qr, kr, v, mask, scores, probs,
                                   attn_out, Hd, Hd, n, D, scaling);
            }
            O1_TIMING_END_GPU("V_b_sdpa");
            /* proj (cuBLAS) over [n, Hd*D] */
            O1_TIMING_BEGIN_GPU("V_b_o_proj");
            hd_linear(attn_out, blk->proj_w, blk->proj_b, attn_resid, n, H, H, 1);
            /* residual: attn_resid = cur_in + attn_out */
            hd_residual_add(cur_in, attn_resid, attn_resid, (size_t)n * H);
            O1_TIMING_END_GPU("V_b_o_proj");
            /* norm2 */
            O1_TIMING_BEGIN_GPU("V_b_layernorm");
            hd_vision_layernorm(attn_resid, blk->norm2_w, blk->norm2_b, ln2, n, H, eps);
            O1_TIMING_END_GPU("V_b_layernorm");
            /* mlp: fc1 -> gelu -> fc2 */
            O1_TIMING_BEGIN_GPU("V_b_fc1");
            hd_linear(ln2, blk->fc1_w, blk->fc1_b, fc1, n, I, H, 1);
            O1_TIMING_END_GPU("V_b_fc1");
            if (i == 0 && ws->block0_snaps && ws->block0_snaps[HD_B0_FC1])
                cudaMemcpy(ws->block0_snaps[HD_B0_FC1], fc1, (size_t)n * I * 2,
                           cudaMemcpyDeviceToDevice);
            O1_TIMING_BEGIN_GPU("V_b_gelu");
            hd_vision_gelu(fc1, fc1, (size_t)n * I);
            O1_TIMING_END_GPU("V_b_gelu");
            O1_TIMING_BEGIN_GPU("V_b_fc2");
            hd_linear(fc1, blk->fc2_w, blk->fc2_b, fc2, n, H, I, 1);
            /* residual: mlp_resid = attn_resid + fc2 */
            hd_residual_add(attn_resid, fc2, mlp_resid, (size_t)n * H);
            O1_TIMING_END_GPU("V_b_fc2");

            /* Block-output debug snapshots (dedicated buffers, captured
             * DURING the forward before the ping-pong copy). */
            if (ws->block_out_snaps && ws->block_out_snaps[i])
                cudaMemcpy(ws->block_out_snaps[i], mlp_resid,
                           (size_t)n * H * 2, cudaMemcpyDeviceToDevice);

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
                    O1_TIMING_BEGIN_GPU("VISION_DEEPSTACK_MERGERS");
                    /* NO spatial permutation: block output [n,1152] is
                     * already block-major; reinterpret as [m,4608] and
                     * norm over 4608 (use_postshuffle_norm=True). */
                    hd_vision_layernorm(mlp_resid, ds->norm_w, ds->norm_b,
                                        ds_fc1, m, 4608, eps);
                    if (ws->ds_merger_snaps && ws->ds_merger_snaps[ds_idx][0])
                        cudaMemcpy(ws->ds_merger_snaps[ds_idx][0], ds_fc1,
                                   (size_t)m * 4608 * 2, cudaMemcpyDeviceToDevice);
                    /* fc1 -> gelu -> fc2. fc1 must NOT be in-place (M>1
                     * GEMM with y==x races across tile blocks); use the
                     * dedicated ds_merged scratch. */
                    hd_linear(ds_fc1, ds->fc1_w, ds->fc1_b, ds_merged, m, 4608, 4608, 1);
                    if (ws->ds_merger_snaps && ws->ds_merger_snaps[ds_idx][1])
                        cudaMemcpy(ws->ds_merger_snaps[ds_idx][1], ds_merged,
                                   (size_t)m * 4608 * 2, cudaMemcpyDeviceToDevice);
                    hd_vision_gelu_exact(ds_merged, ds_merged, (size_t)m * 4608);
                    if (ws->ds_merger_snaps && ws->ds_merger_snaps[ds_idx][2])
                        cudaMemcpy(ws->ds_merger_snaps[ds_idx][2], ds_merged,
                                   (size_t)m * 4608 * 2, cudaMemcpyDeviceToDevice);
                    hd_linear(ds_merged, ds->fc2_w, ds->fc2_b, ds_fc2, m, O, 4608, 1);
                    if (ws->ds_merger_snaps && ws->ds_merger_snaps[ds_idx][3])
                        cudaMemcpy(ws->ds_merger_snaps[ds_idx][3], ds_fc2,
                                   (size_t)m * O * 2, cudaMemcpyDeviceToDevice);
                    cudaMemcpy(deepstack_out[ds_idx], ds_fc2,
                               (size_t)m * O * 2, cudaMemcpyDeviceToDevice);
                    O1_TIMING_END_GPU("VISION_DEEPSTACK_MERGERS");
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
    O1_TIMING_END_GPU("VISION_27_BLOCKS");

    /* ---- final merger ---- */
    O1_TIMING_BEGIN_GPU("VISION_FINAL_MERGER");
    {
        const hd_vision_merger_binding *mg = &bw->merger;
        /* norm over 1152 (use_postshuffle_norm=False): norm applies to the
         * [n, 1152] block output BEFORE the merge view. */
        hd_vision_layernorm(mlp_resid, mg->norm_w, mg->norm_b, merge_norm, n, H, eps);
        if (ws->merger_snaps && ws->merger_snaps[0])
            cudaMemcpy(ws->merger_snaps[0], merge_norm, (size_t)n * H * 2,
                       cudaMemcpyDeviceToDevice);
        /* NO spatial permutation: the vision tokens are already block-major
         * (token order (bh,bw,mh,mw)), so the normed [n,1152] memory is
         * logically [m,4608] by grouping 4 consecutive rows (upstream
         * x.view(-1, 4608)). */
        if (ws->merger_snaps && ws->merger_snaps[1])
            cudaMemcpy(ws->merger_snaps[1], merge_norm, (size_t)m * 4608 * 2,
                       cudaMemcpyDeviceToDevice);
        /* fc1 -> gelu -> fc2 */
        hd_linear(merge_norm, mg->fc1_w, mg->fc1_b, merge_fc1, m, 4608, 4608, 1);
        if (ws->merger_snaps && ws->merger_snaps[2])
            cudaMemcpy(ws->merger_snaps[2], merge_fc1, (size_t)m * 4608 * 2,
                       cudaMemcpyDeviceToDevice);
        hd_vision_gelu_exact(merge_fc1, merge_fc1, (size_t)m * 4608);
        if (ws->merger_snaps && ws->merger_snaps[3])
            cudaMemcpy(ws->merger_snaps[3], merge_fc1, (size_t)m * 4608 * 2,
                       cudaMemcpyDeviceToDevice);
        hd_linear(merge_fc1, mg->fc2_w, mg->fc2_b, merge_fc2, m, O, 4608, 1);
        if (ws->merger_snaps && ws->merger_snaps[4])
            cudaMemcpy(ws->merger_snaps[4], merge_fc2, (size_t)m * O * 2,
                       cudaMemcpyDeviceToDevice);
        cudaMemcpy(image_embeds_out, merge_fc2, (size_t)m * O * 2,
                   cudaMemcpyDeviceToDevice);
    }
    O1_TIMING_END_GPU("VISION_FINAL_MERGER");
    O1_TIMING_END_GPU("VISION_TOTAL");

    return HD_OK;
}