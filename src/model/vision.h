#ifndef HD_VISION_H
#define HD_VISION_H

/*
 * M2 ref-image visual conditioning (Qwen3-VL vision tower).
 *
 * Implements the upstream Qwen3VLVisionModel path that is MISSING from the
 * native forward (see docs/REF_IMAGE_NATIVE_IMPLEMENTATION.md):
 *
 *   pixel_values [N, 1536] (patchified, bf16)
 *     -> patch_embed (Conv3d as matmul)          [N, 1152]
 *     -> + pos_embeds (interpolated)             [N, 1152]
 *     -> 27 x Qwen3VLVisionBlock                 [N, 1152]
 *     -> spatial merge (2x2 unshuffle)           [N/4, 4608]
 *     -> merger (norm + fc1 + GELU + fc2)        [M, 4096]  = image_embeds
 *     -> deepstack mergers at blocks 8/16/24     [M, 4096]  x3
 *
 * Then the caller (hd_forward) replaces the <image_pad> placeholder rows in
 * the text embedding with image_embeds (masked_scatter) and injects the
 * deepstack embeds into decoder layers 0/1/2 (hidden_states[mask] += emb).
 *
 * Numerics: BF16 compute with FP32 accumulation for GEMMs, matching the
 * M1 numerical contract. LayerNorm (not RMSNorm) and GELU (pytorch_tanh)
 * are used by the vision tower.
 *
 * Vision config (frozen, models/dev/config.json vision_config):
 *   depth 27, hidden 1152, heads 16, head_dim 72, intermediate 4304,
 *   out_hidden 4096, spatial_merge_size 2, patch_size 16,
 *   temporal_patch_size 2, num_position_embeddings 2304,
 *   deepstack_visual_indexes [8, 16, 24]
 */

#include <stdint.h>

#include "hd_cuda.h"
#include "hd_cudnn_sdpa.h"
#include "weights.h"

#define HD_VISION_DEPTH        27
#define HD_VISION_HIDDEN       1152
#define HD_VISION_HEADS        16
#define HD_VISION_HEAD_DIM     72
#define HD_VISION_INTERMEDIATE 4304
#define HD_VISION_OUT_HIDDEN   4096
#define HD_VISION_MERGE_SIZE   2
#define HD_VISION_PATCH_SIZE   16
#define HD_VISION_TEMPORAL_PATCH 2
#define HD_VISION_PATCH_DIM    1536   /* 3 * 2 * 16 * 16 */
#define HD_VISION_NUM_POS      2304
#define HD_VISION_NUM_DS       3      /* deepstack mergers */
#define HD_VISION_DS_LAYERS    {8, 16, 24}

/* Block-0 debug snapshot slots (captured DURING the forward, since the
 * workspace scratch buffers are reused by all 27 blocks). */
enum {
    HD_B0_INPUT = 0,   /* cur_in before norm1 */
    HD_B0_NORM1,       /* ln1 */
    HD_B0_QKV,         /* qkv */
    HD_B0_Q,           /* q (head-major) */
    HD_B0_K,           /* k (head-major) */
    HD_B0_V,           /* v (head-major) */
    HD_B0_Q_ROT,       /* qr */
    HD_B0_K_ROT,       /* kr */
    HD_B0_ATTN_HEADS,  /* head-major attention output (qkv temp) */
    HD_B0_ATTN_MERGED, /* attn_out (seq-major) */
    HD_B0_PROJ,        /* attn_resid before residual */
    HD_B0_ATTN_RESID,  /* attn_resid after residual */
    HD_B0_NORM2,       /* ln2 */
    HD_B0_FC1,         /* fc1 */
    HD_B0_FC2,         /* fc2 */
    HD_B0_OUTPUT,      /* mlp_resid */
    HD_B0_SNAP_COUNT
};

/* One vision transformer block's weights (Qwen3VLVisionBlock). */
typedef struct {
    const void *norm1_w;   /* [1152] bf16 LayerNorm weight */
    const void *norm1_b;   /* [1152] bf16 LayerNorm bias   */
    const void *norm2_w;   /* [1152] bf16 LayerNorm weight */
    const void *norm2_b;   /* [1152] bf16 LayerNorm bias   */
    const void *qkv_w;     /* [3456, 1152] bf16 attn.qkv   */
    const void *qkv_b;     /* [3456] bf16 attn.qkv bias    */
    const void *proj_w;    /* [1152, 1152] bf16 attn.proj  */
    const void *proj_b;    /* [1152] bf16 attn.proj bias   */
    const void *fc1_w;     /* [4304, 1152] bf16 mlp.fc1    */
    const void *fc1_b;     /* [4304] bf16 mlp.fc1 bias     */
    const void *fc2_w;     /* [1152, 4304] bf16 mlp.fc2    */
    const void *fc2_b;     /* [1152] bf16 mlp.fc2 bias     */
} hd_vision_block_binding;

/* One vision patch merger (Qwen3VLVisionPatchMerger). */
typedef struct {
    const void *norm_w;    /* [4608] bf16 (post-shuffle) or [1152] (final) */
    const void *norm_b;    /* [4608] or [1152] bf16 */
    const void *fc1_w;     /* [4608, 4608] bf16 */
    const void *fc1_b;     /* [4608] bf16 */
    const void *fc2_w;     /* [4096, 4608] bf16 */
    const void *fc2_b;     /* [4096] bf16 */
} hd_vision_merger_binding;

/* All vision tower weights resolved once at init. */
typedef struct {
    const void *patch_proj_w;  /* [1152, 1536] bf16 (Conv3d as matmul) */
    const void *patch_proj_b;  /* [1152] bf16 */
    const void *pos_embed;     /* [2304, 1152] bf16 */
    hd_vision_block_binding blocks[HD_VISION_DEPTH];
    hd_vision_merger_binding merger;      /* final merger (norm over 1152) */
    hd_vision_merger_binding deepstack[HD_VISION_NUM_DS]; /* norm over 4608 */
} hd_vision_binding;

/*
 * Persistent device workspace for one vision tower forward. Sized for a
 * fixed max token count `n` (grid_t*grid_h*grid_w) and `m` merged tokens
 * (n / merge_size^2). Allocated once by the caller.
 */
typedef struct {
    void *patch_out;   /* [n, 1152] bf16 patch_embed output */
    void *pos_emb;     /* [n, 1152] bf16 interpolated pos embeds */
    void *rot;         /* [n, 1152] bf16 rotary freq (pre cos/sin) */
    void *cosf;        /* [n, 1152] fp32 cos */
    void *sinf;        /* [n, 1152] fp32 sin */
    void *h_a;         /* [n, 1152] bf16 block ping-pong A */
    void *h_b;         /* [n, 1152] bf16 block ping-pong B */
    void *ln1;         /* [n, 1152] bf16 norm1 out */
    void *qkv;         /* [n, 3456] bf16 qkv projection */
    void *q;           /* [16, n, 72] bf16 head-major q */
    void *k;           /* [16, n, 72] bf16 head-major k */
    void *v;           /* [16, n, 72] bf16 head-major v */
    void *qr;          /* [16, n, 72] bf16 rotary q */
    void *kr;          /* [16, n, 72] bf16 rotary k */
    void *scores;      /* [16, n, n] bf16 attention scores */
    void *probs;       /* [16, n, n] bf16 attention probs */
    void *attn_out;    /* [n, 1152] bf16 attention output (seq-major) */
    void *attn_resid;  /* [n, 1152] bf16 residual after attn */
    void *ln2;         /* [n, 1152] bf16 norm2 out */
    void *fc1;         /* [n, 4304] bf16 mlp fc1 out */
    void *fc2;         /* [n, 1152] bf16 mlp fc2 out */
    void *mlp_resid;   /* [n, 1152] bf16 residual after mlp */
    void *merged;      /* [m, 4608] bf16 spatial-merged block output */
    void *merge_norm;  /* [m, 4608] bf16 merger norm out */
    void *merge_fc1;   /* [m, 4608] bf16 merger fc1 out */
    void *merge_fc2;   /* [m, 4096] bf16 merger fc2 out (image_embeds) */
    void *ds_merged;   /* [m, 4608] bf16 deepstack merged input */
    void *ds_fc1;      /* [m, 4608] bf16 deepstack fc1 out */
    void *ds_fc2;      /* [m, 4096] bf16 deepstack fc2 out */
    int64_t bytes;
    /* cuDNN SDPA plan for the vision attention (created once by the caller
     * for the fixed n x n shape; NULL keeps the eager reference backend). */
    hd_sdpa_plan *sdpa;
    /* Optional debug snapshot: if non-NULL, block 0 output (mlp_resid after
     * the first block) is copied here for oracle comparison. */
    void *block0_snap;
    /* Optional debug snapshots of block-0 internal stages, captured DURING
     * the forward (the workspace buffers are reused by later blocks).
     * Each entry is a dedicated device buffer owned by the caller; NULL
     * entries are skipped. */
    void *block0_snaps[HD_B0_SNAP_COUNT];
} hd_vision_workspace;

/* Workspace region offsets (bytes) for a given token count. Exposed for
 * tests that need to read intermediate buffers back from the workspace. */
typedef struct {
    int64_t patch_out, pos_emb, rot, h_a, h_b, ln1, attn_resid, ln2, fc2,
            mlp_resid, q, k, v, qr, kr, qkv, scores, probs, attn_out, fc1,
            cosf, sinf, merged, merge_norm, merge_fc1, merge_fc2, ds_merged,
            ds_fc1, ds_fc2, total_bytes;
} hd_vision_offsets;

void hd_vision_layout(int64_t n, int64_t m, hd_vision_offsets *o);

/*
 * Resolve ALL vision tower weights from the device store into `out` exactly
 * once. Fails closed (HD_ERR_MISSING) if any required tensor is absent.
 */
hd_status hd_vision_resolve(const hd_weight_store *wstore,
                            hd_vision_binding *out);

/* Size (bytes) of the persistent vision workspace for `n` tokens. */
int64_t hd_vision_workspace_bytes(int64_t n);

/*
 * Run the vision tower forward, batch 1, single image (grid_t=1).
 *
 *   pixel_values [n, 1536] bf16 device (patchified, processor layout)
 *   grid_h, grid_w  image grid in patches (e.g. 26, 20)
 *   ws        persistent workspace (sized for n)
 *   bw        resolved vision weights
 *   image_embeds_out [m, 4096] bf16 device (m = n/4)
 *   deepstack_out    [3][m, 4096] bf16 device (may be NULL to skip)
 *
 * Device-resident: no allocation, no H2D/D2H, no sync inside.
 */
hd_status hd_vision_forward(const hd_vision_binding *bw,
                            hd_vision_workspace *ws,
                            const void *pixel_values, int n,
                            int grid_h, int grid_w,
                            void *image_embeds_out,
                            void *deepstack_out[HD_VISION_NUM_DS]);

#endif /* HD_VISION_H */