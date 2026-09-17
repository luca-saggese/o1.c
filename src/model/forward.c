/*
 * M1.4 full HiDream transformer forward (device-resident).
 *
 * Composes the M1.2/M1.3 reference CUDA primitives plus the M1.4 embedding
 * helpers to reproduce the oracle's _forward_generation steps 1-8 in a single
 * device-resident execution path (batch 1):
 *
 *   1. embed = embed_tokens(input_ids)                    [text, H]
 *   2. t_emb = t_embedder1(timestep)                      [H]
 *   3. h_text = where(input_ids==tms, t_emb, embed)       [text, H]
 *   4. vemb = x_embedder(vinputs)                         [img, H]
 *   5. h_full = cat([h_text, vemb])                       [S, H]  -> hidden_a
 *   6. n_layers x hd_decoder_block (hidden A/B ping-pong)
 *   7. norm(h_last)                                       [S, H]  RMSNorm
 *   8. x_pred = final_layer2(norm_out)                    [S, 3072]
 *
 * The forward hot path performs no allocation, no host<->device transfer,
 * no device synchronization and no string-based weight lookup. Weights are
 * resolved once at init (hd_forward_resolve + hd_block_resolve); the single
 * persistent device workspace holds every intermediate.
 *
 * Timestep pipeline matches the oracle exactly:
 *   t_freq = timestep_embedding(t * 1000, 256)      (fp32 table)
 *   t_emb  = mlp(t_freq.to(bf16))                   (bf16 GEMMs + SiLU)
 */

#include "forward.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>

#include "hd_cuda.h"
#include "weights.h"
#include "block.h"
#include "o1_timing.h"

/* ------------------------------------------------------------------ */
/* Binding resolution                                                  */
/* ------------------------------------------------------------------ */

static int64_t forward_layout_offsets(int64_t seq, int text_len, int img_tokens,
                                      int hidden, int heads, int kv_heads,
                                      int ff_hidden, int head_dim,
                                      int64_t out_off[16],
                                      int64_t *block_scratch_bytes);

static const void *resolve_one(const hd_weight_store *ws, const char *name) {
    for (int64_t i = 0; i < ws->n_allocs; i++) {
        if (ws->allocs[i].name[0] &&
            strcmp(ws->allocs[i].name, name) == 0)
            return ws->allocs[i].dev_ptr;
    }
    return NULL;
}

hd_status hd_forward_resolve(const hd_weight_store *wstore, int n_layers,
                             hd_forward_binding *out) {
    if (!wstore || !out || n_layers <= 0) {
        hd_set_error("forward: null argument in resolve");
        return HD_ERR_MISSING;
    }
    hd_forward_binding b;
    memset(&b, 0, sizeof(b));

    b.embed_tokens = resolve_one(wstore, "model.language_model.embed_tokens.weight");
    b.te0_w = resolve_one(wstore, "model.t_embedder1.mlp.0.weight");
    b.te0_b = resolve_one(wstore, "model.t_embedder1.mlp.0.bias");
    b.te2_w = resolve_one(wstore, "model.t_embedder1.mlp.2.weight");
    b.te2_b = resolve_one(wstore, "model.t_embedder1.mlp.2.bias");
    b.xe1_w = resolve_one(wstore, "model.x_embedder.proj1.weight");
    b.xe2_w = resolve_one(wstore, "model.x_embedder.proj2.weight");
    b.xe2_b = resolve_one(wstore, "model.x_embedder.proj2.bias");
    b.fl_w  = resolve_one(wstore, "model.final_layer2.linear.weight");
    b.fl_b  = resolve_one(wstore, "model.final_layer2.linear.bias");
    b.fnorm_w = resolve_one(wstore, "model.language_model.norm.weight");

    if (!b.embed_tokens || !b.te0_w || !b.te0_b || !b.te2_w || !b.te2_b ||
        !b.xe1_w || !b.xe2_w || !b.xe2_b || !b.fl_w || !b.fl_b ||
        !b.fnorm_w) {
        hd_set_error("forward: missing global weight(s) in store");
        return HD_ERR_MISSING;
    }

    b.n_layers = n_layers;
    b.blocks = (hd_block_binding *)calloc((size_t)n_layers,
                                          sizeof(hd_block_binding));
    if (!b.blocks) {
        hd_set_error("forward: oom allocating block bindings");
        return HD_ERR_OOM;
    }
    for (int i = 0; i < n_layers; i++) {
        hd_status s = hd_block_resolve(wstore, i, &b.blocks[i]);
        if (s != HD_OK) {
            hd_forward_binding_free(&b);
            return s;
        }
    }
    *out = b;
    return HD_OK;
}

void hd_forward_binding_free(hd_forward_binding *b) {
    if (!b) return;
    if (b->blocks) { free(b->blocks); b->blocks = NULL; }
    b->n_layers = 0;
}

int hd_forward_num_layers(const hd_forward_binding *bw) {
    return bw ? bw->n_layers : 0;
}

int64_t hd_forward_ws_offset(const char *name, int64_t seq, int text_len,
                             int img_tokens, int heads, int kv_heads,
                             int hidden, int ff_hidden, int head_dim,
                             int64_t *block_scratch_bytes) {
    if (!name) return -1;
    int64_t off[16];
    forward_layout_offsets(seq, text_len, img_tokens, hidden, heads, kv_heads,
                           ff_hidden, head_dim, off, block_scratch_bytes);
    struct { const char *n; int idx; } map[] = {
        {"hidden_a",0},{"hidden_b",1},{"h_text",2},{"norm_out",3},
        {"head_out",4},{"t_emb",5},{"te_hidden",6},{"freq",7},
        {"freq_bf16",8},{"t_scaled",9},{"xe_stage",10},{"xe_out",11},
        {"block_scratch",12},
    };
    for (size_t i = 0; i < sizeof(map)/sizeof(map[0]); i++)
        if (strcmp(map[i].n, name) == 0) return off[map[i].idx];
    return -1;
}

/* ------------------------------------------------------------------ */
/* Workspace layout and sizing                                         */
/* ------------------------------------------------------------------ */

/* Lays out the persistent workspace byte offsets. Region order fixed so
 * hd_forward carves identical pointers every call (no rebinding). */
static int64_t forward_layout_offsets(int64_t seq, int text_len, int img_tokens,
                                      int hidden, int heads, int kv_heads,
                                      int ff_hidden, int head_dim,
                                      int64_t out_off[16],
                                      int64_t *block_scratch_bytes) {
    int64_t o = 0;
    out_off[0]  = o; o += seq * hidden * 2;                  /* 0 hidden_a  */
    out_off[1]  = o; o += seq * hidden * 2;                  /* 1 hidden_b  */
    out_off[2]  = o; o += (int64_t)text_len * hidden * 2;    /* 2 h_text    */
    out_off[3]  = o; o += seq * hidden * 2;                  /* 3 norm_out  */
    out_off[4]  = o; o += seq * 3072 * 2;                    /* 4 head_out  */
    out_off[5]  = o; o += hidden * 2;                        /* 5 t_emb     */
    out_off[6]  = o; o += hidden * 2;                        /* 6 te_hidden */
    out_off[7]  = o; o += 256 * 4;                           /* 7 freq fp32 */
    out_off[8]  = o; o += 256 * 2;                           /* 8 freq_bf16 */
    out_off[9]  = o; o += 1 * 4;                             /* 9 t_scaled  */
    out_off[10] = o; o += (int64_t)img_tokens * 1024 * 2;    /* 10 xe_stage */
    out_off[11] = o; o += (int64_t)img_tokens * hidden * 2;  /* 11 xe_out   */
    /* Align the block scratch sub-arena to 256 bytes: cuDNN SDPA requires
     * >=16B alignment on q/k/v/out, and the block's internal offsets are
     * 16B-aligned relative to this base. The parent arena (cudaMalloc) is
     * 256B-aligned, so aligning this offset keeps every child aligned. */
    o = (o + 255) & ~((int64_t)255);
    out_off[12] = o;                                         /* 12 block_sc */
    int64_t bs = hd_decoder_block_scratch_bytes(seq, heads, kv_heads,
                                                hidden, ff_hidden, head_dim);
    if (block_scratch_bytes) *block_scratch_bytes = bs;
    o += bs;
    return o;
}

int64_t hd_forward_workspace_bytes(int64_t seq, int img_tokens, int heads,
                                   int kv_heads, int hidden, int ff_hidden,
                                   int head_dim, int64_t *block_scratch_bytes) {
    int64_t off[16];
    /* text_len is not known at sizing time here; assume seq-img_tokens.
     * Callers must pass identical dims to hd_forward. */
    int64_t text = seq - img_tokens;
    if (text < 0) text = 0;
    return forward_layout_offsets(seq, text, img_tokens, hidden,
                                  heads, kv_heads, ff_hidden, head_dim,
                                  off, block_scratch_bytes);
}

/* ------------------------------------------------------------------ */
/* Forward (single device-resident execution path, batch 1)            */
/* ------------------------------------------------------------------ */

hd_status hd_forward(const hd_forward_binding *bw,
                     hd_forward_workspace *ws,
                     const int64_t *input_ids, int text_len,
                     const float *pos_f32,
                     const void *mask_dev,
                     const void *vinputs, int img_tokens,
                     const float *timestep,
                     const int64_t *sec_dev,
                     int seq, int heads, int kv_heads,
                     int hidden, int ff_hidden, int head_dim,
                     int tms_token_id,
                     hd_forward_diagnostics *diag,
                     void *out_dev,
                     void *stream) {
    (void)stream; /* single stable compute stream for M1.4 */
    O1_TIMING_BEGIN_GPU("TRANSFORMER_TOTAL");
    if (!bw || !ws || !input_ids || !pos_f32 || !mask_dev || !vinputs ||
        !timestep || !sec_dev || !out_dev) {
        hd_set_error("forward: null argument");
        return HD_ERR_MISSING;
    }
    int S = (int)seq;
    int H = (int)hidden;
    int I = (int)img_tokens;
    int T = (int)text_len;
    float eps = 1e-6f;

    if (T + I != S) {
        hd_set_error("forward: text+img != seq");
        return HD_ERR_MISSING;
    }

    uint8_t *base = (uint8_t *)ws->hidden_a; /* caller-managed buffer */
    if (!base) {
        hd_set_error("forward: workspace buffer not allocated");
        return HD_ERR_MISSING;
    }

    int64_t off[16];
    int64_t bs_unused;
    forward_layout_offsets(S, T, I, H, heads, kv_heads, ff_hidden,
                           head_dim, off, &bs_unused);

    void *hidden_a = base + off[0];
    void *hidden_b = base + off[1];
    void *h_text   = base + off[2];
    void *norm_out = base + off[3];
    void *head_out = base + off[4];
    void *t_emb    = base + off[5];
    void *te_hidden= base + off[6];
    float *freq_f32 = (float *)(base + off[7]);
    void *freq_bf16= base + off[8];
    float *t_scaled= (float *)(base + off[9]);
    void *xe_stage = base + off[10];
    void *xe_out   = base + off[11];
    void *scratch  = base + off[12];

    /* ------------------------------------------------------------------ */
    /* Step 2: t_emb = t_embedder1(timestep), [H] bf16                    */
    /* ------------------------------------------------------------------ */
    O1_TIMING_BEGIN_GPU("EMBEDDING");
    hd_scale_f32(timestep, t_scaled, 1000.0f, 1);
    hd_timestep_embed(t_scaled, freq_f32, 1, 256);
    hd_f32_convert_bf16(freq_f32, freq_bf16, 256);
    /* mlp[0].weight [H,256] (out,in) -> transpose_w=1; bias added. */
    hd_linear(freq_bf16, bw->te0_w, bw->te0_b, te_hidden, 1, H, 256, 1);
    hd_silu(te_hidden, t_emb, (size_t)H);
    /* mlp[2].weight [H,H] (out,in) -> transpose_w=1; bias added.
     * Do NOT write in place into t_emb: an M=1,N=K=H GEMM writing y==x
     * races across tile blocks (later blockIdx.y overwrites rows other
     * blocks still read). Route into the free te_hidden scratch, then
     * D2D-copy the result back into t_emb. */
    hd_linear(t_emb, bw->te2_w, bw->te2_b, te_hidden, 1, H, H, 1);
    cudaMemcpy(t_emb, te_hidden, (size_t)H * 2, cudaMemcpyDeviceToDevice);

    /* ------------------------------------------------------------------ */
    /* Step 1 + 3: embed -> where(tms, t_emb, embed) -> h_text [T,H]      */
    /* ------------------------------------------------------------------ */
    hd_gather_rows(bw->embed_tokens, input_ids, h_text, T, H,
                   (int64_t)151936);
    if (diag && diag->after_embedding)
        cudaMemcpy(diag->after_embedding, h_text, (size_t)T * H * 2,
                   cudaMemcpyDeviceToDevice);
    hd_apply_tms_condition(input_ids, h_text, t_emb, h_text,
                           T, H, tms_token_id);
    if (diag && diag->after_timestep_conditioning)
        cudaMemcpy(diag->after_timestep_conditioning, h_text,
                   (size_t)T * H * 2, cudaMemcpyDeviceToDevice);

    /* ------------------------------------------------------------------ */
    /* Step 4: vemb = x_embedder(vinputs), [I,H] bf16                     */
    /* ------------------------------------------------------------------ */
    /* proj1.weight [1024,3072] (out,in) -> transpose_w=1, no bias. */
    hd_linear(vinputs, bw->xe1_w, NULL, xe_stage, I, 1024, 3072, 1);
    /* proj2.weight [H,1024] (out,in) -> transpose_w=1, bias. */
    hd_linear(xe_stage, bw->xe2_w, bw->xe2_b, xe_out, I, H, 1024, 1);
    if (diag && diag->after_target_embedding)
        cudaMemcpy(diag->after_target_embedding, xe_out, (size_t)I * H * 2,
                   cudaMemcpyDeviceToDevice);

    /* ------------------------------------------------------------------ */
    /* Step 5: hidden_a = cat([h_text, vemb])  [S,H]                      */
    /* ------------------------------------------------------------------ */
    cudaMemcpy(hidden_a, h_text, (size_t)T * H * 2, cudaMemcpyDeviceToDevice);
    cudaMemcpy((uint8_t *)hidden_a + (size_t)T * H * 2, xe_out,
               (size_t)I * H * 2, cudaMemcpyDeviceToDevice);
    O1_TIMING_END_GPU("EMBEDDING");
    if (diag && diag->after_block_0_input)
        cudaMemcpy(diag->after_block_0_input, hidden_a, (size_t)S * H * 2,
                   cudaMemcpyDeviceToDevice);

    /* ------------------------------------------------------------------ */
    /* Step 6: decoder ping-pong. hd_decoder_block writes out_dev via a    */
    /* D2D copy, so input and output must be distinct buffers.            */
    /* layer i: cur_in -> cur_out, then swap. Last write lands in `cur_out`*/
    /* at iteration n-1, then we swap once more leaving it in `cur_in`.   */
    /* ------------------------------------------------------------------ */
    const void *cur_in  = hidden_a;
    void *cur_out       = hidden_b;
    int n = bw->n_layers;
    O1_TIMING_BEGIN_GPU("BLOCKS_TOTAL");
    for (int i = 0; i < n; i++) {
        O1_TIMING_BEGIN_GPU("BLOCK_SINGLE");
        hd_status s = hd_decoder_block(cur_in, pos_f32, mask_dev,
                                       &bw->blocks[i], sec_dev,
                                       scratch, ws->block_scratch_bytes,
                                       NULL, cur_out,
                                       seq, heads, kv_heads,
                                       hidden, ff_hidden, head_dim,
                                       ws->sdpa);
        O1_TIMING_END_GPU("BLOCK_SINGLE");
        if (s != HD_OK) return s;

        if (diag) {
            void *dst = NULL;
            if (i == 0) dst = diag->after_block_0;
            else if (i == n / 2) dst = diag->after_block_mid;
            else if (i == n - 1) dst = diag->after_block_last;
            if (dst)
                cudaMemcpy(dst, cur_out, (size_t)S * H * 2,
                           cudaMemcpyDeviceToDevice);
        }

        void *tmp = cur_out;
        cur_out = (void *)cur_in;
        cur_in = tmp;
    }
    /* After swapping following the final write, cur_in holds the last block
     * output. final_hidden = cur_in. */
    void *final_hidden = (void *)cur_in;
    O1_TIMING_END_GPU("BLOCKS_TOTAL");

    /* ------------------------------------------------------------------ */
    /* Step 7: final RMSNorm                                              */
    /* ------------------------------------------------------------------ */
    O1_TIMING_BEGIN_GPU("FINAL_NORM_HEAD");
    if (diag && diag->before_final_norm)
        cudaMemcpy(diag->before_final_norm, final_hidden, (size_t)S * H * 2,
                   cudaMemcpyDeviceToDevice);
    hd_rmsnorm(final_hidden, bw->fnorm_w, norm_out, S, H, eps);
    if (diag && diag->after_final_norm)
        cudaMemcpy(diag->after_final_norm, norm_out, (size_t)S * H * 2,
                   cudaMemcpyDeviceToDevice);

    /* ------------------------------------------------------------------ */
    /* Step 8: x_pred = final_layer2(norm_out), [S, 3072]                 */
    /* ------------------------------------------------------------------ */
    hd_linear(norm_out, bw->fl_w, bw->fl_b, head_out, S, 3072, H, 1);
    cudaMemcpy(out_dev, head_out, (size_t)S * 3072 * 2, cudaMemcpyDeviceToDevice);
    O1_TIMING_END_GPU("FINAL_NORM_HEAD");
    O1_TIMING_END_GPU("TRANSFORMER_TOTAL");
    if (diag && diag->after_final_head)
        cudaMemcpy(diag->after_final_head, out_dev, (size_t)S * 3072 * 2,
                   cudaMemcpyDeviceToDevice);

    return HD_OK;
}
