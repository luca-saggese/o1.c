#ifndef HD_FORWARD_H
#define HD_FORWARD_H

/*
 * M1.4 full HiDream transformer forward (device-resident).
 *
 * Implements the repository-equivalent of the oracle's
 * _forward_generation over the language model decoder, using the frozen
 * M1.3a/M1.3b execution architecture:
 *   - weights bound once at init (hd_forward_resolve), no string lookup in
 *     the hot path
 *   - persistent device workspace (hidden A/B ping-pong, QKV/attention/MLP
 *     scratch, head scratch)
 *   - no cudaMalloc/cudaFree, no host malloc/free, no H2D/D2H, no file I/O
 *     and no cudaDeviceSynchronize inside the block chain
 *   - diagnostic hooks observe the SAME production path (no debug forward)
 *
 * Forward steps (matching python _forward_generation steps 1-8):
 *   1. embed = embed_tokens(input_ids)                  [text, H]
 *   2. t_emb = t_embedder1(timestep)                    [H]   (sin+mlp)
 *   3. h_text = where(input_ids==tms, t_emb, embed)     [text, H]
 *   4. vemb = x_embedder(vinputs)                       [img, H]
 *   5. h_full = cat([h_text, vemb]) == inputs_embeds    [S, H]
 *   6. decoder: 36 x hd_decoder_block (hidden A/B)
 *   7. norm(h_last)                                     [S, H]  (RMSNorm)
 *   8. x_pred = final_layer2(norm_out)                  [S, 3072] (Linear bias)
 *
 * Batch 1. Batch>1 is out of scope for M1.4.
 */

#include <stdint.h>

#include "hd_cuda.h"
#include "hd_cudnn_sdpa.h"
#include "weights.h"
#include "block.h"

/* Global (non-layer) weights bound by hd_forward_resolve. Names follow the
 * frozen safetensors manifest (config/tensor_manifest_dev.json). */
typedef struct {
    const void *embed_tokens;      /* [151936, H] bf16 lookup          */
    /* t_embedder1: sinusoidal(256) -> linear0(H)x SiLU -> linear2(H)  */
    const void *te0_w;             /* [H, 256] bf16  mlp.0.weight      */
    const void *te0_b;             /* [H]    bf16  mlp.0.bias          */
    const void *te2_w;             /* [H, H]   bf16  mlp.2.weight      */
    const void *te2_b;             /* [H]    bf16  mlp.2.bias          */
    /* x_embedder (BottleneckPatchEmbed): proj1 (no bias), proj2 (bias) */
    const void *xe1_w;             /* [1024, 3072] bf16  proj1.weight  */
    const void *xe2_w;             /* [H, 1024] bf16  proj2.weight     */
    const void *xe2_b;             /* [H]    bf16  proj2.bias          */
    /* final_layer2 (FinalLayer): Linear(H -> 3072, bias)              */
    const void *fl_w;              /* [3072, H] bf16  linear.weight    */
    const void *fl_b;              /* [3072]   bf16  linear.bias       */
    const void *fnorm_w;           /* [H]    bf16  language_model.norm */

    /* per-layer block bindings (M1.3) */
    int n_layers;
    hd_block_binding *blocks;      /* [n_layers] resolved once */
} hd_forward_binding;

/*
 * Persistent device workspace for one forward. Sized for `seq` total tokens
 * and a single fixed profile. Allocated once by the caller (hd_forward
 * workspace sizing helper) and reused across repeated forwards.
 */
typedef struct {
    void *hidden_a;      /* [seq, H] bf16 ping-pong buffer A   */
    void *hidden_b;      /* [seq, H] bf16 ping-pong buffer B   */
    void *h_text;        /* [text, H] bf16 text embedding      */
    void *norm_out;      /* [seq, H] bf16 final norm output    */
    void *head_out;      /* [seq, 3072] bf16 final_layer2 out  */
    void *t_emb;         /* [H] bf16 timestep embedding        */
    void *te_hidden;     /* [H] bf16 t_embedder1 linear0 out   */
    void *freq;          /* [256] fp32 timestep freq table     */
    void *freq_bf16;     /* [256] bf16 freq cast (linear0 in)  */
    void *t_scaled;      /* [1] fp32 scaled timestep (t*1000)  */
    void *xe_stage;      /* [img, 1024] bf16 x_embedder proj1  */
    void *xe_out;        /* [img, H] bf16 x_embedder proj2 out */
    void *block_scratch; /* decoder block scratch (persistent) */
    int64_t block_scratch_bytes;
    /* cuDNN SDPA attention plan (M2 pre-baseline). Created once by the
     * caller (hd_generate) for the fixed shape; NULL keeps the eager
     * reference attention as the backend. */
    hd_sdpa_plan *sdpa;
} hd_forward_workspace;

/* Optional diagnostics on the production path (same forward). */
typedef struct {
    void *after_embedding;            /* [text,H]   step1  */
    void *after_timestep_conditioning;/* [text,H]   step3  */
    void *after_target_embedding;     /* [img,H]    step4  */
    void *after_block_0_input;        /* [S,H]      seq concat, step5 */
    void *after_block_0;              /* [S,H]      layer 0*/
    void *after_block_mid;            /* [S,H]      layer mid*/
    void *after_block_last;           /* [S,H]      layer N-1  */
    void *before_final_norm;          /* [S,H] = after_block_last */
    void *after_final_norm;           /* [S,H]      step7 */
    void *after_final_head;           /* [S,3072]   step8 (== model out) */
} hd_forward_diagnostics;

/*
 * Resolve ALL forward weights from the device store into `out` exactly once.
 * `blocks` and `te/freq` host buffers must live as long as the store.
 * Allocates out->blocks (n_layers) host array. Fails closed on any missing
 * tensor or unsupported layer count.
 */
hd_status hd_forward_resolve(const hd_weight_store *wstore, int n_layers,
                             hd_forward_binding *out);
void hd_forward_binding_free(hd_forward_binding *b);

/*
 * Size (bytes) required for the persistent workspace of one forward.
 * `needs_block_scratch` is returned separately to size block scratch.
 */
int64_t hd_forward_workspace_bytes(int64_t seq, int img_tokens, int heads,
                                   int kv_heads, int hidden, int ff_hidden,
                                   int head_dim, int64_t *block_scratch_bytes);

/* Optional visual conditioning (ref/edit/personalize modes). All device
 * pointers, bf16. NULL visual keeps the plain T2I path. */
typedef struct {
    const void *image_embeds;   /* [V, 4096] bf16 vision tower output */
    const void *deepstack[3];   /* [V, 4096] bf16 each */
    const uint8_t *visual_mask; /* [S] uint8: 1 at <image_pad> rows */
    int v_tokens;               /* V */
} hd_visual_cond;

/*
 * Run the FULL transformer forward, batch 1. Inputs must be device-resident
 * (caller staged). Outputs x_pred written to ws->head_out when bw->head_out_sel
 * != NULL (see below). All pointers live in the workspace / weight store /
 * caller-provided device buffers; no allocation inside forward.
 *
 *   input_ids   [text] int64 device
 *   pos_f32     [3,1,seq] fp32 device  (frozen position ids as fp32)
 *   mask_dev    [1,1,seq,seq] bf16 device attention mask
 *   vinputs     [img,3072] bf16 device (pixel_unshuffled target/input)
 *   visual      optional visual conditioning (NULL for plain T2I)
 *   timestep    [1] fp32 device
 *   sec_dev     [3] int64 device (MRoPE section, resident)
 *   diag        optional diagnostic export buffers (may be NULL)
 *   out_dev     [seq,3072] bf16 device output x_pred
 */
hd_status hd_forward(const hd_forward_binding *bw,
                     hd_forward_workspace *ws,
                     const int64_t *input_ids, int text_len,
                     const float *pos_f32,
                     const void *mask_dev,
                     const void *vinputs, int img_tokens,
                     const hd_visual_cond *visual,
                     const float *timestep,
                     const int64_t *sec_dev,
                     int seq, int heads, int kv_heads,
                     int hidden, int ff_hidden, int head_dim,
                     int tms_token_id,
                     hd_forward_diagnostics *diag,
                     void *out_dev,
                     void *stream);

/* Returns the number of layers bound (for introspection). */
int hd_forward_num_layers(const hd_forward_binding *bw);

/*
 * Return byte offset (from ws->hidden_a base) of a named workspace region,
 * or -1 if unknown. Used by debug harnesses to inspect intermediates in the
 * SAME workspace the forward used (no separate debug forward).
 * Valid names: hidden_a hidden_b h_text norm_out head_out t_emb te_hidden
 * freq freq_bf16 t_scaled xe_stage xe_out block_scratch
 */
int64_t hd_forward_ws_offset(const char *name, int64_t seq, int text_len,
                             int img_tokens, int heads, int kv_heads,
                             int hidden, int ff_hidden, int head_dim,
                             int64_t *block_scratch_bytes);

#endif /* HD_FORWARD_H */