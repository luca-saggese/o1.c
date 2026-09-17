#ifndef HD_BLOCK_H
#define HD_BLOCK_H

/*
 * M1.3 decoder block composition.
 *
 * Wires ONE complete Qwen3-VL text decoder layer (layer index `layer_idx`)
 * by composing the reference CUDA primitives from src/cuda/cuda.h, matching
 * the oracle composition in python/models/qwen3_vl_transformers.py
 * (Qwen3VLTextDecoderLayer.forward, lines 496-543).
 *
 * Numerics: BF16 compute with FP32 accumulation for GEMMs/softmax per
 * docs/M1_NUMERICAL_CONTRACT.md. Residual arcs and RoPE cos/sin follow the
 * oracle's bf16 rounding (class B), attention is class D.
 */

#include <stdint.h>

#include "hd_cuda.h"
#include "hd_cudnn_sdpa.h"
#include "weights.h"

/* Returns the number of device bytes the forward needs for scratch buffers.
 * Weights (from `wstore`) and the output buffer are owned by the caller and
 * are not included. scratch_bytes must equal this value. */
int64_t hd_decoder_block_scratch_bytes(int64_t seq, int heads, int kv_heads,
                                       int hidden, int ff_hidden, int head_dim);

/*
 * Optional internal-tensor export for validation. Each pointer, when non-NULL,
 * must be a caller-owned device buffer of the documented size; the block
 * copies the named intermediate into it. Leave NULL to skip export.
 */
typedef struct {
    void *ln0;          /* [seq, hidden]         input_layernorm(x)       */
    void *attn_hidden;  /* [seq, hidden]         o_proj output (pre-add)  */
    void *attn_resid;   /* [seq, hidden]         residual + attn_hidden   */
    void *post_ln;      /* [seq, hidden]         post_attention_layernorm */
    void *mlp_out;      /* [seq, hidden]         mlp output (pre-add)     */
} hd_block_internals;

/*
 * Pre-resolved device bindings for one decoder layer's weights. Resolved
 * exactly ONCE during init/load (hd_block_resolve) so the forward hot path
 * performs NO string-based tensor lookup. Query/projection/MLP/norm weights
 * are all bias-free and bf16 device-resident.
 */
typedef struct {
    int layer_idx;
    const void *q_proj;   /* [H*D, hidden] bf16  self_attn.q_proj   */
    const void *k_proj;   /* [KV*D, hidden] bf16 self_attn.k_proj   */
    const void *v_proj;   /* [KV*D, hidden] bf16 self_attn.v_proj   */
    const void *o_proj;   /* [hidden, H*D] bf16  self_attn.o_proj   */
    const void *input_ln; /* [hidden] bf16       input_layernorm    */
    const void *post_ln;  /* [hidden] bf16       post_attention_ln  */
    const void *gate_proj;/* [ff_hidden, hidden] mlp.gate_proj      */
    const void *up_proj;  /* [ff_hidden, hidden] mlp.up_proj        */
    const void *down_proj;/* [hidden, ff_hidden] mlp.down_proj      */
    const void *q_norm;   /* [head_dim] bf16     self_attn.q_norm   */
    const void *k_norm;   /* [head_dim] bf16     self_attn.k_norm   */
} hd_block_binding;

/*
 * Resolves a layer's 11 weights from the device store into `out` without
 * allocating or copying. Caller keeps the returned binding alive as long as
 * the store it references. Fails closed (HD_ERR_MISSING) if any required
 * tensor name is absent.
 */
hd_status hd_block_resolve(const hd_weight_store *wstore, int layer_idx,
                           hd_block_binding *out);

/*
 * Forward one decoder block for `seq` tokens (batch 1).
 *
 *   in_dev    [seq, hidden] bf16 device input (block input x)
 *   pos_dev   fp32 [3, seq] position_ids (int64 frozen position_ids cast to
 *             fp32)
 *   mask_dev  bf16 [1,1,seq,seq] attention mask (0.0 / finfo(bf16).min)
 *   bw        pre-resolved weight bindings (hd_block_resolve)
 *   sec_dev   int64 [3] device-resident MRoPE section [24,20,20] (resident
 *             across the whole forward; no HostToDevice copy in the block)
 *   scratch   caller-provided buffer of >= hd_decoder_block_scratch_bytes()
 *   out_dev   [seq, hidden] bf16 device output (block output)
 *
 * The forward is device-resident: no allocation, no host<->device transfer
 * and no device synchronization occurs inside this function (all blocked by
 * this function's caller). `ints` is optional (may be NULL) and receives the
 * intermediate tensors listed in hd_block_internals via DeviceToDevice copy.
 */
hd_status hd_decoder_block(const void *in_dev, const float *pos_dev,
                           const void *mask_dev,
                           const hd_block_binding *bw,
                           const int64_t *sec_dev,
                           void *scratch, int64_t scratch_bytes,
                           hd_block_internals *ints, void *out_dev,
                           int64_t seq, int heads, int kv_heads,
                           int hidden, int ff_hidden, int head_dim,
                           hd_sdpa_plan *sdpa);

#endif /* HD_BLOCK_H */