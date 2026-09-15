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

#include "cuda.h"
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
 * Forward one decoder block for `seq` tokens (batch 1).
 *
 *   in_dev  [seq, hidden] bf16 device input (block input x)
 *   pos_dev fp32 [3, seq] position_ids (int64 frozen position_ids cast to fp32)
 *   mask_dev bf16 [1,1,seq,seq] attention mask (0.0 / finfo(bf16).min)
 *   wstore  loaded device weights (see hd_weights_to_device); the layer's
 *           weights are resolved by frozen name.
 *   layer_idx which decoder layer's weights to use
 *   scratch  caller-provided buffer of >= hd_decoder_block_scratch_bytes()
 *   out_dev  [seq, hidden] bf16 device output (block output)
 *
 * Whole-model forward is NOT performed; this drives only primitive module
 * calls. Returns HD_OK on success, HD_ERR_* otherwise and records a message
 * via hd_last_error(). `ints` is optional (may be NULL) and receives the
 * intermediate tensors listed in hd_block_internals.
 */
hd_status hd_decoder_block(const void *in_dev, const float *pos_dev,
                           const void *mask_dev,
                           const hd_weight_store *wstore, int layer_idx,
                           void *scratch, int64_t scratch_bytes,
                           hd_block_internals *ints, void *out_dev,
                           int64_t seq, int heads, int kv_heads,
                           int hidden, int ff_hidden, int head_dim);

#endif /* HD_BLOCK_H */