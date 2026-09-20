/*
 * M1.3 decoder block composition.
 *
 * Compose the M1.2 reference CUDA primitives to reproduce ONE Qwen3-VL text
 * decoder layer exactly, matching the oracle:
 *
 *   python/models/qwen3_vl_transformers.py
 *     Qwen3VLTextDecoderLayer.forward (lines 496-543):
 *        residual = x
 *        h = input_layernorm(x)                       (RMSNorm)
 *        h = self_attn(h, pos_emb, mask)
 *        h = residual + h                              (residual add)
 *        residual = h
 *        h = post_attention_layernorm(h)              (RMSNorm)
 *        h = mlp(h)                                    (SwiGLU)
 *        h = residual + h                              (residual add)
 *
 *     Qwen3VLTextAttention.forward (lines ~436-465):
 *        q = q_norm(q_proj(h).view(S,H,D)).transpose(1,2)
 *        k = k_norm(k_proj(h).view(S,KV,D)).transpose(1,2)
 *        v = v_proj(h).view(S,KV,D).transpose(1,2)
 *        q, k = apply_rotary_pos_emb(q, k, cos, sin)
 *        attn_out = eager_attention_forward(q, k, v, mask, scaling)
 *        attn_out = attn_out.reshape(S, H*D).contiguous()
 *        attn_out = o_proj(attn_out)
 *
 *     Qwen3VLTextMLP.forward (lines ~491-494):
 *        down_proj(silu(gate_proj(x)) * up_proj(x))
 *
 * Layout: projection output is [S, H*D] with head_dim contiguous (torch
 * .view(S,H,D) layout). We head_split that into head-major [heads,seq,dim]
 * as the eager kernels consume, then head_merge the seq-major attention
 * output back to [S, H*D] for o_proj. See src/cuda/cuda.h + the M1.2 golden
 * contract for layout definitions.
 *
 * Numerics: BF16 compute, FP32 accumulation per docs/M1_NUMERICAL_CONTRACT.md.
 */

#include "block.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>

#include "o1_timing.h"

/* ------------------------------------------------------------------ */
/* Bindings (resolved once at init; never looked up in the hot path)   */
/* ------------------------------------------------------------------ */

static const void *resolve_one(const hd_weight_store *ws, const char *name) {
    for (int64_t i = 0; i < ws->n_allocs; i++) {
        if (ws->allocs[i].name[0] &&
            strcmp(ws->allocs[i].name, name) == 0)
            return ws->allocs[i].dev_ptr;
    }
    return NULL;
}

hd_status hd_block_resolve(const hd_weight_store *wstore, int layer_idx,
                           hd_block_binding *out) {
    if (!wstore || !out) {
        hd_set_error("block: null argument in resolve");
        return HD_ERR_MISSING;
    }
    char name[256];
    const char *slot[11] = {
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
    };
    const void *p[11];
    for (int i = 0; i < 11; i++) {
        snprintf(name, sizeof(name),
                 "model.language_model.layers.%d.%s", layer_idx, slot[i]);
        p[i] = resolve_one(wstore, name);
        if (!p[i]) {
            hd_set_error("block: missing weight %s", name);
            return HD_ERR_MISSING;
        }
    }
    out->layer_idx = layer_idx;
    out->q_proj = p[0]; out->k_proj = p[1]; out->v_proj = p[2];
    out->o_proj = p[3];
    out->input_ln = p[4]; out->post_ln = p[5];
    out->gate_proj = p[6]; out->up_proj = p[7]; out->down_proj = p[8];
    out->q_norm = p[9]; out->k_norm = p[10];
    return HD_OK;
}

/* ------------------------------------------------------------------ */
/* Scratch layout                                                      */
/* ------------------------------------------------------------------ */

int64_t hd_decoder_block_scratch_bytes(int64_t seq, int heads, int kv_heads,
                                       int hidden, int ff_hidden, int head_dim) {
    /* bf16 element count, with 8-element (16-byte) alignment between every
     * region to match the offset layout in hd_decoder_block (cuDNN SDPA
     * requires >=16B alignment on q/k/v/out). */
    int64_t S_H = seq * heads * head_dim;
    int64_t S_K = seq * kv_heads * head_dim;
    int64_t S_S = (int64_t)heads * seq * seq;
    int64_t S_I = seq * ff_hidden;
    int64_t S_HID = seq * hidden;
    int64_t cur = 0;
    int64_t regs[21] = { S_HID, S_H, S_H, S_H, S_K, S_K, S_K, S_K, S_K,
                         S_S, S_S, S_H, S_H, S_HID, S_HID, S_HID,
                         S_I, S_I, S_I, S_HID, S_HID };
    for (int i = 0; i < 21; i++) {
        cur = (cur + 7) & ~(int64_t)7;
        cur += regs[i];
    }
    cur = (cur + 7) & ~(int64_t)7;
    int64_t e = cur;

    int64_t bytes = e * (int64_t)sizeof(uint16_t);
    bytes += 2 * seq * head_dim * (int64_t)sizeof(float); /* fp32 cos/sin */
    bytes += 256;   /* mrope section int64 [3] + slack */
    return bytes;
}

/* ------------------------------------------------------------------ */
/* Forward                                                             */
/* ------------------------------------------------------------------ */

hd_status hd_decoder_block(const void *in_dev, const float *pos_dev,
                           const void *mask_dev,
                           const hd_block_binding *bw,
                           const int64_t *sec_dev,
                           void *scratch, int64_t scratch_bytes,
                           hd_block_internals *ints, void *out_dev,
                           int64_t seq, int heads, int kv_heads,
                           int hidden, int ff_hidden, int head_dim,
                           hd_sdpa_plan *sdpa) {
    if (!in_dev || !pos_dev || !mask_dev || !bw ||
        !sec_dev || !scratch || !out_dev) {
        hd_set_error("block: null argument");
        return HD_ERR_MISSING;
    }
    int64_t need = hd_decoder_block_scratch_bytes(seq, heads, kv_heads,
                                                  hidden, ff_hidden, head_dim);
    if (scratch_bytes < need) {
        hd_set_error("block: scratch too small");
        return HD_ERR_OOM;
    }

    /* Bindings were resolved once at init (hd_block_resolve); the forward
     * hot path performs no string-based tensor lookup. */
    const void *w_q = bw->q_proj, *w_k = bw->k_proj, *w_v = bw->v_proj;
    const void *w_o = bw->o_proj;
    const void *w_in = bw->input_ln, *w_post = bw->post_ln;
    const void *w_g = bw->gate_proj, *w_u = bw->up_proj, *w_d = bw->down_proj;
    const void *w_qn = bw->q_norm, *w_kn = bw->k_norm;

    /* ------------------------------------------------------------ */
    /* Carve scratch pointers. Offsets tracked in bf16 element units. */
    /* ------------------------------------------------------------ */
    int64_t S_H = seq * heads * head_dim;
    int64_t S_K = seq * kv_heads * head_dim;
    int64_t S_S = (int64_t)heads * seq * seq;
    int64_t S_I = seq * ff_hidden;
    int64_t S_HID = seq * hidden;

    int64_t cur = 0;
    /* All offsets are aligned to 8 bf16 elements (16 bytes) so that buffers
     * handed to cuDNN SDPA (q/k/v/out) satisfy the backend's >=16B alignment
     * requirement. Eager kernels only need 2-byte alignment, so this is a
     * strict superset. */
    int64_t o_ln   = cur; cur += S_HID; cur = (cur + 7) & ~(int64_t)7;
    int64_t o_qp   = cur; cur += S_H;   cur = (cur + 7) & ~(int64_t)7;
    int64_t o_q    = cur; cur += S_H;   cur = (cur + 7) & ~(int64_t)7;
    int64_t o_qr   = cur; cur += S_H;   cur = (cur + 7) & ~(int64_t)7;
    int64_t o_kp   = cur; cur += S_K;   cur = (cur + 7) & ~(int64_t)7;
    int64_t o_k    = cur; cur += S_K;   cur = (cur + 7) & ~(int64_t)7;
    int64_t o_kr   = cur; cur += S_K;   cur = (cur + 7) & ~(int64_t)7;
    int64_t o_vp   = cur; cur += S_K;   cur = (cur + 7) & ~(int64_t)7;
    int64_t o_v    = cur; cur += S_K;   cur = (cur + 7) & ~(int64_t)7;
    int64_t o_sco  = cur; cur += S_S;   cur = (cur + 7) & ~(int64_t)7;
    int64_t o_prb  = cur; cur += S_S;   cur = (cur + 7) & ~(int64_t)7;
    int64_t o_as   = cur; cur += S_H;   cur = (cur + 7) & ~(int64_t)7;
    int64_t o_sdpa = cur; cur += S_H;   cur = (cur + 7) & ~(int64_t)7;
    int64_t o_ah   = cur; cur += S_HID; cur = (cur + 7) & ~(int64_t)7;
    int64_t o_ar   = cur; cur += S_HID; cur = (cur + 7) & ~(int64_t)7;
    int64_t o_post = cur; cur += S_HID; cur = (cur + 7) & ~(int64_t)7;
    int64_t o_gate = cur; cur += S_I;   cur = (cur + 7) & ~(int64_t)7;
    int64_t o_up   = cur; cur += S_I;   cur = (cur + 7) & ~(int64_t)7;
    int64_t o_swi  = cur; cur += S_I;   cur = (cur + 7) & ~(int64_t)7;
    int64_t o_mlp  = cur; cur += S_HID; cur = (cur + 7) & ~(int64_t)7;
    int64_t o_mlpr = cur; cur += S_HID; cur = (cur + 7) & ~(int64_t)7;

    uint8_t *b = scratch;
    size_t off(size_t ee) { return ee * sizeof(uint16_t); }
    void *ln    = b + off((size_t)o_ln);
    void *qp    = b + off((size_t)o_qp);
    void *q     = b + off((size_t)o_q);
    void *qr    = b + off((size_t)o_qr);
    void *kp    = b + off((size_t)o_kp);
    void *k     = b + off((size_t)o_k);
    void *kr    = b + off((size_t)o_kr);
    void *vp    = b + off((size_t)o_vp);
    void *v     = b + off((size_t)o_v);
    void *scores = b + off((size_t)o_sco);
    void *probs  = b + off((size_t)o_prb);
    void *as     = b + off((size_t)o_as);
    void *sdpa_out = b + off((size_t)o_sdpa);
    void *ah     = b + off((size_t)o_ah);
    void *ar     = b + off((size_t)o_ar);
    void *post   = b + off((size_t)o_post);
    void *gate   = b + off((size_t)o_gate);
    void *up     = b + off((size_t)o_up);
    void *swi    = b + off((size_t)o_swi);
    void *mlp    = b + off((size_t)o_mlp);
    void *mlpr   = b + off((size_t)o_mlpr);

    size_t bf16_bytes = off((size_t)cur);
    float *cosf = (float *)(b + bf16_bytes);
    float *sinf = cosf + seq * head_dim;

    /* ------------------------------------------------------------------ */
    /* Hyper-parameter constants (from config/dev.json + oracle)           */
    /* ------------------------------------------------------------------ */
    float eps     = 1e-6f;
    float theta   = 5000000.0f;
    float scaling = (float)(1.0 / sqrt((double)head_dim));
    float attn_scaling = 1.0f;   /* MRoPE attention_scaling for this rope init */

    int S = (int)seq;
    int H = (int)heads;
    int KV = (int)kv_heads;
    int HD = (int)hidden;
    int D = (int)head_dim;

    /* ------------------------------------------------------------ */
    /* 1. input RMSNorm                                             */
    /* ------------------------------------------------------------ */
    O1_BTIMING_BEGIN_GPU("B_input_norm");
    hd_rmsnorm(in_dev, w_in, ln, S, HD, eps);
    O1_BTIMING_END_GPU("B_input_norm");

    /* 2. q/k/v projections (bias-free) -> [S, H*D] / [S, KV*D]       */
    O1_BTIMING_BEGIN_GPU("B_qkv_proj");
    hd_linear(ln, w_q, NULL, qp, S, H * D, HD, 1);
    hd_linear(ln, w_k, NULL, kp, S, KV * D, HD, 1);
    hd_linear(ln, w_v, NULL, vp, S, KV * D, HD, 1);
    O1_BTIMING_END_GPU("B_qkv_proj");

    /* 3. head split -> [H,S,D] / [KV,S,D]                            */
    O1_BTIMING_BEGIN_GPU("B_head_split");
    hd_head_split(qp, q, S, H, D);
    hd_head_split(kp, k, S, KV, D);
    hd_head_split(vp, v, S, KV, D);
    O1_BTIMING_END_GPU("B_head_split");

    /* 4. q/k RMSNorm over head dim (rows x cols = S*H x D)           */
    O1_BTIMING_BEGIN_GPU("B_qk_norm");
    hd_rmsnorm(q, w_qn, qr, S * H, D, eps);
    hd_rmsnorm(k, w_kn, kr, S * KV, D, eps);
    O1_BTIMING_END_GPU("B_qk_norm");

    /* 5. MRoPE cos/sin (fp32), section [24,20,20] interleaved. The section
     *    array is already device-resident (sec_dev, bound once at init);
     *    no HostToDevice transfer happens inside the forward.            */
    O1_BTIMING_BEGIN_GPU("B_mrope");
    hd_mrope_cos_sin(pos_dev, 1, S, sec_dev, 3, D, theta, attn_scaling,
                     1, cosf, sinf);
    O1_BTIMING_END_GPU("B_mrope");

    /* 6. apply rotary to q/k (head-major)                            */
    O1_BTIMING_BEGIN_GPU("B_rope");
    hd_apply_rotary(qr, cosf, sinf, q, H, S, D);
    hd_apply_rotary(kr, cosf, sinf, k, KV, S, D);
    O1_BTIMING_END_GPU("B_rope");

    /* 7. attention -> out [S,H,D] seq-major, already contiguous
     *    as [S, H*D] (torch .view(S,H,D) layout) ready for o_proj.
     *
     *    Backend selection:
     *      - cuDNN SDPA (M2 pre-baseline): hd_sdpa_execute writes the
     *        head-major [H,S,D] output into `sdpa_out`, then hd_head_merge
     *        transposes it to seq-major [S,H*D] in `as`. The plan is built
     *        once by the caller (hd_generate) for the fixed shape.
     *      - eager reference (default): hd_attention_eager internally
     *        transposes head-major to seq-major, writing `as` directly.   */
    O1_BTIMING_BEGIN_GPU("B_sdpa");
    if (sdpa) {
        int rc = hd_sdpa_execute(sdpa, q, k, v, mask_dev, sdpa_out, 0);
        if (rc != 0) {
            /* cuDNN SDPA may reject non-standard shapes (e.g. large
             * keep-original-aspect sequences); fall back to the eager
             * reference backend rather than aborting generation. */
            O1_TIMING_COUNTER_ADD("DEC_SDPA_EAGER", 1);
            hd_attention_eager(q, k, v, mask_dev, scores, probs, as,
                               H, KV, S, D, scaling);
        } else {
            O1_TIMING_COUNTER_ADD("DEC_SDPA_CUDNN", 1);
            hd_head_merge(sdpa_out, as, S, H, D);
        }
    } else {
        O1_TIMING_COUNTER_ADD("DEC_SDPA_EAGER_NOPLAN", 1);
        hd_attention_eager(q, k, v, mask_dev, scores, probs, as,
                           H, KV, S, D, scaling);
    }
    O1_BTIMING_END_GPU("B_sdpa");

    /* 8. o_proj over [S, H*D] contiguous attention output. Note: the
     *    eager kernel's internal transpose (attn.cu) already produces the
     *    seq-major [S,H,D] == torch .view(S,H,D) layout, so this feeds
     *    hd_attention_eager's output directly into o_proj.             */
    O1_BTIMING_BEGIN_GPU("B_o_proj");
    hd_linear(as, w_o, NULL, ah, S, HD, HD, 1);
    O1_BTIMING_END_GPU("B_o_proj");

    /* 9.  residual: attn_r = x + attn_hidden                         */
    O1_BTIMING_BEGIN_GPU("B_attn_resid");
    hd_residual_add(in_dev, ah, ar, (size_t)S_HID);
    O1_BTIMING_END_GPU("B_attn_resid");

    /* 10. post-attention RMSNorm                                     */
    O1_BTIMING_BEGIN_GPU("B_post_norm");
    hd_rmsnorm(ar, w_post, post, S, HD, eps);
    O1_BTIMING_END_GPU("B_post_norm");

    /* 11. SwiGLU MLP                                                  */
    O1_BTIMING_BEGIN_GPU("B_mlp_gate_up");
    hd_linear(post, w_g, NULL, gate, S, (int)ff_hidden, HD, 1);
    hd_linear(post, w_u, NULL, up, S, (int)ff_hidden, HD, 1);
    O1_BTIMING_END_GPU("B_mlp_gate_up");
    O1_BTIMING_BEGIN_GPU("B_swiglu");
    hd_swiglu(gate, up, swi, (size_t)S_I);
    O1_BTIMING_END_GPU("B_swiglu");
    O1_BTIMING_BEGIN_GPU("B_mlp_down");
    hd_linear(swi, w_d, NULL, mlp, S, HD, (int)ff_hidden, 1);
    O1_BTIMING_END_GPU("B_mlp_down");

    /* 12. residual: mlpr = attn_r + mlp                              */
    O1_BTIMING_BEGIN_GPU("B_mlp_resid");
    hd_residual_add(ar, mlp, mlpr, (size_t)S_HID);
    O1_BTIMING_END_GPU("B_mlp_resid");

    /* copy result to caller output (device-to-device, async; the caller /
     * orchestrator owns synchronization — no sync inside the block)     */
    O1_BTIMING_BEGIN_GPU("B_final_copy");
    cudaError_t e = cudaMemcpy(out_dev, mlpr, (size_t)S_HID * 2,
                               cudaMemcpyDeviceToDevice);
    O1_BTIMING_END_GPU("B_final_copy");
    if (e != cudaSuccess) {
        hd_set_error("block: final copy %s", cudaGetErrorString(e));
        return HD_ERR_IO;
    }

    /* optional internal-tensor export (validation harness)             */
    if (ints) {
        if (ints->ln0)
            e = cudaMemcpy(ints->ln0, ln, (size_t)S_HID * 2,
                           cudaMemcpyDeviceToDevice);
        if (e == cudaSuccess && ints->attn_hidden)
            e = cudaMemcpy(ints->attn_hidden, ah, (size_t)S_HID * 2,
                           cudaMemcpyDeviceToDevice);
        if (e == cudaSuccess && ints->attn_resid)
            e = cudaMemcpy(ints->attn_resid, ar, (size_t)S_HID * 2,
                           cudaMemcpyDeviceToDevice);
        if (e == cudaSuccess && ints->post_ln)
            e = cudaMemcpy(ints->post_ln, post, (size_t)S_HID * 2,
                           cudaMemcpyDeviceToDevice);
        if (e == cudaSuccess && ints->mlp_out)
            e = cudaMemcpy(ints->mlp_out, mlp, (size_t)S_HID * 2,
                           cudaMemcpyDeviceToDevice);
        if (e != cudaSuccess) {
            hd_set_error("block: internal export sync %s",
                         cudaGetErrorString(e));
            return HD_ERR_IO;
        }
    }
    return HD_OK;
}