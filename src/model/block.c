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
#include <string.h>

#include <cuda_runtime.h>

/* ------------------------------------------------------------------ */
/* Weight resolution                                                   */
/* ------------------------------------------------------------------ */

static const void *weight_ptr(const hd_weight_store *ws, const char *name) {
    if (!ws) return NULL;
    for (int64_t i = 0; i < ws->n_allocs; i++) {
        if (ws->allocs[i].name[0] && strcmp(ws->allocs[i].name, name) == 0)
            return ws->allocs[i].dev_ptr;
    }
    return NULL;
}

static hd_status require_weight(const hd_weight_store *ws, const char *name,
                                const void **out) {
    const void *p = weight_ptr(ws, name);
    if (!p) {
        hd_set_error("block: missing weight %s", name);
        return HD_ERR_MISSING;
    }
    *out = p;
    return HD_OK;
}

/* ------------------------------------------------------------------ */
/* Scratch layout                                                      */
/* ------------------------------------------------------------------ */

int64_t hd_decoder_block_scratch_bytes(int64_t seq, int heads, int kv_heads,
                                       int hidden, int ff_hidden, int head_dim) {
    /* bf16 element count */
    int64_t e = 0;
    e += seq * hidden;                      /* in_ln       */
    e += seq * heads * head_dim;            /* qp, q, qr   */
    e += seq * heads * head_dim;
    e += seq * heads * head_dim;
    e += seq * kv_heads * head_dim;         /* kp, k, kr   */
    e += seq * kv_heads * head_dim;
    e += seq * kv_heads * head_dim;
    e += seq * kv_heads * head_dim;         /* vp, v       */
    e += seq * kv_heads * head_dim;
    e += (int64_t)heads * seq * seq;        /* scores      */
    e += (int64_t)heads * seq * seq;        /* probs       */
    e += seq * heads * head_dim;            /* attn_sm     */
    e += seq * hidden;                      /* attn_m, attn_h, attn_r */
    e += seq * hidden;
    e += seq * hidden;
    e += seq * hidden;                      /* post        */
    e += seq * ff_hidden;                   /* gate, up, swi */
    e += seq * ff_hidden;
    e += seq * ff_hidden;
    e += seq * hidden;                      /* mlp, mlp_r  */
    e += seq * hidden;

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
                           const hd_weight_store *wstore, int layer_idx,
                           void *scratch, int64_t scratch_bytes,
                           hd_block_internals *ints, void *out_dev,
                           int64_t seq, int heads, int kv_heads,
                           int hidden, int ff_hidden, int head_dim) {
    if (!in_dev || !pos_dev || !mask_dev || !wstore || !scratch || !out_dev) {
        hd_set_error("block: null argument");
        return HD_ERR_MISSING;
    }
    int64_t need = hd_decoder_block_scratch_bytes(seq, heads, kv_heads,
                                                  hidden, ff_hidden, head_dim);
    if (scratch_bytes < need) {
        hd_set_error("block: scratch too small");
        return HD_ERR_OOM;
    }

    /* ------------------------------------------------------------ */
    /* Resolve this layer's weights (by frozen name; all bias-free). */
    /* ------------------------------------------------------------ */
    const void *W[11];
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
    char name[256];
    hd_status st;
    for (int i = 0; i < 11; i++) {
        snprintf(name, sizeof(name),
                 "model.language_model.layers.%d.%s", layer_idx, slot[i]);
        st = require_weight(wstore, name, &W[i]);
        if (st != HD_OK) return st;
    }
    const void *w_q = W[0], *w_k = W[1], *w_v = W[2], *w_o = W[3];
    const void *w_in = W[4], *w_post = W[5];
    const void *w_g = W[6], *w_u = W[7], *w_d = W[8];
    const void *w_qn = W[9], *w_kn = W[10];

    /* ------------------------------------------------------------ */
    /* Carve scratch pointers. Offsets tracked in bf16 element units. */
    /* ------------------------------------------------------------ */
    int64_t S_H = seq * heads * head_dim;
    int64_t S_K = seq * kv_heads * head_dim;
    int64_t S_S = (int64_t)heads * seq * seq;
    int64_t S_I = seq * ff_hidden;
    int64_t S_HID = seq * hidden;

    int64_t e = 0;
    int64_t o_ln   = e; e += S_HID;
    int64_t o_qp   = e; e += S_H;
    int64_t o_q    = e; e += S_H;
    int64_t o_qr   = e; e += S_H;
    int64_t o_kp   = e; e += S_K;
    int64_t o_k    = e; e += S_K;
    int64_t o_kr   = e; e += S_K;
    int64_t o_vp   = e; e += S_K;
    int64_t o_v    = e; e += S_K;
    int64_t o_sco  = e; e += S_S;
    int64_t o_prb  = e; e += S_S;
    int64_t o_as   = e; e += S_H;
    int64_t o_ah   = e; e += S_HID;
    int64_t o_ar   = e; e += S_HID;
    int64_t o_post = e; e += S_HID;
    int64_t o_gate = e; e += S_I;
    int64_t o_up   = e; e += S_I;
    int64_t o_swi  = e; e += S_I;
    int64_t o_mlp  = e; e += S_HID;
    int64_t o_mlpr = e; e += S_HID;

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
    void *ah     = b + off((size_t)o_ah);
    void *ar     = b + off((size_t)o_ar);
    void *post   = b + off((size_t)o_post);
    void *gate   = b + off((size_t)o_gate);
    void *up     = b + off((size_t)o_up);
    void *swi    = b + off((size_t)o_swi);
    void *mlp    = b + off((size_t)o_mlp);
    void *mlpr   = b + off((size_t)o_mlpr);

    size_t bf16_bytes = off((size_t)e);
    float *cosf = (float *)(b + bf16_bytes);
    float *sinf = cosf + seq * head_dim;
    int64_t *sec = (int64_t *)(sinf + seq * head_dim);

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
    hd_rmsnorm(in_dev, w_in, ln, S, HD, eps);

    /* 2. q/k/v projections (bias-free) -> [S, H*D] / [S, KV*D]       */
    hd_linear(ln, w_q, NULL, qp, S, H * D, HD, 1);
    hd_linear(ln, w_k, NULL, kp, S, KV * D, HD, 1);
    hd_linear(ln, w_v, NULL, vp, S, KV * D, HD, 1);

    /* 3. head split -> [H,S,D] / [KV,S,D]                            */
    hd_head_split(qp, q, S, H, D);
    hd_head_split(kp, k, S, KV, D);
    hd_head_split(vp, v, S, KV, D);

    /* 4. q/k RMSNorm over head dim (rows x cols = S*H x D)           */
    hd_rmsnorm(q, w_qn, qr, S * H, D, eps);
    hd_rmsnorm(k, w_kn, kr, S * KV, D, eps);

    /* 5. MRoPE cos/sin (fp32), section [24,20,20] interleaved         */
    {
        int64_t section[3] = {24, 20, 20};
        cudaMemcpy(sec, section, sizeof(section), cudaMemcpyHostToDevice);
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            hd_set_error("block: sec copy %s", cudaGetErrorString(e));
            return HD_ERR_IO;
        }
    }
    /* input shape [3, 1, seq] fp32; batch=1. */
    hd_mrope_cos_sin(pos_dev, 1, S, sec, 3, D, theta, attn_scaling,
                     1, cosf, sinf);

    /* 6. apply rotary to q/k (head-major)                            */
    hd_apply_rotary(qr, cosf, sinf, q, H, S, D);
    hd_apply_rotary(kr, cosf, sinf, k, KV, S, D);

    /* 7. eager attention -> out [S,H,D] seq-major, already contiguous
     *    as [S, H*D] (torch .view(S,H,D) layout) ready for o_proj.
     *    The eager kernel internally transposes head-major to seq-major, so
     *    NO head_merge is needed here (see attn.cu hd_attn_transpose_kernel
     *    and contract 5.4: attn_out [S,H,D]).                         */
    hd_attention_eager(q, k, v, mask_dev, scores, probs, as,
                       H, KV, S, D, scaling);

    /* 8. o_proj over [S, H*D] contiguous attention output. Note: the
     *    eager kernel's internal transpose (attn.cu) already produces the
     *    seq-major [S,H,D] == torch .view(S,H,D) layout, so this feeds
     *    hd_attention_eager's output directly into o_proj.             */
    hd_linear(as, w_o, NULL, ah, S, HD, HD, 1);

    /* 9.  residual: attn_r = x + attn_hidden                         */
    hd_residual_add(in_dev, ah, ar, (size_t)S_HID);

    /* 10. post-attention RMSNorm                                     */
    hd_rmsnorm(ar, w_post, post, S, HD, eps);

    /* 11. SwiGLU MLP                                                  */
    hd_linear(post, w_g, NULL, gate, S, (int)ff_hidden, HD, 1);
    hd_linear(post, w_u, NULL, up, S, (int)ff_hidden, HD, 1);
    hd_swiglu(gate, up, swi, (size_t)S_I);
    hd_linear(swi, w_d, NULL, mlp, S, HD, (int)ff_hidden, 1);

    /* 12. residual: mlpr = attn_r + mlp                              */
    hd_residual_add(ar, mlp, mlpr, (size_t)S_HID);

    /* copy result to caller output                                    */
    {
        cudaError_t e = cudaMemcpy(out_dev, mlpr, (size_t)S_HID * 2,
                                   cudaMemcpyDeviceToDevice);
        e = cudaDeviceSynchronize();
        if (e != cudaSuccess) {
            hd_set_error("block: final sync %s", cudaGetErrorString(e));
            return HD_ERR_IO;
        }
    }

    /* optional internal-tensor export (validation harness)             */
    if (ints) {
        cudaError_t e = cudaSuccess;
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