#ifndef HD_CUDA_H
#define HD_CUDA_H

/*
 * Public C ABI for the M1.2 reference CUDA transformer primitives.
 *
 * This is a *reference implementation* engine for validating the native
 * pipeline against frozen golden fixtures. Grade is BF16 compute with FP32
 * accumulation for GEMMs / softmax (see docs/M1_NUMERICAL_CONTRACT.md).
 * Kernels are hand-written CUDA; cuBLAS is NOT used for any primitive.
 *
 * Every primitive is a small kernel-launch wrapper. Pointers must be device
 * pointers in the current CUDA context (callers stage host data). Reference
 * numeric values are exposed as host helpers too, so the C harness can
 * cross-check against cuBLAS without invoking any model forward.
 */

#include <stddef.h>
#include <stdint.h>

#include "hidream.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* BF16 host helpers                                                   */
/* ------------------------------------------------------------------ */

typedef uint16_t hd_bf16;

/* IEEE-754 fp32 <-> BF16 (round-to-nearest-even on the way down). */
float hd_bf16_to_f32(uint16_t b);
uint16_t hd_f32_to_bf16(float f);

/* Raw little-endian payload host readers used by the fixture harness. */
void hd_bf16_buf_to_f32(const void *src, float *dst, size_t n);
void hd_f32_buf_to_bf16(const float *src, void *dst, size_t n);

/* ------------------------------------------------------------------ */
/* Error reporting                                                     */
/* ------------------------------------------------------------------ */

const char *hd_cuda_last_error(void);
void hd_cuda_clear_error(void);

/* ------------------------------------------------------------------ */
/* Device-layout primitives (reference kernels, current stream)        */
/* ------------------------------------------------------------------ */

/*
 * RMSNorm (class B): y[row, k] = w[k] * x * rsqrt(mean(x^2, dim=-1) + eps),
 * fp32 internal, bf16 in/out. x is [rows, cols], weight length cols, reduced
 * over the last (contiguous) dim.
 */
void hd_rmsnorm(const void *x_dev, const void *w_dev, void *y_dev,
                int rows, int cols, float eps);

/* SiLU (class B): y = x * sigmoid(x), bf16 elementwise. */
void hd_silu(const void *x_dev, void *y_dev, size_t n);

/* SwiGLU (class B): y = silu(gate) * up, bf16 elementwise. */
void hd_swiglu(const void *gate_dev, const void *up_dev, void *y_dev, size_t n);

/* Residual add (class B): y[i] = x[i] + a[i] elementwise, bf16. */
void hd_residual_add(const void *x_dev, const void *a_dev, void *y_dev, size_t n);

/*
 * Linear / GEMM (class C): y[M,N] = x[M,K] x W[K,N]^T with fp32 accumulation.
 * transpose_w == 1: W stored [N,K] (out,in) -> compute x^T . W^T.
 * transpose_w == 0: W stored [K,N] (in,out) -> direct mac.
 * All operands are device pointers (bf16).
 */
void hd_linear(const void *x_dev, const void *w_dev, const void *bias_dev,
               void *y_dev, int M, int N, int K, int transpose_w);

/*
 * Timestep sinusoidal embedding (class B): produce the [N, dim] fp32 table.
 * t is [N] fp32. y is fp32.
 */
void hd_timestep_embed(const float *t_dev, float *y_dev, int N, int dim);

/*
 * MRoPE cos/sin table (class B): given fp32 position_ids [3, bs, seq],
 * fp32 inv_freq [dim/2], mrope_section, produce cos/sin [1, seq, dim]
 * (fp32 outputs; oracle casts to bf16 only at use). Returns fp32.
 */
void hd_mrope_cos_sin(const float *pos_dev, int bs, int seq,
                      const int64_t *section_dev, int n_section,
                      int head_dim, float theta, float attention_scaling,
                      int interleaved, float *cos_dev, float *sin_dev);

/*
 * apply_rotary_pos_emb (class B). q/k shape [heads, seq, dim] device bf16.
 * cos/sin [1, seq, dim] fp32. y same layout, bf16.
 */
void hd_apply_rotary(const void *x_dev, const float *cos_dev,
                     const float *sin_dev, void *y_dev,
                     int heads, int seq, int dim);

/*
 * Eager attention (class D): q[heads,seq,dim], k/v[kv_heads,seq,dim],
 * mask[1,1,seq,seq]. Produces scores[heads,seq,seq], probs[heads,seq,seq],
 * out[seq,heads,dim] (seq-major). All bf16, softmax fp32.
 */
void hd_attention_eager(const void *q_dev, const void *k_dev, const void *v_dev,
                        const void *mask_dev, void *scores_dev,
                        void *probs_dev, void *out_dev,
                        int heads, int kv_heads, int seq, int dim,
                        float scaling);

/*
 * head_split / head_merge (class A, bit-exact). in [1,seq,H,D] bf16 ->
 * out [H,seq,D] bf16, and inverse. Byte-level permutation only.
 */
void hd_head_split(const void *in_dev, void *out_dev, int seq, int heads, int dim);
void hd_head_merge(const void *in_dev, void *out_dev, int seq, int heads, int dim);

/*
 * Row-gather embedding lookup (class A, byte-copy). table [nrows, cols]
 * bf16, idx [M] int64 -> out [M, cols] bf16. Out-of-range idx fall back to
 * padding row 0. Caller provides device buffers; no allocation.
 */
void hd_gather_rows(const void *table_dev, const void *idx_dev, void *out_dev,
                    int M, int cols, int64_t nrows);

/*
 * Timestep conditioning (class A copy): out[row] = t_emb when idx[row]==tms_id
 * else emb[row] (broadcast of the single [H] t_emb across matching rows).
 * Used to reproduce `torch.where(tms_mask, t_emb_expanded, inputs_embeds)`.
 */
void hd_apply_tms_condition(const void *idx_dev, const void *emb_dev,
                            const void *t_emb_dev, void *out_dev,
                            int M, int H, int64_t tms_id);

/* y[i] = x[i] * s elementwise fp32 (used to scale timestep by 1000). */
void hd_scale_f32(const float *in_dev, float *out_dev, float s, int n);
/* y = f32-to-bf16 device conversion (used for the freq table cast). */
void hd_f32_convert_bf16(const float *in_dev, void *out_dev, int n);

/* ------------------------------------------------------------------ */
/* cuBLAS cross-check helper (used by the harness, not the kernels)    */
/* ------------------------------------------------------------------ */

/* y[M,N] = x[M,K] @ W[N,K]^T in bf16 via cuBLAS BF16 GEMM. Returns HD_OK. */
hd_status hd_cublas_gemm_bf16(const void *x_dev, const void *w_dev, void *y_dev,
                              int M, int N, int K, int transpose_w);

#ifdef __cplusplus
}
#endif

#endif /* HD_CUDA_H */