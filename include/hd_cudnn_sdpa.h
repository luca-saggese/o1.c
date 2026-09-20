#ifndef HD_CUDNN_SDPA_H
#define HD_CUDNN_SDPA_H

/*
 * cuDNN SDPA attention backend (M2 pre-baseline).
 *
 * Thin C ABI over the cuDNN C++ Frontend (cudnn_frontend v1.22.1, header-only,
 * linked against libcudnn.so). The plan is built ONCE in hd_sdpa_create()
 * (graph construction + workspace sizing) and reused for every execute.
 *
 * Layout contract (matches the engine's head-split output):
 *   q [B, Hq, Sq, D] bf16 contiguous (B=1: [Hq, Sq, D])
 *   k [B, Hkv, Skv, D] bf16 contiguous (B=1: [Hkv, Skv, D])
 *   v [B, Hkv, Skv, D] bf16 contiguous
 *   out [B, Hq, Sq, D] bf16 contiguous (B=1: [Hq, Sq, D])
 *
 * The mask is passed as an additive bias tensor [1, 1, Sq, Skv] bf16
 * (0.0 / bf16 min), matching the engine's attention mask semantics.
 *
 * No cudnn_frontend types leak into the rest of the engine.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct hd_sdpa_plan hd_sdpa_plan;

/*
 * Builds the SDPA graph for the given shape and scale.
 *   scale = 1/sqrt(head_dim) (same as the eager reference).
 * Returns 0 on success, non-zero on failure (errbuf set via hd_cuda_errbuf).
 */
int hd_sdpa_create(hd_sdpa_plan **out,
                   int batch, int q_heads, int kv_heads,
                   int seq_q, int seq_kv, int head_dim,
                   float scale);

/*
 * Executes the plan on the given stream. q/k/v/out are device pointers in
 * the layout above; mask is the [1,1,Sq,Skv] bf16 additive bias (may be
 * NULL for no mask). Returns 0 on success.
 */
int hd_sdpa_execute(hd_sdpa_plan *plan,
                    const void *q, const void *k, const void *v,
                    const void *mask, void *out,
                    void *stream);

/* Frees the plan and its workspace. */
void hd_sdpa_destroy(hd_sdpa_plan *plan);

/*
 * Two-pass SDPA. Instead of one graph with a mixed additive bias, the query
 * range is split at `ar_len`:
 *
 *   pass 1 (causal): Q[0:ar_len] against K/V[0:ar_len]
 *   pass 2 (full):   Q[ar_len:seq] against K/V[0:seq]
 *
 * This is exactly equivalent to the masked single-graph form for the mask
 * produced by hd_seq_t2i (rows >= text_len-1 are fully unmasked, earlier
 * rows are causal). Both graphs are built here and only executed afterwards,
 * so the repeated path performs no plan build, allocation or sync.
 */
int hd_sdpa_create_split(hd_sdpa_plan **out,
                         int batch, int q_heads, int kv_heads,
                         int seq, int ar_len, int head_dim,
                         float scale);

/* Runs both passes. q/k/v/out use the same [H,S,D] layout as
 * hd_sdpa_execute; no mask is needed. Returns 0 on success. */
int hd_sdpa_execute_split(hd_sdpa_plan *plan,
                          const void *q, const void *k, const void *v,
                          void *out, void *stream);

/* Runs a single pass (0 = causal head, 1 = full tail) into the same output
 * layout. For benchmarking/validation only. */
int hd_sdpa_execute_split_pass(hd_sdpa_plan *plan, int which,
                               const void *q, const void *k, const void *v,
                               void *out, void *stream);

/*
 * Production entry point. Builds the split (two-pass) plan when
 * 0 < ar_len < seq, otherwise falls back to the masked single-graph plan.
 * Executed through the ordinary hd_sdpa_execute(), which dispatches to the
 * two passes when the plan is split (the mask argument is then ignored,
 * since the split reproduces the mask structurally).
 */
int hd_sdpa_create_prod(hd_sdpa_plan **out,
                        int batch, int q_heads, int kv_heads,
                        int seq, int ar_len, int head_dim,
                        float scale);

#ifdef __cplusplus
}
#endif

#endif /* HD_CUDNN_SDPA_H */
