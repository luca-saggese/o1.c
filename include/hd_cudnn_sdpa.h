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

#ifdef __cplusplus
}
#endif

#endif /* HD_CUDNN_SDPA_H */
