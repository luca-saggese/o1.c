#ifndef HD_GEMM_H
#define HD_GEMM_H

/*
 * M2 production GEMM backend.
 *
 * hd_linear() dispatches to a persistent cuBLASLt runtime when one is
 * active (hd_gemm_runtime_init), otherwise falls back to the hand-written
 * reference kernel (hd_linear_reference). The runtime owns:
 *   - one cublasHandle_t / cublasLtHandle_t (created once, destroyed once)
 *   - a persistent workspace (no cudaMalloc/free in the forward)
 *   - a cached cuBLASLt matmul plan per (M,N,K) shape, with a tuned and
 *     cached algorithm selected once per shape
 *
 * The reference kernel is kept as the correctness/debug backend.
 */

#include <stddef.h>
#include <stdint.h>

#include "hidream.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque persistent GEMM runtime. */
typedef struct hd_gemm_runtime hd_gemm_runtime;

/*
 * Creates the persistent cuBLAS/cuBLASLt handles and allocates the workspace.
 * Call ONCE at runtime init (e.g. inside hd_weights_to_device). Returns NULL
 * on failure (errbuf set via hd_cuda_errbuf).
 */
hd_gemm_runtime *hd_gemm_runtime_init(int device_id, size_t workspace_bytes);

/* Destroys the runtime and frees the workspace. Call ONCE at shutdown. */
void hd_gemm_runtime_destroy(hd_gemm_runtime *rt);

/* Returns the runtime's persistent workspace (may be NULL if 0 bytes). */
void *hd_gemm_workspace(hd_gemm_runtime *rt);
size_t hd_gemm_workspace_bytes(hd_gemm_runtime *rt);

/*
 * Reference linear (class C): y[M,N] = x[M,K] . w[N,K]^T, bf16 in/out,
 * fp32 accumulate. Hand-written tiled kernel. Kept for correctness/debug.
 */
void hd_linear_reference(const void *x_dev, const void *w_dev,
                         const void *bias_dev, void *y_dev,
                         int M, int N, int K, int transpose_w);

/*
 * Production linear: dispatches to the persistent cuBLAS/cuBLASLt backend
 * when the runtime is active, else falls back to hd_linear_reference.
 * Same contract as hd_linear_reference.
 */
void hd_linear(const void *x_dev, const void *w_dev, const void *bias_dev,
               void *y_dev, int M, int N, int K, int transpose_w);

/* Selects the backend used by hd_linear (default: production). */
void hd_gemm_set_backend(int use_production);

/*
 * Selects the production sub-backend: 0 = cuBLASLt (default, tuned+cached
 * per shape), 1 = cublasGemmEx (fallback/debug). Only meaningful when the
 * production backend is active.
 */
void hd_gemm_set_prod_backend(int backend);

#ifdef __cplusplus
}
#endif

#endif /* HD_GEMM_H */