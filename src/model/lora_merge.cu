/*
 * M3 LoRA GPU merge (merge-on-load).
 *
 * For each linear target:
 *   delta = up @ down            (FP32, row-chunked)
 *   W_bf16 = bf16(float(W_bf16) + scale * delta)
 *
 * Uses cuBLAS for the delta GEMM and a small add/cast kernel. The resident
 * base weights are mutated in place; the forward path is unchanged.
 */

#include "hd_lora.h"
#include "cuda_internal.h"
#include "o1_timing.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Row-chunk budget for the FP32 delta scratch (bytes). */
#define HD_LORA_SCRATCH_BUDGET (64u * 1024u * 1024u)

__global__ void hd_lora_add_bf16_kernel(__nv_bfloat16 *w, const float *delta,
                                        size_t n, float scale) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float base = __bfloat162float(w[i]);
        w[i] = __float2bfloat16_rn(base + scale * delta[i]);
    }
}

hd_status hd_lora_merge_gpu(hd_lora_entry *entries, size_t count,
                            cudaStream_t stream) {
    if (count == 0) return HD_OK;

    cublasHandle_t cublas = NULL;
    cublasStatus_t cs = cublasCreate(&cublas);
    if (cs != CUBLAS_STATUS_SUCCESS) {
        snprintf(hd_cuda_errbuf(), 512, "lora: cublasCreate failed");
        return HD_ERR_IO;
    }
    cublasSetStream(cublas, stream);

    /* Scratch for one row chunk of the delta. */
    float *scratch = NULL;
    size_t scratch_cap = 0;

    for (size_t i = 0; i < count; i++) {
        hd_lora_entry *e = &entries[i];
        int R = (int)e->rank, K = (int)e->in_dim, N = (int)e->out_dim;
        float scale = e->multiplier * e->alpha / (float)e->rank;

        /* Row chunk size from the scratch budget. */
        int chunk_rows = (int)(HD_LORA_SCRATCH_BUDGET / (4u * (size_t)K));
        if (chunk_rows < 1) chunk_rows = 1;
        if (chunk_rows > N) chunk_rows = N;

        size_t need = (size_t)chunk_rows * K * 4;
        if (need > scratch_cap) {
            if (scratch) cudaFree(scratch);
            cudaError_t ce = cudaMalloc(&scratch, need);
            if (ce != cudaSuccess) {
                snprintf(hd_cuda_errbuf(), 512, "lora: scratch cudaMalloc: %s",
                         cudaGetErrorString(ce));
                cublasDestroy(cublas);
                return HD_ERR_OOM;
            }
            scratch_cap = need;
        }

        __nv_bfloat16 *w = (__nv_bfloat16 *)e->base_dev;
        const float *up = (const float *)e->up_dev;     /* [N,R] row-major */
        const float *down = (const float *)e->down_dev; /* [R,K] row-major */

        for (int row0 = 0; row0 < N; row0 += chunk_rows) {
            int C = (N - row0) < chunk_rows ? (N - row0) : chunk_rows;

            /* delta[C,K] = up[row0:row0+C, :] @ down[R,K]
             * cuBLAS col-major: C'[K,C] = A[K,R] @ B[R,C]
             *   A = down [R,K] row-major == down^T [K,R] col-major, ld=K, op=N
             *   B = up_chunk [C,R] row-major == up_chunk^T [R,C] col-major, ld=C, op=N
             *   C = delta [C,K] row-major == delta^T [K,C] col-major, ld=K */
            float alpha = 1.0f, beta = 0.0f;
            cs = cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                             K, C, R,
                             &alpha,
                             down, K,      /* A [R,K] row-major, ld=K */
                             up + (size_t)row0 * R, C, /* B [C,R] row-major, ld=C */
                             &beta,
                             scratch, K);  /* C [C,K] row-major, ld=K */
            if (cs != CUBLAS_STATUS_SUCCESS) {
                snprintf(hd_cuda_errbuf(), 512, "lora: cublasSgemm failed: %d", (int)cs);
                if (scratch) cudaFree(scratch);
                cublasDestroy(cublas);
                return HD_ERR_IO;
            }

            size_t n = (size_t)C * K;
            hd_lora_add_bf16_kernel<<<(unsigned)((n + 255) / 256), 256, 0, stream>>>(
                w + (size_t)row0 * K, scratch, n, scale);
        }
    }

    if (scratch) cudaFree(scratch);
    cublasDestroy(cublas);
    return HD_OK;
}