/*
 * M2 production GEMM backend.
 *
 * hd_linear() dispatches to a persistent cuBLAS/cuBLASLt runtime when one is
 * active (hd_gemm_runtime_init), otherwise falls back to the hand-written
 * reference kernel (hd_linear_reference). The reference kernel is kept as
 * the correctness/debug backend (class C: bf16 in/out, fp32 accumulate).
 *
 * The runtime owns one cublasHandle_t / cublasLtHandle_t (created once,
 * destroyed once) plus a persistent workspace, so the forward hot path does
 * no handle creation and no cudaMalloc/free.
 */

#include "cuda_internal.h"
#include "gemm.h"

#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>

/* ------------------------------------------------------------------ */
/* Reference tiled matmul y = x @ w^T (w stored [N,K])                 */
/* ------------------------------------------------------------------ */

#define HD_GEMM_TILE 16

__global__ void hd_gemm_wT_kernel(const uint16_t *__restrict__ x,   /* [M,K] */
                                  const uint16_t *__restrict__ w,   /* [N,K] */
                                  const uint16_t *__restrict__ bias,/* [N] */
                                  uint16_t *__restrict__ y,         /* [M,N] */
                                  int M, int N, int K) {
    __shared__ float xs[HD_GEMM_TILE][HD_GEMM_TILE];
    __shared__ float ws[HD_GEMM_TILE][HD_GEMM_TILE];

    int row = blockIdx.x * HD_GEMM_TILE + threadIdx.x; /* M dim */
    int col = blockIdx.y * HD_GEMM_TILE + threadIdx.y; /* N dim */
    int txi = threadIdx.x, tyi = threadIdx.y;

    float acc = 0.0f;

    for (int kk = 0; kk < K; kk += HD_GEMM_TILE) {
        /* Load x tile [HD_GEMM_TILE x HD_GEMM_TILE] at (row, kk). */
        int gx = kk + tyi;             /* K coord */
        xs[txi][tyi] = (row < M && gx < K) ? hd_dev_bf16_to_f32(x[(size_t)row * K + gx]) : 0.0f;
        /* Load w tile [HD_GEMM_TILE x HD_GEMM_TILE] at (col, kk) of stored [N,K]. */
        int gw = blockIdx.y * HD_GEMM_TILE + txi;  /* N coord */
        int gk = kk + tyi;                          /* K coord */
        ws[txi][tyi] = (gw < N && gk < K) ? hd_dev_bf16_to_f32(w[(size_t)gw * K + gk]) : 0.0f;
        __syncthreads();

        if (row < M && col < N) {
            for (int k = 0; k < HD_GEMM_TILE; k++) {
                acc += xs[txi][k] * ws[tyi][k];
            }
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        if (bias) acc += hd_dev_bf16_to_f32(bias[col]);
        y[(size_t)row * N + col] = hd_dev_f32_to_bf16(acc);
    }
}

void hd_linear_reference(const void *x_dev, const void *w_dev,
                         const void *bias_dev, void *y_dev,
                         int M, int N, int K, int transpose_w) {
    if (!x_dev || !w_dev || !y_dev || M <= 0 || N <= 0 || K <= 0) {
        snprintf(hd_cuda_errbuf(), 512, "linear: bad args");
        return;
    }
    if (!transpose_w) {
        snprintf(hd_cuda_errbuf(), 512, "linear: transpose_w=0 unsupported");
        return;
    }
    dim3 block(HD_GEMM_TILE, HD_GEMM_TILE);
    dim3 grid((M + HD_GEMM_TILE - 1) / HD_GEMM_TILE,
              (N + HD_GEMM_TILE - 1) / HD_GEMM_TILE);
    hd_gemm_wT_kernel<<<grid, block>>>(
        (const uint16_t *)x_dev, (const uint16_t *)w_dev,
        (const uint16_t *)bias_dev, (uint16_t *)y_dev, M, N, K);
}

/* ------------------------------------------------------------------ */
/* Persistent cuBLAS/cuBLASLt runtime                                  */
/* ------------------------------------------------------------------ */

struct hd_gemm_runtime {
    cublasHandle_t cublas;
    cublasLtHandle_t lt;
    void *workspace;
    size_t workspace_bytes;
    int device_id;
};

/* Active runtime (NULL = reference backend). Set by hd_gemm_runtime_init. */
static hd_gemm_runtime *g_rt = NULL;
/* Backend selection: 1 = production (cuBLAS/cuBLASLt), 0 = reference. */
static int g_use_production = 1;

hd_gemm_runtime *hd_gemm_runtime_init(int device_id, size_t workspace_bytes) {
    hd_gemm_runtime *rt = (hd_gemm_runtime *)calloc(1, sizeof(hd_gemm_runtime));
    if (!rt) { snprintf(hd_cuda_errbuf(), 512, "gemm rt oom"); return NULL; }
    rt->device_id = device_id;

    cudaError_t ce = cudaSetDevice(device_id);
    if (ce != cudaSuccess) {
        snprintf(hd_cuda_errbuf(), 512, "gemm rt cudaSetDevice: %s", cudaGetErrorString(ce));
        free(rt);
        return NULL;
    }
    cublasStatus_t cs = cublasCreate(&rt->cublas);
    if (cs != CUBLAS_STATUS_SUCCESS) {
        snprintf(hd_cuda_errbuf(), 512, "gemm rt cublasCreate failed");
        free(rt);
        return NULL;
    }
    cublasStatus_t ls = cublasLtCreate(&rt->lt);
    if (ls != CUBLAS_STATUS_SUCCESS) {
        snprintf(hd_cuda_errbuf(), 512, "gemm rt cublasLtCreate failed");
        cublasDestroy(rt->cublas);
        free(rt);
        return NULL;
    }
    if (workspace_bytes > 0) {
        ce = cudaMalloc(&rt->workspace, workspace_bytes);
        if (ce != cudaSuccess) {
            snprintf(hd_cuda_errbuf(), 512, "gemm rt workspace cudaMalloc: %s",
                     cudaGetErrorString(ce));
            cublasLtDestroy(rt->lt);
            cublasDestroy(rt->cublas);
            free(rt);
            return NULL;
        }
        rt->workspace_bytes = workspace_bytes;
    }
    g_rt = rt;
    return rt;
}

void hd_gemm_runtime_destroy(hd_gemm_runtime *rt) {
    if (!rt) return;
    if (g_rt == rt) g_rt = NULL;
    if (rt->workspace) cudaFree(rt->workspace);
    if (rt->lt) cublasLtDestroy(rt->lt);
    if (rt->cublas) cublasDestroy(rt->cublas);
    free(rt);
}

void *hd_gemm_workspace(hd_gemm_runtime *rt) { return rt ? rt->workspace : NULL; }
size_t hd_gemm_workspace_bytes(hd_gemm_runtime *rt) { return rt ? rt->workspace_bytes : 0; }

void hd_gemm_set_backend(int use_production) { g_use_production = use_production; }

/* ------------------------------------------------------------------ */
/* cuBLAS GEMM (BF16 in/out, FP32 accumulate)                          */
/* ------------------------------------------------------------------ */

__global__ void hd_gemm_bias_add_kernel(const uint16_t *__restrict__ bias,
                                        uint16_t *__restrict__ y,
                                        int M, int N) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)M * N;
    if (idx >= total) return;
    int n = (int)(idx % N);
    float b = hd_dev_bf16_to_f32(bias[n]);
    float v = hd_dev_bf16_to_f32(y[idx]);
    y[idx] = hd_dev_f32_to_bf16(v + b);
}

static void hd_gemm_cublas(const void *x_dev, const void *w_dev,
                           const void *bias_dev, void *y_dev,
                           int M, int N, int K) {
    /* y[M,N] = x[M,K] . w[N,K]^T, all row-major.
     *
     * In cuBLAS column-major terms the same memory is:
     *   W [N,K] row-major  == W^T [K,N] col-major, ld=K
     *   X [M,K] row-major  == X^T [K,M] col-major, ld=K
     *   Y [M,N] row-major  == Y^T [N,M] col-major, ld=N
     * and Y[m,n] = sum_k X[m,k]*W[n,k]  =>  Y^T[n,m] = sum_k W^T[k,n]*X^T[k,m]
     *   = sum_k A[k,n]*B[k,m] with A=W^T [K,N], B=X^T [K,M]
     *   => C[N,M] = op(A)[N,K] . op(B)[K,M] with op(A)=T (A=W^T), op(B)=T (B=X^T). */
    const __nv_bfloat16 *A = (const __nv_bfloat16 *)w_dev; /* [N,K] ld=K */
    const __nv_bfloat16 *B = (const __nv_bfloat16 *)x_dev; /* [M,K] ld=K */
    __nv_bfloat16 *C = (__nv_bfloat16 *)y_dev;             /* [M,N] result */
    float alpha = 1.0f;
    float beta = 0.0f;

    cublasStatus_t cs = cublasGemmEx(g_rt->cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                     N, M, K,
                                     &alpha, A, CUDA_R_16BF, K,
                                             B, CUDA_R_16BF, K,
                                     &beta,  C, CUDA_R_16BF, N,
                                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (cs != CUBLAS_STATUS_SUCCESS) {
        snprintf(hd_cuda_errbuf(), 512, "cublasGemmEx failed: %d", (int)cs);
        return;
    }
    if (bias_dev) {
        /* Bias add: y[M,N] += bias[N] broadcast over rows. */
        hd_gemm_bias_add_kernel<<<(M * N + 255) / 256, 256>>>(
            (const uint16_t *)bias_dev, (uint16_t *)y_dev, M, N);
    }
}

/* ------------------------------------------------------------------ */
/* Production linear dispatch                                          */
/* ------------------------------------------------------------------ */

void hd_linear(const void *x_dev, const void *w_dev, const void *bias_dev,
               void *y_dev, int M, int N, int K, int transpose_w) {
    if (!x_dev || !w_dev || !y_dev || M <= 0 || N <= 0 || K <= 0) {
        snprintf(hd_cuda_errbuf(), 512, "linear: bad args");
        return;
    }
    if (!transpose_w) {
        snprintf(hd_cuda_errbuf(), 512, "linear: transpose_w=0 unsupported");
        return;
    }
    if (g_use_production && g_rt && g_rt->cublas) {
        hd_gemm_cublas(x_dev, w_dev, bias_dev, y_dev, M, N, K);
        return;
    }
    hd_linear_reference(x_dev, w_dev, bias_dev, y_dev, M, N, K, transpose_w);
}

/* ------------------------------------------------------------------ */
/* Timestep sinusoidal embedding (class B)                             */
/* ------------------------------------------------------------------ */

__global__ void hd_timestep_embed_kernel(const float *__restrict__ t,
                                         float *__restrict__ y,
                                         int N, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * dim) return;
    int n = idx / dim;
    int d = idx % dim;
    int half = dim / 2;
    int p = d < half ? d : d - half;
    float freq = expf(-logf(10000.0f) * (float)p / (float)half);
    float arg = t[n] * freq;
    float v = (d < half) ? cosf(arg) : sinf(arg);
    if (dim % 2 && d == dim - 1) v = 0.0f;
    y[idx] = v;
}

void hd_timestep_embed(const float *t_dev, float *y_dev, int N, int dim) {
    if (!t_dev || !y_dev || N <= 0 || dim <= 0) {
        snprintf(hd_cuda_errbuf(), 512, "timestep_embed: bad args");
        return;
    }
    hd_timestep_embed_kernel<<<(N * dim + 255) / 256, 256>>>(t_dev, y_dev, N, dim);
}
