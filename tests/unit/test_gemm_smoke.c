/*
 * Minimal cuBLAS GEMM smoke test (M2).
 *
 * Verifies the transpose mapping used by hd_gemm_cublas:
 *
 *   Y_rm[M,N] = X_rm[M,K] * W_rm[N,K]^T
 *
 * via cublasGemmEx with:
 *   transA = CUBLAS_OP_T   (A = W, [N,K] row-major)
 *   transB = CUBLAS_OP_N   (B = X, [M,K] row-major)
 *   m = N, n = M, k = K
 *   lda = K, ldb = K, ldc = N
 *   A/B/C type = CUDA_R_16BF, compute = CUBLAS_COMPUTE_32F
 *   alpha/beta = FP32
 *
 * Case 1: M=N=K=2
 *   X = [[1,2],[3,4]]  W = [[5,6],[7,8]]
 *   Y = X W^T = [[17,23],[39,53]]
 *
 * Case 2: M=4,N=3,K=5 (the shape that previously produced garbage).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

static uint16_t f2b(float f) {
    /* round-to-nearest-even bf16 conversion */
    uint32_t u;
    memcpy(&u, &f, 4);
    uint32_t r = (u + 0x7fff + ((u >> 16) & 1)) >> 16;
    return (uint16_t)r;
}

static float b2f(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

static int run_case(const char *name, int M, int N, int K,
                    const uint16_t *X, const uint16_t *W,
                    const uint16_t *Y_ref) {
    cublasHandle_t h;
    cublasStatus_t hs = cublasCreate(&h);
    if (hs != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[%s] cublasCreate failed: %d\n", name, (int)hs);
        return 1;
    }

    uint16_t *Xd, *Wd, *Yd;
    cudaError_t ce;
    ce = cudaMalloc(&Xd, (size_t)M * K * 2);
    if (ce != cudaSuccess) { fprintf(stderr, "[%s] cudaMalloc X: %s\n", name, cudaGetErrorString(ce)); return 1; }
    ce = cudaMalloc(&Wd, (size_t)N * K * 2);
    if (ce != cudaSuccess) { fprintf(stderr, "[%s] cudaMalloc W: %s\n", name, cudaGetErrorString(ce)); return 1; }
    ce = cudaMalloc(&Yd, (size_t)M * N * 2);
    if (ce != cudaSuccess) { fprintf(stderr, "[%s] cudaMalloc Y: %s\n", name, cudaGetErrorString(ce)); return 1; }

    ce = cudaMemcpy(Xd, X, (size_t)M * K * 2, cudaMemcpyHostToDevice);
    if (ce != cudaSuccess) { fprintf(stderr, "[%s] H2D X: %s\n", name, cudaGetErrorString(ce)); return 1; }
    ce = cudaMemcpy(Wd, W, (size_t)N * K * 2, cudaMemcpyHostToDevice);
    if (ce != cudaSuccess) { fprintf(stderr, "[%s] H2D W: %s\n", name, cudaGetErrorString(ce)); return 1; }

    float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t st = cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N,
                                     N, M, K,
                                     &alpha, Wd, CUDA_R_16BF, K,
                                             Xd, CUDA_R_16BF, K,
                                     &beta,  Yd, CUDA_R_16BF, N,
                                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[%s] FAIL: cublasGemmEx status %d\n", name, (int)st);
        return 1;
    }

    ce = cudaStreamSynchronize(0);
    if (ce != cudaSuccess) {
        fprintf(stderr, "[%s] FAIL: sync %s\n", name, cudaGetErrorString(ce));
        return 1;
    }

    uint16_t *Y = malloc((size_t)M * N * 2);
    ce = cudaMemcpy(Y, Yd, (size_t)M * N * 2, cudaMemcpyDeviceToHost);
    if (ce != cudaSuccess) { fprintf(stderr, "[%s] D2H Y: %s\n", name, cudaGetErrorString(ce)); return 1; }

    int bad = 0;
    for (int i = 0; i < M * N; i++) {
        float got = b2f(Y[i]), want = b2f(Y_ref[i]);
        if (fabsf(got - want) > 0.01f) {
            fprintf(stderr, "[%s] Y[%d] got %.4f want %.4f\n", name, i, got, want);
            bad++;
        }
    }
    if (bad) {
        fprintf(stderr, "[%s] FAIL: %d mismatches\n", name, bad);
        return 1;
    }
    printf("[%s] PASS\n", name);
    return 0;
}

int main(void) {
    int fails = 0;

    /* Case 1: M=N=K=2 */
    {
        const int M = 2, N = 2, K = 2;
        uint16_t X[M * K], W[N * K], Y_ref[M * N];
        float Xf[4] = { 1, 2, 3, 4 };
        float Wf[4] = { 5, 6, 7, 8 };
        for (int i = 0; i < M * K; i++) X[i] = f2b(Xf[i]);
        for (int i = 0; i < N * K; i++) W[i] = f2b(Wf[i]);
        /* Y = X W^T = [[17,23],[39,53]] */
        float Yf[4] = { 17, 23, 39, 53 };
        for (int i = 0; i < M * N; i++) Y_ref[i] = f2b(Yf[i]);
        fails += run_case("2x2", M, N, K, X, W, Y_ref);
    }

    /* Case 2: M=4,N=3,K=5 (previously garbage) */
    {
        const int M = 4, N = 3, K = 5;
        uint16_t X[M * K], W[N * K], Y_ref[M * N];
        for (int i = 0; i < M * K; i++) X[i] = f2b((float)(i + 1));
        for (int i = 0; i < N * K; i++) W[i] = f2b((float)(i + 1));
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++) {
                float acc = 0.0f;
                for (int k = 0; k < K; k++)
                    acc += b2f(X[m * K + k]) * b2f(W[n * K + k]);
                Y_ref[m * N + n] = f2b(acc);
            }
        fails += run_case("4x3x5", M, N, K, X, W, Y_ref);
    }

    printf("\n%s\n", fails ? "SMOKE FAIL" : "SMOKE PASS");
    return fails ? 1 : 0;
}