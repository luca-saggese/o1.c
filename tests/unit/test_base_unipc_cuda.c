/*
 * Base "default" (FlowUniPC) CUDA kernel validation harness.
 *
 * Drives the production UniPC CUDA kernels (hd_unipc_convert/correct/predict,
 * hd_sched_cfg_guided) and compares them bit-for-bit / within fp32 epsilon
 * against the CPU reference (hd_scheduler_unipc_step, which is itself
 * validated against the frozen Python oracle in scheduler_matrix.c). This
 * isolates the scheduler arithmetic from the transformer forward, so any
 * divergence is attributed to the CUDA port, never to network drift.
 *
 * The harness:
 *   1. derives the canonical base schedule (shift=3.0, 50 steps)
 *   2. drives N steps alternating a synthetic model_output through BOTH the
 *      CPU reference and the CUDA production path
 *   3. requires the fp32 lattices to match within 1e-5 (they are the same
 *      fp64->fp32 scalar math; elementwise fp32 is expected near-exact).
 */

#include "hidream.h"
#include "scheduler.h"
#include "hd_cuda.h"

#include <cuda_runtime.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 1024
#define STEPS 5
#define SHIFT 3.0f
#define NSTEPS 50

static int failures = 0;
static int passes = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("FAIL: %s\n", msg); failures++; } \
    else { printf("ok: %s\n", msg); passes++; } \
} while (0)

static void *dev_alloc(size_t bytes) {
    void *p = NULL;
    if (cudaMalloc(&p, bytes) != cudaSuccess) return NULL;
    return p;
}
static void dev_free(void *p) { if (p) cudaFree(p); }

static void copy_dev_to_host(const void *dev, float *host, int n) {
    cudaMemcpy(host, dev, (size_t)n * 4, cudaMemcpyDeviceToHost);
}

int main(void) {
    /* ---- schedule ---- */
    hd_scheduler s;
    int n_sig = hd_scheduler_derive_default(&s, NSTEPS, SHIFT, 8.0f);
    CHECK(n_sig == NSTEPS + 1, "derive_default 50+1 sigmas");

    /* synthetic sample (host) */
    float z[N], mo[N];
    srand(7);
    for (int i = 0; i < N; i++) { z[i] = (rand() / (float)RAND_MAX - 0.5f) * 2.f; }
    for (int i = 0; i < N; i++) { mo[i] = (rand() / (float)RAND_MAX - 0.5f) * 0.6f; }

    /* ---- CUDA buffers ---- */
    void *zdev = dev_alloc((size_t)N * 4);
    void *modev = dev_alloc((size_t)N * 4);
    void *cur = dev_alloc((size_t)N * 4);
    void *corr_out = dev_alloc((size_t)N * 4);
    void *prev_out = dev_alloc((size_t)N * 4);
    void *hist = dev_alloc(3 * (size_t)N * 4);
    void *last = dev_alloc((size_t)N * 4);
    void *cfg0 = dev_alloc((size_t)N * 4);
    void *cfg1 = dev_alloc((size_t)N * 4);
    void *cfgout = dev_alloc((size_t)N * 4);
    if (!zdev || !modev || !cur || !corr_out || !prev_out || !hist || !last ||
        !cfg0 || !cfg1 || !cfgout) {
        printf("OOM\n");
        return 1;
    }
    cudaMemcpy(zdev, z, (size_t)N * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(modev, mo, (size_t)N * 4, cudaMemcpyHostToDevice);

    /* ---- CPU reference state ---- */
    float hist0[N], hist1[N], lasts[N], cscratch[6 * N];
    float sample[N], prev[N];
    memcpy(sample, z, sizeof(z));
    hd_scheduler_unipc u;
    memset(&u, 0, sizeof(u));
    u.model_outputs[0] = hist0;
    u.model_outputs[1] = hist1;
    u.last_sample = lasts;

    /* ---- CUDA persistent state ---- */
    float *slot0 = (float *)hist;
    float *slot1 = (float *)hist + N;
    float *slot2 = (float *)hist + 2 * N;
    float *unipc_cur = (float *)cur;
    float *unipc_corr = (float *)corr_out;
    float *unipc_prev = (float *)prev_out;
    int lower = 0, this_order = 0, have_hist = 0;
    float ref_out[N];

    float max_dev = 0.0f;
    for (int i = 0; i < STEPS; i++) {
        float sigma_cur = s.sigmas[i];

        /* ---- CPU reference ---- */
        hd_scheduler_unipc_step(&s, &u, mo, sample, prev, N, cscratch);
        memcpy(sample, prev, sizeof(prev));

        /* ---- CUDA path ---- */
        hd_unipc_plan plan;
        hd_scheduler_unipc_plan(&s, i, this_order, lower, &plan);

        hd_unipc_convert(zdev, modev, sigma_cur, unipc_cur, N);
        cudaMemcpy(slot0, unipc_cur, (size_t)N * 4, cudaMemcpyDeviceToDevice);
        if (i > 0 && this_order > 0 && have_hist) {
            hd_unipc_correct(last, slot1, slot2, unipc_cur,
                             (float)plan.c_sig_t, (float)plan.c_sig_s0,
                             (float)plan.c_alpha_t,
                             (float)plan.c_h_phi_1, (float)plan.c_B_h,
                             (float)plan.c_rhos0, (float)plan.c_rhos1,
                             (float)plan.c_inv_rks0, plan.corr_order,
                             unipc_corr, N);
        } else {
            cudaMemcpy(unipc_corr, zdev, (size_t)N * 4, cudaMemcpyDeviceToDevice);
        }
        cudaMemcpy(last, unipc_corr, (size_t)N * 4, cudaMemcpyDeviceToDevice);
        hd_unipc_predict(unipc_corr, slot0, slot1, (float)plan.p_sig_t,
                         (float)plan.p_sig_s0, (float)plan.p_alpha_t,
                         (float)plan.p_h_phi_1, (float)plan.p_B_h,
                         (float)plan.p_rhos_p, (float)plan.p_inv_rks0,
                         plan.pred_order, unipc_prev, N);
        cudaMemcpy(slot2, slot1, (size_t)N * 4, cudaMemcpyDeviceToDevice);
        cudaMemcpy(slot1, slot0, (size_t)N * 4, cudaMemcpyDeviceToDevice);
        this_order = plan.next_this_order;
        lower = plan.next_lower_order;
        have_hist = 1;

        /* z for next step = predictor output cast to bf16 then fp32 (mirror) */
        copy_dev_to_host(unipc_prev, ref_out, N);
        /* the CUDA scalar math (fp64->fp32) is the same as the CPU reference;
           the only difference is the fp32 accumulation order inside each
           kernel, which for these pure pointwise ops is identical. */
        for (int k = 0; k < N; k++) {
            float d = fabsf(ref_out[k] - prev[k]);
            if (d > max_dev) max_dev = d;
        }
        /* feed next step: production uses z_next as next z_prev */
        cudaMemcpy(zdev, unipc_prev, (size_t)N * 4, cudaMemcpyDeviceToDevice);
    }
    printf("CUDA vs CPU UniPC 5-step max abs diff = %.3e (target <= 1e-5)\n",
           max_dev);
    CHECK(max_dev <= 1e-5f, "CUDA UniPC matches CPU reference (<= 1e-5)");

    /* ---- CFG combine sanity ---- */
    float u0[N], u1[N];
    for (int i = 0; i < N; i++) { u0[i] = 0.3f; u1[i] = -0.1f; }
    cudaMemcpy(cfg0, u0, (size_t)N * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(cfg1, u1, (size_t)N * 4, cudaMemcpyHostToDevice);
    hd_sched_cfg_guided((float *)cfg0, (float *)cfg1, 5.0f, (float *)cfgout, N);
    float got[N];
    copy_dev_to_host(cfgout, got, N);
    float expect = (1.0f - 5.0f) * (-0.1f) + 5.0f * 0.3f;
    CHECK(fabsf(got[0] - expect) < 1e-5f &&
          fabsf(got[N - 1] - expect) < 1e-5f, "cfg_guided combine correct");

    /* ---- FlowMatch Euler sanity ---- */
    for (int i = 0; i < N; i++) {
        u0[i] = 0.25f;
        u1[i] = -0.5f;
    }
    cudaMemcpy(cfg0, u0, (size_t)N * 4, cudaMemcpyHostToDevice);
    hd_f32_convert_bf16((float *)cfg0, cfg1, N);
    cudaMemcpy(cfg0, u1, (size_t)N * 4, cudaMemcpyHostToDevice);
    hd_sched_flow_match_step(cfg1, (float *)cfg0, 0.9f, 0.7f, cfgout, N);
    hd_sched_bf16_upcast(cfgout, (float *)cur, N);
    copy_dev_to_host(cur, got, N);
    expect = 0.25f + (0.7f - 0.9f) * -0.5f;
    CHECK(fabsf(got[0] - expect) < 2e-3f &&
          fabsf(got[N - 1] - expect) < 2e-3f,
          "flow_match Euler step correct");

    dev_free(zdev); dev_free(modev); dev_free(cur); dev_free(corr_out);
    dev_free(prev_out); dev_free(hist); dev_free(last);
    dev_free(cfg0); dev_free(cfg1); dev_free(cfgout);

    printf("RESULT: %s (%d pass, %d fail)\n",
           failures == 0 ? "PASS" : "FAIL", passes, failures);
    return failures == 0 ? 0 : 1;
}