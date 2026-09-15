/*
 * M1.5 scheduler kernels (reference implementation).
 *
 * Reproduce the oracle FlashFlowMatchEulerDiscreteScheduler.step() and the
 * pipeline v_cond/model_output arithmetic in fp32, matching torch's exact
 * operation order (left-to-right) so the frozen-noise denoising chain is
 * bit-reproducible from the golden fixture (contract section 7).
 *
 *   denoised[i] = z[i] - model_output[i] * sigma
 *   z_next[i]   = (sigma_next * noise[i]) * s_noise
 *                 + (1.0f - sigma_next) * denoised[i]
 *   model_output[i] = (z[i] - x_pred_masked[i]) / sigma
 *
 * All arithmetic fp32; the final z_next cast to bf16 reuses
 * hd_f32_convert_bf16 (round-to-nearest-even, same as torch .to(bf16)).
 */

#include "cuda_internal.h"

#include <stdio.h>

/* z_f32[i] = bf16_to_f32(z_bf16[i])  (upcast of the current sample). */
__global__ void hd_bf16_upcast_kernel(const uint16_t *__restrict__ in,
                                      float *__restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = hd_dev_bf16_to_f32(in[i]);
}

/* denoised[i] = z[i] - model_output[i] * sigma  (fp32, torch order). */
__global__ void hd_denoised_kernel(const float *__restrict__ z,
                                   const float *__restrict__ mo,
                                   float sigma, float *__restrict__ denoised,
                                   int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) denoised[i] = z[i] - mo[i] * sigma;
}

/* z_next[i] = (sigma_next*noise[i])*s_noise + (1.0f-sigma_next)*denoised[i]
 * fp32, torch left-to-right: sigma_next * noise * s_noise. */
__global__ void hd_z_next_kernel(const float *__restrict__ noise,
                                 const float *__restrict__ denoised,
                                 float sigma_next, float s_noise,
                                 float *__restrict__ z_next, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        z_next[i] = (sigma_next * noise[i]) * s_noise +
                    (1.0f - sigma_next) * denoised[i];
}

/* model_output[i] = (z[i] - xp[i]) / sigma  (fp32, fp32 upcast of bf16 inputs).
 * Folded from oracle v_cond = (x_pred_masked - z)/sigma;
 * model_output = -v_guided = (z - xp)/sigma (bit-exact negation). */
__global__ void hd_vcond_kernel(const uint16_t *__restrict__ z,
                                const uint16_t *__restrict__ xp,
                                float sigma, float *__restrict__ mo, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float zf = hd_dev_bf16_to_f32(z[i]);
        float xf = hd_dev_bf16_to_f32(xp[i]);
        mo[i] = (zf - xf) / sigma;
    }
}

void hd_sched_bf16_upcast(const void *in_dev, float *out_dev, int n) {
    if (!in_dev || !out_dev || n <= 0) {
        snprintf(hd_cuda_errbuf(), 512, "sched_bf16_upcast: bad args");
        return;
    }
    hd_bf16_upcast_kernel<<<(n + 255) / 256, 256>>>(
        (const uint16_t *)in_dev, out_dev, n);
}

void hd_sched_denoised(const float *z_dev, const float *mo_dev, float sigma,
                       float *denoised_dev, int n) {
    if (!z_dev || !mo_dev || !denoised_dev || n <= 0) {
        snprintf(hd_cuda_errbuf(), 512, "sched_denoised: bad args");
        return;
    }
    hd_denoised_kernel<<<(n + 255) / 256, 256>>>(z_dev, mo_dev, sigma,
                                                 denoised_dev, n);
}

void hd_sched_z_next(const float *noise_dev, const float *denoised_dev,
                     float sigma_next, float s_noise, float *z_next_dev, int n) {
    if (!noise_dev || !denoised_dev || !z_next_dev || n <= 0) {
        snprintf(hd_cuda_errbuf(), 512, "sched_z_next: bad args");
        return;
    }
    hd_z_next_kernel<<<(n + 255) / 256, 256>>>(noise_dev, denoised_dev,
                                               sigma_next, s_noise,
                                               z_next_dev, n);
}

void hd_sched_vcond(const void *z_dev, const void *xp_dev, float sigma,
                    float *mo_dev, int n) {
    if (!z_dev || !xp_dev || !mo_dev || n <= 0) {
        snprintf(hd_cuda_errbuf(), 512, "sched_vcond: bad args");
        return;
    }
    hd_vcond_kernel<<<(n + 255) / 256, 256>>>(
        (const uint16_t *)z_dev, (const uint16_t *)xp_dev, sigma, mo_dev, n);
}