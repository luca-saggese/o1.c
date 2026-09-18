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

/* ------------------------------------------------------------------ */
/* CFG guidance combine (M1-post base path)                            */
/* ------------------------------------------------------------------ */

/*
 * model_output = -v_guided = (1-g)*mo_uncond + g*mo_cond   (fp32).
 * The two mo arrays are the folded vcond outputs (mo = -v): mo_cond = -v_cond,
 * mo_uncond = -v_uncond; oracle v_guided = v_uncond + g*(v_cond-v_uncond), so
 * model_output = -v_guided = (1-g)*mo_uncond + g*mo_cond. Matches upstream
 * `model_output = -v_guided` (pipeline.py). fp32, torch left-to-right.
 */
__global__ void hd_cfg_guided_kernel(const float *__restrict__ mo_cond,
                                     const float *__restrict__ mo_uncond,
                                     float g, float *__restrict__ mo_guided,
                                     int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        mo_guided[i] = (1.0f - g) * mo_uncond[i] + g * mo_cond[i];
}

void hd_sched_cfg_guided(const float *mo_cond, const float *mo_uncond, float g,
                         float *mo_guided, int n) {
    if (!mo_cond || !mo_uncond || !mo_guided || n <= 0) {
        snprintf(hd_cuda_errbuf(), 512, "sched_cfg_guided: bad args");
        return;
    }
    hd_cfg_guided_kernel<<<(n + 255) / 256, 256>>>(mo_cond, mo_uncond, g,
                                                   mo_guided, n);
}

/* ------------------------------------------------------------------ */
/* FlowUniPC multistep (base "default" path)                           */
/* ------------------------------------------------------------------ */

/* bf16->f32 upcast aliases the flash upcast kernel; reuse hd_sched_bf16_upcast. */

/* conv[j] = sample_f32[j] - sigma_cur * model_output[j]  (convert_model_output,
 * predict_x0, flow_prediction; sigma_cur = sigmas[step_index]). fp32. */
__global__ void hd_unipc_convert_kernel(const float *__restrict__ sample,
                                        const float *__restrict__ mo,
                                        float sigma_cur,
                                        float *__restrict__ conv, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) conv[i] = sample[i] - sigma_cur * mo[i];
}

/*
 * UniC corrector (multistep_uni_c_bh_update, predict_x0, bh2, order 1 or 2).
 *
 *   x_t_   = sig_t/sig_s0 * last - alpha_t*h_phi_1*m0
 *   order1: corr = rhos_c1 * (conv - m0)                    (rhos_c=[0.5])
 *   order2: corr = rhos_c0 * (m_old - m0) * inv_rks0
 *                 + rhos_c1 * (conv - m0)
 *   out    = x_t_ - alpha_t * B_h * corr
 *
 * sigma_t=sigmas[si], sigma_s0=sigmas[si-1] (per oracle corrector).
 */
__global__ void hd_unipc_correct_kernel(
    const float *__restrict__ last, const float *__restrict__ mo0,
    const float *__restrict__ mo_old, const float *__restrict__ conv,
    float sig_t, float sig_s0, float alpha_t, float h_phi_1, float B_h,
    float rhos_c0, float rhos_c1, float inv_rks0, int order_c,
    float *__restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x_t = (sig_t / sig_s0) * last[i] - alpha_t * h_phi_1 * mo0[i];
        float D1_t = conv[i] - mo0[i];
        float corr;
        if (order_c == 2) {
            float D1s = (mo_old[i] - mo0[i]) * inv_rks0;
            corr = rhos_c0 * D1s + rhos_c1 * D1_t;
        } else {
            corr = rhos_c1 * D1_t;
        }
        out[i] = x_t - alpha_t * B_h * corr;
    }
}

/*
 * UniP predictor (multistep_uni_p_bh_update, predict_x0, bh2, order 1 or 2).
 * m0 = conv (current converted output, newest); m_old = previous converted.
 *
 *   x_t_   = sig_t/sig_s0 * sample - alpha_t*h_phi_1*m0
 *   order2: pred_res = rhos_p * (m_old - m0) * inv_rks0   (rhos_p=[0.5])
 *   order1: pred_res = 0
 *   out    = x_t_ - alpha_t * B_h * pred_res
 *
 * sigma_t=sigmas[si+1], sigma_s0=sigmas[si] (per oracle predictor).
 */
__global__ void hd_unipc_predict_kernel(
    const float *__restrict__ sample, const float *__restrict__ mo0,
    const float *__restrict__ mo_old, float sig_t, float sig_s0,
    float alpha_t, float h_phi_1, float B_h, float rhos_p, float inv_rks0,
    int order_p, float *__restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x_t = (sig_t / sig_s0) * sample[i] - alpha_t * h_phi_1 * mo0[i];
        if (order_p == 2) {
            float D1s = (mo_old[i] - mo0[i]) * inv_rks0;
            x_t -= alpha_t * B_h * (rhos_p * D1s);
        }
        out[i] = x_t;
    }
}

void hd_unipc_convert(const float *sample_dev, const float *mo_dev,
                      float sigma_cur, float *conv_dev, int n) {
    if (!sample_dev || !mo_dev || !conv_dev || n <= 0) {
        snprintf(hd_cuda_errbuf(), 512, "unipc_convert: bad args");
        return;
    }
    hd_unipc_convert_kernel<<<(n + 255) / 256, 256>>>(sample_dev, mo_dev,
                                                      sigma_cur, conv_dev, n);
}

void hd_unipc_correct(const float *last_dev, const float *mo0_dev,
                      const float *mo_old_dev, const float *conv_dev,
                      float sig_t, float sig_s0, float alpha_t, float h_phi_1,
                      float B_h, float rhos_c0, float rhos_c1, float inv_rks0,
                      int order_c, float *out_dev, int n) {
    if (!last_dev || !mo0_dev || !conv_dev || !out_dev || n <= 0) {
        snprintf(hd_cuda_errbuf(), 512, "unipc_correct: bad args");
        return;
    }
    hd_unipc_correct_kernel<<<(n + 255) / 256, 256>>>(
        last_dev, mo0_dev, mo_old_dev, conv_dev, sig_t, sig_s0, alpha_t,
        h_phi_1, B_h, rhos_c0, rhos_c1, inv_rks0, order_c, out_dev, n);
}

void hd_unipc_predict(const float *sample_dev, const float *mo0_dev,
                      const float *mo_old_dev, float sig_t, float sig_s0,
                      float alpha_t, float h_phi_1, float B_h, float rhos_p,
                      float inv_rks0, int order_p, float *out_dev, int n) {
    if (!sample_dev || !mo0_dev || !out_dev || n <= 0) {
        snprintf(hd_cuda_errbuf(), 512, "unipc_predict: bad args");
        return;
    }
    hd_unipc_predict_kernel<<<(n + 255) / 256, 256>>>(
        sample_dev, mo0_dev, mo_old_dev, sig_t, sig_s0, alpha_t, h_phi_1,
        B_h, rhos_p, inv_rks0, order_p, out_dev, n);
}