/*
 * M1.5 native scheduler implementation (see scheduler.h).
 *
 * The step() arithmetic is delegated to the reference CUDA kernels in
 * src/cuda/sched.cu (fp32, torch operation order, contract section 7).
 */

#include "scheduler.h"

#include <math.h>
#include <string.h>

#include "hd_cuda.h"

/* torch.linalg-style 2x2 solve (fp64), returns determinant; singular -> 0. */
static double hd_solve2x2(double a, double b, double c, double d,
                          double e, double f, double *x0, double *x1) {
    double det = a * d - b * c;
    if (det == 0.0) return 0.0;
    *x0 = (e * d - b * f) / det;
    *x1 = (a * f - e * c) / det;
    return det;
}

/* Frozen Dev recipe timesteps (pipeline.py DEFAULT_TIMESTEPS). */
static const int HD_DEV_DEFAULT_TIMESTEPS[28] = {
    999, 987, 974, 960, 945, 929, 913, 895, 877, 857, 836, 814, 790, 764,
    737, 707, 675, 640, 602, 560, 515, 464, 409, 347, 278, 199, 110, 8,
};

hd_status hd_scheduler_init(hd_scheduler *s, const float *sigmas,
                            int num_steps, float noise_clip_std) {
    if (!s || !sigmas || num_steps <= 0 || num_steps > HD_SCHED_MAX_STEPS) {
        hd_set_error("scheduler: bad init args");
        return HD_ERR_MISSING;
    }
    memset(s, 0, sizeof(*s));
    memcpy(s->sigmas, sigmas, (size_t)num_steps * sizeof(float));
    s->num_steps = num_steps;
    s->step_index = 0;
    s->noise_clip_std = noise_clip_std;
    return HD_OK;
}

/*
 * Production sigma derivation: sigmas = timesteps/1000 plus a trailing 0.0
 * (pipeline.py build_scheduler with explicit timesteps_list). The Dev recipe
 * uses the frozen DEFAULT_TIMESTEPS; other recipes derive from the scheduler
 * class (flash: linspace sigma_max->sigma_min then shift). For now the Dev
 * recipe is the only production path, so this fills sigmas from
 * DEFAULT_TIMESTEPS. Returns the number of sigmas written (num_steps+1).
 */
int hd_scheduler_derive_dev(hd_scheduler *s, float noise_clip_std) {
    if (!s) return 0;
    memset(s, 0, sizeof(*s));
    for (int i = 0; i < 28; i++) {
        s->sigmas[i] = (float)HD_DEV_DEFAULT_TIMESTEPS[i] / 1000.0f;
    }
    s->sigmas[28] = 0.0f;
    s->num_steps = 29;
    s->step_index = 0;
    s->noise_clip_std = noise_clip_std;
    return 29;
}

float hd_scheduler_sigma(const hd_scheduler *s) {
    if (!s || s->step_index < 0 || s->step_index >= s->num_steps) return 0.0f;
    return s->sigmas[s->step_index];
}

hd_status hd_scheduler_step(hd_scheduler *s,
                            const void *z_bf16, const float *model_output,
                            const float *noise_post_clamp, float s_noise,
                            void *z_next_bf16, int n, float *scratch) {
    if (!s || !z_bf16 || !model_output || !noise_post_clamp || !z_next_bf16 ||
        !scratch || n <= 0) {
        hd_set_error("scheduler: null argument");
        return HD_ERR_MISSING;
    }
    if (s->step_index < 0 || s->step_index >= s->num_steps - 1) {
        hd_set_error("scheduler: step_index out of range");
        return HD_ERR_MISSING;
    }
    float sigma = s->sigmas[s->step_index];
    float sigma_next = s->sigmas[s->step_index + 1];

    float *z_f32 = scratch;
    float *denoised = scratch + n;
    float *z_next_f32 = scratch + 2 * n;

    hd_sched_bf16_upcast(z_bf16, z_f32, n);
    hd_sched_denoised(z_f32, model_output, sigma, denoised, n);
    hd_sched_z_next(noise_post_clamp, denoised, sigma_next, s_noise,
                   z_next_f32, n);
    hd_f32_convert_bf16(z_next_f32, z_next_bf16, n);

    s->step_index++;
    return HD_OK;
}

/*
 * Generic timestep-space sigma derivation used by both the "flash" (Dev) and
 * "flow_match" (Dev-edit) schedulers. Matches FlashFlowMatchEulerDiscreteScheduler
 * (python/models/flash_scheduler.py lines 195-210) and diffusers'
 * FlowMatchEulerDiscreteScheduler.set_timesteps (lines 333-340, 355, 373):
 *
 *   t_i     = linspace(sigma_max*1000, sigma_min*1000, steps)   (fp64)
 *   sig_i   = t_i / 1000
 *   sig_i   = shift*sig_i / (1 + (shift-1)*sig_i)               (fp64)
 *   -> fp32; append terminal 0.0
 *
 * sigma_max=1.0, sigma_min=0.0010000000474974513 at init for shift=1.0
 * (timesteps=linspace(1,1000,1000)/1000 with shift applied).
 * `steps` must be >= 2 (the schedule is a ramp between two distinct sigmas).
 * Returns num_steps+1 sigmas written.
 */
static int hd_scheduler_derive_flash_fm(hd_scheduler *s, int steps,
                                        float shift, float noise_clip_std) {
    if (!s || steps < 2 || steps > HD_SCHED_MAX_STEPS - 1) {
        hd_set_error("scheduler: derive_flash: bad steps");
        return 0;
    }
    memset(s, 0, sizeof(*s));
    /* Init: sigmas = linspace(1,1000,1000)[::-1]/1000, then shift applied.
       sigma_max stays 1.0 under the shift; sigma_min becomes the shifted
       value of 1/1000. set_timesteps then ramps sigma_max->sigma_min in
       timestep space and applies the shift a second time. */
    const double sh = (double)shift;
    const double sigma_max = 1.0;
    const double raw_min = 0.0010000000474974513; /* fp32 of 1/1000 */
    const double sigma_min = sh * raw_min / (1.0 + (sh - 1.0) * raw_min);

    double lo = sigma_max * 1000.0;
    double hi = sigma_min * 1000.0;
    double d = (hi - lo) / (double)(steps - 1);

    for (int i = 0; i < steps; i++) {
        double t = lo + d * (double)i;
        double sig = t / 1000.0;
        double shifted = sh * sig / (1.0 + (sh - 1.0) * sig);
        s->sigmas[i] = (float)shifted;
    }
    s->sigmas[steps] = 0.0f;
    s->num_steps = steps + 1;
    s->step_index = 0;
    s->noise_clip_std = noise_clip_std;
    return steps + 1;
}

int hd_scheduler_derive_flash(hd_scheduler *s, int steps, float shift,
                              float noise_clip_std) {
    return hd_scheduler_derive_flash_fm(s, steps, shift, noise_clip_std);
}

int hd_scheduler_derive_flow_match(hd_scheduler *s, int steps, float shift,
                                   float noise_clip_std) {
    return hd_scheduler_derive_flash_fm(s, steps, shift, noise_clip_std);
}

/*
 * "default" (Full/Base) sigma derivation matching
 * FlowUniPCMultistepScheduler.set_timesteps (python/models/fm_solvers_unipc.py
 * lines 166-215). Unlike flash/flow_match the ramp is linear in *sigma* space:
 *
 *   sig_i   = linspace(sigma_max, sigma_min, steps+1)[:-1]     (fp64)
 *   sig_i   = shift*sig_i / (1 + (shift-1)*sig_i)               (fp64)
 *   -> fp32; append terminal 0.0
 *
 * sigma_max=0.9996663928031921, sigma_min=0.0 at init for shift=3.0
 * (alphas=linspace(1,1/1000,1000)[::-1], sigmas=1-alphas cast to fp32, then
 * shift applied). Returns num_steps+1 sigmas written.
 */
int hd_scheduler_derive_default(hd_scheduler *s, int steps, float shift,
                                float noise_clip_std) {
    if (!s || steps < 1 || steps > HD_SCHED_MAX_STEPS - 1) {
        hd_set_error("scheduler: derive_default: bad steps");
        return 0;
    }
    memset(s, 0, sizeof(*s));
    const double sigma_max = 0.9996664524078369;
    const double sigma_min = 0.0;
    const double sh = (double)shift;

    double d = (sigma_min - sigma_max) / (double)steps;
    for (int i = 0; i < steps; i++) {
        double sig = sigma_max + d * (double)i;
        double shifted = sh * sig / (1.0 + (sh - 1.0) * sig);
        s->sigmas[i] = (float)shifted;
    }
    s->sigmas[steps] = 0.0f;
    s->num_steps = steps + 1;
    s->step_index = 0;
    s->noise_clip_std = noise_clip_std;
    return steps + 1;
}

float hd_scheduler_noise_scale(const hd_scheduler *s, int step_index,
                               float noise_scale_start, float noise_scale_end) {
    (void)s;
    if (step_index < 0) step_index = 0;
    /* num_steps here = number of inference steps = s->num_steps - 1. */
    int num_steps = (s && s->num_steps > 1) ? s->num_steps - 1 : 1;
    if (num_steps <= 1) return noise_scale_start;
    if (step_index >= num_steps) step_index = num_steps - 1;
    return noise_scale_start +
           (noise_scale_end - noise_scale_start) * (float)step_index /
               (float)(num_steps - 1);
}

/* ------------------------------------------------------------------ */
/* FlowUniPC multistep solver (Full/Base default path)                */
/* ------------------------------------------------------------------ */

/* lambda(sigma) = log(1 - sigma) - log(sigma), fp64 (oracle torch.log). */
static double hd_unipc_lambda(double sigma) {
    return log(1.0 - sigma) - log(sigma);
}

/*
 * UniP predictor (fm_solvers_unipc.py multistep_uni_p_bh_update, predict_x0,
 * bh2). sigma_t=sigmas[step+1], sigma_s0=sigmas[step]. Applies the converted
 * model_output history (ring) stored in u->model_outputs; the raw (unconverted)
 * model_output is ignored because predict_x0 uses only converted outputs.
 * On entry sample is the post-corrector sample and m0 = model_outputs[ORDER-1].
 * Writes prev_sample.
 */
static void hd_unipc_predict(const hd_scheduler *s, hd_scheduler_unipc *u,
                             const float *sample, int n, int order,
                             float *prev_sample, float *tmp) {
    int si = u->step_index;
    double sigma_t = (double)s->sigmas[si + 1];
    double sigma_s0 = (double)s->sigmas[si];
    double alpha_t = 1.0 - sigma_t;
    double alpha_s0 = 1.0 - sigma_s0;

    double lambda_t = hd_unipc_lambda(sigma_t);
    double lambda_s0 = hd_unipc_lambda(sigma_s0);
    double h = lambda_t - lambda_s0;

    double hh = -h; /* predict_x0 */
    double h_phi_1 = expm1(hh);
    double h_phi_k = h_phi_1 / hh - 1.0;
    double B_h = expm1(hh); /* bh2 */

    const float *m0 = u->model_outputs[HD_UNIPC_ORDER - 1];
    (void)alpha_s0;

    /* rks / D1s for i in 1..order-1 */
    double rks[HD_UNIPC_ORDER];
    double rk0 = 1.0;
    int k = 0;
    for (int i = 1; i < order; i++) {
        int ssi = si - i;
        double sigma_si = (double)s->sigmas[ssi];
        double lambda_si = hd_unipc_lambda(sigma_si);
        double rk = (lambda_si - lambda_s0) / h;
        rks[k++] = rk;
    }
    rks[k++] = 1.0;

    /* rhos_p: for order 2 -> [0.5]; otherwise solve R[:-1,:-1] b[:-1].
       Only order 1 and 2 occur (solver_order=2). */
    double rhos_p = 0.5; /* only used when order==2 and k>=1 */

    /* x_t_ = sigma_t/sigma_s0 * x - alpha_t * h_phi_1 * m0 */
    double scale = sigma_t / sigma_s0;
    double coef = alpha_t * h_phi_1;
    for (int j = 0; j < n; j++) {
        prev_sample[j] = (float)(scale * (double)sample[j] - coef * (double)m0[j]);
    }
    if (order == 2) {
        /* pred_res = rhos_p * D1s[0]; D1s[0] = (m_{step-1} - m0)/rks[0] */
        const float *m1 = u->model_outputs[HD_UNIPC_ORDER - 2];
        double inv_rk = 1.0 / rks[0];
        double Bc = alpha_t * B_h * rhos_p;
        for (int j = 0; j < n; j++) {
            double D1 = ((double)m1[j] - (double)m0[j]) * inv_rk;
            prev_sample[j] = (float)((double)prev_sample[j] - Bc * D1);
        }
    }
    (void)tmp;
}

/*
 * UniC corrector (fm_solvers_unipc.py multistep_uni_c_bh_update, predict_x0,
 * bh2). sigma_t=sigmas[step], sigma_s0=sigmas[step-1]. this_model_output is
 * the converted output; last_sample is pre-predictor; this_sample is
 * post-predictor. Writes corrected sample into sample (in place).
 */
static void hd_unipc_correct(const hd_scheduler *s, hd_scheduler_unipc *u,
                             const float *this_model_output,
                             const float *last_sample, float *this_sample,
                             int n, int order, float *tmp) {
    int si = u->step_index;
    double sigma_t = (double)s->sigmas[si];
    double sigma_s0 = (double)s->sigmas[si - 1];
    double alpha_t = 1.0 - sigma_t;
    double alpha_s0 = 1.0 - sigma_s0;

    double lambda_t = hd_unipc_lambda(sigma_t);
    double lambda_s0 = hd_unipc_lambda(sigma_s0);
    double h = lambda_t - lambda_s0;

    double hh = -h;
    double h_phi_1 = expm1(hh);
    double h_phi_k = h_phi_1 / hh - 1.0;
    double B_h = expm1(hh);

    const float *m0 = u->model_outputs[HD_UNIPC_ORDER - 1];
    (void)alpha_s0;

    /* D1s for i in 1..order-1 using si - (i+1) */
    double rks[HD_UNIPC_ORDER];
    int k = 0;
    for (int i = 1; i < order; i++) {
        int ssi = si - (i + 1);
        double sigma_si = (double)s->sigmas[ssi];
        double lambda_si = hd_unipc_lambda(sigma_si);
        double rk = (lambda_si - lambda_s0) / h;
        rks[k++] = rk;
    }
    rks[k++] = 1.0;

    /* rhos_c: order 1 -> [0.5]; order 2 -> solve 2x2 (R, b) */
    double rhos_c[2] = {0.5, 0.0};
    int rhos_n = 1;
    if (order == 2) {
        double factorial_i = 1.0;
        double b0, b1;
        /* R = [rks^(i-1) for i in 1..order+1], rks=[rk0,1.0] for order 2:
             i=1 -> [1,1]; i=2 -> [rk0, 1.0].
           b = [h_phi_k*factorial_i/B_h] iterated. */
        double R00 = 1.0, R01 = 1.0;               /* i=1: rks^0 */
        double R10 = rks[0], R11 = rks[1];         /* i=2: rks^1 (rks[1]==1.0) */
        (void)factorial_i;
        /* h_phi_k iterated: i=1 h_phi_k0=h_phi_1/hh-1, then update */
        double hpk = h_phi_1 / hh - 1.0;          /* i=1 */
        b0 = hpk * 1.0 / B_h;                      /* factorial_i=1 */
        hpk = hpk / hh - 1.0 / 2.0;                /* i=2, factorial=2! */
        b1 = hpk * 2.0 / B_h;                      /* factorial_i=2 */
        double x0, x1;
        if (hd_solve2x2(R00, R01, R10, R11, b0, b1, &x0, &x1) != 0.0) {
            rhos_c[0] = x0;
            rhos_c[1] = x1;
            rhos_n = 2;
        } else {
            rhos_n = 1;
        }
    }

    /* x_t_ = sigma_t/sigma_s0*last_sample - alpha_t*h_phi_1*m0 */
    double scale = sigma_t / sigma_s0;
    double coef = alpha_t * h_phi_1;
    for (int j = 0; j < n; j++) {
        this_sample[j] =
            (float)(scale * (double)last_sample[j] - coef * (double)m0[j]);
    }

    /* D1_t = this_model_output - m0 */
    if (rhos_n == 2) {
        /* corr_res = rhos_c[0]*D1s[0]; D1s[0] = (m_{si-2} - m0)/rks[0] */
        const float *m2 = u->model_outputs[HD_UNIPC_ORDER - 2];
        double inv_rk = 1.0 / rks[0];
        double Bc = alpha_t * B_h;
        for (int j = 0; j < n; j++) {
            double D1 = ((double)m2[j] - (double)m0[j]) * inv_rk;
            double D1_t = (double)this_model_output[j] - (double)m0[j];
            double corr = rhos_c[0] * D1 + rhos_c[1] * D1_t;
            this_sample[j] = (float)((double)this_sample[j] - Bc * corr);
        }
    } else {
        /* order 1: rhos_c = [0.5], no D1s -> corr_res=0, D1_t term only */
        double Bc = alpha_t * B_h * 0.5;
        for (int j = 0; j < n; j++) {
            double D1_t = (double)this_model_output[j] - (double)m0[j];
            this_sample[j] = (float)((double)this_sample[j] - Bc * D1_t);
        }
    }
}

hd_status hd_scheduler_unipc_step(const hd_scheduler *s, hd_scheduler_unipc *u,
                                  const float *model_output, const float *sample,
                                  float *prev_sample, int n, float *scratch) {
    if (!s || !u || !model_output || !sample || !prev_sample || !scratch ||
        n <= 0) {
        hd_set_error("scheduler: unipc: null argument");
        return HD_ERR_MISSING;
    }
    if (u->step_index < 0 || u->step_index >= s->num_steps - 1) {
        hd_set_error("scheduler: unipc: step_index out of range");
        return HD_ERR_MISSING;
    }
    if (!u->model_outputs[HD_UNIPC_ORDER - 1] || !u->last_sample) {
        hd_set_error("scheduler: unipc: history not initialized");
        return HD_ERR_MISSING;
    }

    int si = u->step_index;

    /* convert_model_output: x0_pred = sample - sigma_t * model_output.
       Store into the (temporary) current slot used as m0 source. We write the
       converted output into model_outputs[ORDER-1]'s buffer AFTER computing the
       current conversion so the corrector sees the *previous* history. */
    float sigma_cur = s->sigmas[si];
    float *conv = scratch; /* current converted model output (n) */
    for (int j = 0; j < n; j++) {
        conv[j] = sample[j] - sigma_cur * model_output[j];
    }

    /* use_corrector = step_index>0 && last_sample is not None */
    int use_corrector =
        (si > 0) && u->has_last_sample && (u->this_order > 0);

    /* Working sample copy (post-corrector = this_sample). We compute into
       prev_sample's buffer for the corrector input, then predictor output. */
    float *this_sample = prev_sample; /* scratch reuse: post-corr sample */
    for (int j = 0; j < n; j++) this_sample[j] = sample[j];

    if (use_corrector) {
        int order = u->this_order;
        hd_unipc_correct(s, u, conv, u->last_sample, this_sample, n, order,
                         scratch);
    }

    /* history shift: move converted output into the ring (newest at end) */
    float *sink = u->model_outputs[HD_UNIPC_ORDER - 1];
    for (int i = 0; i < HD_UNIPC_ORDER - 1; i++) {
        /* shift ring: model_outputs[i] = model_outputs[i+1] by copying
           contents (buffers are persistent pointers). */
        float *dst = u->model_outputs[i];
        float *src = u->model_outputs[i + 1];
        for (int j = 0; j < n; j++) dst[j] = src[j];
    }
    for (int j = 0; j < n; j++) sink[j] = conv[j];

    /* this_order: min(solver_order, len(timesteps)-step_index) then warmup */
    int to = HD_UNIPC_ORDER;
    int len = s->num_steps - 1; /* number of inference steps */
    if (len - si < to) to = len - si;
    int this_order = to;
    if (this_order > u->lower_order_nums + 1)
        this_order = u->lower_order_nums + 1;
    u->this_order = this_order;

    /* last_sample = this_sample (post-corrector) before predictor */
    if (u->has_last_sample) {
        float *ls = u->last_sample;
        for (int j = 0; j < n; j++) ls[j] = this_sample[j];
    } else {
        for (int j = 0; j < n; j++) u->last_sample[j] = this_sample[j];
        u->has_last_sample = 1;
    }

    /* predictor: needs m0 = model_outputs[-1] (now the new converted) and
       m_{step-1} = model_outputs[-2]. But the predictor must use m0 = the
       converted output *of the current step* which we just stored as sink. */
    hd_unipc_predict(s, u, this_sample, n, this_order, prev_sample, scratch);

    if (u->lower_order_nums < HD_UNIPC_ORDER) u->lower_order_nums++;

    u->step_index++;
    return HD_OK;
}