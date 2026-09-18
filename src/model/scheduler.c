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
 * Plan the host-side scalar coefficients for one UniPC step. The gamma terms
 * (expm1, log) use fp64 like torch's scalar promotion; the per-element
 * arithmetic is executed by the CUDA kernels (hd_unipc_*). Mirror of
 * fm_solvers_unipc.py step()/multistep_* with solver_order=2, predict_x0,
 * bh2, lower_order_final=True, disable_corrector=[].
 *
 *   si             current step index
 *   this_order     predictor order persisted from the last step (used by the
 *                  current step's corrector); 0 -> no corrector
 *   lower_order_nums  warmup counter persisted from the last step
 */
void hd_scheduler_unipc_plan(const hd_scheduler *s, int si, int this_order,
                             int lower_order_nums, hd_unipc_plan *plan) {
    if (!s || !plan) return;
    memset(plan, 0, sizeof(*plan));

    int n_inf = s->num_steps - 1; /* number of inference steps */
    if (si < 0 || si >= n_inf) return;

    /* ---- corrector: use_corrector = step_index>0 && last_sample not None
       (disable_corrector empty). The corrector order is the last step's
       predictor order (self.this_order), clamped to solver_order=2. ----
       Warmup: legacy step computed this_order = min(2, len-si) then took
       min with lower_order_nums+1; that *predictor* order became the next
       step's *corrector* order. Here the caller passes exactly that
       persisted value (defaults 0 -> no corrector on step 0). */
    if (si > 0 && this_order > 0) {
        int co = (this_order >= 2) ? 2 : 1;
        double sig_c_t = (double)s->sigmas[si];       /* sigmas[step_index]   */
        double sig_c_s0 = (double)s->sigmas[si - 1];  /* sigmas[step_index-1] */
        double alpha_c_t = 1.0 - sig_c_t;
        double l_t = hd_unipc_lambda(sig_c_t);
        double l_s0 = hd_unipc_lambda(sig_c_s0);
        double hc = l_t - l_s0;
        double hh = -hc;                 /* predict_x0 */
        double h_phi_1 = expm1(hh);
        double B_h = expm1(hh);          /* bh2 */

        double rk0 = 1.0, rhos_c0 = 0.0, rhos_c1 = 0.5;
        if (co == 2) {
            double sig_m = (si - 2 >= 0) ? (double)s->sigmas[si - 2]
                                         : (double)s->sigmas[0];
            double l_m = hd_unipc_lambda(sig_m);
            rk0 = (l_m - l_s0) / hc;
            /* rhos_c via 2x2 solve (mirrors legacy hd_unipc_correct). */
            double h_phi_k = h_phi_1 / hh - 1.0;
            double b0 = h_phi_k * 1.0 / B_h;
            h_phi_k = h_phi_k / hh - 1.0 / 2.0;
            double b1 = h_phi_k * 2.0 / B_h;
            double x0, x1;
            if (hd_solve2x2(1.0, 1.0, rk0, 1.0, b0, b1, &x0, &x1) != 0.0) {
                rhos_c0 = x0; rhos_c1 = x1;
            } else {
                rhos_c0 = 0.0; rhos_c1 = 0.5;
            }
        }
        plan->corr_order = co;
        plan->c_sig_t = sig_c_t; plan->c_sig_s0 = sig_c_s0;
        plan->c_alpha_t = alpha_c_t;
        plan->c_h_phi_1 = h_phi_1; plan->c_B_h = B_h;
        plan->c_rhos0 = rhos_c0; plan->c_rhos1 = rhos_c1;
        plan->c_inv_rks0 = (rk0 == 0.0) ? 0.0 : 1.0 / rk0;
    }

    /* ---- predictor ----
       lower_order_final: this_order = min(solver_order, n_inf - si); then
       min(this_order, lower_order_nums + 1)  (warmup). */
    int to = 2;
    if (n_inf - si < to) to = n_inf - si;
    int pred_order = to;
    if (pred_order > lower_order_nums + 1) pred_order = lower_order_nums + 1;
    if (pred_order < 1) pred_order = 1;

    double sig_p_t = (double)s->sigmas[si + 1];   /* sigmas[step_index+1] */
    double sig_p_s0 = (double)s->sigmas[si];      /* sigmas[step_index]   */
    double alpha_p_t = 1.0 - sig_p_t;
    double l_pt = hd_unipc_lambda(sig_p_t);
    double l_ps0 = hd_unipc_lambda(sig_p_s0);
    double hp = l_pt - l_ps0;
    double hh = -hp;                 /* predict_x0 */
    double h_phi_1 = expm1(hh);
    double B_h = expm1(hh);          /* bh2 */

    double rk0 = 1.0, rhos_p = 0.0;
    if (pred_order == 2) {
        int ssi = si - 1;
        double sig_m = (ssi >= 0) ? (double)s->sigmas[ssi] : (double)s->sigmas[0];
        double l_m = hd_unipc_lambda(sig_m);
        rk0 = (l_m - l_ps0) / hp;
        if (rk0 != 0.0) rhos_p = 0.5;  /* order-2 simplified rhos_p=[0.5] */
    }
    plan->pred_order = pred_order;
    plan->p_sig_t = sig_p_t; plan->p_sig_s0 = sig_p_s0;
    plan->p_alpha_t = alpha_p_t;
    plan->p_h_phi_1 = h_phi_1; plan->p_B_h = B_h;
    plan->p_rhos_p = rhos_p; plan->p_inv_rks0 = (rk0 == 0.0) ? 0.0 : 1.0 / rk0;

    /* ---- persist ----
       The next step's corrector uses this predictor's order; the warmup
       counter increments once up to solver_order. */
    plan->next_this_order = pred_order;
    int lower_next = lower_order_nums;
    if (lower_next < 2) lower_next++;
    plan->next_lower_order = lower_next;
}

/* ------------------------------------------------------------------ */
/* CPU-only reference FlowUniPC step (UnitPC fixture/scheduler_matrix) */
/* ------------------------------------------------------------------ */

/*
 * Reference UniPC scalar math on plain host float arrays, kept for the
 * CPU-only scheduler_matrix fixture. The static helpers are suffixed `_cpu`
 * to avoid collision with the CUDA device wrappers (hd_unipc_predict/correct)
 * used by generate.c.
 */

static double hd_unipc_lambda_cpu(double sigma) {
    return log(1.0 - sigma) - log(sigma);
}

static void hd_unipc_predict_cpu(const hd_scheduler *s, hd_scheduler_unipc *u,
                                 const float *sample, int n, int order,
                                 float *prev_sample, float *tmp) {
    int si = u->step_index;
    double sigma_t = (double)s->sigmas[si + 1];
    double sigma_s0 = (double)s->sigmas[si];
    double alpha_t = 1.0 - sigma_t;
    double lambda_t = hd_unipc_lambda_cpu(sigma_t);
    double lambda_s0 = hd_unipc_lambda_cpu(sigma_s0);
    double h = lambda_t - lambda_s0;
    double hh = -h;
    double h_phi_1 = expm1(hh);
    double B_h = expm1(hh);
    const float *m0 = u->model_outputs[HD_UNIPC_ORDER - 1];

    double rks[HD_UNIPC_ORDER];
    int k = 0;
    rks[k++] = 1.0;
    for (int i = 1; i < order; i++) {
        int ssi = si - i;
        double sigma_si = (double)s->sigmas[ssi];
        double lambda_si = hd_unipc_lambda_cpu(sigma_si);
        double rk = (lambda_si - lambda_s0) / h;
        rks[k++] = rk;
    }
    double rhos_p = 0.5;
    double scale = sigma_t / sigma_s0;
    double coef = alpha_t * h_phi_1;
    for (int j = 0; j < n; j++)
        prev_sample[j] = (float)(scale * (double)sample[j] - coef * (double)m0[j]);
    if (order == 2) {
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

static void hd_unipc_correct_cpu(const hd_scheduler *s, hd_scheduler_unipc *u,
                                 const float *this_model_output,
                                 const float *last_sample, float *this_sample,
                                 int n, int order, float *tmp) {
    int si = u->step_index;
    double sigma_t = (double)s->sigmas[si];
    double sigma_s0 = (double)s->sigmas[si - 1];
    double alpha_t = 1.0 - sigma_t;
    double lambda_t = hd_unipc_lambda_cpu(sigma_t);
    double lambda_s0 = hd_unipc_lambda_cpu(sigma_s0);
    double h = lambda_t - lambda_s0;
    double hh = -h;
    double h_phi_1 = expm1(hh);
    double B_h = expm1(hh);
    const float *m0 = u->model_outputs[HD_UNIPC_ORDER - 1];

    double rks[HD_UNIPC_ORDER];
    int k = 0;
    rks[k++] = 1.0;
    for (int i = 1; i < order; i++) {
        int ssi = si - (i + 1);
        double sigma_si = (double)s->sigmas[ssi];
        double lambda_si = hd_unipc_lambda_cpu(sigma_si);
        double rk = (lambda_si - lambda_s0) / h;
        rks[k++] = rk;
    }

    double rhos_c0 = 0.0, rhos_c1 = 0.5;
    if (order == 2) {
        double h_phi_k = h_phi_1 / hh - 1.0;
        double b0 = h_phi_k * 1.0 / B_h;
        h_phi_k = h_phi_k / hh - 1.0 / 2.0;
        double b1 = h_phi_k * 2.0 / B_h;
        double x0, x1;
        if (hd_solve2x2(1.0, 1.0, rks[0], 1.0, b0, b1, &x0, &x1) != 0.0) {
            rhos_c0 = x0; rhos_c1 = x1;
        }
    }

    double scale = sigma_t / sigma_s0;
    double coef = alpha_t * h_phi_1;
    for (int j = 0; j < n; j++)
        this_sample[j] = (float)(scale * (double)last_sample[j] - coef * (double)m0[j]);

    if (order == 2) {
        const float *m2 = u->model_outputs[HD_UNIPC_ORDER - 2];
        double inv_rk = 1.0 / rks[0];
        double Bc = alpha_t * B_h;
        for (int j = 0; j < n; j++) {
            double D1 = ((double)m2[j] - (double)m0[j]) * inv_rk;
            double D1_t = (double)this_model_output[j] - (double)m0[j];
            double corr = rhos_c0 * D1 + rhos_c1 * D1_t;
            this_sample[j] = (float)((double)this_sample[j] - Bc * corr);
        }
    } else {
        double Bc = alpha_t * B_h * 0.5;
        for (int j = 0; j < n; j++) {
            double D1_t = (double)this_model_output[j] - (double)m0[j];
            this_sample[j] = (float)((double)this_sample[j] - Bc * D1_t);
        }
    }
    (void)tmp;
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
    float sigma_cur = s->sigmas[si];
    float *conv = scratch; /* current converted model output */
    for (int j = 0; j < n; j++)
        conv[j] = sample[j] - sigma_cur * model_output[j];

    int use_corrector =
        (si > 0) && u->has_last_sample && (u->this_order > 0);

    float *this_sample = prev_sample;
    for (int j = 0; j < n; j++) this_sample[j] = sample[j];

    if (use_corrector) {
        hd_unipc_correct_cpu(s, u, conv, u->last_sample, this_sample, n,
                             u->this_order, scratch);
    }

    /* history shift (ring, newest at end) */
    float *sink = u->model_outputs[HD_UNIPC_ORDER - 1];
    for (int i = 0; i < HD_UNIPC_ORDER - 1; i++) {
        float *dst = u->model_outputs[i];
        float *src = u->model_outputs[i + 1];
        for (int j = 0; j < n; j++) dst[j] = src[j];
    }
    for (int j = 0; j < n; j++) sink[j] = conv[j];

    int to = HD_UNIPC_ORDER;
    int len = s->num_steps - 1;
    if (len - si < to) to = len - si;
    int this_order = to;
    if (this_order > u->lower_order_nums + 1)
        this_order = u->lower_order_nums + 1;
    u->this_order = this_order;

    if (u->has_last_sample) {
        for (int j = 0; j < n; j++) u->last_sample[j] = this_sample[j];
    } else {
        for (int j = 0; j < n; j++) u->last_sample[j] = this_sample[j];
        u->has_last_sample = 1;
    }

    hd_unipc_predict_cpu(s, u, this_sample, n, this_order, prev_sample, scratch);

    if (u->lower_order_nums < HD_UNIPC_ORDER) u->lower_order_nums++;
    u->step_index++;
    return HD_OK;
}
