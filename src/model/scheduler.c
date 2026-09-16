/*
 * M1.5 native scheduler implementation (see scheduler.h).
 *
 * The step() arithmetic is delegated to the reference CUDA kernels in
 * src/cuda/sched.cu (fp32, torch operation order, contract section 7).
 */

#include "scheduler.h"

#include <string.h>

#include "cuda.h"

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