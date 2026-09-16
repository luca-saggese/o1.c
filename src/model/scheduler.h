#ifndef HD_SCHEDULER_H
#define HD_SCHEDULER_H

/*
 * M1.5 native scheduler (FlashFlowMatchEulerDiscreteScheduler parity).
 *
 * Reproduces the oracle scheduler step() arithmetic in fp32 with the frozen
 * noise contract (docs/M1_NUMERICAL_CONTRACT.md section 7): the native
 * scheduler consumes the golden post-clamp noise tensor verbatim, so the
 * x_pred -> z_next chain divergence from the golden is attributable only to
 * native forward/embedding drift, never to RNG mismatch.
 *
 *   denoised = z - model_output * sigma
 *   z_next   = (sigma_next * noise) * s_noise + (1 - sigma_next) * denoised
 *
 * sigma/sigma_next come from the manifest schedule (numpy fp64 -> fp32,
 * matching build_scheduler). All arithmetic fp32; z_next cast to bf16 with
 * round-to-nearest-even (torch .to(bf16) semantics).
 */

#include <stdint.h>

#include "hidream.h"

#define HD_SCHED_MAX_STEPS 32

typedef struct {
    float sigmas[HD_SCHED_MAX_STEPS]; /* manifest schedule (fp64->fp32) */
    int num_steps;                    /* number of sigmas (incl. trailing 0) */
    int step_index;                   /* current sigma index */
    float noise_clip_std;             /* manifest noise_clip_std (8.0) */
} hd_scheduler;

/*
 * Initialize from the manifest sigmas array (num_steps entries, the last
 * being the trailing 0.0). Copies the schedule; no allocation.
 */
hd_status hd_scheduler_init(hd_scheduler *s, const float *sigmas,
                            int num_steps, float noise_clip_std);

/*
 * One denoising step. Consumes the frozen post-clamp noise verbatim.
 *
 *   z_bf16          [n] bf16 device  current sample (z_prev)
 *   model_output    [n] fp32 device  -v_guided (folded vcond)
 *   noise_post_clamp[n] fp32 device  frozen post-clamp noise
 *   s_noise         fp32             noise_scale_schedule[step_index]
 *   z_next_bf16     [n] bf16 device  output sample
 *   scratch         [3*n] fp32 device scratch (z_f32, denoised, z_next_f32)
 *
 * Advances step_index. Fails closed if step_index >= num_steps-1.
 */
hd_status hd_scheduler_step(hd_scheduler *s,
                            const void *z_bf16, const float *model_output,
                            const float *noise_post_clamp, float s_noise,
                            void *z_next_bf16, int n, float *scratch);

/* Current sigma (sigmas[step_index]) for the caller's v_cond computation. */
float hd_scheduler_sigma(const hd_scheduler *s);

/*
 * Production sigma derivation for the Dev recipe: sigmas = DEFAULT_TIMESTEPS/1000
 * plus a trailing 0.0 (pipeline.py build_scheduler with explicit timesteps_list).
 * Fills `s` and returns the number of sigmas written (29 for Dev).
 */
int hd_scheduler_derive_dev(hd_scheduler *s, float noise_clip_std);

#endif /* HD_SCHEDULER_H */