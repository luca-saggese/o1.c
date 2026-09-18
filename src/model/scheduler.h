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

#define HD_SCHED_MAX_STEPS 64 /* Full/Base FlowUniPC needs 50+1=51 sigmas */

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

/*
 * Generic "flash" (Dev) sigma derivation matching
 * FlashFlowMatchEulerDiscreteScheduler.set_timesteps (python/models/flash_scheduler.py
 * lines 195-245). The schedule is a linear ramp in *timestep* space from
 * sigma_max to sigma_min (both already shift-scaled at init), divided by
 * num_train_timesteps (1000), then the shift is applied again, and a terminal
 * 0.0 is appended. `shift` is the shift factor. Returns num_steps+1 sigmas.
 */
int hd_scheduler_derive_flash(hd_scheduler *s, int steps, float shift,
                              float noise_clip_std);

/*
 * "flow_match" (Dev-edit) sigma derivation matching
 * FlowMatchEulerDiscreteScheduler.set_timesteps. Identical arithmetic to the
 * flash derivation (same shift formula); kept separate for clarity. Returns
 * num_steps+1 sigmas.
 */
int hd_scheduler_derive_flow_match(hd_scheduler *s, int steps, float shift,
                                   float noise_clip_std);

/*
 * "default" (Full/Base) sigma derivation matching
 * FlowUniPCMultistepScheduler.set_timesteps (python/models/fm_solvers_unipc.py
 * lines 166-215). Unlike flash/flow_match, the ramp is linear in *sigma*
 * space from sigma_max to sigma_min over num_inference_steps+1 points,
 * dropped to the first num_inference_steps, then shift applied, then a
 * terminal 0.0 appended. Returns num_steps+1 sigmas.
 */
int hd_scheduler_derive_default(hd_scheduler *s, int steps, float shift,
                                float noise_clip_std);

/*
 * noise_scale_schedule value for step index i (pipeline oracle):
 *   [start + (end-start)*i/(num_steps-1) for i in range(num_steps)]  if num_steps>1
 *   [start]                                                          otherwise
 * num_steps is the number of inference steps (not sigmas). Returns the s_noise
 * for the given step index.
 */
float hd_scheduler_noise_scale(const hd_scheduler *s, int step_index,
                               float noise_scale_start, float noise_scale_end);

/* ------------------------------------------------------------------ */
/* FlowUniPC multistep solver (Full/Base default path)                */
/* ------------------------------------------------------------------ */

#define HD_UNIPC_ORDER 2 /* solver_order for FlowUniPCMultistepScheduler */

/*
 * Persistent UniPC solver state. model_outputs is a ring buffer of
 * solver_order entries (newest at index ORDER-1); each entry is a caller
 * scratch buffer of length n. last_sample is also caller scratch (length n).
 * Reset by zeroing the struct and re-pointing the buffers before the first
 * step.
 */
typedef struct {
    float *model_outputs[HD_UNIPC_ORDER]; /* ring, newest at index ORDER-1 */
    float *last_sample;                   /* pre-predictor sample (scratch) */
    int has_last_sample;
    int this_order;         /* effective UniC order for the current step */
    int lower_order_nums;   /* warmup counter for multistep */
    int step_index;         /* current step (0-based) */
} hd_scheduler_unipc;

/*
 * One UniPC multistep step for the "default" FlowUniPCMultistepScheduler
 * (python/models/fm_solvers_unipc.py, solver_order=2, predict_x0=True,
 * solver_type="bh2", lower_order_final=True). Implements the convert ->
 * (optional) UniC corrector -> history shift -> UniP predictor chain in fp32
 * matching the oracle's torch operation order. The scalar coefficient
 * derivation runs on the host (hd_scheduler_unipc_plan); the pointwise
 * math is executed by the CUDA kernels (hd_unipc_*).
 */

/*
 * Per-step UniPC coefficient plan derived on the host from the sigma schedule
 * exactly like the oracle (flow prediction, predict_x0, solver bh2).
 * Computed for the current step index si; the caller supplies the scheduler's
 * persistent `this_order` (last step's predictor order) used by the corrector.
 */
typedef struct {
    int corr_order;   /* effective order for the corrector (this_order from last step; 0 -> none) */
    int pred_order;   /* effective order for the predictor  (1 during warmup, 2 after) */
    int next_this_order;  /* this_order to persist for the next step's corrector */
    int next_lower_order; /* lower_order_nums to persist (incremented up to solver_order) */

    /* corrector coefficients */
    double c_sig_t, c_sig_s0, c_alpha_t;   /* sigma_t=sigmas[si], sigma_s0=sigmas[si-1] */
    double c_h_phi_1, c_B_h;
    double c_rhos0, c_rhos1, c_inv_rks0;   /* rhos_c and 1/rks[0] (order 2) */

    /* predictor coefficients */
    double p_sig_t, p_sig_s0, p_alpha_t;   /* sigma_t=sigmas[si+1], sigma_s0=sigmas[si] */
    double p_h_phi_1, p_B_h;
    double p_rhos_p, p_inv_rks0;           /* rhos_p (0.5 for order 2) and 1/rks[0] */
} hd_unipc_plan;

/*
 * Compute the UniPC per-step coefficient plan for step index `si`.
 *   this_order   scheduler->this_order as persisted by the last step
 *                (used by the corrector; 0 on the first step -> no corrector)
 * Fills `plan` including the predictor order and the order/counter to persist.
 */
void hd_scheduler_unipc_plan(const hd_scheduler *s, int si, int this_order,
                             int lower_order_nums, hd_unipc_plan *plan);

/* Legacy unified CPU step (predictor+corrector) — still used by scheduler_matrix.c. */
hd_status hd_scheduler_unipc_step(const hd_scheduler *s, hd_scheduler_unipc *u,
                                  const float *mo, const float *sample,
                                  float *prev, int n,
                                  float *scratch);

#endif /* HD_SCHEDULER_H */