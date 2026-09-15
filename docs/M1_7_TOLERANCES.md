# M1.7 — Full Dev Generation: Quantitative Tolerances and Comparison Protocol

**Status:** FROZEN (defined before the V5 gate runs, per MILESTONES.md §M1.7)

**Milestone:** M1.7 — M1 closure: full Dev inference
**Validation level:** V5 (one oracle run + one native run, non-iterative)
**Commit subject (gate):** `test(m1): validate full dev inference against frozen oracle`

---

## 1. Scope

This document defines, **before** the gate executes, the quantitative tensor
tolerances and perceptual/image metrics for the single controlled full 28-step
Dev generation at the validation resolution (64×64, 4 image tokens, seq 23)
with the canonical prompt/seed:

- prompt: `a red fox sits under a cherry blossom tree`
- seed: `42`
- guidance_scale: `1.0` (single-sample path, no `v_uncond` term)
- scheduler: manifest `flash` Euler, 28 steps, timesteps
  `[999, 987, 974, ..., 8]`, 29 sigmas (last `0.0`), `s_noise = 8.0`,
  `noise_clip_std = 8.0`, `t_eps = 0.001`.

The native run replays the frozen per-step post-clamp noise verbatim
(contract §7): any z-chain divergence is attributable to native forward drift,
never to RNG/scheduler mismatch.

## 2. Reference measurements (input to this spec)

All from the frozen oracle / prior milestones (no new runs):

| Source | Quantity | Value |
|--------|----------|-------|
| M1.5 golden (V4, 3 steps) | step model_output NRMSE / cos | 0.0032 / 0.99999 |
| M1.5 golden (V4, 3 steps) | step z_next NRMSE / cos | 0.00043 / 0.9999999 |
| M1.4 golden (V3, 1 forward) | complete_output envelope | NRMSE ≤ 0.18, cos ≥ 0.99 |
| contract §4 class E | complete forward | NRMSE ≤ 2e-2, cos ≥ 0.995 |
| contract §4 class D | attention / block | NRMSE ≤ 1e-2, cos ≥ 0.999 |

## 3. Tensor tolerances (per-step, native vs golden)

### 3.1 Per-step scalar/metadata assertions (exact)

The native run must consume the golden per-step scalars and reproduce them
structurally (byte exact, class A):

- `stepXX_scheduler_timestep` f32[1] — scheduler time (999, 987, …)
- `stepXX_sigma` f32[1] — `clamp_min(t_eps)` of `step_t/1000`
- `stepXX_model_timestep` f32[1] — `1 - step_t/1000` (model domain)
- `stepXX_timestep_embedder_input` f32[1] — `model_timestep * 1000`
- `stepXX_noise_std` f32[1], `stepXX_clip_val` f32[1] — frozen noise params

These are **inputs** to the native run; the test asserts the native reading
path selects the same values (guards against manifest/metadata drift).

### 3.2 Per-step z_next (the 28-step chain)

Chain invariant: `step0 z_prev == M1.4 vinputs` (byte-equal, proven).

Tolerance (native `z_next` vs golden `z_next`, bf16→f32 upcast):

| step range | NRMSE ≤ | cos ≥ |
|-----------|---------|-------|
| steps 0–15 | 5e-3  | 0.9999 |
| steps 16–23 | 1e-2 | 0.9995 |
| steps 24–27 | 2.5e-2 | 0.999 |

The staircase mirrors the measured per-step forward drift (M1.5 step
model_output NRMSE ≈ 3.2e-3) compounded through the decaying `(1 − σ_next)`
contraction. The model-based envelope (spec appendix) predicts final z NRMSE
≈ 2.6e-2; the gate limit 2.5e-2 on the final step is set at the predicted
value with the relaxation derived from the arithmetic contraction.

Additionally, every step must satisfy, on the z_next tensor:

- `max_abs_err` ≤ 0.05 (bf16 quantization floor is 2^-8 ≈ 0.0039 per value)
- no NaN/Inf in native output

### 3.3 Per-step model_output (diagnostic, informational)

`model_output` is reconstructed natively as `(z − x_pred_masked)/σ` and
compared for diagnosis only (not a gate stop condition on its own; the z_next
chain is the gate). Informational envelope: NRMSE ≤ 5e-2, cos ≥ 0.995 per step
(measured single-step 3.2e-3, widened ×15 for the input-amplified tail).

## 4. Final image comparison (the "agreed final comparison")

The final denoised `z` (step 27 output, bf16) is decoded to a 64×64 RGB image
with the oracle formula `img = (z+1)/2` rearranged
`B (H W) (C p1 p2) -> B C (H p1) (W p2)` with `p1=p2=32`, `H=W=2`:

```c
/* native decode: z bf16[4,3072] -> uint8[64,64,3] */
for tok in 0..3:  for c in 0..3:  for (p1,p2) in 0..32:
    dst[tok/2*32+p1][tok%2*32+p2][c] =
        clamp(round((bf16_to_f32(z[tok][c*1024+p1*32+p2]) + 1)/2 * 255), 0, 255)
```

Note: `einops.rearrange` with `p1=32, p2=32` and `H=W=2` maps token `t` (0..3)
to the spatial block `(t/2, t%2)` (block offset `t/2*32`, `t%2*32`), inner
offset `(p1,p2)`; channel `c` to `(C p1 p2)` stride `1024`. Verified against
`einops.rearrange` (max diff 0.0).

### 4.1 Image metrics (golden decode vs native decode)

| metric | formula | gate limit |
|--------|---------|-----------|
| pixel MAE | mean(|golden_px − native_px|) over 64×64×3 | ≤ 4.0 (0–255 scale) |
| PSNR | 20·log10(255 / RMSE_px) | ≥ 32 dB |
| SSIM (3×3) | scipy-free: computed from local mean/var/cov over 8×8 windows, sliding step 4, Gaussian-free box kernels | ≥ 0.90 |

SSIM implementation is self-contained in the test (no skimage dependency):
per 8×8 window, `ssim = (2·μx·μy + C1)(2·σxy + C2)/((μx²+μy²+C1)(σx²+σy²+C2))`,
`C1=(0.01·255)²`, `C2=(0.03·255)²`, averaged over all windows.

Rationale: pixel MAE ≈ 1.0 predicted from the z-chain envelope (≈0.026 final
NRMSE on latents of RMS ≈ 0.3). Gate limit 4.0 keeps ≥ 4× headroom while still
rejecting a broken chain (a divergent tail yields MAE ≫ 4).

### 4.2 Perceptual note

SSIM ≥ 0.90 at 64×64 is consistent with a visually near-identical image
(per-structure correlation preserved). The gate does not require the user to
visually inspect; metrics are the "agreed final comparison."

## 5. Run budget

- 1 fresh full-model oracle load, 1 V5 28-step oracle run → frozen golden
  (28×16 tensors, single `.bin` + JSON manifest + sha256, git-ignored under
  `artifacts/`).
- 1 native run driving `hd_forward` 28× with frozen per-step
  model_timestep/noise; no Python in the native path.
- No iterative full-run debugging (MILESTONES M1.7). Any failure is analyzed
  from the persisted per-step tensors, not by re-running the oracle.

## 6. Stop conditions

Stop and record evidence before proceeding if:

- the single oracle run fails or produces non-finite tensors;
- the golden capture's manifest `model_timestep` domain check fails
  (timestep-domain regression);
- the native run cannot establish the step0 invariant
  (`step0 z_prev == M1.4 vinputs` byte-equal);
- the final step semantics diverge from the oracle's
  `step_index < num_inference_steps` noise-skip rule (final σ_next = 0.0 →
  `z_next = denoised`, native must replicate via the manifest's σ[28] = 0.0
  path, not by hardcoding a step-index skip).

## Appendix A — error-envelope model (informative)

Error recursion per step (z-chain NRMSE, ε on z, δ on model_output):

```
ε_{k+1} = (1 − σ_{k+1}) · (ε_k + σ_k · δ_k),   δ_k = 2·ε_k + β
β = 3.0e-3  (per-step forward drift on masked rows, M1.5 measured)
```

| step | σ_next | ε (predicted) |
|------|--------|---------------|
| 0  | 0.987 | 3.9e-5 |
| 8  | 0.857 | 5.6e-4 |
| 16 | 0.640 | 2.4e-3 |
| 23 | 0.278 | 1.3e-2 |
| 26 | 0.008 | 2.5e-2 |
| 27 | 0.000 | 2.6e-2 (final) |

Predicted image impact: pixel MAE ≈ 1.0, PSNR ≈ 48 dB. Gate limits in §4.1
keep ≥ 4× headroom over the prediction while still detecting a broken chain.
